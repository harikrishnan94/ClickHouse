#include <Interpreters/HashJoin/KeyGetter.h>

#if (defined(__aarch64__) && defined(__ARM_NEON)) || defined(__AVX2__)
#    include <base/unaligned.h>

#    include <algorithm>
#    include <array>
#    include <bit>
#    include <limits>
#    include <type_traits>
#    include <utility>

#    if defined(__aarch64__)
#        include <arm_neon.h>
#    endif
#endif


namespace DB
{

#if (defined(__aarch64__) && defined(__ARM_NEON)) || defined(__AVX2__)

namespace
{

static_assert(std::endian::native == std::endian::little);

/// The keys of a range are built 256 rows at a time, so that the row numbers and the buffers fit on the stack.
constexpr size_t chunk_rows = 256;

/// A row is hashed when it has at least 2 full words.
static_assert(PackedStringJoinKey::max_inline_size == 2 * sizeof(UInt64) - 1);

/// Rows are sorted by their number of 8-byte words: one bucket per word count below `exact_buckets`, then 8 buckets per
/// doubling of the word count, up to `length_buckets - 1`.
constexpr size_t exact_buckets = 64;
constexpr size_t length_buckets = 256;

/// The initial state of the unkeyed `sipHash128Padded`.
constexpr UInt64 sip_v0 = 0x736f6d6570736575ULL;
constexpr UInt64 sip_v1 = 0x646f72616e646f6dULL;
constexpr UInt64 sip_v2 = 0x6c7967656e657261ULL;
constexpr UInt64 sip_v3 = 0x7465646279746573ULL;

/// A group is `group_vectors` vectors of `lanes` strings, one string per 64-bit lane, plus `group_scalars` strings in
/// general registers. The operations of one vector form a chain, so two vectors hide each other's latency. On aarch64 a
/// vector is a NEON register of 2 strings. The vector code leaves the integer pipes idle, and 4 scalar keys use them.
/// With AVX2 a vector holds 4 strings, and no strings go in general registers. On Ice Lake the vector operations already
/// fill ports 0, 1 and 5, and the integer operations need the same ports.
#    if defined(__AVX2__)
constexpr size_t lanes = 4;
constexpr size_t group_scalars = 0;
#    else
constexpr size_t lanes = 2;
constexpr size_t group_scalars = 4;
#    endif
constexpr size_t group_vectors = 2;
constexpr size_t group_size = lanes * group_vectors + group_scalars;

/// Clang vector extensions: an operation on a whole vector compiles to one NEON or AVX2 instruction.
using UInt64Vec = UInt64 __attribute__((vector_size(8 * lanes)));
using Int64Vec = Int64 __attribute__((vector_size(8 * lanes)));
using UInt32Vec = UInt32 __attribute__((vector_size(8 * lanes)));
using UInt8Vec = UInt8 __attribute__((vector_size(8 * lanes)));
using UInt64x2 = UInt64 __attribute__((vector_size(16)));
using UInt8x16 = UInt8 __attribute__((vector_size(16)));
using Int8x16 = Int8 __attribute__((vector_size(16)));

template <size_t count, typename F>
ALWAYS_INLINE void unrolledFor(F && f)
{
    [&]<size_t... i>(std::index_sequence<i...>) ALWAYS_INLINE
    { (f(std::integral_constant<size_t, i>{}), ...); }(std::make_index_sequence<count>{});
}

/// Lane `i` = `f(i)`.
template <typename F>
ALWAYS_INLINE UInt64Vec makeLanes(F && f)
{
    return [&]<size_t... i>(std::index_sequence<i...>) ALWAYS_INLINE { return UInt64Vec{f(i)...}; }(std::make_index_sequence<lanes>{});
}

/// Lane `i` = the 8 bytes at `address(i)`. Each lane is loaded as a vector of one lane, and the vectors are joined as
/// `vcombine_u64` joins them. Building the vector from 64-bit integers would load them into general registers first.
template <typename F>
ALWAYS_INLINE UInt64Vec loadLanes(F && address)
{
    using UInt64x1 = UInt64 __attribute__((vector_size(8)));
    const auto pair = [&](size_t i) ALWAYS_INLINE
    { return __builtin_shufflevector(unalignedLoad<UInt64x1>(address(i)), unalignedLoad<UInt64x1>(address(i + 1)), 0, 1); };
    static_assert(lanes == 2 || lanes == 4);
    if constexpr (lanes == 2)
        return pair(0);
    else
        return __builtin_shufflevector(pair(0), pair(2), 0, 1, 2, 3);
}

/// Element `i` = element `index(i)` of `x`.
template <auto index, typename Vec>
ALWAYS_INLINE Vec permute(Vec x)
{
    return [&]<size_t... i>(std::index_sequence<i...>) ALWAYS_INLINE
    { return __builtin_shufflevector(x, x, index(i)...); }(std::make_index_sequence<sizeof(Vec) / sizeof(x[0])>{});
}

template <int bits>
ALWAYS_INLINE UInt64Vec rotl(UInt64Vec x)
{
#    if defined(__aarch64__)
    /// SHL and SRI. Clang lowers the portable rotate to SHL and USRA, and USRA makes the chain of a round a cycle longer.
    return std::bit_cast<UInt64Vec>(vsriq_n_u64(std::bit_cast<uint64x2_t>(x << bits), std::bit_cast<uint64x2_t>(x), 64 - bits));
#    else
    return (x << bits) | (x >> (64 - bits));
#    endif
}

/// REV64 or VPSHUFD.
ALWAYS_INLINE UInt64Vec rotl32(UInt64Vec x)
{
    return std::bit_cast<UInt64Vec>(permute<[](size_t i) { return i ^ 1; }>(std::bit_cast<UInt32Vec>(x)));
}

/// One TBL or VPSHUFB instead of two shifts that depend on each other: byte i of a lane gets byte (i - 2) mod 8.
ALWAYS_INLINE UInt64Vec rotl16(UInt64Vec x)
{
    return std::bit_cast<UInt64Vec>(permute<[](size_t i) { return (i & ~size_t(7)) | ((i + 6) & 7); }>(std::bit_cast<UInt8Vec>(x)));
}

/// The SipHash state of `lanes` strings, one per lane. `offsets` as in `buildChunk`.
struct VectorKeys
{
    UInt64Vec v0;
    UInt64Vec v1;
    UInt64Vec v2;
    UInt64Vec v3;
    const char * data[lanes];
    size_t size[lanes];
    /// The full words of the shortest and of the longest string.
    size_t common_words;
    size_t longer_words;

    ALWAYS_INLINE void init(const char * chars, const UInt64 * offsets, const UInt32 * rows)
    {
        v0 = UInt64Vec{} + sip_v0;
        v1 = UInt64Vec{} + sip_v1;
        v2 = UInt64Vec{} + sip_v2;
        v3 = UInt64Vec{} + sip_v3;
        common_words = std::numeric_limits<size_t>::max();
        longer_words = 0;
        unrolledFor<lanes>(
            [&](auto i) ALWAYS_INLINE
            {
                const size_t row = rows[i];
                data[i] = chars + offsets[row];
                size[i] = offsets[row + 1] - offsets[row];
                common_words = std::min(common_words, size[i] / 8);
                longer_words = std::max(longer_words, size[i] / 8);
            });
    }

    /// `SIPROUND` on all lanes.
    ALWAYS_INLINE void round()
    {
        const UInt64Vec a0 = v0 + v1;
        const UInt64Vec b1 = rotl<13>(v1) ^ a0;
        const UInt64Vec a2 = v2 + v3;
        const UInt64Vec b3 = rotl16(v3) ^ a2;
        v0 = rotl32(a0) + b3;
        v3 = rotl<21>(b3) ^ v0;
        const UInt64Vec d2 = a2 + b1;
        v1 = rotl<17>(b1) ^ d2;
        v2 = rotl32(d2);
    }

    ALWAYS_INLINE void compress(UInt64Vec words)
    {
        v3 ^= words;
        round();
        round();
        v0 ^= words;
    }

    ALWAYS_INLINE UInt64Vec word(size_t k) const
    {
        return loadLanes([&](size_t i) ALWAYS_INLINE { return data[i] + 8 * k; });
    }

    ALWAYS_INLINE UInt64Vec sizes() const
    {
        return makeLanes([&](size_t i) ALWAYS_INLINE { return UInt64(size[i]); });
    }

    /// The lanes where `active` is 0 get the state `saved` back.
    ALWAYS_INLINE void keepState(Int64Vec active, const std::array<UInt64Vec, 4> & saved)
    {
        v0 = active ? v0 : saved[0];
        v1 = active ? v1 : saved[1];
        v2 = active ? v2 : saved[2];
        v3 = active ? v3 : saved[3];
    }

    /// The full words of the longer strings past `common_words`. A lane past the end of its string runs on the words of
    /// another string and then gets its state back. With 2 lanes only the longer string has such words: they are loaded
    /// once into both lanes, and the other lane gets its state back at the end. With 4 lanes several strings can have
    /// them: a lane loads its own string while it has words. Past its end it loads the longest string and gets its state
    /// back after each word.
    ALWAYS_INLINE void compressLongerWords()
    {
        if (longer_words == common_words)
            return;
        if constexpr (lanes == 2)
        {
            const std::array saved{v0, v1, v2, v3};
            const char * longer = size[0] > size[1] ? data[0] : data[1];
            for (size_t k = common_words; k < longer_words; ++k)
                compress(UInt64Vec{} + unalignedLoad<UInt64>(longer + 8 * k));
            const Int64Vec lane_sizes = std::bit_cast<Int64Vec>(sizes());
            keepState(lane_sizes > permute<[](size_t i) { return i ^ 1; }>(lane_sizes), saved);
        }
        else
        {
            const char * longest = data[0];
            unrolledFor<lanes>(
                [&](auto i) ALWAYS_INLINE
                {
                    if (size[i] / 8 == longer_words)
                        longest = data[i];
                });
            const Int64Vec words = std::bit_cast<Int64Vec>(sizes() >> 3);
            for (size_t k = common_words; k < longer_words; ++k)
            {
                const std::array saved{v0, v1, v2, v3};
                compress(loadLanes([&](size_t i) ALWAYS_INLINE { return (k < size[i] / 8 ? data[i] : longest) + 8 * k; }));
                keepState(words > static_cast<Int64>(k), saved);
            }
        }
    }

    /// The last partial word with the size in the high byte. The strings have at least 16 bytes, so it is the last 8
    /// bytes of the string shifted right by 64 - 8 * (size % 8). When no bytes remain, that shift gives 0. It is split in
    /// two, because a vector shift by 64 is undefined.
    ALWAYS_INLINE UInt64Vec lastWord() const
    {
        const UInt64Vec lane_sizes = sizes();
        const UInt64Vec words = loadLanes([&](size_t i) ALWAYS_INLINE { return data[i] + size[i] - 8; });
        return ((words >> 8) >> (56 - ((lane_sizes & 7) << 3))) | (lane_sizes << 56);
    }

    /// The keys of the lanes to `keys[rows[i]]`. `even` gets the keys of the even lanes and `odd` of the odd lanes, one
    /// per 16 bytes, as ZIP1 and ZIP2 or VPUNPCKLQDQ and VPUNPCKHQDQ do.
    ALWAYS_INLINE void store(const UInt32 * rows, UInt128 * keys) const
    {
        const UInt64Vec low = v0 ^ v1;
        const UInt64Vec high = (v2 ^ v3) | PackedStringJoinKey::hashed_flag;
        const auto [even, odd] = [&]<size_t... i>(std::index_sequence<i...>) ALWAYS_INLINE
        {
            return std::pair{
                __builtin_shufflevector(low, high, (i % 2 ? lanes + i - 1 : i)...),
                __builtin_shufflevector(low, high, (i % 2 ? lanes + i : i + 1)...)};
        }(std::make_index_sequence<lanes>{});
        unrolledFor<lanes>(
            [&](auto i) ALWAYS_INLINE
            {
                const UInt64Vec & pairs = i % 2 ? odd : even;
                unalignedStore<UInt64x2>(keys + rows[i], __builtin_shufflevector(pairs, pairs, i / 2 * 2, i / 2 * 2 + 1));
            });
    }
};

/// The SipHash state of one string in general registers. With AVX2 there are no scalar keys, and the functions are
/// unused.
struct ScalarKey
{
    UInt64 v0;
    UInt64 v1;
    UInt64 v2;
    UInt64 v3;
    const char * data;
    size_t size;
    size_t words;

    [[maybe_unused]] ALWAYS_INLINE void init(const char * chars, const UInt64 * offsets, size_t row)
    {
        v0 = sip_v0;
        v1 = sip_v1;
        v2 = sip_v2;
        v3 = sip_v3;
        data = chars + offsets[row];
        size = offsets[row + 1] - offsets[row];
        words = size / 8;
    }

    [[maybe_unused]] ALWAYS_INLINE void round() { SIPROUND; }

    [[maybe_unused]] ALWAYS_INLINE void compressWord(size_t k)
    {
        const UInt64 word = unalignedLoad<UInt64>(data + 8 * k);
        v3 ^= word;
        SIPROUND;
        SIPROUND;
        v0 ^= word;
    }

    [[maybe_unused]] ALWAYS_INLINE UInt64 lastWord() const
    {
        const size_t remaining = size - 8 * words;
        const UInt64 last = remaining ? unalignedLoad<UInt64>(data + 8 * words) & ((1ULL << (remaining * 8)) - 1) : 0;
        return last | (static_cast<UInt64>(static_cast<uint8_t>(size)) << 56);
    }
};

/// The keys of the rows `rows[0..group_size)` of the range. All of them run their common full words in one loop, then
/// each the rest of its words. The last words and the final rounds of all of them are interleaved round by round.
ALWAYS_INLINE void hashGroup(const char * chars, const UInt64 * offsets, const UInt32 * rows, UInt128 * keys)
{
    VectorKeys vectors[group_vectors];
    std::array<ScalarKey, group_scalars> scalars; // NOLINT(cppcoreguidelines-pro-type-member-init,hicpp-member-init) - set by `init`
    size_t common_words = std::numeric_limits<size_t>::max();
    unrolledFor<group_vectors>(
        [&](auto j) ALWAYS_INLINE
        {
            vectors[j].init(chars, offsets, rows + lanes * j);
            common_words = std::min(common_words, vectors[j].common_words);
        });
    unrolledFor<group_scalars>(
        [&](auto t) ALWAYS_INLINE
        {
            scalars[t].init(chars, offsets, rows[lanes * group_vectors + t]);
            common_words = std::min(common_words, scalars[t].words);
        });

    for (size_t k = 0; k < common_words; ++k)
    {
        unrolledFor<group_vectors>([&](auto j) ALWAYS_INLINE { vectors[j].compress(vectors[j].word(k)); });
        unrolledFor<group_scalars>([&](auto t) ALWAYS_INLINE { scalars[t].compressWord(k); });
    }
    unrolledFor<group_vectors>(
        [&](auto j) ALWAYS_INLINE
        {
            for (size_t k = common_words; k < vectors[j].common_words; ++k)
                vectors[j].compress(vectors[j].word(k));
            vectors[j].compressLongerWords();
        });
    unrolledFor<group_scalars>(
        [&](auto t) ALWAYS_INLINE
        {
            for (size_t k = common_words; k < scalars[t].words; ++k)
                scalars[t].compressWord(k);
        });

    UInt64Vec vector_last[group_vectors];
    std::array<UInt64, group_scalars> scalar_last; // NOLINT(cppcoreguidelines-pro-type-member-init,hicpp-member-init) - set below
    unrolledFor<group_vectors>(
        [&](auto j) ALWAYS_INLINE
        {
            vector_last[j] = vectors[j].lastWord();
            vectors[j].v3 ^= vector_last[j];
        });
    unrolledFor<group_scalars>(
        [&](auto t) ALWAYS_INLINE
        {
            scalar_last[t] = scalars[t].lastWord();
            scalars[t].v3 ^= scalar_last[t];
        });
    for (int i = 0; i < 2; ++i)
    {
        unrolledFor<group_vectors>([&](auto j) ALWAYS_INLINE { vectors[j].round(); });
        unrolledFor<group_scalars>([&](auto t) ALWAYS_INLINE { scalars[t].round(); });
    }
    unrolledFor<group_vectors>(
        [&](auto j) ALWAYS_INLINE
        {
            vectors[j].v0 ^= vector_last[j];
            vectors[j].v2 ^= 0xff;
        });
    unrolledFor<group_scalars>(
        [&](auto t) ALWAYS_INLINE
        {
            scalars[t].v0 ^= scalar_last[t];
            scalars[t].v2 ^= 0xff;
        });
    for (int i = 0; i < 4; ++i)
    {
        unrolledFor<group_vectors>([&](auto j) ALWAYS_INLINE { vectors[j].round(); });
        unrolledFor<group_scalars>([&](auto t) ALWAYS_INLINE { scalars[t].round(); });
    }

    unrolledFor<group_vectors>([&](auto j) ALWAYS_INLINE { vectors[j].store(rows + lanes * j, keys); });
    unrolledFor<group_scalars>(
        [&](auto t) ALWAYS_INLINE
        {
            UInt128 & key = keys[rows[lanes * group_vectors + t]];
            key.items[0] = scalars[t].v0 ^ scalars[t].v1;
            key.items[1] = (scalars[t].v2 ^ scalars[t].v3) | PackedStringJoinKey::hashed_flag;
        });
}

/// `buckets[r]` = min(words of row r, `exact_buckets`) for the rows [0, num_rows) of the range, 16 rows per step. The
/// words narrow to bytes in saturating steps, one UQXTN each on NEON. The 128-byte vectors stay in this function: passing
/// them to another function would change its ABI.
ALWAYS_INLINE void lengthBuckets(const UInt64 * offsets, size_t num_rows, uint8_t * buckets)
{
    using UInt64x16 = UInt64 __attribute__((vector_size(128)));
    using UInt32x16 = UInt32 __attribute__((vector_size(64)));
    using UInt16x16 = UInt16 __attribute__((vector_size(32)));
    size_t r = 0;
    for (; r + 16 <= num_rows; r += 16)
    {
        UInt64x16 ends;
        UInt64x16 begins;
        memcpy(&ends, offsets + r + 1, sizeof(ends));
        memcpy(&begins, offsets + r, sizeof(begins));
        const UInt64x16 words = (ends - begins) >> 3;
        const auto words32 = __builtin_convertvector(__builtin_elementwise_min(words, UInt64x16{} + 0xFFFFFFFF), UInt32x16);
        const auto words16 = __builtin_convertvector(__builtin_elementwise_min(words32, UInt32x16{} + 0xFFFF), UInt16x16);
        const auto words8 = __builtin_convertvector(__builtin_elementwise_min(words16, UInt16x16{} + 0xFF), UInt8x16);
        unalignedStore<UInt8x16>(buckets + r, __builtin_elementwise_min(words8, UInt8x16{} + UInt8(exact_buckets)));
    }
    for (; r < num_rows; ++r)
        buckets[r] = static_cast<uint8_t>(std::min<UInt64>((offsets[r + 1] - offsets[r]) / 8, exact_buckets));
}

/// Whether the rows can be hashed in their order: a vector runs as many words as the longest of its `lanes` rows, so
/// rows of different lengths in one vector waste the difference. It also gives the smallest and the largest bucket.
/// `buckets` is readable up to a multiple of 32 past `num_rows`, filled with its last bucket.
ALWAYS_INLINE bool isBalanced(const uint8_t * buckets, size_t num_rows, unsigned & min_bucket, unsigned & max_bucket)
{
    using UInt16Widened = UInt16 __attribute__((vector_size(16 * lanes)));
    static_assert(std::has_single_bit(lanes));
    UInt16Widened wasted{};
    UInt16Widened total{};
    UInt8Vec largest{};
    UInt8Vec smallest = ~UInt8Vec{};
    for (size_t r = 0; r < num_rows; r += sizeof(UInt8Vec))
    {
        const UInt8Vec x = unalignedLoad<UInt8Vec>(buckets + r);
        /// The largest bucket of the `lanes` rows of each vector, in all of their bytes.
        UInt8Vec group_max = x;
        unrolledFor<std::countr_zero(lanes)>(
            [&](auto level) ALWAYS_INLINE
            {
                constexpr size_t distance = size_t(1) << decltype(level)::value;
                group_max = __builtin_elementwise_max(group_max, permute<[](size_t i) { return i ^ distance; }>(group_max));
            });
        wasted += __builtin_convertvector(group_max - x, UInt16Widened);
        total += __builtin_convertvector(x, UInt16Widened);
        largest = __builtin_elementwise_max(largest, x);
        smallest = __builtin_elementwise_min(smallest, x);
    }
    min_bucket = __builtin_reduce_min(smallest);
    max_bucket = __builtin_reduce_max(largest);
    /// The rows stay in their order when the waste is at most 1/16 of their words plus 3/16 of a word per row. Both sums
    /// include the padding up to a multiple of `sizeof(UInt8Vec)` rows. The waste counts the rows of the scalar keys as
    /// if they were in vectors too. The sums are UInt16: they hold at most `chunk_rows` padded rows of `exact_buckets` words.
    static_assert(chunk_rows % sizeof(UInt8Vec) == 0 && chunk_rows * exact_buckets <= std::numeric_limits<UInt16>::max());
    return UInt64(__builtin_reduce_add(wasted)) * 16 <= UInt64(__builtin_reduce_add(total)) + 3 * num_rows;
}

/// Whether all the rows [0, num_rows) have at most 15 bytes: then their lengths OR to at most 15.
ALWAYS_INLINE bool allShort(const UInt64 * offsets, size_t num_rows)
{
    size_t r = 0;
    for (; r + 16 <= num_rows; r += 16)
    {
        UInt64Vec lengths{};
        for (size_t q = 0; q < 16 / lanes; ++q)
            lengths |= unalignedLoad<UInt64Vec>(offsets + r + lanes * q + 1) - unalignedLoad<UInt64Vec>(offsets + r + lanes * q);
        if (__builtin_reduce_or(lengths & ~UInt64(PackedStringJoinKey::max_inline_size)))
            return false;
    }
    UInt64 lengths = 0;
    for (; r < num_rows; ++r)
        lengths |= offsets[r + 1] - offsets[r];
    return lengths <= PackedStringJoinKey::max_inline_size;
}

/// `UnhashedPackedStringRef::build` of a string of up to 15 bytes into `key`, without branches: the bytes from `size`
/// on are cleared, and byte 15 gets the length shifted left by `SMALL_LEN_SHIFT_IN_BYTE`. It reads 16 bytes. `size`
/// must be at most 15; the byte positions compare as signed bytes.
ALWAYS_INLINE void shortKey(const char * data, size_t size, UInt128 * key)
{
    static constexpr Int8x16 positions = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
    Int8x16 bytes = unalignedLoad<Int8x16>(data);
    bytes &= positions < static_cast<Int8>(size);
    bytes[15] = static_cast<Int8>(size << UnhashedPackedStringRef::SMALL_LEN_SHIFT_IN_BYTE);
    unalignedStore<Int8x16>(key, bytes);
}

/// The bucket of a row of at least `exact_buckets` words: 8 buckets per doubling, by the 3 bits after the leading one.
ALWAYS_INLINE uint8_t longBucket(UInt64 words)
{
    const unsigned log = std::bit_width(words) - 1;
    const UInt64 bucket = exact_buckets + 8 * (log - std::countr_zero(exact_buckets)) + ((words >> (log - 3)) & 7);
    return static_cast<uint8_t>(std::min<UInt64>(bucket, length_buckets - 1));
}

/// `isBalanced` on the real word counts of the rows; the buckets cap them at `exact_buckets`. `words` is readable up to
/// `lanes` past `num_rows`, filled with its last count.
ALWAYS_INLINE bool isBalancedWords(const UInt64 * words, size_t num_rows)
{
    UInt64 wasted = 0;
    UInt64 total = 0;
    for (size_t j = 0; j < num_rows; j += lanes)
    {
        UInt64 longest = 0;
        UInt64 sum = 0;
        for (size_t lane = 0; lane < lanes; ++lane)
        {
            longest = std::max(longest, words[j + lane]);
            sum += words[j + lane];
        }
        wasted += lanes * longest - sum;
        total += sum;
    }
    return wasted * 16 <= total + 3 * num_rows;
}

/// A stable counting sort of `rows` by bucket into `sorted`. Each quarter of the rows counts in its own histogram, so
/// that a run of equal buckets does not make each increment wait for the previous one. The rows of a bucket keep their
/// order, so that a group reads neighbouring strings.
ALWAYS_INLINE void
sortByBucket(const uint8_t * buckets, const UInt32 * rows, size_t num_rows, unsigned min_bucket, unsigned max_bucket, UInt32 * sorted)
{
    const size_t quarter = num_rows / 4;
    UInt16 counts[4][length_buckets];
    for (auto & count : counts)
        for (unsigned b = min_bucket; b <= max_bucket; ++b)
            count[b] = 0;
    for (size_t i = 0; i < quarter; ++i)
        for (size_t q = 0; q < 4; ++q)
            ++counts[q][buckets[q * quarter + i]];
    for (size_t j = 4 * quarter; j < num_rows; ++j)
        ++counts[3][buckets[j]];
    UInt16 position = 0;
    for (unsigned b = min_bucket; b <= max_bucket; ++b)
        for (auto & count : counts)
            position += std::exchange(count[b], position);
    for (size_t i = 0; i < quarter; ++i)
        for (size_t q = 0; q < 4; ++q)
            sorted[counts[q][buckets[q * quarter + i]]++] = rows[q * quarter + i];
    for (size_t j = 4 * quarter; j < num_rows; ++j)
        sorted[counts[3][buckets[j]]++] = rows[j];
}

/// The keys of at most `chunk_rows` rows. `offsets` is that of the first row minus one, so row `r` is
/// [offsets[r], offsets[r + 1]).
void buildChunk(const char * chars, const UInt64 * offsets, size_t num_rows, UInt128 * __restrict keys)
{
    alignas(16) uint8_t row_buckets[chunk_rows + 32];
    alignas(16) uint8_t long_buckets[chunk_rows + 32];
    UInt32 long_rows[chunk_rows];
    UInt32 sorted_rows[chunk_rows];

    if (allShort(offsets, num_rows))
    {
        for (size_t r = 0; r < num_rows; ++r)
            shortKey(chars + offsets[r], offsets[r + 1] - offsets[r], keys + r);
        return;
    }

    lengthBuckets(offsets, num_rows, row_buckets);
    size_t num_long = 0;
    for (size_t r = 0; r < num_rows; ++r)
        num_long += row_buckets[r] >= 2;

    uint8_t * buckets = row_buckets;
    if (num_long == num_rows)
    {
        for (size_t r = 0; r < num_rows; ++r)
            long_rows[r] = static_cast<UInt32>(r);
    }
    else
    {
        /// Every row gets its short key, and the long rows are listed; a long row gets its hash below.
        num_long = 0;
        for (size_t r = 0; r < num_rows; ++r)
        {
            const uint8_t bucket = row_buckets[r];
            long_rows[num_long] = static_cast<UInt32>(r);
            long_buckets[num_long] = bucket;
            num_long += bucket >= 2;
            shortKey(chars + offsets[r], std::min<size_t>(offsets[r + 1] - offsets[r], PackedStringJoinKey::max_inline_size), keys + r);
        }
        buckets = long_buckets;
    }

    for (size_t j = num_long; j < ((num_long + 31) & ~size_t(31)); ++j)
        buckets[j] = buckets[num_long - 1];
    const UInt32 * rows = long_rows;
    unsigned min_bucket = 0;
    unsigned max_bucket = 0;
    bool sort = !isBalanced(buckets, num_long, min_bucket, max_bucket);
    if (max_bucket == exact_buckets && size_t(std::count(buckets, buckets + num_long, exact_buckets)) >= group_size)
    {
        /// At least a group of rows have `exact_buckets` words or more, and their buckets cap their lengths. Rows that the
        /// buckets find balanced are checked again with their real word counts. Before a sort, these long rows get
        /// buckets by length ratio, so that strings of very different lengths do not share a group. Fewer such rows
        /// cannot fill a group, and sorting them would gain nothing.
        if (!sort)
        {
            UInt64 words[chunk_rows + lanes];
            for (size_t j = 0; j < num_long; ++j)
                words[j] = (offsets[long_rows[j] + 1] - offsets[long_rows[j]]) / 8;
            for (size_t j = num_long; j < num_long + lanes; ++j)
                words[j] = words[num_long - 1];
            sort = !isBalancedWords(words, num_long);
        }
        if (sort)
        {
            for (size_t j = 0; j < num_long; ++j)
            {
                const UInt64 words = (offsets[long_rows[j] + 1] - offsets[long_rows[j]]) / 8;
                const uint8_t long_bucket = longBucket(std::max<UInt64>(words, exact_buckets));
                buckets[j] = buckets[j] == exact_buckets ? long_bucket : buckets[j];
                max_bucket = std::max<unsigned>(max_bucket, buckets[j]);
            }
        }
    }
    if (sort)
    {
        sortByBucket(buckets, long_rows, num_long, min_bucket, max_bucket, sorted_rows);
        rows = sorted_rows;
    }

    size_t j = 0;
    for (; j + group_size <= num_long; j += group_size)
        hashGroup(chars, offsets, rows + j, keys);
    for (; j < num_long; ++j)
        keys[rows[j]] = PackedStringJoinKey::build(chars + offsets[rows[j]], offsets[rows[j] + 1] - offsets[rows[j]]);
}

}

#endif

void PackedStringJoinKey::buildRange(
    const char * chars, const IColumn::Offset * offsets, size_t begin, size_t end, UInt128 * __restrict keys)
{
#if (defined(__aarch64__) && defined(__ARM_NEON)) || defined(__AVX2__)
    for (size_t chunk_begin = begin; chunk_begin < end; chunk_begin += chunk_rows)
    {
        const size_t chunk_end = std::min(end, chunk_begin + chunk_rows);
        buildChunk(chars, offsets + chunk_begin - 1, chunk_end - chunk_begin, keys + (chunk_begin - begin));
    }

#    ifndef NDEBUG
    for (size_t i = begin; i < end; ++i)
        chassert(keys[i - begin] == build(chars + offsets[ssize_t(i) - 1], offsets[i] - offsets[ssize_t(i) - 1]));
#    endif
#else
    for (size_t i = begin; i < end; ++i)
        keys[i - begin] = build(chars + offsets[ssize_t(i) - 1], offsets[i] - offsets[ssize_t(i) - 1]);
#endif
}

}
