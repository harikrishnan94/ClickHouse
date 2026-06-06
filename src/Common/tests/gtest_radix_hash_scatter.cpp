#include <gtest/gtest.h>

#include <Common/RadixShuffle/Scatter.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <vector>

using namespace DB;
using namespace DB::RadixShuffle;

namespace
{

/// One 64 B-aligned backing buffer carved into exact-sized, 64 B-aligned per-partition regions —
/// exactly the shape the NT stores require (spec section 4.5). `base(p)` is the partition base, with
/// capacity `roundUpTo64(hist[p] * elem_size)` bytes. Allocated fresh (not reused) and zero-filled.
struct PartitionArena
{
    void * buf = nullptr;
    size_t bytes = 0;
    std::vector<size_t> off; /// byte offset of each partition's base within `buf`

    PartitionArena(const std::vector<size_t> & hist, size_t elem_size)
    {
        off.resize(hist.size());
        size_t o = 0;
        for (size_t p = 0; p < hist.size(); ++p)
        {
            off[p] = o;
            o += roundUpTo64(hist[p] * elem_size);
        }
        bytes = std::max<size_t>(o, LINE_BYTES);
        if (posix_memalign(&buf, LINE_BYTES, bytes) != 0 || buf == nullptr)
            throw std::bad_alloc{};
        std::memset(buf, 0, bytes);
    }

    ~PartitionArena() { std::free(buf); }

    PartitionArena(const PartitionArena &) = delete;
    PartitionArena & operator=(const PartitionArena &) = delete;

    char * base(size_t p) const { return static_cast<char *>(buf) + off[p]; }
};

/// Routing function: `part = (hash >> shift) & mask`.
UInt32 routeOf(UInt32 hash, UInt32 shift, UInt32 mask)
{
    return (hash >> shift) & mask;
}

/// Deterministic per-element byte fill: every byte depends on both the source row `j` and its byte
/// index, so distinct rows produce distinct content and the whole `W`-byte element is exercised. The
/// differential oracle below never decodes this — it compares raw bytes against a scalar reference.
void fillElem(char * dst, size_t j, size_t width)
{
    for (size_t b = 0; b < width; ++b)
        dst[b] = static_cast<char>((j * 1099087573ull + b * 2246822519ull + (j >> ((b % 8) * 8))) & 0xFF);
}

/// Generate `n` random 32-bit hash values whose bottom `total_bits` bits are in `[0, 2^total_bits)`.
/// (For scatter tests where `shift = 0`, routing is `hash & mask`, so only the bottom bits matter.)
std::vector<UInt32> makeHash(size_t n, UInt32 total_bits, uint64_t seed)
{
    std::vector<UInt32> hashes(n);
    std::mt19937_64 rng(seed);
    const UInt32 m = total_bits == 0 ? 0u : ((1u << total_bits) - 1u);
    for (size_t j = 0; j < n; ++j)
        hashes[j] = static_cast<UInt32>(rng() & m);
    return hashes;
}

std::vector<char> makeColumn(size_t n, size_t width)
{
    std::vector<char> col(n * width);
    for (size_t j = 0; j < n; ++j)
        fillElem(col.data() + j * width, j, width);
    return col;
}

std::vector<size_t> histogram(const std::vector<UInt32> & hashes, UInt32 shift, UInt32 mask, size_t partitions)
{
    std::vector<size_t> hist(partitions, 0);
    for (UInt32 h : hashes)
        ++hist[routeOf(h, shift, mask)];
    return hist;
}

/// Ground truth: a plain per-partition scatter preserving arrival (increasing-`j`) order. The kernel
/// under test must reproduce these bytes exactly (conservation + routing + order + content, at once).
std::vector<std::vector<char>> referenceScatter(
    const std::vector<UInt32> & hashes, UInt32 shift, UInt32 mask, size_t n, const char * src, size_t width, size_t partitions)
{
    std::vector<std::vector<char>> out(partitions);
    for (size_t j = 0; j < n; ++j)
    {
        const UInt32 p = routeOf(hashes[j], shift, mask);
        out[p].insert(out[p].end(), src + j * width, src + j * width + width);
    }
    return out;
}

/// Run one column scatter into a fresh arena and assert it is byte-identical to the scalar reference.
void expectColumnMatchesReference(
    const std::vector<UInt32> & hashes, UInt32 shift, UInt32 mask, size_t n, size_t width, size_t partitions, bool use_swwc, uint64_t seed)
{
    const std::vector<char> col = makeColumn(n, width);
    const std::vector<size_t> hist = histogram(hashes, shift, mask, partitions);
    const auto ref = referenceScatter(hashes, shift, mask, n, col.data(), width, partitions);

    PartitionArena arena(hist, width);
    std::vector<void *> base(partitions);
    for (size_t p = 0; p < partitions; ++p)
    {
        base[p] = arena.base(p);
        if (use_swwc)
            ASSERT_EQ(reinterpret_cast<uintptr_t>(base[p]) % 64, 0u) << "W=" << width << " p=" << p;
    }

    ScatterScratch scratch(partitions);
    scatterColumn(hashes.data(), shift, mask, n, col.data(), width, partitions, base.data(), scratch, use_swwc);

    for (size_t p = 0; p < partitions; ++p)
    {
        ASSERT_EQ(ref[p].size(), hist[p] * width) << "W=" << width << " p=" << p;
        if (!ref[p].empty())
            ASSERT_EQ(0, std::memcmp(base[p], ref[p].data(), ref[p].size()))
                << "W=" << width << " p=" << p << " use_swwc=" << use_swwc << " seed=" << seed;
    }
}

}

/// The core differential test: across a width sweep of multiples of 4 that exercises every code path —
/// tiled SWWC (`width | 64`), multi-line NT streaming (`width` a multiple of 64), and the direct
/// batched scatter (templated + generic, and the path SWWC routes non-`÷64` widths to) — both the SWWC
/// and the non-SWWC scatter must reproduce a scalar reference byte-for-byte. This validates
/// conservation, routing (`part = (hash>>shift)&mask`), arrival order, and content simultaneously, and
/// (transitively) that SWWC == direct. (In a non-NT build `use_swwc=true` runs the direct path, so the
/// NT SWWC kernel itself is exercised only in a multitarget build.)
TEST(RadixHashScatter, ColumnWidthSweepMatchesReference)
{
    constexpr UInt32 total_bits = 10;
    constexpr UInt32 shift = 0;
    constexpr size_t partitions = 1u << total_bits; /// 1024
    constexpr UInt32 mask = partitions - 1;
    const size_t n = 200003; /// deliberately not a multiple of any slot/line count

    const std::vector<UInt32> hashes = makeHash(n, total_bits, 0xABCDEF);

    /// Multiples of 4: divisors of 64 (4,8,16,32,64), non-divisor multiples of 4 (12,20,24,28,36,48,60),
    /// and multiples of 64 (64,128,192,256).
    for (size_t width : {size_t{4}, size_t{8}, size_t{12}, size_t{16}, size_t{20}, size_t{24}, size_t{28},
                         size_t{32}, size_t{36}, size_t{48}, size_t{60}, size_t{64}, size_t{128}, size_t{192},
                         size_t{256}})
    {
        expectColumnMatchesReference(hashes, shift, mask, n, width, partitions, /*use_swwc=*/true, width);
        expectColumnMatchesReference(hashes, shift, mask, n, width, partitions, /*use_swwc=*/false, width);
    }
}

/// `nt_store_bytes` is reported and non-zero whenever SWWC flushes whole lines (the divisor and
/// multiple-of-64 widths flush every element); the direct path reports zero. SWWC exists only when NT
/// stores do, so this is meaningful only in a multitarget build.
TEST(RadixHashScatter, NTStoreBytesAccounting)
{
    if (!ntStoresAvailable())
        GTEST_SKIP() << "NT stores unavailable in this build (x86-64-v2 / ENABLE_MULTITARGET_CODE=0); "
                        "scatterColumn(use_swwc=true) runs the direct path";

    constexpr UInt32 total_bits = 8;
    constexpr UInt32 shift = 0;
    constexpr size_t partitions = 1u << total_bits; /// 256
    constexpr UInt32 mask = partitions - 1;
    const size_t n = 100000;

    const std::vector<UInt32> hashes = makeHash(n, total_bits, 0x777);

    for (size_t width : {size_t{8}, size_t{16}, size_t{64}, size_t{128}})
    {
        const std::vector<char> col = makeColumn(n, width);
        const std::vector<size_t> hist = histogram(hashes, shift, mask, partitions);
        PartitionArena arena(hist, width);
        std::vector<void *> base(partitions);
        for (size_t p = 0; p < partitions; ++p)
            base[p] = arena.base(p);

        ScatterScratch scratch(partitions);
        const ScatterStats sw = scatterColumn(hashes.data(), shift, mask, n, col.data(), width, partitions, base.data(), scratch, true);
        EXPECT_GT(sw.nt_store_bytes, 0u) << "W=" << width;
        EXPECT_EQ(sw.nt_store_bytes % 64, 0u) << "W=" << width << " NT flushes whole 64 B lines";
        EXPECT_LE(sw.nt_store_bytes, n * width) << "W=" << width;

        const ScatterStats dir = scatterColumn(hashes.data(), shift, mask, n, col.data(), width, partitions, base.data(), scratch, false);
        EXPECT_EQ(dir.nt_store_bytes, 0u) << "W=" << width << " direct path emits no NT stores";
    }
}

/// Two-column key + `BuildRef` scatter (the production shape): each column is scattered separately
/// (column-major), but both share the same `hash`. Verifies (a) key bytes and ref values match the
/// scalar reference per partition, (b) key[i] and ref[i] in a partition come from the SAME source row
/// (pairing preserved), and (c) the multiset of scattered `row_no` is a bijection onto [0, n).
TEST(RadixHashScatter, TwoColumnKeyRefPairingAndBijection)
{
    constexpr UInt32 total_bits = 10;
    constexpr UInt32 shift = 0;
    constexpr size_t partitions = 1u << total_bits; /// 1024
    constexpr UInt32 mask = partitions - 1;
    const size_t n = 300007;

    const std::vector<UInt32> hashes = makeHash(n, total_bits, 0x123456);
    const std::vector<size_t> hist = histogram(hashes, shift, mask, partitions);

    /// Two representative key widths: 8 B (UInt64) and 16 B (UInt128/UUID).
    for (size_t key_width : {size_t{8}, size_t{16}})
    {
        const std::vector<char> keys = makeColumn(n, key_width);
        std::vector<BuildRef> refs(n);
        for (size_t j = 0; j < n; ++j)
            refs[j] = BuildRef{0, static_cast<UInt32>(j)}; /// row_no == source index, for bijection check

        const auto key_ref = referenceScatter(hashes, shift, mask, n, keys.data(), key_width, partitions);

        PartitionArena keys_out(hist, key_width);
        PartitionArena refs_out(hist, sizeof(BuildRef));
        std::vector<void *> kbase(partitions);
        std::vector<BuildRef *> rbase(partitions);
        for (size_t p = 0; p < partitions; ++p)
        {
            kbase[p] = keys_out.base(p);
            rbase[p] = reinterpret_cast<BuildRef *>(refs_out.base(p));
        }

        ScatterScratch scratch(partitions);
        const ScatterStats stats = scatterKeyRefTwoColumn(
            hashes.data(), shift, mask, n, keys.data(), key_width, refs.data(), partitions, kbase.data(), rbase.data(), scratch, true);
        /// NT stores only in a multitarget build; without NT, use_swwc=true runs the direct path (0 NT bytes).
        if (ntStoresAvailable())
            EXPECT_GT(stats.nt_store_bytes, 0u);

        std::vector<char> seen(n, 0);
        size_t total = 0;
        for (size_t p = 0; p < partitions; ++p)
        {
            ASSERT_EQ(key_ref[p].size(), hist[p] * key_width) << "key_width=" << key_width << " p=" << p;
            if (!key_ref[p].empty())
                ASSERT_EQ(0, std::memcmp(kbase[p], key_ref[p].data(), key_ref[p].size())) << "key bytes p=" << p;

            total += hist[p];
            for (size_t i = 0; i < hist[p]; ++i)
            {
                const UInt32 j = rbase[p][i].row_no;
                ASSERT_LT(j, n);
                EXPECT_EQ(routeOf(hashes[j], shift, mask), p) << "ref row landed in wrong partition";
                /// Pairing: the key at slot i must equal the source key of the row the ref points at.
                ASSERT_EQ(0, std::memcmp(static_cast<const char *>(kbase[p]) + i * key_width, keys.data() + static_cast<size_t>(j) * key_width, key_width))
                    << "key/ref pairing broken p=" << p << " i=" << i;
                EXPECT_EQ(seen[j], 0) << "row scattered more than once";
                seen[j] = 1;
            }
        }
        EXPECT_EQ(total, n);
        EXPECT_EQ(std::count(seen.begin(), seen.end(), 1), static_cast<long>(n)) << "every row must appear exactly once";
    }
}

/// Small inputs exercise residual drains: tiled SWWC (W=4,8 — drains whole-element residuals),
/// a non-divisor multiple of 4 routed to direct (W=24), and multi-line streaming (W=64). Empty and
/// residual-only partitions must still match the reference.
TEST(RadixHashScatter, SmallAndResidualDrain)
{
    for (size_t width : {size_t{4}, size_t{8}, size_t{24}, size_t{64}})
    {
        for (size_t n : {size_t{0}, size_t{1}, size_t{7}, size_t{8}, size_t{9}, size_t{63}, size_t{64}, size_t{65}, size_t{1000}})
        {
            constexpr UInt32 total_bits = 6;
            constexpr UInt32 shift = 0;
            constexpr size_t partitions = 1u << total_bits; /// 64
            constexpr UInt32 mask = partitions - 1;

            const std::vector<UInt32> hashes = makeHash(n, total_bits, 4242 + n * 31 + width);
            expectColumnMatchesReference(hashes, shift, mask, n, width, partitions, /*use_swwc=*/true, n);
            expectColumnMatchesReference(hashes, shift, mask, n, width, partitions, /*use_swwc=*/false, n);
        }
    }
}

/// A full two-pass {6,5} shuffle (spec section 5.3) reproduces num_leaves membership identical to a
/// scalar reference: every row lands in leaf == hash & (num_leaves-1), conserved and bijective. Uses
/// an 8 B key so the per-pass leaf id is recoverable from the carried key value.
TEST(RadixHashScatter, MultiPassMembership)
{
    constexpr UInt32 total_bits = 11;
    const UInt32 pass_bits[2] = {6, 5};
    const size_t num_leaves = 1u << total_bits; /// 2048
    const size_t n = 400000;
    constexpr size_t key_width = sizeof(UInt64);

    /// 8 B keys whose top-`total_bits` bits ARE the routing key (so a child pass can recompute it).
    std::vector<UInt64> keys(n);
    std::vector<UInt32> hashes(n);
    std::mt19937_64 rng(0xFACADE); /// NOLINT(cert-msc32-c,cert-msc51-cpp,bugprone-random-generator-seed) -- deterministic test
    auto hash_of_key = [&](UInt64 key) { return static_cast<UInt32>((key * 0x9E3779B97F4A7C15ULL) >> (64 - total_bits)); };
    for (size_t j = 0; j < n; ++j)
    {
        keys[j] = rng();
        hashes[j] = hash_of_key(keys[j]);
    }
    std::vector<BuildRef> refs(n);
    for (size_t j = 0; j < n; ++j)
        refs[j] = BuildRef{0, static_cast<UInt32>(j)};

    /// Pass 0: top 6 bits of the 11-bit routing key (shift within [0, 2047] range).
    const UInt32 shift0 = total_bits - pass_bits[0]; /// 5
    const UInt32 mask0 = (1u << pass_bits[0]) - 1; /// 63
    const size_t p0 = 1u << pass_bits[0]; /// 64
    const std::vector<size_t> hist0 = histogram(hashes, shift0, mask0, p0);

    PartitionArena keys0(hist0, key_width);
    PartitionArena refs0(hist0, sizeof(BuildRef));
    std::vector<void *> kb0(p0);
    std::vector<BuildRef *> rb0(p0);
    for (size_t p = 0; p < p0; ++p)
    {
        kb0[p] = keys0.base(p);
        rb0[p] = reinterpret_cast<BuildRef *>(refs0.base(p));
    }
    ScatterScratch scratch0(p0);
    scatterKeyRefTwoColumn(
        hashes.data(), shift0, mask0, n, keys.data(), key_width, refs.data(), p0, kb0.data(), rb0.data(), scratch0, true);

    /// Pass 1: next 5 bits, within each pass-0 partition; routing key recomputed from the carried key.
    const UInt32 shift1 = total_bits - pass_bits[0] - pass_bits[1]; /// 0
    const UInt32 mask1 = (1u << pass_bits[1]) - 1; /// 31
    const size_t p1 = 1u << pass_bits[1]; /// 32

    std::vector<char> seen(n, 0);
    size_t total = 0;
    ScatterScratch scratch1(p1);

    for (size_t parent = 0; parent < p0; ++parent)
    {
        const size_t rows = hist0[parent];
        if (rows == 0)
            continue;

        const auto * pkeys = reinterpret_cast<const UInt64 *>(kb0[parent]);
        std::vector<UInt32> sub_hash(rows);
        for (size_t i = 0; i < rows; ++i)
            sub_hash[i] = hash_of_key(pkeys[i]);

        const std::vector<size_t> hist1 = histogram(sub_hash, shift1, mask1, p1);
        PartitionArena keys1(hist1, key_width);
        PartitionArena refs1(hist1, sizeof(BuildRef));
        std::vector<void *> kb1(p1);
        std::vector<BuildRef *> rb1(p1);
        for (size_t p = 0; p < p1; ++p)
        {
            kb1[p] = keys1.base(p);
            rb1[p] = reinterpret_cast<BuildRef *>(refs1.base(p));
        }
        scatterKeyRefTwoColumn(
            sub_hash.data(), shift1, mask1, rows, kb0[parent], key_width, rb0[parent], p1, kb1.data(), rb1.data(), scratch1, true);

        for (size_t sub = 0; sub < p1; ++sub)
        {
            const size_t leaf = parent * p1 + sub;
            ASSERT_LT(leaf, num_leaves);
            const auto * lkeys = reinterpret_cast<const UInt64 *>(kb1[sub]);
            for (size_t i = 0; i < hist1[sub]; ++i)
            {
                const size_t j = rb1[sub][i].row_no;
                ASSERT_LT(j, n);
                EXPECT_EQ(lkeys[i], keys[j]);
                EXPECT_EQ(hashes[j], static_cast<UInt32>(leaf)) << "final leaf must equal routing key (identical to scalar reference)";
                EXPECT_EQ(seen[j], 0);
                seen[j] = 1;
                ++total;
            }
        }
    }
    EXPECT_EQ(total, n);
    EXPECT_EQ(std::count(seen.begin(), seen.end(), 1), static_cast<long>(n));
}
