#include <Common/RadixShuffle/ColumnPrimitives/FixedWidth.h>

#include <Columns/ColumnDecimal.h>
#include <Columns/ColumnFixedString.h>
#include <Columns/ColumnVector.h>
#include <Common/RadixShuffle/HashCombiner.h>
#include <Common/assert_cast.h>

#include <cstring>

namespace DB::RadixShuffle
{

namespace
{

/// Per-row finalizer used by the fixed-width hash column primitives. This is the
/// MurmurHash3 64-bit finalizer of `intHash64`, applied bit-by-bit to a
/// fixed-width payload up to 8 bytes. For wider types we fold the bytes
/// through the same finalizer in 8-byte chunks. The result is then mixed
/// into the caller's output via `hashCombine` (§3.4).
[[gnu::always_inline]] inline uint64_t intHash64Local(uint64_t x) noexcept
{
    x ^= x >> 33;
    x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33;
    x *= 0xc4ceb9fe1a85ec53ULL;
    x ^= x >> 33;
    return x;
}


/// Generic mixer over a contiguous T value. Loads through unaligned
/// memcpy to avoid undefined behavior on packed/wide types.
template <typename T>
[[gnu::always_inline]] inline uint64_t hashOne(const T & v) noexcept
{
    if constexpr (sizeof(T) <= sizeof(uint64_t))
    {
        uint64_t buf = 0;
        std::memcpy(&buf, &v, sizeof(T));
        return intHash64Local(buf);
    }
    else
    {
        const auto * bytes = reinterpret_cast<const unsigned char *>(&v);
        uint64_t acc = 0;
        for (size_t i = 0; i < sizeof(T); i += sizeof(uint64_t))
        {
            uint64_t word = 0;
            std::memcpy(&word, bytes + i, std::min(sizeof(uint64_t), sizeof(T) - i));
            acc ^= intHash64Local(word + acc + 0x9e3779b97f4a7c15ULL);
        }
        return acc;
    }
}


/// Bound on the partition count handled by a single scatter call. The
/// scatter primitives materialize one write pointer per partition in a
/// stack-resident array; the bound caps that array's size. The spec's
/// workload sweep tops out at P=256, so 1024 gives slack for future
/// configurations without growing the stack footprint beyond ~8 KiB.
constexpr size_t MAX_PARTITIONS = 1024;


/// Scatter for a fixed-width T value column. The inner loop is the
/// 5-µops/row branch-free pattern from the `phj-bench` reference: bump
/// per-partition write pointers indexed by `pids[j]`. We refresh the
/// pointer array once per call from the caller-supplied destinations.
template <typename T>
[[gnu::hot]] void
scatterFixed(const ColumnPrimitives & /*self*/, const IColumn & src_, const uint32_t * pids, size_t n, size_t partitions, Reservation * dst)
{
    const auto & col = assert_cast<const ColumnVector<T> &>(src_);
    const T * src = col.getData().data();

    /// `dst[p].chunk->primary` is the chunk's primary buffer; the slot
    /// starts at byte `dst[p].begin_row * sizeof(T)`. We pre-compute a
    /// per-partition write pointer array. The caller's destination array
    /// covers `[0, partitions)` indexed by pid; the precondition
    /// `pids[j] < partitions` (§3.2) means we only index that range.
    chassert(partitions <= MAX_PARTITIONS);
    T * ptrs[MAX_PARTITIONS];
    for (size_t p = 0; p < partitions; ++p)
    {
        if (dst[p].chunk != nullptr)
            ptrs[p] = static_cast<T *>(dst[p].chunk->primary) + dst[p].begin_row;
        else
            ptrs[p] = nullptr;
    }

    /// Branch-free row loop.
    for (size_t j = 0; j < n; ++j)
        *ptrs[pids[j]]++ = src[j];
}


/// Specialization for `ColumnFixedString(n)`. The element size is dynamic
/// (`n`), so we cannot use the templated scatter. Each row is a contiguous
/// `n`-byte block in the column's `getChars()` buffer.
[[gnu::hot]] void scatterFixedString(
    const ColumnPrimitives & /*self*/, const IColumn & src_, const uint32_t * pids, size_t n_rows, size_t partitions, Reservation * dst)
{
    const auto & col = assert_cast<const ColumnFixedString &>(src_);
    const size_t n = col.getN();
    const auto * src = reinterpret_cast<const unsigned char *>(col.getChars().data());

    chassert(partitions <= MAX_PARTITIONS);
    unsigned char * ptrs[MAX_PARTITIONS];
    for (size_t p = 0; p < partitions; ++p)
    {
        if (dst[p].chunk != nullptr)
            ptrs[p] = static_cast<unsigned char *>(dst[p].chunk->primary) + dst[p].begin_row * n;
        else
            ptrs[p] = nullptr;
    }

    for (size_t j = 0; j < n_rows; ++j)
    {
        unsigned char * out = ptrs[pids[j]];
        std::memcpy(out, src + j * n, n);
        ptrs[pids[j]] = out + n;
    }
}


template <typename T>
ResumePosition
reconstructFixed(const ColumnPrimitives & /*self*/, const ChunkRangeView * views, size_t n_views, ResumePosition start, IColumn & target)
{
    auto & col = assert_cast<ColumnVector<T> &>(target);
    auto & data = col.getData();
    const size_t cap = data.capacity();
    size_t cur = data.size();

    size_t vi = start.view_index;
    size_t in_view = start.rows_consumed_in_view;
    while (vi < n_views)
    {
        const ChunkRangeView & v = views[vi];
        const size_t available = v.end - v.begin - in_view;
        const size_t room = cap - cur;
        if (room == 0)
            break;
        const size_t take = std::min(available, room);
        const T * src = static_cast<const T *>(v.chunk->primary) + v.begin + in_view;

        data.resize_assume_reserved(cur + take);
        std::memcpy(data.data() + cur, src, take * sizeof(T));
        cur += take;

        in_view += take;
        if (in_view == v.end - v.begin)
        {
            ++vi;
            in_view = 0;
        }
        else
        {
            /// Target full mid-view; return resume cursor pointing inside.
            break;
        }
    }
    return ResumePosition{vi, in_view};
}


template <typename T>
ResumePosition
reconstructDecimal(const ColumnPrimitives & /*self*/, const ChunkRangeView * views, size_t n_views, ResumePosition start, IColumn & target)
{
    auto & col = assert_cast<ColumnDecimal<T> &>(target);
    auto & data = col.getData();
    const size_t cap = data.capacity();
    size_t cur = data.size();

    size_t vi = start.view_index;
    size_t in_view = start.rows_consumed_in_view;
    while (vi < n_views)
    {
        const ChunkRangeView & v = views[vi];
        const size_t available = v.end - v.begin - in_view;
        const size_t room = cap - cur;
        if (room == 0)
            break;
        const size_t take = std::min(available, room);
        const T * src = static_cast<const T *>(v.chunk->primary) + v.begin + in_view;

        data.resize_assume_reserved(cur + take);
        std::memcpy(data.data() + cur, src, take * sizeof(T));
        cur += take;

        in_view += take;
        if (in_view == v.end - v.begin)
        {
            ++vi;
            in_view = 0;
        }
        else
        {
            break;
        }
    }
    return ResumePosition{vi, in_view};
}


ResumePosition reconstructFixedString(
    const ColumnPrimitives & /*self*/, const ChunkRangeView * views, size_t n_views, ResumePosition start, IColumn & target)
{
    auto & col = assert_cast<ColumnFixedString &>(target);
    auto & chars = col.getChars();
    const size_t n = col.getN();
    const size_t cap_rows = chars.capacity() / n;
    size_t cur_rows = chars.size() / n;

    size_t vi = start.view_index;
    size_t in_view = start.rows_consumed_in_view;
    while (vi < n_views)
    {
        const ChunkRangeView & v = views[vi];
        const size_t available = v.end - v.begin - in_view;
        const size_t room = cap_rows - cur_rows;
        if (room == 0)
            break;
        const size_t take = std::min(available, room);
        const auto * src = static_cast<const unsigned char *>(v.chunk->primary) + (v.begin + in_view) * n;

        chars.resize_assume_reserved((cur_rows + take) * n);
        auto * dst = reinterpret_cast<unsigned char *>(chars.data()) + cur_rows * n;
        std::memcpy(dst, src, take * n);
        cur_rows += take;

        in_view += take;
        if (in_view == v.end - v.begin)
        {
            ++vi;
            in_view = 0;
        }
        else
        {
            break;
        }
    }
    return ResumePosition{vi, in_view};
}


template <typename T>
void hashFixed(const ColumnPrimitives & /*self*/, const IColumn & src_, size_t n, uint64_t * out)
{
    const auto & col = assert_cast<const ColumnVector<T> &>(src_);
    const T * data = col.getData().data();
    for (size_t i = 0; i < n; ++i)
        out[i] = hashCombine(out[i], hashOne(data[i]));
}


template <typename T>
void hashDecimal(const ColumnPrimitives & /*self*/, const IColumn & src_, size_t n, uint64_t * out)
{
    const auto & col = assert_cast<const ColumnDecimal<T> &>(src_);
    const auto & data = col.getData();
    for (size_t i = 0; i < n; ++i)
        out[i] = hashCombine(out[i], hashOne(data[i].value));
}


void hashFixedString(const ColumnPrimitives & /*self*/, const IColumn & src_, size_t n_rows, uint64_t * out)
{
    const auto & col = assert_cast<const ColumnFixedString &>(src_);
    const size_t n = col.getN();
    const auto * data = reinterpret_cast<const unsigned char *>(col.getChars().data());
    for (size_t i = 0; i < n_rows; ++i)
    {
        uint64_t acc = 0;
        for (size_t j = 0; j < n; j += sizeof(uint64_t))
        {
            uint64_t word = 0;
            std::memcpy(&word, data + i * n + j, std::min(sizeof(uint64_t), n - j));
            acc = intHash64Local(word + acc + 0x9e3779b97f4a7c15ULL) ^ acc;
        }
        out[i] = hashCombine(out[i], acc);
    }
}


/// ColumnDecimal scatter: identical body to scatterFixed<NativeT>; we
/// route it through the same template by selecting the underlying scalar
/// type, but ColumnDecimal stores `Decimal<T>` which is a wrapper. We
/// scatter by raw element width.
template <typename T>
[[gnu::hot]] void scatterDecimal(
    const ColumnPrimitives & /*self*/, const IColumn & src_, const uint32_t * pids, size_t n, size_t partitions, Reservation * dst)
{
    const auto & col = assert_cast<const ColumnDecimal<T> &>(src_);
    const T * src = col.getData().data();

    chassert(partitions <= MAX_PARTITIONS);
    T * ptrs[MAX_PARTITIONS];
    for (size_t p = 0; p < partitions; ++p)
    {
        if (dst[p].chunk != nullptr)
            ptrs[p] = static_cast<T *>(dst[p].chunk->primary) + dst[p].begin_row;
        else
            ptrs[p] = nullptr;
    }

    for (size_t j = 0; j < n; ++j)
        *ptrs[pids[j]]++ = src[j];
}

}


/// Public entry points. Each returns a ColumnPrimitives triple bound to a column type.
template <typename T>
ColumnPrimitives makeFixedWidth()
{
    ColumnPrimitives column_primitives;
    column_primitives.scatter = &scatterFixed<T>;
    column_primitives.reconstruct = &reconstructFixed<T>;
    column_primitives.hash = &hashFixed<T>;
    column_primitives.column_desc.element_size = sizeof(T);
    column_primitives.column_desc.alignment = alignof(T);
    column_primitives.column_desc.has_offsets = false;
    column_primitives.column_desc.has_null_map = false;
    column_primitives.column_desc.variable_length = false;
    return column_primitives;
}


/// Explicit instantiations for every numeric type used by ColumnVector.
template ColumnPrimitives makeFixedWidth<UInt8>();
template ColumnPrimitives makeFixedWidth<UInt16>();
template ColumnPrimitives makeFixedWidth<UInt32>();
template ColumnPrimitives makeFixedWidth<UInt64>();
template ColumnPrimitives makeFixedWidth<UInt128>();
template ColumnPrimitives makeFixedWidth<UInt256>();
template ColumnPrimitives makeFixedWidth<Int8>();
template ColumnPrimitives makeFixedWidth<Int16>();
template ColumnPrimitives makeFixedWidth<Int32>();
template ColumnPrimitives makeFixedWidth<Int64>();
template ColumnPrimitives makeFixedWidth<Int128>();
template ColumnPrimitives makeFixedWidth<Int256>();
template ColumnPrimitives makeFixedWidth<BFloat16>();
template ColumnPrimitives makeFixedWidth<Float32>();
template ColumnPrimitives makeFixedWidth<Float64>();
template ColumnPrimitives makeFixedWidth<UUID>();
template ColumnPrimitives makeFixedWidth<IPv4>();
template ColumnPrimitives makeFixedWidth<IPv6>();


/// ColumnDecimal column primitives use the Decimal value-type as their scatter
/// element. We provide one helper template per Decimal width and resolve
/// in the dispatcher.
template <typename T>
ColumnPrimitives makeDecimal()
{
    ColumnPrimitives column_primitives;
    column_primitives.scatter = &scatterDecimal<T>;
    column_primitives.reconstruct = &reconstructDecimal<T>;
    column_primitives.hash = &hashDecimal<T>;
    column_primitives.column_desc.element_size = sizeof(T);
    column_primitives.column_desc.alignment = alignof(T);
    column_primitives.column_desc.has_offsets = false;
    column_primitives.column_desc.has_null_map = false;
    column_primitives.column_desc.variable_length = false;
    return column_primitives;
}


template ColumnPrimitives makeDecimal<Decimal32>();
template ColumnPrimitives makeDecimal<Decimal64>();
template ColumnPrimitives makeDecimal<Decimal128>();
template ColumnPrimitives makeDecimal<Decimal256>();
template ColumnPrimitives makeDecimal<DateTime64>();
template ColumnPrimitives makeDecimal<Time64>();


ColumnPrimitives makeFixedString(size_t n)
{
    ColumnPrimitives column_primitives;
    column_primitives.scatter = &scatterFixedString;
    column_primitives.reconstruct = &reconstructFixedString;
    column_primitives.hash = &hashFixedString;
    column_primitives.column_desc.element_size = n;
    column_primitives.column_desc.alignment = 1;
    column_primitives.column_desc.has_offsets = false;
    column_primitives.column_desc.has_null_map = false;
    column_primitives.column_desc.variable_length = false;
    column_primitives.aux = n;
    return column_primitives;
}

}
