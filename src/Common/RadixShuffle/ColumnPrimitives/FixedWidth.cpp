#include <Common/RadixShuffle/ColumnPrimitives/FixedWidth.h>

#include <Columns/ColumnDecimal.h>
#include <Columns/ColumnFixedString.h>
#include <Columns/ColumnVector.h>
#include <Common/RadixShuffle/HashCombiner.h>
#include <Common/RadixShuffle/HashKernels.h>
#include <Common/assert_cast.h>

#include <cstring>

namespace DB::RadixShuffle
{

namespace
{

/// Maximum partition count per scatter call.  The scatter primitives
/// materialise one write pointer per partition in a stack array; this
/// bound caps the array's size.  The v1 workload sweep tops out at
/// P=256; 1024 gives slack for future configurations.
constexpr size_t MAX_PARTITIONS = 1024;


/// Refresh the fixed-chunk write pointer for partition p.
/// elem_size is in bytes (sizeof(T) for fixed-width, n for FixedString).
[[gnu::always_inline]] inline void
refreshFixedPtr(char * & ptr, size_t p, size_t slot_idx, const PartReservation * dst, size_t elem_size)
{
    if (dst[p].fixed != nullptr)
    {
        const size_t slot_off = dst[p].fixed->slot_byte_offsets[slot_idx];
        ptr = static_cast<char *>(dst[p].fixed->data) + slot_off + dst[p].begin_row * elem_size;
    }
    else
    {
        ptr = nullptr;
    }
}


/// Refresh all stale or uninitialised fixed-chunk write pointers.
/// On the first call (state.initialized == false) every partition is
/// refreshed with an O(P) loop.  On subsequent calls only the stale
/// partitions (flagged in stale_fixed_bitset) are touched.
template <size_t ElemSize>
[[gnu::always_inline]] inline void refreshFixedPtrs(
    ScatterState & state,
    size_t slot_idx,
    size_t partitions,
    const PartReservation * dst,
    const uint64_t * stale_fixed_bitset)
{
    if (!state.initialized)
    {
        for (size_t p = 0; p < partitions; ++p)
            refreshFixedPtr(state.fixed_ptrs[p], p, slot_idx, dst, ElemSize);
        state.initialized = true;
        return;
    }

    // Selective refresh: walk only set bits in the stale bitset.
    const size_t words = (partitions + 63) / 64;
    for (size_t word = 0; word < words; ++word)
    {
        uint64_t bits = stale_fixed_bitset[word];
        while (bits)
        {
            const size_t bit = static_cast<size_t>(__builtin_ctzll(bits));
            const size_t p = word * 64 + bit;
            if (p < partitions)
                refreshFixedPtr(state.fixed_ptrs[p], p, slot_idx, dst, ElemSize);
            bits &= bits - 1;
        }
    }
}


template <typename T>
[[gnu::hot]] void scatterFixed(
    const ColumnPrimitives & self,
    const PartSchema & /*schema*/,
    const IColumn & src_,
    const uint16_t * pids,
    size_t n,
    size_t partitions,
    const PartReservation * dst,
    ScatterState & state,
    const uint64_t * stale_fixed_bitset)
{
    const auto & col = assert_cast<const ColumnVector<T> &>(src_);
    const T * src = col.getData().data();

    const size_t slot_idx = self.fixed_slot_indices[0];
    chassert(partitions <= MAX_PARTITIONS);

    refreshFixedPtrs<sizeof(T)>(state, slot_idx, partitions, dst, stale_fixed_bitset);

    // Load cached pointers into a stack array for the branch-free inner loop.
    T * ptrs[MAX_PARTITIONS];
    for (size_t p = 0; p < partitions; ++p)
        ptrs[p] = reinterpret_cast<T *>(state.fixed_ptrs[p]);

    for (size_t j = 0; j < n; ++j)
        *ptrs[pids[j]]++ = src[j];

    // Write back the advanced pointers so the cache is up-to-date for the next batch.
    for (size_t p = 0; p < partitions; ++p)
        state.fixed_ptrs[p] = reinterpret_cast<char *>(ptrs[p]);
}


[[gnu::hot]] void scatterFixedString(
    const ColumnPrimitives & self,
    const PartSchema & /*schema*/,
    const IColumn & src_,
    const uint16_t * pids,
    size_t n_rows,
    size_t partitions,
    const PartReservation * dst,
    ScatterState & state,
    const uint64_t * stale_fixed_bitset)
{
    const auto & col = assert_cast<const ColumnFixedString &>(src_);
    const size_t n = col.getN();
    const auto * src = reinterpret_cast<const unsigned char *>(col.getChars().data());

    const size_t slot_idx = self.fixed_slot_indices[0];
    chassert(partitions <= MAX_PARTITIONS);

    // Element size is dynamic (n), so we cannot use the compile-time template.
    if (!state.initialized)
    {
        for (size_t p = 0; p < partitions; ++p)
            refreshFixedPtr(state.fixed_ptrs[p], p, slot_idx, dst, n);
        state.initialized = true;
    }
    else
    {
        const size_t words = (partitions + 63) / 64;
        for (size_t word = 0; word < words; ++word)
        {
            uint64_t bits = stale_fixed_bitset[word];
            while (bits)
            {
                const size_t bit = static_cast<size_t>(__builtin_ctzll(bits));
                const size_t p = word * 64 + bit;
                if (p < partitions)
                    refreshFixedPtr(state.fixed_ptrs[p], p, slot_idx, dst, n);
                bits &= bits - 1;
            }
        }
    }

    unsigned char * ptrs[MAX_PARTITIONS];
    for (size_t p = 0; p < partitions; ++p)
        ptrs[p] = reinterpret_cast<unsigned char *>(state.fixed_ptrs[p]);

    for (size_t j = 0; j < n_rows; ++j)
    {
        unsigned char * out = ptrs[pids[j]];
        std::memcpy(out, src + j * n, n);
        ptrs[pids[j]] = out + n;
    }

    for (size_t p = 0; p < partitions; ++p)
        state.fixed_ptrs[p] = reinterpret_cast<char *>(ptrs[p]);
}


template <typename T>
[[gnu::hot]] void scatterDecimal(
    const ColumnPrimitives & self,
    const PartSchema & /*schema*/,
    const IColumn & src_,
    const uint16_t * pids,
    size_t n,
    size_t partitions,
    const PartReservation * dst,
    ScatterState & state,
    const uint64_t * stale_fixed_bitset)
{
    const auto & col = assert_cast<const ColumnDecimal<T> &>(src_);
    const T * src = col.getData().data();

    const size_t slot_idx = self.fixed_slot_indices[0];
    chassert(partitions <= MAX_PARTITIONS);

    refreshFixedPtrs<sizeof(T)>(state, slot_idx, partitions, dst, stale_fixed_bitset);

    T * ptrs[MAX_PARTITIONS];
    for (size_t p = 0; p < partitions; ++p)
        ptrs[p] = reinterpret_cast<T *>(state.fixed_ptrs[p]);

    for (size_t j = 0; j < n; ++j)
        *ptrs[pids[j]]++ = src[j];

    for (size_t p = 0; p < partitions; ++p)
        state.fixed_ptrs[p] = reinterpret_cast<char *>(ptrs[p]);
}


template <typename T>
ResumePosition reconstructFixed(
    const ColumnPrimitives & self,
    const PartSchema & /*schema*/,
    const PartReservationView * views,
    size_t n_views,
    ResumePosition start,
    IColumn & target)
{
    auto & col = assert_cast<ColumnVector<T> &>(target);
    auto & data = col.getData();
    const size_t cap = data.capacity();
    size_t cur = data.size();

    const size_t slot_idx = self.fixed_slot_indices[0];

    size_t vi = start.view_index;
    size_t in_view = start.rows_consumed_in_view;
    while (vi < n_views)
    {
        const PartReservationView & v = views[vi];
        const size_t view_rows = v.row_end - v.row_begin;
        const size_t available = view_rows - in_view;
        const size_t room = cap - cur;
        if (room == 0)
            break;
        const size_t take = std::min(available, room);

        const size_t slot_off = v.fixed->slot_byte_offsets[slot_idx];
        const T * src = reinterpret_cast<const T *>(
                            static_cast<const char *>(v.fixed->data) + slot_off)
            + v.row_begin + in_view;

        data.resize_assume_reserved(cur + take);
        std::memcpy(data.data() + cur, src, take * sizeof(T));
        cur += take;

        in_view += take;
        if (in_view == view_rows)
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
ResumePosition reconstructDecimal(
    const ColumnPrimitives & self,
    const PartSchema & /*schema*/,
    const PartReservationView * views,
    size_t n_views,
    ResumePosition start,
    IColumn & target)
{
    auto & col = assert_cast<ColumnDecimal<T> &>(target);
    auto & data = col.getData();
    const size_t cap = data.capacity();
    size_t cur = data.size();

    const size_t slot_idx = self.fixed_slot_indices[0];

    size_t vi = start.view_index;
    size_t in_view = start.rows_consumed_in_view;
    while (vi < n_views)
    {
        const PartReservationView & v = views[vi];
        const size_t view_rows = v.row_end - v.row_begin;
        const size_t available = view_rows - in_view;
        const size_t room = cap - cur;
        if (room == 0)
            break;
        const size_t take = std::min(available, room);

        const size_t slot_off = v.fixed->slot_byte_offsets[slot_idx];
        const T * src = reinterpret_cast<const T *>(
                            static_cast<const char *>(v.fixed->data) + slot_off)
            + v.row_begin + in_view;

        data.resize_assume_reserved(cur + take);
        std::memcpy(data.data() + cur, src, take * sizeof(T));
        cur += take;

        in_view += take;
        if (in_view == view_rows)
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
    const ColumnPrimitives & self,
    const PartSchema & /*schema*/,
    const PartReservationView * views,
    size_t n_views,
    ResumePosition start,
    IColumn & target)
{
    auto & col = assert_cast<ColumnFixedString &>(target);
    auto & chars = col.getChars();
    const size_t n = col.getN();
    const size_t cap_rows = chars.capacity() / n;
    size_t cur_rows = chars.size() / n;

    const size_t slot_idx = self.fixed_slot_indices[0];

    size_t vi = start.view_index;
    size_t in_view = start.rows_consumed_in_view;
    while (vi < n_views)
    {
        const PartReservationView & v = views[vi];
        const size_t view_rows = v.row_end - v.row_begin;
        const size_t available = view_rows - in_view;
        const size_t room = cap_rows - cur_rows;
        if (room == 0)
            break;
        const size_t take = std::min(available, room);

        const size_t slot_off = v.fixed->slot_byte_offsets[slot_idx];
        const auto * src = static_cast<const unsigned char *>(v.fixed->data) + slot_off
            + (v.row_begin + in_view) * n;

        chars.resize_assume_reserved((cur_rows + take) * n);
        auto * dst_ptr = reinterpret_cast<unsigned char *>(chars.data()) + cur_rows * n;
        std::memcpy(dst_ptr, src, take * n);
        cur_rows += take;

        in_view += take;
        if (in_view == view_rows)
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
void hashFixed(
    const ColumnPrimitives & /*self*/,
    const PartSchema & /*schema*/,
    const IColumn & src_,
    size_t n,
    uint32_t * out)
{
    const auto & col = assert_cast<const ColumnVector<T> &>(src_);
    const T * data = col.getData().data();
    for (size_t i = 0; i < n; ++i)
        out[i] = hashCombine(out[i], hashOne32(data[i]));
}


template <typename T>
void hashDecimal(
    const ColumnPrimitives & /*self*/,
    const PartSchema & /*schema*/,
    const IColumn & src_,
    size_t n,
    uint32_t * out)
{
    const auto & col = assert_cast<const ColumnDecimal<T> &>(src_);
    const auto & data = col.getData();
    for (size_t i = 0; i < n; ++i)
        out[i] = hashCombine(out[i], hashOne32(data[i].value));
}


void hashFixedString(
    const ColumnPrimitives & /*self*/,
    const PartSchema & /*schema*/,
    const IColumn & src_,
    size_t n_rows,
    uint32_t * out)
{
    const auto & col = assert_cast<const ColumnFixedString &>(src_);
    const size_t n = col.getN();
    const auto * data = reinterpret_cast<const unsigned char *>(col.getChars().data());
    for (size_t i = 0; i < n_rows; ++i)
        out[i] = hashCombine(out[i], hashBytes32(data + i * n, n));
}

} // namespace


template <typename T>
ColumnPrimitives makeFixedWidth()
{
    ColumnPrimitives cp;
    cp.scatter = &scatterFixed<T>;
    cp.reconstruct = &reconstructFixed<T>;
    cp.hash = &hashFixed<T>;
    return cp;
}

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


template <typename T>
ColumnPrimitives makeDecimal()
{
    ColumnPrimitives cp;
    cp.scatter = &scatterDecimal<T>;
    cp.reconstruct = &reconstructDecimal<T>;
    cp.hash = &hashDecimal<T>;
    return cp;
}

template ColumnPrimitives makeDecimal<Decimal32>();
template ColumnPrimitives makeDecimal<Decimal64>();
template ColumnPrimitives makeDecimal<Decimal128>();
template ColumnPrimitives makeDecimal<Decimal256>();
template ColumnPrimitives makeDecimal<DateTime64>();
template ColumnPrimitives makeDecimal<Time64>();


ColumnPrimitives makeFixedString(size_t n)
{
    ColumnPrimitives cp;
    cp.scatter = &scatterFixedString;
    cp.reconstruct = &reconstructFixedString;
    cp.hash = &hashFixedString;
    cp.aux = n;
    return cp;
}

}
