#pragma once

#include <Common/RadixShuffle/IScatterColumn.h>

#include <cstddef>
#include <cstdint>


namespace DB::RadixShuffle
{

/// Fixed-width numeric scatter column for radix partitioning.
///
/// Generalises `UInt64Column` from `radix_part_vs_memcpy.cpp` to any POD type
/// `T`.  The SWWC scatter path (`scatterStaged`) uses `_mm512_stream_si512`
/// NT stores on x86_64-v4 hardware when `sizeof(T) == 8` (matching the
/// reference for `T = uint64_t`).  On other architectures or narrower types
/// `scatterStaged` falls back to scalar.
///
/// Lifecycle:
///   1. Construct with P (number of partitions).
///   2. Call `onGrow` when the owning `RadixPartitionOperator` allocates a new
///      `OutBlock` for a partition.
///   3. Call `scatterDirect` or `scatterStaged` for each batch.
///   4. Call `drainOne` before `onGrow` and at finish to flush partial SWWC
///      staging buffers.
template <typename T>
class NumericScatterColumn final : public IScatterColumn
{
public:
    explicit NumericScatterColumn(size_t P);
    ~NumericScatterColumn() override;

    NumericScatterColumn(const NumericScatterColumn &) = delete;
    NumericScatterColumn & operator=(const NumericScatterColumn &) = delete;

    void onGrow(size_t p, void * col_base) override;
    void drainOne(size_t p, uint32_t cnt) override;
    void scatterDirect(const uint32_t * pids, const void * src, int n) override;
    void scatterStaged(const uint32_t * pids, const uint32_t * positions, const void * src, int n) override;

private:
    size_t num_partitions_;
    T * staging_; ///< [P × 8] 64B-aligned SWWC staging buffer.
    T ** out_;    ///< Per-partition write-destination pointer.
};

/// Explicit instantiation declarations; definitions live in NumericScatterColumn.cpp.
extern template class NumericScatterColumn<uint8_t>;
extern template class NumericScatterColumn<uint16_t>;
extern template class NumericScatterColumn<uint32_t>;
extern template class NumericScatterColumn<uint64_t>;
extern template class NumericScatterColumn<int8_t>;
extern template class NumericScatterColumn<int16_t>;
extern template class NumericScatterColumn<int32_t>;
extern template class NumericScatterColumn<int64_t>;
extern template class NumericScatterColumn<float>;
extern template class NumericScatterColumn<double>;

} // namespace DB::RadixShuffle
