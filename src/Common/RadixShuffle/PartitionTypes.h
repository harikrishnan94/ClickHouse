#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <vector>


namespace DB::RadixShuffle
{

/// One chunk in a per-partition fixed chain.  Holds all fixed-width arrays
/// (values, offsets, null maps) for all columns in the partition, laid out
/// column-major (planar): each slot's array is contiguous.
///
/// Byte address of row r in slot s:
///     slot_byte_offsets[s] + r * PartSchema::fixed_slots[s].element_size
struct FixedChunk
{
    void * data = nullptr;
    size_t row_capacity = 0;

    /// Per-slot byte offsets within `data` at row 0, computed at allocation
    /// time for this chunk's specific row_capacity.  Length equals
    /// PartSchema::fixed_slots.size().  Points into the handle's arena.
    const size_t * slot_byte_offsets = nullptr;
};


/// One chunk in a per-partition data chain.  Holds all varlen character
/// payloads for all varlen columns in the partition as a single byte stream.
struct DataChunk
{
    unsigned char * bytes = nullptr;
    size_t byte_capacity = 0;
};


/// Writable destination for one partition returned by Handle::reserve.
struct PartReservation
{
    FixedChunk * fixed = nullptr;
    size_t begin_row = 0;
    size_t reserved_rows = 0;

    DataChunk * data = nullptr; ///< null for fixed-only partitions.
    size_t begin_byte = 0;
    size_t reserved_bytes = 0;
};


/// Full result of one Handle::reserve call for a single partition.
struct PartReserveGrant
{
    size_t granted_rows = 0;
    size_t granted_varlen_bytes = 0;
    PartReservation slice;
    bool fully_satisfied = false;
};


/// Per-partition view tuple consumed by reconstruct.  Identifies which rows
/// (and which byte range) of a chunk were actually written by scatter.
struct PartReservationView
{
    const FixedChunk * fixed = nullptr;
    size_t row_begin = 0;
    size_t row_end = 0;

    const DataChunk * data = nullptr; ///< null for fixed-only primitives.
    size_t byte_begin = 0;
    size_t byte_end = 0;
};


/// Resume cursor for reconstruct's pump pattern.  (0, 0) is the natural
/// start; the value returned by reconstruct is the next position into
/// the input view list.
struct ResumePosition
{
    size_t view_index = 0;
    size_t rows_consumed_in_view = 0;
};


/// Per-thread, per-column mutable write-pointer cache for scatter.
/// Owned by the operator thread alongside Handle; one instance per
/// ColumnPrimitives per shuffle phase.
///
/// The write pointers cached here survive across batches.  On each call
/// scatter refreshes only the partitions flagged stale in the bitset
/// returned by Handle::reserve (fixed-chunk reallocations) plus a
/// DataChunk-pointer comparison for varlen columns.  This eliminates the
/// O(P) unconditional setup loop on steady-state batches.
///
/// Lifecycle: construct with ScatterState(P), pass to scatter on every
/// batch.  The scatter function lazily initialises varlen fields (data_ptrs,
/// cached_data) and the nested ScatterState on the first call.
struct ScatterState
{
    /// Type-erased write pointer per partition for the primary fixed slot.
    ///   - ColumnVector<T>  / ColumnDecimal<T>: stores T *
    ///   - ColumnString:                         stores uint64_t * (Offsets slot)
    ///   - ColumnFixedString:                    stores unsigned char *
    ///   - ColumnNullable:                       stores uint8_t * (NullMap slot)
    /// Stored as void* so `state.fixed_ptrs[p]` can be passed as `void*&` to
    /// `flushStagedNTInPlace`, letting it update in-place without an explicit
    /// load-cast-store round-trip.  Byte-level arithmetic uses a local char* cast.
    std::vector<void *> fixed_ptrs;

    /// DataChunk write-pointer cache for varlen columns (lazily sized on
    /// the first scatter call that needs varlen state).
    std::vector<unsigned char *> data_ptrs;

    /// Last DataChunk * seen per partition.  When dst[p].data differs,
    /// data_ptrs[p] is stale and is recomputed from the new chunk.
    std::vector<const DataChunk *> cached_data;

    /// Nested ScatterState for Nullable(X); null for leaf column types.
    /// Lazily constructed on the first scatter call for Nullable columns.
    std::unique_ptr<ScatterState> nested;

    /// False until the first scatter call fully initialises fixed_ptrs.
    bool initialized = false;

    /// Per-partition write-pointer cache for the OutBlock raw scatter path.
    /// Plain raw pointer (not std::vector) so it can be loaded once into a
    /// callee-saved register at function entry and accessed as *(ptr+p) —
    /// one load, same depth as the baseline's `T** out_` member in
    /// `NumericScatterColumn`.  A std::vector would require an extra
    /// `vector.data()` load per access, causing the pids/positions stack spill.
    /// Lazily allocated by `on_grow_raw` on the first call.
    void ** raw_write_ptrs = nullptr;

    /// SWWC staging buffer for `scatter_raw_swwc`.  Lazily allocated on the
    /// first `scatter_raw_swwc` call; layout is [P × 8 × elem_size],
    /// 64-byte aligned so that NT stores can address it.
    char * swwc_staging = nullptr;
    bool swwc_staging_initialized = false;

    explicit ScatterState(size_t P)
        : fixed_ptrs(P, nullptr)
    {
    }

    ScatterState(const ScatterState &) = delete;
    ScatterState & operator=(const ScatterState &) = delete;

    ScatterState(ScatterState && other) noexcept
        : fixed_ptrs(std::move(other.fixed_ptrs))
        , data_ptrs(std::move(other.data_ptrs))
        , cached_data(std::move(other.cached_data))
        , nested(std::move(other.nested))
        , initialized(other.initialized)
        , raw_write_ptrs(other.raw_write_ptrs)
        , swwc_staging(other.swwc_staging)
        , swwc_staging_initialized(other.swwc_staging_initialized)
    {
        other.raw_write_ptrs = nullptr;
        other.swwc_staging = nullptr;
    }

    ScatterState & operator=(ScatterState && other) noexcept
    {
        if (this != &other)
        {
            std::free(raw_write_ptrs);
            std::free(swwc_staging);
            fixed_ptrs = std::move(other.fixed_ptrs);
            data_ptrs = std::move(other.data_ptrs);
            cached_data = std::move(other.cached_data);
            nested = std::move(other.nested);
            initialized = other.initialized;
            raw_write_ptrs = other.raw_write_ptrs;
            swwc_staging = other.swwc_staging;
            swwc_staging_initialized = other.swwc_staging_initialized;
            other.raw_write_ptrs = nullptr;
            other.swwc_staging = nullptr;
        }
        return *this;
    }

    ~ScatterState()
    {
        std::free(raw_write_ptrs);
        std::free(swwc_staging);
    }
};

}
