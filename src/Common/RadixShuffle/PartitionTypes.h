#pragma once

#include <cstddef>
#include <cstdint>


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

}
