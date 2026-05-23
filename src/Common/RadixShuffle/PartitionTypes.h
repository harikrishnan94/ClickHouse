#pragma once

#include <cstddef>
#include <cstdint>


namespace DB::RadixShuffle
{

/// Per-column description used by the allocator to size chunks and route
/// reservation byte requests. Captured at construction time and never
/// rebuilt per call.
struct ColumnDesc
{
    /// Bytes per "primary" row. For fixed-width columns this is the element
    /// width; for variable-length columns this is 0 and the bytes a row
    /// consumes is determined by the caller's per-batch reservation request.
    size_t element_size = 0;

    /// Alignment required by the primary buffer. Always a power of two.
    size_t alignment = 1;

    /// Whether the column has a parallel offsets array (one UInt64 per row).
    /// True for `ColumnString` (variable-length); false otherwise.
    bool has_offsets = false;

    /// Whether the column has a parallel null map (one UInt8 per row). True
    /// for `ColumnNullable(X)`; false otherwise.
    bool has_null_map = false;

    /// True when the column is variable-length (the primary buffer is sized
    /// by bytes, not rows).
    bool variable_length = false;
};


/// One chunk in a chain. Allocated by `Allocator`; owned by the allocator's
/// per-handle arena. The caller observes a chunk through its primary/offset/
/// null pointers and a row capacity (and, for variable-length columns, a
/// byte capacity for the primary region).
struct Chunk
{
    /// Primary data buffer. For fixed-width columns this is `row_capacity *
    /// element_size` bytes. For variable-length columns this is
    /// `byte_capacity` bytes.
    void * primary = nullptr;

    /// Per-row offsets buffer (variable-length only). Each entry stores the
    /// end-position within `primary` of one row's bytes. By the convention
    /// used by `ColumnString`, the implicit start position for row 0 is 0.
    /// Reservations within a chunk write offsets that are local to that
    /// chunk; reconstruct decodes per-row sizes via successive differences.
    uint64_t * offsets = nullptr;

    /// Per-row null map buffer (nullable only). Each entry is 0 (not null)
    /// or 1 (null). Caller writes one entry per scattered row.
    uint8_t * null_map = nullptr;

    /// Row capacity of the chunk. Reservations carve subranges
    /// `[begin_row, begin_row + reserved_rows)` from this range.
    size_t row_capacity = 0;

    /// Byte capacity of the `primary` region (variable-length only). Equal
    /// to `row_capacity * element_size` for fixed-width columns and tracked
    /// only for parity in that case.
    size_t byte_capacity = 0;
};


/// One row-block view: a chunk plus a half-open row range `[begin, end)`
/// identifying which rows the operator actually filled. Reconstruct never
/// sees reservation gaps; the operator passes only the genuinely-occupied
/// part of each chunk.
struct ChunkRangeView
{
    const Chunk * chunk = nullptr;
    size_t begin = 0;
    size_t end = 0;
};


/// Caller's per-partition reservation request. Variable-length columns must
/// also supply `bytes`; fixed-width columns ignore it.
struct ReservationRequest
{
    size_t rows = 0;
    size_t bytes = 0;
};


/// Allocator's reservation output: the carved slot for one (column,
/// partition) on one batch. The slot occupies rows
/// `[begin_row, begin_row + reserved_rows)` of `chunk` and (variable-length)
/// bytes `[begin_byte, begin_byte + reserved_bytes)` of `chunk->primary`.
struct Reservation
{
    Chunk * chunk = nullptr;
    size_t begin_row = 0;
    size_t reserved_rows = 0;

    /// Variable-length only: starting byte offset within `chunk->primary`
    /// for the slot's content. Equal to `chunk->offsets[begin_row - 1]` or
    /// 0 if `begin_row == 0`. Cached here so scatter does not have to read
    /// the offsets array back to find its write position.
    size_t begin_byte = 0;
    size_t reserved_bytes = 0;
};


/// Resume cursor for reconstruct's pump pattern (§3.3). `(0, 0)` is the
/// natural start; the value returned by reconstruct is the next position
/// into the input buffer list.
struct ResumePosition
{
    size_t view_index = 0;
    size_t rows_consumed_in_view = 0;
};

}
