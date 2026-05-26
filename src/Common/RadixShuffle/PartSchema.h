#pragma once

#include <cstddef>
#include <vector>


namespace DB
{

enum class SlotRole
{
    Values, ///< Fixed-width value array (ColumnVector<T> or ColumnDecimal<T>).
    Offsets, ///< Per-row byte end-position array (ColumnString).
    NullMap, ///< Per-row null flag (ColumnNullable(X)).
    FixedStringChars, ///< Fixed-width character block (ColumnFixedString(n)).
};


/// Description of one fixed slot inside a partition's fixed chunk.
struct FixedSlot
{
    size_t col_idx; ///< Index of the originating column.
    SlotRole role;
    size_t element_size; ///< Bytes per row for this slot.
    size_t alignment; ///< Required alignment (power of two).
};


/// Static layout descriptor built once at operator construction.
/// Batch code only passes dynamic row counts and varlen byte totals;
/// all buffer sizing is derived from this schema.
///
/// Fixed-chunk layout is column-major (planar): each slot's array is
/// stored contiguously.  The byte address of row r in slot s is:
///
///     fixed->slot_byte_offsets[s] + r * fixed_slots[s].element_size
///
/// where `slot_byte_offsets` is stored per-chunk in FixedChunk because
/// it depends on the chunk's row capacity.  `slot_byte_offset` below
/// stores the 1-row layout values used by the schema builder.
struct PartSchema
{
    std::vector<FixedSlot> fixed_slots;

    /// Byte offset of each slot at row 0 in a 1-row layout chunk.
    /// Used by the schema builder; actual per-chunk offsets live in
    /// FixedChunk::slot_byte_offsets.
    std::vector<size_t> slot_byte_offset;

    /// Approximate per-row byte total across all slots (1-row layout,
    /// including inter-slot alignment padding).  Used for chunk-size hints;
    /// exact chunk sizes are computed at allocation time.
    size_t fixed_bytes_per_row = 0;

    bool has_varlen_portion = false;
};

}
