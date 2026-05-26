#pragma once

#include <Common/RadixShuffle/ColumnPrimitives.h>


namespace DB
{

/// Build the column primitives for `ColumnString`. The scatter primitive
/// writes:
///   - Per-row offsets: the cumulative byte end-position within the
///     chunk's `primary` buffer (chunk-global). This lets multiple
///     reservations within the same chunk coexist without disambiguation.
///   - Per-row characters: appended contiguously into the chunk's primary
///     buffer starting at `Reservation::begin_byte`.
///
/// Reconstruct re-bases offsets into the target `chars` buffer using the
/// target's existing tail position; it walks each chunk-range view's
/// offsets to determine the per-row byte slice.
ColumnPrimitives makeString();

}
