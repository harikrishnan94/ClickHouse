#pragma once

#include <Common/RadixShuffle/ColumnPrimitives.h>


namespace DB::RadixShuffle
{

/// Build column primitives for `ColumnVector<T>` (fixed-width PODArray-backed
/// numeric / IP / UUID / floating-point columns). `T` is the element type
/// as stored by the column (e.g., `UInt32`, `Float64`).
template <typename T>
ColumnPrimitives makeFixedWidth();


/// Build column primitives for `ColumnDecimal<T>` (Decimal32/64/128/256,
/// DateTime64, Time64).
template <typename T>
ColumnPrimitives makeDecimal();


/// Build column primitives for `ColumnFixedString` with row width `n`.
ColumnPrimitives makeFixedString(size_t n);

}
