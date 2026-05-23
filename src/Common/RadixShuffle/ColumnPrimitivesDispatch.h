#pragma once

#include <Common/RadixShuffle/ColumnPrimitives.h>


namespace DB
{
class IDataType;
}


namespace DB::RadixShuffle
{

/// Resolve the column-primitive triple for a given column data type
/// (§3 last bullet).
///
/// Supported types (scope D, §2 non-goal #5):
///   - `ColumnVector<T>` for every numeric T: UInt8/16/32/64/128/256,
///     Int8/16/32/64/128/256, BFloat16, Float32/64, UUID, IPv4, IPv6.
///   - `ColumnDecimal<T>` for every decimal T: Decimal32/64/128/256,
///     DateTime64, Time64.
///   - `ColumnFixedString(n)`.
///   - `ColumnString`.
///   - `ColumnNullable(X)` wrapping each of the above.
///
/// Unsupported types throw a `LOGICAL_ERROR` exception. Reset / rebuild
/// the column primitives on a fresh `IDataType &` per shuffle pass.
[[nodiscard]] ColumnPrimitives resolveColumnPrimitives(const IDataType & type);

}
