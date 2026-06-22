#pragma once

#if defined(OS_LINUX)

#include <Storages/SharedMemorySource/Wire/Layout.h>
#include <Core/TypeId.h>

#include <optional>


namespace DB::SharedMemoryWire
{

/// Single source of truth mapping a ClickHouse `TypeIndex` to the wire column
/// tag the SHM-adoption ABI uses for it, or `std::nullopt` when the type is
/// outside the supported set. Shared by:
///   - the SQL-side type gate (`TableFunctionShm::executeImpl`),
///   - the handshake membership re-check (`PollableShmSource::ensureAttached`),
///   - the adoption dispatch (`AdoptionLayer::adopt`),
///   - producers (e.g. the in-process test producer and the pg_clickhouse one).
///
/// Date/DateTime/Date32 map to their own wire tags (not the tag of their
/// fixed-width storage column) so the per-block descriptor's `type` is
/// unambiguous; the consumer adopts each into the matching `ColumnVector<T>`
/// (UInt16/UInt32/Int32). Decimal32/64/128 and DateTime64 adopt into the
/// matching `ColumnDecimal<T>`; their scale is NOT on the wire and is taken
/// from the (cross-validated) SQL/handshake DataType at adoption time.
/// `Decimal256` is intentionally absent: it is declined at the SQL gate (the
/// adopted-column path covers widths up to 16 bytes), so an attempt to offload
/// a `Decimal256` column fails closed rather than corrupting. String is a
/// separate, variable-width case.
inline std::optional<WireColumnType> tryWireColumnTypeForTypeIndex(TypeIndex type_id) noexcept
{
    switch (type_id)
    {
        case TypeIndex::UInt8:      return WireColumnType::UInt8;
        case TypeIndex::UInt16:     return WireColumnType::UInt16;
        case TypeIndex::UInt32:     return WireColumnType::UInt32;
        case TypeIndex::UInt64:     return WireColumnType::UInt64;
        case TypeIndex::Int8:       return WireColumnType::Int8;
        case TypeIndex::Int16:      return WireColumnType::Int16;
        case TypeIndex::Int32:      return WireColumnType::Int32;
        case TypeIndex::Int64:      return WireColumnType::Int64;
        case TypeIndex::Float32:    return WireColumnType::Float32;
        case TypeIndex::Float64:    return WireColumnType::Float64;
        case TypeIndex::Date:       return WireColumnType::Date;
        case TypeIndex::DateTime:   return WireColumnType::DateTime;
        case TypeIndex::Date32:     return WireColumnType::Date32;
        case TypeIndex::Decimal32:  return WireColumnType::Decimal32;
        case TypeIndex::Decimal64:  return WireColumnType::Decimal64;
        case TypeIndex::Decimal128: return WireColumnType::Decimal128;
        case TypeIndex::DateTime64: return WireColumnType::DateTime64;
        case TypeIndex::String:     return WireColumnType::String;
        default:                    return std::nullopt;
    }
}

/// True iff `type_id` is in the supported SHM-adoption set.
inline bool isSupportedShmType(TypeIndex type_id) noexcept
{
    return tryWireColumnTypeForTypeIndex(type_id).has_value();
}

/// Human-readable list of supported SQL types, for error messages.
inline const char * supportedShmTypeList() noexcept
{
    return "{UInt8, UInt16, UInt32, UInt64, Int8, Int16, Int32, Int64, "
           "Float32, Float64, Date, DateTime, Date32, "
           "Decimal32, Decimal64, Decimal128, DateTime64, String}";
}

}

#endif
