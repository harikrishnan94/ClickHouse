#pragma once

#include <Common/RadixShuffle/ColumnPrimitives.h>
#include <Common/RadixShuffle/PartSchema.h>

#include <vector>


namespace DB
{
class IDataType;
using DataTypePtr = std::shared_ptr<const IDataType>;
}


namespace DB
{

/// Resolve the column-primitive triple for a given column data type.
///
/// The returned ColumnPrimitives has valid scatter/reconstruct/hash
/// function pointers but EMPTY fixed_slot_indices — callers that need
/// a fully-bound primitive should use buildSchemaAndPrimitives instead.
///
/// Supported types:
///   - ColumnVector<T> for every numeric T.
///   - ColumnDecimal<T> for every decimal T.
///   - ColumnFixedString(n).
///   - ColumnString.
///   - ColumnNullable(X) wrapping each of the above.
///
/// Throws NOT_IMPLEMENTED for unsupported types.
[[nodiscard]] ColumnPrimitives resolveColumnPrimitives(const IDataType & type);


/// Schema and primitives built together so slot indices are consistent.
struct SchemaAndPrimitives
{
    PartSchema schema;
    std::vector<ColumnPrimitives> primitives;
};

/// Canonical entry point: resolve primitives AND build the PartSchema in
/// one pass.  After this call each primitive's fixed_slot_indices is
/// populated and consistent with schema.fixed_slots.
///
/// NullMap slot is always first in PartSchema::fixed_slots for Nullable
/// columns, so Nullable::scatter can access fixed_slot_indices[0] for the
/// null map without any additional lookup.
[[nodiscard]] SchemaAndPrimitives buildSchemaAndPrimitives(const std::vector<DataTypePtr> & types);

}
