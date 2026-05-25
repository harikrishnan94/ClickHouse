#include <Common/RadixShuffle/ColumnPrimitivesDispatch.h>

#include <Common/RadixShuffle/ColumnPrimitives/FixedWidth.h>
#include <Common/RadixShuffle/ColumnPrimitives/Nullable.h>
#include <Common/RadixShuffle/ColumnPrimitives/String.h>

#include <Core/TypeId.h>
#include <Core/Types.h>
#include <DataTypes/DataTypeFixedString.h>
#include <DataTypes/DataTypeNullable.h>
#include <DataTypes/IDataType.h>
#include <base/Decimal.h>
#include <base/IPv4andIPv6.h>
#include <Common/Exception.h>

#include <utility>


namespace DB
{

namespace ErrorCodes
{
extern const int NOT_IMPLEMENTED;
extern const int LOGICAL_ERROR;
}

}


namespace DB::RadixShuffle
{

namespace
{

[[nodiscard]] ColumnPrimitives resolveLeaf(const IDataType & type)
{
    switch (type.getColumnType())
    {
        case TypeIndex::UInt8:
            return makeFixedWidth<UInt8>();
        case TypeIndex::UInt16:
            return makeFixedWidth<UInt16>();
        case TypeIndex::UInt32:
            return makeFixedWidth<UInt32>();
        case TypeIndex::UInt64:
            return makeFixedWidth<UInt64>();
        case TypeIndex::UInt128:
            return makeFixedWidth<UInt128>();
        case TypeIndex::UInt256:
            return makeFixedWidth<UInt256>();
        case TypeIndex::Int8:
            return makeFixedWidth<Int8>();
        case TypeIndex::Int16:
            return makeFixedWidth<Int16>();
        case TypeIndex::Int32:
            return makeFixedWidth<Int32>();
        case TypeIndex::Int64:
            return makeFixedWidth<Int64>();
        case TypeIndex::Int128:
            return makeFixedWidth<Int128>();
        case TypeIndex::Int256:
            return makeFixedWidth<Int256>();
        case TypeIndex::BFloat16:
            return makeFixedWidth<BFloat16>();
        case TypeIndex::Float32:
            return makeFixedWidth<Float32>();
        case TypeIndex::Float64:
            return makeFixedWidth<Float64>();
        case TypeIndex::UUID:
            return makeFixedWidth<UUID>();
        case TypeIndex::IPv4:
            return makeFixedWidth<IPv4>();
        case TypeIndex::IPv6:
            return makeFixedWidth<IPv6>();

        case TypeIndex::Decimal32:
            return makeDecimal<Decimal32>();
        case TypeIndex::Decimal64:
            return makeDecimal<Decimal64>();
        case TypeIndex::Decimal128:
            return makeDecimal<Decimal128>();
        case TypeIndex::Decimal256:
            return makeDecimal<Decimal256>();
        case TypeIndex::DateTime64:
            return makeDecimal<DateTime64>();
        case TypeIndex::Time64:
            return makeDecimal<Time64>();

        case TypeIndex::Date:
            return makeFixedWidth<UInt16>();
        case TypeIndex::Date32:
            return makeFixedWidth<Int32>();
        case TypeIndex::DateTime:
            return makeFixedWidth<UInt32>();
        case TypeIndex::Time:
            return makeFixedWidth<Int32>();
        case TypeIndex::Enum8:
            return makeFixedWidth<Int8>();
        case TypeIndex::Enum16:
            return makeFixedWidth<Int16>();
        case TypeIndex::Interval:
            return makeFixedWidth<Int64>();

        case TypeIndex::String:
            return makeString();

        case TypeIndex::FixedString: {
            const auto & fixed = static_cast<const DataTypeFixedString &>(type);
            return makeFixedString(fixed.getN());
        }

        default:
            throw Exception(
                ErrorCodes::NOT_IMPLEMENTED, "RadixShuffle::resolveColumnPrimitives: unsupported column data type '{}'", type.getName());
    }
}


/// Returns (element_size, alignment) for any scope-D leaf type.
/// Used by buildSchemaAndPrimitives to register the correct slot metadata.
std::pair<size_t, size_t> leafElementSizeAlign(const IDataType & type)
{
    switch (type.getColumnType())
    {
        case TypeIndex::UInt8:
        case TypeIndex::Int8:
        case TypeIndex::Enum8:
            return {1, 1};

        case TypeIndex::UInt16:
        case TypeIndex::Int16:
        case TypeIndex::Enum16:
        case TypeIndex::Date:
        case TypeIndex::BFloat16:
            return {2, 2};

        case TypeIndex::UInt32:
        case TypeIndex::Int32:
        case TypeIndex::Float32:
        case TypeIndex::Date32:
        case TypeIndex::DateTime:
        case TypeIndex::Time:
        case TypeIndex::Decimal32:
        case TypeIndex::IPv4:
            return {4, 4};

        case TypeIndex::UInt64:
        case TypeIndex::Int64:
        case TypeIndex::Float64:
        case TypeIndex::DateTime64:
        case TypeIndex::Time64:
        case TypeIndex::Decimal64:
        case TypeIndex::Interval:
            return {8, 8};

        case TypeIndex::UInt128:
        case TypeIndex::Int128:
        case TypeIndex::Decimal128:
        case TypeIndex::UUID:
        case TypeIndex::IPv6:
            return {16, 8};

        case TypeIndex::UInt256:
        case TypeIndex::Int256:
        case TypeIndex::Decimal256:
            return {32, 8};

        case TypeIndex::FixedString: {
            const auto & fs = static_cast<const DataTypeFixedString &>(type);
            return {fs.getN(), 1};
        }

        default:
            throw Exception(ErrorCodes::LOGICAL_ERROR, "RadixShuffle: leafElementSizeAlign: unhandled type '{}'", type.getName());
    }
}


/// Incrementally builds PartSchema by appending slots and tracking the
/// running 1-row reference offsets.
class PartSchemaBuilder
{
public:
    /// Add one slot; returns the slot index (index into fixed_slots).
    /// NullMap must be added before any other slot for a Nullable column.
    size_t addSlot(size_t col_idx, SlotRole role, size_t element_size, size_t alignment)
    {
        current_off_ = alignUp(current_off_, alignment);
        schema_.fixed_slots.push_back({col_idx, role, element_size, alignment});
        schema_.slot_byte_offset.push_back(current_off_);
        current_off_ += element_size;
        return schema_.fixed_slots.size() - 1;
    }

    PartSchema finish(bool has_varlen)
    {
        schema_.fixed_bytes_per_row = current_off_;
        schema_.has_varlen_portion = has_varlen;
        return std::move(schema_);
    }

private:
    static constexpr size_t alignUp(size_t n, size_t align) noexcept { return (n + (align - 1)) & ~(align - 1); }

    PartSchema schema_;
    size_t current_off_ = 0;
};


/// Walk the primitive tree rooted at `prim` (which was produced by
/// resolveLeaf/resolveColumnPrimitives), assign slot indices, and
/// populate prim.fixed_slot_indices / prim.writes_varlen.
///
/// For Nullable(X): adds the NullMap slot first, then recurses into X.
/// The nested primitive is replaced with a freshly-indexed copy so that
/// its fixed_slot_indices are correct.
void assignSlotIndices(ColumnPrimitives & prim, PartSchemaBuilder & builder, size_t col_idx, const IDataType & type)
{
    if (type.getTypeId() == TypeIndex::Nullable)
    {
        const auto & nullable_type = static_cast<const DataTypeNullable &>(type);
        const IDataType & nested_type = *nullable_type.getNestedType();

        /// NullMap slot must be first so that Nullable::scatter can
        /// address it via fixed_slot_indices[0].
        const size_t null_slot = builder.addSlot(col_idx, SlotRole::NullMap, sizeof(uint8_t), alignof(uint8_t));
        prim.fixed_slot_indices = {null_slot};

        /// Build a fresh indexed copy of the nested primitive.
        ColumnPrimitives nested_prim = resolveLeaf(nested_type);
        assignSlotIndices(nested_prim, builder, col_idx, nested_type);
        prim.writes_varlen = nested_prim.writes_varlen;
        prim.nested = std::make_shared<const ColumnPrimitives>(std::move(nested_prim));
        return;
    }

    if (type.getTypeId() == TypeIndex::String)
    {
        const size_t slot = builder.addSlot(col_idx, SlotRole::Offsets, sizeof(uint64_t), alignof(uint64_t));
        prim.fixed_slot_indices = {slot};
        prim.writes_varlen = true;
        return;
    }

    if (type.getTypeId() == TypeIndex::FixedString)
    {
        const auto & fs_type = static_cast<const DataTypeFixedString &>(type);
        const size_t n = fs_type.getN();
        const size_t slot = builder.addSlot(col_idx, SlotRole::FixedStringChars, n, 1);
        prim.fixed_slot_indices = {slot};
        prim.writes_varlen = false;
        return;
    }

    // ColumnVector / ColumnDecimal — single Values slot.
    auto [elem_size, elem_align] = leafElementSizeAlign(type);
    const size_t slot = builder.addSlot(col_idx, SlotRole::Values, elem_size, elem_align);
    prim.fixed_slot_indices = {slot};
    prim.writes_varlen = false;
}

} // namespace


ColumnPrimitives resolveColumnPrimitives(const IDataType & type)
{
    if (type.getTypeId() == TypeIndex::Nullable)
    {
        const auto & nullable = static_cast<const DataTypeNullable &>(type);
        ColumnPrimitives nested = resolveLeaf(*nullable.getNestedType());
        return makeNullable(std::move(nested));
    }
    return resolveLeaf(type);
}


SchemaAndPrimitives buildSchemaAndPrimitives(const std::vector<DataTypePtr> & types)
{
    PartSchemaBuilder builder;
    std::vector<ColumnPrimitives> primitives;
    bool has_varlen = false;

    for (size_t col_idx = 0; col_idx < types.size(); ++col_idx)
    {
        ColumnPrimitives prim = resolveColumnPrimitives(*types[col_idx]);
        assignSlotIndices(prim, builder, col_idx, *types[col_idx]);
        if (prim.writes_varlen)
            has_varlen = true;
        primitives.push_back(std::move(prim));
    }

    return {builder.finish(has_varlen), std::move(primitives)};
}

}
