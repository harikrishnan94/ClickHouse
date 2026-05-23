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


namespace DB
{

namespace ErrorCodes
{
extern const int NOT_IMPLEMENTED;
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

}


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

}
