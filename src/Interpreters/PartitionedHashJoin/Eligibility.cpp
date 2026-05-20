#include <Interpreters/PartitionedHashJoin/Eligibility.h>

#include <algorithm>
#include <Common/assert_cast.h>

#include <DataTypes/DataTypeArray.h>
#include <DataTypes/DataTypeFixedString.h>
#include <DataTypes/DataTypeLowCardinality.h>
#include <DataTypes/DataTypeMap.h>
#include <DataTypes/DataTypeNullable.h>
#include <DataTypes/DataTypeString.h>
#include <DataTypes/DataTypesNumber.h>

namespace DB
{

/// Recursively unwrap Nullable and LowCardinality to reach the underlying numeric/fixed type.
/// Returns 0 if the bottom type is variable-width (String, Array, Map, Tuple-with-variable, etc.).
size_t fixedElemBytes(const DataTypePtr & dt)
{
    if (!dt)
        return 0;

    /// Peel Nullable: the null map is handled separately; the data contributes its inner width.
    if (dt->isNullable())
    {
        const auto & nullable = assert_cast<const DataTypeNullable &>(*dt);
        return fixedElemBytes(nullable.getNestedType());
    }

    /// Peel LowCardinality: treat as its dictionary value type.
    if (dt->getTypeId() == TypeIndex::LowCardinality)
    {
        const DataTypePtr inner = removeLowCardinality(dt);
        if (inner->isNullable())
            return fixedElemBytes(assert_cast<const DataTypeNullable &>(*inner).getNestedType());
        return fixedElemBytes(inner);
    }

    /// FixedString(N): fixed, but only supported scatter sizes are 1,2,4,8,16.
    if (dt->getTypeId() == TypeIndex::FixedString)
    {
        const auto & fs = assert_cast<const DataTypeFixedString &>(*dt);
        const size_t n = fs.getN();
        if (n == 1 || n == 2 || n == 4 || n == 8 || n == 16)
            return n;
        return 0; /// >16 byte FixedString rejected
    }

    /// All other types: use haveMaximumSizeOfValue() to check fixed-width.
    if (!dt->haveMaximumSizeOfValue())
        return 0;

    const size_t sz = dt->getMaximumSizeOfValueInMemory();
    /// We support 1, 2, 4, 8, and 16 byte fixed types.
    if (sz == 1 || sz == 2 || sz == 4 || sz == 8 || sz == 16)
        return sz;

    return 0; /// wider fixed types (e.g. Decimal256, UInt256) rejected
}

bool isSupportedByColumns(const Block & right_sample, const Names & key_names, const Names & kept_payload_names)
{
    /// Spec §2.1: sum of key fixed widths ≤ 16 bytes (128 bits).
    size_t key_bytes_total = 0;
    for (const auto & name : key_names)
    {
        if (!right_sample.has(name))
            return false;

        const auto & col = right_sample.getByName(name);
        const size_t sz = fixedElemBytes(col.type);
        if (sz == 0)
            return false; /// variable-width key → ineligible
        key_bytes_total += sz;
    }
    if (key_bytes_total > 16) /// 128 bits
        return false;

    /// Spec §2.2: every kept payload column must also be fixed-width.
    /// Unknown columns are ignored (conservative: fall through to the next algorithm).
    return std::ranges::all_of(
        kept_payload_names,
        [&](const auto & name)
        {
            if (!right_sample.has(name))
                return true;
            return fixedElemBytes(right_sample.getByName(name).type) != 0;
        });
}

}
