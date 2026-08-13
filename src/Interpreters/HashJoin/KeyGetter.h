#pragma once
#include <Interpreters/HashJoin/HashJoin.h>
#include <Common/ColumnsHashing.h>
#include <Columns/ColumnLowCardinality.h>
#include <Columns/ColumnsNumber.h>


namespace DB
{

namespace ErrorCodes
{
    extern const int LOGICAL_ERROR;
}

template <HashJoin::Type type, typename Value, typename Mapped, bool use_offset>
struct KeyGetterForTypeImpl;

/// Key getter for a single LowCardinality column in HashJoin. Unlike aggregation's
/// `HashMethodSingleLowCardinalityColumn`: const-correct on probe, offset-carrying `FindResult`
/// for `JoinUsedFlags`, no null-key path, and per-dictionary-index dedup of HT work within a
/// block. Probe may be plain T vs LowCardinality(T) — then the base method runs directly.
template <typename BaseMethod, typename Mapped, bool use_offset>
struct LowCardinalityKeyGetterForJoin
{
    using MappedNonConst = std::remove_const_t<Mapped>;
    static constexpr bool has_mapped = !std::is_same_v<Mapped, void>;
    using EmplaceResult = BaseMethod::EmplaceResult;
    using FindResult = BaseMethod::FindResult;

    /// Dictionary-index lookup is not cheap; keeps probe software prefetch off (it fights the cache).
    static constexpr bool has_cheap_key_calculation = false;

    BaseMethod base;
    const IColumn * positions = nullptr;
    size_t size_of_index_type = 0;
    const UInt64 * saved_hash = nullptr;
    ColumnPtr dictionary_holder;

    /// Pointers into HT cells (stable during probe / lazy emit), not copies — copies would dangle
    /// and would not work for move-only `AsofRowRefs`.
    PaddedPODArray<UInt8> visit_cache;       /// 0 = not visited, 1 = found, 2 = not found
    PaddedPODArray<Mapped *> mapped_cache;
    PaddedPODArray<size_t> offset_cache;

    /// Nested dictionary column for LC keys; the column itself for plain keys.
    static const IColumn * getBaseColumn(const IColumn * column)
    {
        if (const auto * low_cardinality_column = typeid_cast<const ColumnLowCardinality *>(column))
            return low_cardinality_column->getDictionary().getNestedNotNullableColumn().get();
        return column;
    }

    LowCardinalityKeyGetterForJoin(const ColumnRawPtrs & key_columns, const Sizes & key_sizes, const ColumnsHashing::HashMethodContextPtr &)
        : base({getBaseColumn(key_columns[0])}, key_sizes, nullptr)
    {
        /// Build key is always LC; probe may be plain T (no dictionary / no dedup). Map stores
        /// key values, so plain probe still matches dictionary-encoded build.
        const auto * low_cardinality_column = typeid_cast<const ColumnLowCardinality *>(key_columns[0]);
        if (!low_cardinality_column)
            return;

        const auto & dictionary = low_cardinality_column->getDictionary();
        dictionary_holder = low_cardinality_column->getDictionaryPtr();
        saved_hash = dictionary.tryGetSavedHash();
        size_of_index_type = low_cardinality_column->getSizeOfIndexType();
        positions = low_cardinality_column->getIndexesPtr().get();

        const size_t dictionary_size = dictionary.getNestedNotNullableColumn()->size();
        visit_cache.assign(dictionary_size, static_cast<UInt8>(0));
        mapped_cache.assign(dictionary_size, static_cast<Mapped *>(nullptr));
        if constexpr (use_offset)
            offset_cache.assign(dictionary_size, static_cast<size_t>(0));
    }

    /// True when the current column is LowCardinality (dictionary path).
    ALWAYS_INLINE bool isLowCardinality() const { return positions != nullptr; }

    ALWAYS_INLINE size_t getIndexAt(size_t row) const
    {
        switch (size_of_index_type)
        {
            case sizeof(UInt8):  return assert_cast<const ColumnUInt8 *>(positions)->getElement(row);
            case sizeof(UInt16): return assert_cast<const ColumnUInt16 *>(positions)->getElement(row);
            case sizeof(UInt32): return assert_cast<const ColumnUInt32 *>(positions)->getElement(row);
            case sizeof(UInt64): return assert_cast<const ColumnUInt64 *>(positions)->getElement(row);
            default: throw Exception(ErrorCodes::LOGICAL_ERROR, "Unexpected size of index type for low cardinality column.");
        }
    }

    ALWAYS_INLINE auto getKeyHolder(size_t row, Arena & pool) const
    {
        return base.getKeyHolder(isLowCardinality() ? getIndexAt(row) : row, pool);
    }

    template <typename Data>
    ALWAYS_INLINE size_t routingHashForRow(const Data & data, size_t row_, Arena & pool) const
    {
        if (!isLowCardinality())
        {
            auto key_holder = base.getKeyHolder(row_, pool);
            return data.hash(keyHolderGetKey(key_holder));
        }

        const size_t row = getIndexAt(row_);
        /// Reuse the dictionary's saved hash so routing matches `emplace`.
        if (saved_hash)
            return saved_hash[row];

        auto key_holder = base.getKeyHolder(row, pool);
        return data.hash(keyHolderGetKey(key_holder));
    }

    /// Build inserts every row into the real cell — no per-index dedup (unlike aggregation).
    /// Dictionary speedup is probe-only.
    template <typename Data>
    ALWAYS_INLINE EmplaceResult emplaceKey(Data & data, size_t row_, Arena & pool)
    {
        /// Plain key on build is rare (build is LC); handled by the base method.
        if (!isLowCardinality())
            return base.emplaceKey(data, row_, pool);

        const size_t row = getIndexAt(row_);

        auto key_holder = base.getKeyHolder(row, pool);

        typename Data::LookupResult it;
        bool inserted = false;
        data.emplace(key_holder, it, inserted, routingHashForRow(data, row_, pool));

        auto & mapped = it->getMapped();
        if (inserted)
            new (&mapped) MappedNonConst();
        return EmplaceResult(mapped, mapped, inserted);
    }

    template <typename Data>
    ALWAYS_INLINE FindResult findKey(Data & data, size_t row_, Arena & pool)
    {
        /// Plain probe: map stores key values, so this still hits dictionary-encoded build rows.
        if (!isLowCardinality())
            return base.findKey(data, row_, pool);

        const size_t row = getIndexAt(row_);

        if (visit_cache[row] != 0)
        {
            size_t cached_offset = 0;
            if constexpr (use_offset)
                cached_offset = offset_cache[row];
            return FindResult(mapped_cache[row], visit_cache[row] == 1, cached_offset);
        }

        auto key_holder = base.getKeyHolder(row, pool);
        const auto key = keyHolderGetKey(key_holder);

        auto it = saved_hash ? data.find(key, saved_hash[row]) : data.find(key);

        const bool found = it;
        Mapped * mapped = found ? &it->getMapped() : nullptr;

        size_t offset = 0;
        /// Offset only for used flags; needs current bucket-prefix state.
        if constexpr (use_offset)
            offset = found ? data.offsetInternalUnsafe(it) : 0;

        visit_cache[row] = found ? 1 : 2;
        mapped_cache[row] = mapped;
        if constexpr (use_offset)
            offset_cache[row] = offset;
        return FindResult(mapped, found, offset);
    }
};

template <typename Value, typename Mapped, bool use_offset>
struct KeyGetterForTypeImpl<HashJoin::Type::key8, Value, Mapped, use_offset>
{
    using Type = ColumnsHashing::HashMethodOneNumber<Value, Mapped, UInt8, false, use_offset>;
};
template <typename Value, typename Mapped, bool use_offset>
struct KeyGetterForTypeImpl<HashJoin::Type::key16, Value, Mapped, use_offset>
{
    using Type = ColumnsHashing::HashMethodOneNumber<Value, Mapped, UInt16, false, use_offset>;
};
template <typename Value, typename Mapped, bool use_offset>
struct KeyGetterForTypeImpl<HashJoin::Type::key32, Value, Mapped, use_offset>
{
    using Type = ColumnsHashing::HashMethodOneNumber<Value, Mapped, UInt32, false, use_offset>;
};
template <typename Value, typename Mapped, bool use_offset>
struct KeyGetterForTypeImpl<HashJoin::Type::key64, Value, Mapped, use_offset>
{
    using Type = ColumnsHashing::HashMethodOneNumber<Value, Mapped, UInt64, false, use_offset>;
};
template <typename Value, typename Mapped, bool use_offset>
struct KeyGetterForTypeImpl<HashJoin::Type::key_string, Value, Mapped, use_offset>
{
    using Type = ColumnsHashing::HashMethodString<Value, Mapped, true, false, use_offset>;
};
template <typename Value, typename Mapped, bool use_offset>
struct KeyGetterForTypeImpl<HashJoin::Type::key_fixed_string, Value, Mapped, use_offset>
{
    using Type = ColumnsHashing::HashMethodFixedString<Value, Mapped, true, false, use_offset>;
};
template <typename Value, typename Mapped, bool use_offset>
struct KeyGetterForTypeImpl<HashJoin::Type::keys32, Value, Mapped, use_offset>
{
    using Type = ColumnsHashing::HashMethodKeysFixed<Value, UInt32, Mapped, false, false, false, use_offset>;
};
template <typename Value, typename Mapped, bool use_offset>
struct KeyGetterForTypeImpl<HashJoin::Type::keys64, Value, Mapped, use_offset>
{
    using Type = ColumnsHashing::HashMethodKeysFixed<Value, UInt64, Mapped, false, false, false, use_offset>;
};
template <typename Value, typename Mapped, bool use_offset>
struct KeyGetterForTypeImpl<HashJoin::Type::keys128, Value, Mapped, use_offset>
{
    using Type = ColumnsHashing::HashMethodKeysFixed<Value, UInt128, Mapped, false, false, false, use_offset>;
};
template <typename Value, typename Mapped, bool use_offset>
struct KeyGetterForTypeImpl<HashJoin::Type::keys256, Value, Mapped, use_offset>
{
    using Type = ColumnsHashing::HashMethodKeysFixed<Value, UInt256, Mapped, false, false, false, use_offset>;
};
template <typename Value, typename Mapped, bool use_offset>
struct KeyGetterForTypeImpl<HashJoin::Type::hashed, Value, Mapped, use_offset>
{
    using Type = ColumnsHashing::HashMethodHashed<Value, Mapped, false, use_offset>;
};
template <typename Value, typename Mapped, bool use_offset>
struct KeyGetterForTypeImpl<HashJoin::Type::low_cardinality_key_string, Value, Mapped, use_offset>
{
    using Type
        = LowCardinalityKeyGetterForJoin<ColumnsHashing::HashMethodString<Value, Mapped, true, false, use_offset>, Mapped, use_offset>;
};
template <typename Value, typename Mapped, bool use_offset>
struct KeyGetterForTypeImpl<HashJoin::Type::low_cardinality_key_fixed_string, Value, Mapped, use_offset>
{
    using Type
        = LowCardinalityKeyGetterForJoin<ColumnsHashing::HashMethodFixedString<Value, Mapped, true, false, use_offset>, Mapped, use_offset>;
};
#define KEYGETTER_RANGE_IMPL(TYPE, FIELD_TYPE) \
    template <typename Value, typename Mapped, bool use_offset> \
    struct KeyGetterForTypeImpl<HashJoin::Type::TYPE, Value, Mapped, use_offset> \
    { \
        using Type = ColumnsHashing::HashMethodOneNumberInRange<Value, Mapped, FIELD_TYPE, false, use_offset>; \
    };
KEYGETTER_RANGE_IMPL(range8_key32, UInt32)
KEYGETTER_RANGE_IMPL(range16_key32, UInt32)
KEYGETTER_RANGE_IMPL(range17_key32, UInt32)
KEYGETTER_RANGE_IMPL(range18_key32, UInt32)
KEYGETTER_RANGE_IMPL(range8_key64, UInt64)
KEYGETTER_RANGE_IMPL(range16_key64, UInt64)
KEYGETTER_RANGE_IMPL(range17_key64, UInt64)
KEYGETTER_RANGE_IMPL(range18_key64, UInt64)
#undef KEYGETTER_RANGE_IMPL

#define KEYGETTER_TWO_LEVEL_IMPL(NAME) \
    template <typename Value, typename Mapped, bool use_offset> \
    struct KeyGetterForTypeImpl<HashJoin::Type::two_level_##NAME, Value, Mapped, use_offset> \
        : KeyGetterForTypeImpl<HashJoin::Type::NAME, Value, Mapped, use_offset> \
    { \
    };
APPLY_FOR_SINGLE_LEVEL_JOIN_VARIANTS(KEYGETTER_TWO_LEVEL_IMPL)
#undef KEYGETTER_TWO_LEVEL_IMPL

template <HashJoin::Type type, typename Data, bool use_offset>
struct KeyGetterForType
{
    using Value = Data::value_type;
    using Mapped_t = Data::mapped_type;
    using Mapped = std::conditional_t<std::is_const_v<Data>, const Mapped_t, Mapped_t>;
    using Type = KeyGetterForTypeImpl<type, Value, Mapped, use_offset>::Type;
};
}
