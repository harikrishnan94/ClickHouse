#include <Interpreters/UnifiedHashJoin/SlotScatter.h>

#include <Columns/IColumn.h>
#include <Common/Arena.h>
#include <Common/PODArray.h>

#include <base/defines.h>

#include <optional>
#include <utility>
#include <vector>

namespace DB
{
namespace Unified
{
namespace
{

/// `routingHashForRow` expects a map-like object whose `hash` matches placement; forward to the
/// traits' statics instead of a concrete map.
template <typename Traits>
struct ScatterHashAdapter
{
    template <typename K>
    size_t hash(const K & key) const { return Traits::hash(key); }
};

template <typename Traits>
SlotScatter scatterImpl(
    const ColumnRawPtrs & key_columns,
    const Sizes & key_sizes,
    const ScatteredBlock::Selector & selector,
    size_t num_slots,
    bool is_asof)
{
    using KeyGetter = typename Traits::KeyGetter;

    /// Bucket selection must use the same hash as insertion; a precomputed hash could disagree with map placement.
    if constexpr (requires { KeyGetter::has_pre_computed_hashes; })
        static_assert(!KeyGetter::has_pre_computed_hashes, "Bucket routing assumes the map computes the hash it places by");

    /// The ASOF slicing that `createKeyGetter<KeyGetter, true>` applied: the last key column is
    /// the inequality column. `dense_keys` below still sees the full column list, as before.
    ColumnRawPtrs getter_columns = key_columns;
    Sizes getter_sizes = key_sizes;
    if (is_asof)
    {
        getter_columns.pop_back();
        getter_sizes.pop_back();
    }

    /// Build-side range getters use the default key range (no shift), matching the previous
    /// `createKeyGetter` call with a default `KeyRange`.
    KeyGetter key_getter(getter_columns, getter_sizes, nullptr);

    static constexpr ScatterHashAdapter<Traits> hash_adapter{};

    /// Key holders are only read here, never persisted into the map, so nothing allocated through
    /// this outlives the call. String key holders do not touch it at all.
    Arena scratch_pool;

    const size_t rows = selector.size();

    PODArray<UInt32> row_to_slot(rows);
    std::vector<size_t> counts(num_slots, 0);
    for (size_t i = 0; i < rows; ++i)
    {
        auto key_holder = key_getter.getKeyHolder(selector[i], scratch_pool);
        const auto & key = keyHolderGetKey(key_holder);

        size_t hash_value = 0;
        if constexpr (requires { key_getter.routingHashForRow(hash_adapter, selector[i], scratch_pool); })
            hash_value = key_getter.routingHashForRow(hash_adapter, selector[i], scratch_pool);
        else
            hash_value = Traits::hash(key);

        const size_t bucket = Traits::getBucketFromHash(Traits::bucketRoutingHash(key, hash_value));
        const auto slot = static_cast<UInt32>(slotForBucket(bucket, num_slots));
        row_to_slot[i] = slot;
        ++counts[slot];
    }

    std::vector<ScatteredBlock::Selector::IndexesPtr> indexes;
    indexes.reserve(num_slots);
    for (size_t slot = 0; slot < num_slots; ++slot)
    {
        auto column = ScatteredBlock::Selector::Indexes::create();
        column->getData().reserve(counts[slot]);
        indexes.push_back(std::move(column));
    }

    for (size_t i = 0; i < rows; ++i)
        indexes[row_to_slot[i]]->getData().push_back(selector[i]);

    SlotScatter result;
    result.selectors.reserve(num_slots);
    for (auto & column : indexes)
        result.selectors.emplace_back(std::move(column));

    constexpr size_t threshold = sizeof(IColumn::Selector::value_type);
    size_t max_bytes_per_row = 0;
    for (const auto * column : key_columns)
        max_bytes_per_row += (column->valuesHaveFixedSize() && !column->lowCardinality()) ? column->sizeOfValueIfFixed() : threshold + 1;

    const bool selector_is_identity = selector.isContinuousRange() && selector.getRange().first == 0
        && !key_columns.empty() && selector.getRange().second == key_columns[0]->size();

    if (max_bytes_per_row <= threshold && selector_is_identity)
    {
        IColumn::Selector column_selector(rows);
        for (size_t i = 0; i < rows; ++i)
            column_selector[i] = row_to_slot[i];

        result.dense_keys.resize(num_slots);
        for (const auto * column : key_columns)
        {
            auto parts = column->scatter(num_slots, column_selector);
            chassert(parts.size() == num_slots);
            for (size_t slot = 0; slot < num_slots; ++slot)
                result.dense_keys[slot].push_back(std::move(parts[slot]));
        }
    }

    return result;
}

}

SlotScatter scatterBlockBySlot(
    HashJoin::Type type,
    MapsKind maps_kind,
    const ColumnRawPtrs & key_columns,
    const Sizes & key_sizes,
    const ScatteredBlock::Selector & selector,
    size_t num_slots,
    bool is_asof)
{
    switch (type)
    {
#define M(NAME) \
        case HashJoin::Type::NAME: \
        { \
            using MapOne = typename decltype(std::declval<HashJoin::MapsOne>().NAME)::element_type; \
            if constexpr (MapOne::isFixedRangeStorage()) \
            { \
                switch (maps_kind) \
                { \
                    case MapsKind::One: \
                        return scatterImpl<SlotScatterTraits<HashJoin::Type::NAME, MapOne>>( \
                            key_columns, key_sizes, selector, num_slots, is_asof); \
                    case MapsKind::All: \
                    { \
                        using MapAll = typename decltype(std::declval<HashJoin::MapsAll>().NAME)::element_type; \
                        return scatterImpl<SlotScatterTraits<HashJoin::Type::NAME, MapAll>>( \
                            key_columns, key_sizes, selector, num_slots, is_asof); \
                    } \
                    case MapsKind::Asof: \
                    { \
                        using MapAsof = typename decltype(std::declval<HashJoin::MapsAsof>().NAME)::element_type; \
                        return scatterImpl<SlotScatterTraits<HashJoin::Type::NAME, MapAsof>>( \
                            key_columns, key_sizes, selector, num_slots, is_asof); \
                    } \
                } \
            } \
            else \
                return scatterImpl<SlotScatterTraits<HashJoin::Type::NAME, MapOne>>( \
                    key_columns, key_sizes, selector, num_slots, is_asof); \
        }

            UNIFIED_APPLY_FOR_JOIN_VARIANTS(M)
#undef M
    }
    UNREACHABLE();
}

}
}
