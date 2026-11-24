#include <Interpreters/HashJoin/HashJoinPartitionedBuild.h>

#include <Interpreters/HashJoin/AddedColumns.h>
#include <Interpreters/HashJoin/HashJoinPartitionedBuildKeys.h>
#include <base/getL1CacheSize.h>
#include <base/getL2CacheSize.h>

#include <algorithm>
#include <bit>
#include <cstring>

namespace DB
{

namespace ErrorCodes
{
extern const int LOGICAL_ERROR;
}

namespace JoinPartitionedBuild
{

namespace
{

using detail::KeyKind;

/// The skip bytes of the rows of the key columns: 1 for a row with a NULL key, a row that the `ON` mask filters out, and a
/// row that `selector` does not list (such as a row with a NULL `ASOF` value). Returns nullptr when no row is skipped.
/// Sets `skipped_null_key` when a listed row has a NULL key.
const UInt8 * buildSkip(
    const WaitingBlockRecord::Clause & clause,
    const ScatteredBlock::Selector & selector,
    size_t num_rows,
    PaddedPODArray<UInt8> & skip,
    bool & skipped_null_key)
{
    IColumn::Filter buffer;
    const UInt8 * base = buildRowSkipData(clause.null_map, clause.join_mask, buffer, 0, num_rows);
    const bool all_listed = selector.isContinuousRange() && selector.getRange() == ScatteredBlock::Selector::Range{0, num_rows};

    skipped_null_key = false;
    if (clause.null_map)
    {
        const UInt8 * null_map = clause.null_map->data();
        UInt8 any_null = 0;
        if (all_listed)
            for (size_t i = 0; i < num_rows; ++i)
                any_null |= null_map[i];
        else
            for (const size_t row : selector)
                any_null |= null_map[row];
        skipped_null_key = any_null != 0;
    }

    if (all_listed && !base)
        return nullptr;

    skip.resize(num_rows);
    if (all_listed)
        memcpy(skip.data(), base, num_rows);
    else
    {
        std::fill(skip.begin(), skip.end(), UInt8(1));
        for (const size_t row : selector)
            skip[row] = base ? base[row] : 0;
    }

    UInt8 any = 0;
    for (size_t i = 0; i < num_rows; ++i)
        any |= skip[i];
    if (!any)
    {
        skip.clear();
        return nullptr;
    }
    return skip.data();
}

/// Calls `row(i)` for each row that is inserted, `skipped(i)` for the others. Two loops, so that the usual case, with no
/// skipped row, has no test per row.
template <typename Row, typename Skipped>
ALWAYS_INLINE void forRows(size_t num_rows, const UInt8 * skip, Row && row, Skipped && skipped)
{
    if (skip)
    {
        for (size_t i = 0; i < num_rows; ++i)
        {
            if (skip[i])
                skipped(i);
            else
                row(i);
        }
    }
    else
    {
        for (size_t i = 0; i < num_rows; ++i)
            row(i);
    }
}

/// Hashes the key of each row with the map's own hash and calls `sink(i, h)` for the rows that are inserted and
/// `skipped(i)` for the others. `key16` passes the key itself as `h`. `key128` receives the keys of `hashed` and packed
/// `String` maps.
template <HashJoin::Type TYPE, typename Maps, typename Sink, typename Skipped>
void hashRows(
    const HashJoinTable & table,
    const WaitingBlockRecord::Clause & clause,
    const ScatteredBlock::Selector & selector,
    size_t num_rows,
    const UInt8 * skip,
    PaddedPODArray<UInt128> & key128,
    Sink && sink,
    Skipped && skipped)
{
    constexpr KeyKind kind = detail::keyKindOf<TYPE>();
    constexpr bool is_asof_join = std::is_same_v<Maps, HashJoin::MapsAsof>;
    using Map = JoinPartMapOf<TYPE, Maps>;

    if constexpr (kind == KeyKind::Fixed16)
    {
        const auto * keys = reinterpret_cast<const UInt16 *>(clause.key_columns.at(0)->getRawData().data());
        forRows(num_rows, skip, [&](size_t i) { sink(i, keys[i]); }, skipped);
    }
    else if constexpr (kind == KeyKind::Number)
    {
        using Key = typename Map::key_type;
        const auto * keys = reinterpret_cast<const Key *>(clause.key_columns.at(0)->getRawData().data());
        const detail::MapHash<Map> hash{};
        forRows(num_rows, skip, [&](size_t i) { sink(i, hash(keys[i])); }, skipped);
    }
    else if constexpr (kind == KeyKind::Prepared)
    {
        using KeyGetter = typename KeyGetterForType<TYPE, Map>::Type;
        KeyGetter key_getter = createJoinKeyGetter<KeyGetter, is_asof_join>(clause.key_columns, table.keySizes());
        key_getter.buildAllKeys(selector, key128);
        const UInt128 * keys = key128.data();
        const detail::MapHash<Map> hash{};
        forRows(num_rows, skip, [&](size_t i) { sink(i, hash(keys[i])); }, skipped);
    }
    else if constexpr (kind == KeyKind::Packed || kind == KeyKind::String)
    {
        using KeyGetter = typename KeyGetterForType<TYPE, Map>::Type;
        KeyGetter key_getter = createJoinKeyGetter<KeyGetter, is_asof_join>(clause.key_columns, table.keySizes());
        /// The key holders are hashed and dropped, so the pool stays empty.
        Arena pool;
        const detail::MapHash<Map> hash{};
        forRows(
            num_rows,
            skip,
            [&](size_t i)
            {
                auto key_holder = key_getter.getKeyHolder(i, pool);
                sink(i, hash(keyHolderGetKey(key_holder)));
            },
            skipped);
    }
    else
        static_assert(kind != KeyKind::None, "Only the fill pass of a partitioned key type hashes rows");
}

template <HashJoin::Type TYPE, typename Maps>
void fillTyped(const HashJoinTable & table, WaitingBlockRecord::Clause & clause, const ScatteredBlock::Selector & selector,
               JoinKeySketch & sketch)
{
    constexpr KeyKind kind = detail::keyKindOf<TYPE>();
    const size_t num_rows = clause.key_columns.at(0)->size();
    const UInt8 * skip = buildSkip(clause, selector, num_rows, clause.skip, clause.skipped_null_key);

    clause.route.resize(num_rows);
    UInt16 * route = clause.route.data();
    const auto skipped = [&](size_t i) { route[i] = 0; };

    if constexpr (kind == KeyKind::Fixed16)
    {
        /// The partitions of `key16` are the cache-line buckets of the flat table. The route keeps the bucket in its top
        /// bits, where the scatter reads the partition of any other key.
        constexpr size_t route_shift = 16 - HashJoin::key16_partition_bits;
        hashRows<TYPE, Maps>(table, clause, selector, num_rows, skip, clause.key128,
            [&](size_t i, UInt16 key)
            {
                route[i] = static_cast<UInt16>(JoinPartMapOf<TYPE, Maps>::bucketOfKey(key) << route_shift);
                sketch.insert(key);
            },
            skipped);
    }
    else
    {
        UInt64 * hash64 = nullptr;
        if constexpr (kind == KeyKind::String)
        {
            clause.hash64.resize(num_rows);
            hash64 = clause.hash64.data();
        }
        hashRows<TYPE, Maps>(table, clause, selector, num_rows, skip, clause.key128,
            [&](size_t i, UInt64 h)
            {
                route[i] = static_cast<UInt16>(static_cast<UInt32>(h) >> 16);
                /// The owner's emplace takes the saved hash, so a string is hashed once.
                if constexpr (kind == KeyKind::String)
                    hash64[i] = h;
                sketch.insert(static_cast<UInt32>(h));
            },
            skipped);
    }
}

template <HashJoin::Type TYPE, typename Maps>
void sketchTyped(const HashJoinTable & table, const WaitingBlockRecord::Clause & clause, const ScatteredBlock::Selector & selector,
                 JoinKeySketch & sketch)
{
    constexpr KeyKind kind = detail::keyKindOf<TYPE>();
    const size_t num_rows = clause.key_columns.at(0)->size();
    PaddedPODArray<UInt8> skip_buffer;
    bool skipped_null_key = false;
    const UInt8 * skip = buildSkip(clause, selector, num_rows, skip_buffer, skipped_null_key);
    const auto skipped = [](size_t) {};

    if constexpr (TYPE == HashJoin::Type::key8)
    {
        const auto * keys = reinterpret_cast<const UInt8 *>(clause.key_columns.at(0)->getRawData().data());
        forRows(num_rows, skip, [&](size_t i) { sketch.insert(keys[i]); }, skipped);
    }
    else if constexpr (TYPE == HashJoin::Type::low_cardinality_key_string || TYPE == HashJoin::Type::low_cardinality_key_fixed_string)
    {
        /// The `LowCardinality` maps hold the values with the hash of the `String` map.
        const detail::MapHash<JoinPartMapOf<HashJoin::Type::key_string, Maps>> hash{};
        const IColumn & column = *clause.key_columns.at(0);
        forRows(num_rows, skip, [&](size_t i) { sketch.insert(static_cast<UInt32>(hash(column.getDataAt(i)))); }, skipped);
    }
    else if constexpr (kind == KeyKind::None)
        throw Exception(ErrorCodes::LOGICAL_ERROR, "The join key type {} has no key sketch", static_cast<int>(TYPE));
    else
    {
        PaddedPODArray<UInt128> key128;
        hashRows<TYPE, Maps>(table, clause, selector, num_rows, skip, key128,
            [&](size_t, UInt64 h) { sketch.insert(static_cast<UInt32>(h)); }, skipped);
    }
}

}

void fillClause(const HashJoinTable & table, bool is_asof, WaitingBlockRecord::Clause & clause,
                const ScatteredBlock::Selector & selector, JoinKeySketch & sketch)
{
    if (is_asof != table.holds<HashJoin::MapsAsof>())
        throw Exception(ErrorCodes::LOGICAL_ERROR, "The fill pass of an ASOF join got a table of other maps, or the reverse");
    detail::dispatchMaps(table, [&]<typename Maps>()
    {
        table.dispatchType([&]<HashJoin::Type TYPE>()
        {
            if constexpr (detail::keyKindOf<TYPE>() == KeyKind::None)
                throw Exception(ErrorCodes::LOGICAL_ERROR, "The join key type {} is not built in partitions", static_cast<int>(TYPE));
            else
                fillTyped<TYPE, Maps>(table, clause, selector, sketch);
        });
    });
}

void sketchClause(const HashJoinTable & table, bool is_asof, const WaitingBlockRecord::Clause & clause,
                  const ScatteredBlock::Selector & selector, JoinKeySketch & sketch)
{
    if (is_asof != table.holds<HashJoin::MapsAsof>())
        throw Exception(ErrorCodes::LOGICAL_ERROR, "The key sketch of an ASOF join got a table of other maps, or the reverse");
    detail::dispatchMaps(table, [&]<typename Maps>()
    {
        table.dispatchType([&]<HashJoin::Type TYPE>() { sketchTyped<TYPE, Maps>(table, clause, selector, sketch); });
    });
}

}

size_t maxJoinPartitions()
{
    /// A probe row reads one 16-byte descriptor of its partition. The descriptors take at most a quarter of the L1, so that
    /// the cell they point at and the key and result columns stay in it too.
    const size_t l1_bytes = std::max<size_t>(getL1CacheSize(), 32 << 10);
    const size_t partitions = std::bit_floor(l1_bytes / 4 / 16);
    /// Partition ids are `UInt16`, and the scatter takes id P for the rows that are not inserted.
    return std::min<size_t>(partitions, size_t(1) << 15);
}

JoinPartitionPlan choosePartitionPlan(
    size_t estimate,
    const char * estimate_source,
    size_t waiting_rows,
    size_t num_build_streams,
    size_t parallel_hash_join_threshold,
    size_t max_threads,
    bool early,
    bool robin_hood_setting,
    size_t (*buffer_bytes_for_reserve)(size_t keys, bool robin_hood))
{
    const size_t cap = maxJoinPartitions();
    JoinPartitionPlan plan;
    plan.estimate = estimate;
    plan.estimate_source = estimate_source;

    if (early)
    {
        /// The estimate of an early build can be a small part of the final key count, so the tables are not sized from it.
        /// They start small and grow, and each may switch to the Robin Hood mode on its own.
        const size_t partitions = std::min(std::max<size_t>(256, std::bit_ceil(2 * std::max<size_t>(max_threads, 1))), cap);
        plan.partition_bits = std::countr_zero(partitions);
        plan.reserve_per_partition = 0;
        plan.robin_hood = robin_hood_setting;
        return plan;
    }

    /// Keys reserved per table for `partitions` tables: ceil(E * 1.2 / P). The margin covers the spread between partitions,
    /// so that no table grows during the build.
    const auto reserve_for = [&](size_t partitions) { return (estimate * 6 + 5 * partitions - 1) / (5 * partitions); };
    /// A reserve of more than 2^15 keys gives the grower 2^16 cells or more, where it switches to the Robin Hood mode.
    const auto robin_hood_for = [&](size_t reserve) { return robin_hood_setting && reserve > 32768; };

    /// The fewest partitions whose table buffer for its reserve, in the mode that reserve gets, fits 0.8 of the L2. The rest of
    /// the L2 holds the piece that the owner reads and its arena.
    const size_t l2_share = std::max<size_t>(getL2CacheSize(), 1 << 20) * 4 / 5;
    size_t bits = 0;
    while (bits < 15)
    {
        const size_t r = reserve_for(size_t(1) << bits);
        if (buffer_bytes_for_reserve(r, robin_hood_for(r)) <= l2_share)
            break;
        ++bits;
    }

    const size_t streams = std::bit_ceil(std::max<size_t>(num_build_streams, 1));
    size_t partitions = size_t(1) << bits;
    if (bits > 0)
        /// Every build stream gets a partition to fill.
        partitions = std::max(partitions, streams);
    else if (waiting_rows >= parallel_hash_join_threshold)
        /// Few keys and many rows: without this floor, one thread would insert all the rows.
        partitions = std::min(streams, std::bit_ceil(std::max<size_t>(estimate, 1)));
    /// At least two partitions: the parallel layout never has a single table.
    partitions = std::min(std::max<size_t>(partitions, 2), cap);

    plan.partition_bits = std::countr_zero(partitions);
    const size_t r = reserve_for(partitions);
    plan.reserve_per_partition = r;
    /// The same number decides the partition count and the mode, so all tables of the build share one mode. In the Robin Hood
    /// mode every table is reserved for `r` keys, which the grower turns into 2^16 cells or more.
    plan.robin_hood = robin_hood_for(r);
    return plan;
}

}
