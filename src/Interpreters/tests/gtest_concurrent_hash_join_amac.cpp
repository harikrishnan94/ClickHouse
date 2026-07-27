#include <gtest/gtest.h>

#include <fmt/format.h>

#include <algorithm>
#include <tuple>
#include <vector>

#include <Columns/ColumnString.h>
#include <Columns/ColumnsNumber.h>
#include <Core/Block.h>
#include <Core/Settings.h>
#include <DataTypes/DataTypeString.h>
#include <DataTypes/DataTypesNumber.h>
#include <Interpreters/ConcurrentHashJoin.h>
#include <Interpreters/HashJoin/AmacMode.h>
#include <Interpreters/TableJoin.h>
#include <Common/ProfileEvents.h>
#include <Common/assert_cast.h>

namespace ProfileEvents
{
extern const Event ConcurrentHashJoinAmacBuildRows;
extern const Event ConcurrentHashJoinAmacBuildRingGrowths;
}

using namespace DB;

namespace
{

constexpr size_t block_rows = 65536;
constexpr size_t num_slots = 4;

/// The tests run single-threaded with no `ThreadStatus`, so every `ProfileEvents::increment`
/// of the build lands directly in the global counters; a delta around the build isolates the
/// events of one join from the rest of the test binary.
UInt64 eventValue(ProfileEvents::Event event)
{
    return ProfileEvents::global_counters[event];
}

/// One joined output row: (k, probe_id, rk, build_id). The multiset of these tuples over the
/// whole drain is an exact identity: a dropped, duplicated or cross-wired build row changes it.
template <typename Key>
using JoinedRows = std::vector<std::tuple<Key, UInt64, Key, UInt64>>;

Block uintKeyBlock(const String & key_name, const String & id_name, const std::vector<UInt64> & keys, const std::vector<UInt64> & ids)
{
    auto key_column = ColumnUInt64::create();
    auto id_column = ColumnUInt64::create();
    key_column->getData().assign(keys.begin(), keys.end());
    id_column->getData().assign(ids.begin(), ids.end());
    Block block;
    block.insert({std::move(key_column), std::make_shared<DataTypeUInt64>(), key_name});
    block.insert({std::move(id_column), std::make_shared<DataTypeUInt64>(), id_name});
    return block;
}

Block stringKeyBlock(const String & key_name, const String & id_name, const std::vector<String> & keys, const std::vector<UInt64> & ids)
{
    auto key_column = ColumnString::create();
    for (const auto & key : keys)
        key_column->insertData(key.data(), key.size());
    auto id_column = ColumnUInt64::create();
    id_column->getData().assign(ids.begin(), ids.end());
    Block block;
    block.insert({std::move(key_column), std::make_shared<DataTypeString>(), key_name});
    block.insert({std::move(id_column), std::make_shared<DataTypeUInt64>(), id_name});
    return block;
}

Block makeKeyBlock(const String & key_name, const String & id_name, const std::vector<UInt64> & keys, const std::vector<UInt64> & ids)
{
    return uintKeyBlock(key_name, id_name, keys, ids);
}

Block makeKeyBlock(const String & key_name, const String & id_name, const std::vector<String> & keys, const std::vector<UInt64> & ids)
{
    return stringKeyBlock(key_name, id_name, keys, ids);
}

template <typename Key>
Key columnElement(const IColumn & column, size_t i)
{
    if constexpr (std::is_same_v<Key, UInt64>)
        return assert_cast<const ColumnUInt64 &>(column).getElement(i);
    else
        return String(assert_cast<const ColumnString &>(column).getDataAt(i));
}

template <typename Key>
void accumulateRows(const Block & block, JoinedRows<Key> & rows)
{
    if (!block.rows())
        return;
    const ColumnPtr k = block.getByName("k").column->convertToFullColumnIfReplicated();
    const ColumnPtr probe_id = block.getByName("probe_id").column->convertToFullColumnIfReplicated();
    const ColumnPtr rk = block.getByName("rk").column->convertToFullColumnIfReplicated();
    const ColumnPtr build_id = block.getByName("build_id").column->convertToFullColumnIfReplicated();
    for (size_t i = 0; i < block.rows(); ++i)
        rows.emplace_back(
            columnElement<Key>(*k, i),
            columnElement<UInt64>(*probe_id, i),
            columnElement<Key>(*rk, i),
            columnElement<UInt64>(*build_id, i));
}

template <typename Key>
void drainResult(IJoinResult & result, JoinedRows<Key> & rows)
{
    while (true)
    {
        auto r = result.next();
        accumulateRows(r.block, rows);
        if (r.is_last)
            return;
    }
}

std::shared_ptr<TableJoin> makeTableJoin(const Block & left_header, const Block & right_header)
{
    Settings settings;
    auto table_join = std::make_shared<TableJoin>(settings, /*tmp_volume=*/nullptr, /*tmp_data=*/nullptr);
    table_join->setKind(JoinKind::Inner);
    table_join->getTableJoin().strictness = JoinStrictness::All;
    table_join->addDisjunct();
    table_join->getClauses().back().addKey(
        left_header.getByPosition(0).name, right_header.getByPosition(0).name, /*null_safe_comparison=*/false);

    NamesAndTypesList left_columns;
    NamesAndTypesList right_columns;
    Names used_columns;
    for (const auto & col : left_header)
    {
        left_columns.emplace_back(col.name, col.type);
        used_columns.push_back(col.name);
    }
    for (const auto & col : right_header)
    {
        right_columns.emplace_back(col.name, col.type);
        used_columns.push_back(col.name);
    }
    table_join->setInputColumns(std::move(left_columns), std::move(right_columns));
    table_join->setUsedColumns(used_columns);
    return table_join;
}

struct BuiltJoin
{
    std::shared_ptr<TableJoin> table_join;
    std::shared_ptr<ConcurrentHashJoin> join;
    UInt64 amac_build_rows = 0;
    UInt64 amac_ring_growths = 0;
};

/// Builds a `ConcurrentHashJoin` over the given build keys (`duplicates` adjacent copies each),
/// fed in `block_rows`-sized blocks through the real `IJoin` build interface, with the AMAC
/// engagement counters snapshotted around the build. No statistics hint is passed
/// (`StatsCollectingParams{}`), so the per-slot maps start at their minimal size and grow while
/// rows are in flight in the rings.
template <typename Key>
BuiltJoin buildJoin(const std::vector<Key> & distinct_keys, size_t duplicates, AmacMode mode, size_t build_block_rows = block_rows)
{
    setAmacModeForTests(mode);

    const Block left_header = makeKeyBlock("k", "probe_id", std::vector<Key>{}, {});
    const Block right_header = makeKeyBlock("rk", "build_id", std::vector<Key>{}, {});

    BuiltJoin result;
    result.table_join = makeTableJoin(left_header, right_header);
    result.join = std::make_shared<ConcurrentHashJoin>(
        result.table_join, num_slots, std::make_shared<const Block>(right_header), StatsCollectingParams{});

    const UInt64 rows_before = eventValue(ProfileEvents::ConcurrentHashJoinAmacBuildRows);
    const UInt64 growths_before = eventValue(ProfileEvents::ConcurrentHashJoinAmacBuildRingGrowths);

    std::vector<Key> keys;
    std::vector<UInt64> ids;
    keys.reserve(build_block_rows);
    ids.reserve(build_block_rows);
    UInt64 id = 0;
    for (const auto & key : distinct_keys)
    {
        for (size_t d = 0; d < duplicates; ++d)
        {
            keys.push_back(key);
            ids.push_back(id++);
            if (keys.size() == build_block_rows)
            {
                EXPECT_TRUE(result.join->addBlockToJoin(makeKeyBlock("rk", "build_id", keys, ids), /*check_limits=*/true));
                keys.clear();
                ids.clear();
            }
        }
    }
    if (!keys.empty())
        EXPECT_TRUE(result.join->addBlockToJoin(makeKeyBlock("rk", "build_id", keys, ids), /*check_limits=*/true));

    result.join->onBuildPhaseFinish();

    result.amac_build_rows = eventValue(ProfileEvents::ConcurrentHashJoinAmacBuildRows) - rows_before;
    result.amac_ring_growths = eventValue(ProfileEvents::ConcurrentHashJoinAmacBuildRingGrowths) - growths_before;
    return result;
}

/// Probes every distinct key once (plus `misses` keys absent from the build) and returns the
/// multiset of joined rows, sorted for comparison.
template <typename Key>
JoinedRows<Key> probeAll(BuiltJoin & built, const std::vector<Key> & distinct_keys, const std::vector<Key> & misses)
{
    JoinedRows<Key> actual;
    std::vector<Key> keys;
    std::vector<UInt64> ids;
    size_t probe_id = 0;
    auto flush = [&](bool force)
    {
        if (keys.empty() || (!force && keys.size() < block_rows))
            return;
        auto result = built.join->joinBlock(makeKeyBlock("k", "probe_id", keys, ids));
        drainResult(*result, actual);
        keys.clear();
        ids.clear();
    };
    for (const auto & key : distinct_keys)
    {
        keys.push_back(key);
        ids.push_back(probe_id++);
        flush(false);
    }
    for (const auto & key : misses)
    {
        keys.push_back(key);
        ids.push_back(probe_id++);
        flush(false);
    }
    flush(true);
    std::sort(actual.begin(), actual.end());
    return actual;
}

/// The exact multiset an INNER ALL join must produce when every distinct key is probed once:
/// key `i` (probe id `i`) against its `duplicates` adjacent build rows.
template <typename Key>
JoinedRows<Key> expectedRows(const std::vector<Key> & distinct_keys, size_t duplicates)
{
    JoinedRows<Key> expected;
    expected.reserve(distinct_keys.size() * duplicates);
    for (size_t i = 0; i < distinct_keys.size(); ++i)
        for (size_t d = 0; d < duplicates; ++d)
            expected.emplace_back(distinct_keys[i], i, distinct_keys[i], i * duplicates + d);
    std::sort(expected.begin(), expected.end());
    return expected;
}

std::vector<UInt64> uintKeys(size_t count)
{
    std::vector<UInt64> keys(count);
    for (size_t i = 0; i < count; ++i)
        keys[i] = i * 2654435761ULL + 1;
    return keys;
}

/// Miss keys with a +2 offset cannot collide with built keys: `i * K + 2 == j * K + 1` would
/// need `i - j` to be the (huge) modular inverse of `-K`, far outside these small ranges.
std::vector<UInt64> uintMisses(size_t count, size_t first)
{
    std::vector<UInt64> keys(count);
    for (size_t i = 0; i < count; ++i)
        keys[i] = (first + i) * 2654435761ULL + 2;
    return keys;
}

}

TEST(ConcurrentHashJoinAmac, RingGrowthResume)
{
    /// Growth resume mid-ring: with no statistics hint every per-slot map starts at its minimal
    /// size and must grow repeatedly while ~600K keys stream in, so the insert rings hit the
    /// grower boundary with rows in flight, drain, resize and re-seed many times. The exact
    /// joined-row multiset check is count-exact: a build row lost (two in-flight rows claiming
    /// one cell) or duplicated by the re-seed cannot pass.
    const auto distinct_keys = uintKeys(600000);
    auto built = buildJoin(distinct_keys, /*duplicates=*/1, AmacMode::Force);

    EXPECT_EQ(built.amac_build_rows, distinct_keys.size()) << "every build row must go through the ring under Force";
    EXPECT_GT(built.amac_ring_growths, 0u) << "an unhinted 600K-key build must grow the per-slot maps mid-ring";

    const auto actual = probeAll(built, distinct_keys, uintMisses(10000, distinct_keys.size()));
    const auto expected = expectedRows(distinct_keys, /*duplicates=*/1);
    ASSERT_EQ(actual.size(), expected.size());
    ASSERT_TRUE(actual == expected);
}

TEST(ConcurrentHashJoinAmac, DuplicateHeavyBuildParityVsSequential)
{
    /// Duplicate-heavy build (16 adjacent rows per key) forced through the ring, cross-checked
    /// against the same build on the sequential loop. Each key's duplicates are adjacent, so
    /// same-key rows are permanently in flight together - the fused read -> act step invariant
    /// is what keeps the counts exact (a batched read-then-act would let two of them claim one
    /// cell). The Off arm also proves the engagement counters stay silent when the hook is off.
    constexpr size_t duplicates = 16;
    auto distinct_keys = uintKeys(200000);
    /// Key 0 is the zero sentinel of the numeric maps: its 16 duplicates take the synchronous
    /// `start` path (never a ring slot) and must still all reach the zero-value cell.
    distinct_keys[0] = 0;
    const auto misses = uintMisses(1000, distinct_keys.size());

    auto ring_built = buildJoin(distinct_keys, duplicates, AmacMode::Force);
    EXPECT_EQ(ring_built.amac_build_rows, distinct_keys.size() * duplicates);

    auto sequential_built = buildJoin(distinct_keys, duplicates, AmacMode::Off);
    EXPECT_EQ(sequential_built.amac_build_rows, 0u);
    EXPECT_EQ(sequential_built.amac_ring_growths, 0u);

    const auto ring_rows = probeAll(ring_built, distinct_keys, misses);
    const auto sequential_rows = probeAll(sequential_built, distinct_keys, misses);
    const auto expected = expectedRows(distinct_keys, duplicates);
    ASSERT_EQ(ring_rows.size(), expected.size());
    ASSERT_TRUE(ring_rows == expected);
    ASSERT_TRUE(sequential_rows == expected);
}

TEST(ConcurrentHashJoinAmac, TinySectionGrowthDuringSweepTail)
{
    /// Regression test for the steady sweep stepping an emptied slot. With tiny build blocks a
    /// per-slot section is only a couple of sweeps long, so the sweep whose failed refill
    /// deactivates a slot mid-sweep (the section tail) frequently also hits a growth. The
    /// growth's re-seed must put the in-flight rows back into their OWN slots: re-filling "the
    /// first free slots" instead moves a row into the dead slot, leaves a not-yet-swept slot
    /// empty, and the rest of the sweep dereferences the inactive-row sentinel - a segfault on
    /// string keys (`offsets[2^32 - 2]`), a silent garbage insert on numeric ones.
    const auto uint_keys = uintKeys(300000);
    auto uint_built = buildJoin(uint_keys, /*duplicates=*/1, AmacMode::Force, /*build_block_rows=*/256);
    EXPECT_EQ(uint_built.amac_build_rows, uint_keys.size());
    EXPECT_GT(uint_built.amac_ring_growths, 0u) << "the tiny-section build must still grow mid-ring for the tail window to be exercised";
    const auto uint_rows = probeAll(uint_built, uint_keys, uintMisses(1000, uint_keys.size()));
    const auto uint_expected = expectedRows(uint_keys, /*duplicates=*/1);
    ASSERT_EQ(uint_rows.size(), uint_expected.size());
    ASSERT_TRUE(uint_rows == uint_expected);

    std::vector<String> string_keys;
    string_keys.reserve(150000);
    for (size_t i = 0; i < 150000; ++i)
        string_keys.push_back(fmt::format("key_{}_{}", i, String(i % 23, 'x')));
    auto string_built = buildJoin(string_keys, /*duplicates=*/1, AmacMode::Force, /*build_block_rows=*/256);
    EXPECT_GT(string_built.amac_ring_growths, 0u);
    const auto string_rows = probeAll(string_built, string_keys, std::vector<String>{});
    const auto string_expected = expectedRows(string_keys, /*duplicates=*/1);
    ASSERT_EQ(string_rows.size(), string_expected.size());
    ASSERT_TRUE(string_rows == string_expected);
}

TEST(ConcurrentHashJoinAmac, StringKeyBuildEngagement)
{
    /// String keys exercise the persist-once path: key holders are fetched per ring visit
    /// without persisting, and `cursorClaim` persists exactly once at the claim; a key persisted
    /// twice wastes arena but a key not persisted at all dangles into the source block and the
    /// probe below misses it. The empty-string key is a zero-LENGTH key, not the zero sentinel
    /// (a column-backed `StringRef` always has a non-null data pointer), so it rides the ring
    /// like any other key and covers the empty-payload persist.
    constexpr size_t duplicates = 2;
    std::vector<String> distinct_keys;
    distinct_keys.reserve(150001);
    distinct_keys.emplace_back("");
    for (size_t i = 0; i < 150000; ++i)
        distinct_keys.push_back(fmt::format("key_{}_{}", i, String(i % 23, 'x')));

    std::vector<String> misses;
    misses.reserve(1000);
    for (size_t i = 0; i < 1000; ++i)
        misses.push_back(fmt::format("miss_{}", i));

    auto ring_built = buildJoin(distinct_keys, duplicates, AmacMode::Force);
    EXPECT_EQ(ring_built.amac_build_rows, distinct_keys.size() * duplicates);

    auto sequential_built = buildJoin(distinct_keys, duplicates, AmacMode::Off);
    EXPECT_EQ(sequential_built.amac_build_rows, 0u);

    const auto ring_rows = probeAll(ring_built, distinct_keys, misses);
    const auto sequential_rows = probeAll(sequential_built, distinct_keys, misses);
    const auto expected = expectedRows(distinct_keys, duplicates);
    ASSERT_EQ(ring_rows.size(), expected.size());
    ASSERT_TRUE(ring_rows == expected);
    ASSERT_TRUE(sequential_rows == expected);
}
