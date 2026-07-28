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
extern const Event ConcurrentHashJoinAmacProbeRows;
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

Block makeKeyBlock(const String & key_name, const String & id_name, const std::vector<UInt64> & keys, const std::vector<UInt64> & ids)
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

Block makeKeyBlock(const String & key_name, const String & id_name, const std::vector<String> & keys, const std::vector<UInt64> & ids)
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
void drainResult(ConcurrentHashJoin & join, IJoinResult & result, JoinedRows<Key> & rows)
{
    while (true)
    {
        auto r = result.next();
        accumulateRows(r.block, rows);
        if (r.is_last)
        {
            /// A `max_joined_block_rows` remainder is re-fed the way `JoiningTransform` does it.
            if (r.next_block && r.next_block->rows())
            {
                r.next_block->filterBySelector();
                Block next = std::move(*r.next_block).getSourceBlock();
                auto next_result = join.joinBlock(std::move(next));
                drainResult(join, *next_result, rows);
            }
            return;
        }
    }
}

std::shared_ptr<TableJoin> makeTableJoin(
    const Block & left_header, const Block & right_header, JoinKind kind = JoinKind::Inner, JoinStrictness strictness = JoinStrictness::All)
{
    Settings settings;
    auto table_join = std::make_shared<TableJoin>(settings, /*tmp_volume=*/nullptr, /*tmp_data=*/nullptr);
    table_join->setKind(kind);
    table_join->getTableJoin().strictness = strictness;
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
/// fed in `build_block_rows`-sized blocks through the real `IJoin` build interface, with the AMAC
/// engagement counters snapshotted around the build. No statistics hint is passed
/// (`StatsCollectingParams{}`), so the per-slot maps start at their minimal size and grow while
/// rows are in flight in the rings.
template <typename Key>
BuiltJoin buildJoin(
    const std::vector<Key> & distinct_keys,
    size_t duplicates,
    AmacMode mode,
    size_t build_block_rows = block_rows,
    JoinKind kind = JoinKind::Inner,
    JoinStrictness strictness = JoinStrictness::All,
    size_t slots = num_slots)
{
    setAmacModeForTests(mode);

    const Block left_header = makeKeyBlock("k", "probe_id", std::vector<Key>{}, {});
    const Block right_header = makeKeyBlock("rk", "build_id", std::vector<Key>{}, {});

    BuiltJoin result;
    result.table_join = makeTableJoin(left_header, right_header, kind, strictness);
    result.join = std::make_shared<ConcurrentHashJoin>(
        result.table_join, slots, std::make_shared<const Block>(right_header), StatsCollectingParams{});

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
        drainResult(*built.join, *result, actual);
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

/// The output header of the join: left columns then right columns (the shape
/// `getNonJoinedBlocks` maps right columns into by name).
template <typename Key>
Block makeResultHeader()
{
    Block header = makeKeyBlock("k", "probe_id", std::vector<Key>{}, {});
    for (const auto & col : makeKeyBlock("rk", "build_id", std::vector<Key>{}, {}))
        header.insert(col);
    return header;
}

/// Drains the non-joined (RIGHT/FULL) stream into the sorted multiset of (rk, build_id).
template <typename Key>
std::vector<std::pair<Key, UInt64>> drainNonJoined(BuiltJoin & built)
{
    const Block left_header = makeKeyBlock("k", "probe_id", std::vector<Key>{}, {});
    std::vector<std::pair<Key, UInt64>> rows;
    auto stream = built.join->getNonJoinedBlocks(left_header, makeResultHeader<Key>(), block_rows);
    if (!stream)
        return rows;
    while (true)
    {
        Block block = stream->next();
        if (block.empty())
            break;
        const ColumnPtr rk = block.getByName("rk").column->convertToFullColumnIfReplicated();
        const ColumnPtr build_id = block.getByName("build_id").column->convertToFullColumnIfReplicated();
        for (size_t i = 0; i < block.rows(); ++i)
            rows.emplace_back(columnElement<Key>(*rk, i), columnElement<UInt64>(*build_id, i));
    }
    std::sort(rows.begin(), rows.end());
    return rows;
}

/// The exact ALL-join match multiset when probing `probed_indices` (probe ids by position):
/// the build rows of key `j` carry ids `j * duplicates .. j * duplicates + duplicates - 1`.
template <typename Key>
JoinedRows<Key> expectedRowsForProbe(
    const std::vector<Key> & distinct_keys, size_t duplicates, const std::vector<size_t> & probed_indices)
{
    JoinedRows<Key> expected;
    expected.reserve(probed_indices.size() * duplicates);
    for (size_t p = 0; p < probed_indices.size(); ++p)
    {
        const size_t j = probed_indices[p];
        for (size_t d = 0; d < duplicates; ++d)
            expected.emplace_back(distinct_keys[j], p, distinct_keys[j], j * duplicates + d);
    }
    std::sort(expected.begin(), expected.end());
    return expected;
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

TEST(ConcurrentHashJoinAmac, ProbeRingParityUIntKeys)
{
    /// The find ring vs the sequential routed loop, INNER ALL over uint keys - the flagless
    /// word-mapped lazy shape, so the Force arm runs the dispatch-free `word_loop` phase B.
    /// Key 0 is the zero sentinel of the numeric maps and takes the synchronous find path.
    constexpr size_t duplicates = 2;
    auto distinct_keys = uintKeys(300000);
    distinct_keys[0] = 0;
    const auto misses = uintMisses(10000, distinct_keys.size());
    const auto expected = expectedRows(distinct_keys, duplicates);

    auto ring_built = buildJoin(distinct_keys, duplicates, AmacMode::Force);
    const UInt64 ring_probe_before = eventValue(ProfileEvents::ConcurrentHashJoinAmacProbeRows);
    const auto ring_rows = probeAll(ring_built, distinct_keys, misses);
    const UInt64 ring_probe_rows = eventValue(ProfileEvents::ConcurrentHashJoinAmacProbeRows) - ring_probe_before;
    EXPECT_EQ(ring_probe_rows, distinct_keys.size() + misses.size())
        << "every probe row must be resolved by the find pass under Force";

    auto sequential_built = buildJoin(distinct_keys, duplicates, AmacMode::Off);
    const UInt64 sequential_probe_before = eventValue(ProfileEvents::ConcurrentHashJoinAmacProbeRows);
    const auto sequential_rows = probeAll(sequential_built, distinct_keys, misses);
    EXPECT_EQ(eventValue(ProfileEvents::ConcurrentHashJoinAmacProbeRows), sequential_probe_before)
        << "the find pass must not engage when the hook is off";

    ASSERT_EQ(ring_rows.size(), expected.size());
    ASSERT_TRUE(ring_rows == expected);
    ASSERT_TRUE(sequential_rows == expected);
}

TEST(ConcurrentHashJoinAmac, ProbeRingParityStringKeys)
{
    /// String keys pin the saved-hash ring lane (the ring carries the hash for cells that store
    /// one) and the stored-key-as-view contract of the find policy. The empty string is a
    /// zero-LENGTH key, not the zero sentinel, and rides the ring like any other key.
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

    const auto expected = expectedRows(distinct_keys, duplicates);

    auto ring_built = buildJoin(distinct_keys, duplicates, AmacMode::Force);
    const UInt64 ring_probe_before = eventValue(ProfileEvents::ConcurrentHashJoinAmacProbeRows);
    const auto ring_rows = probeAll(ring_built, distinct_keys, misses);
    const UInt64 ring_probe_rows = eventValue(ProfileEvents::ConcurrentHashJoinAmacProbeRows) - ring_probe_before;
    EXPECT_EQ(ring_probe_rows, distinct_keys.size() + misses.size());

    auto sequential_built = buildJoin(distinct_keys, duplicates, AmacMode::Off);
    const UInt64 sequential_probe_before = eventValue(ProfileEvents::ConcurrentHashJoinAmacProbeRows);
    const auto sequential_rows = probeAll(sequential_built, distinct_keys, misses);
    EXPECT_EQ(eventValue(ProfileEvents::ConcurrentHashJoinAmacProbeRows), sequential_probe_before);

    ASSERT_EQ(ring_rows.size(), expected.size());
    ASSERT_TRUE(ring_rows == expected);
    ASSERT_TRUE(sequential_rows == expected);
}

namespace
{

/// RIGHT/FULL ALL parity between the find ring and the sequential routed loop, joined stream
/// AND non-joined stream: the flagged shapes exercise the ring's slot-local `found_offset`
/// recording, the per-slot used flags, and the untouched `NotJoinedHash` iteration. Probes
/// every even-indexed key (odd ones flow into the non-joined stream) plus misses (dropped by
/// RIGHT, emitted with default right columns by FULL).
void runFlaggedShapeParity(JoinKind kind)
{
    constexpr size_t duplicates = 2;
    const auto distinct_keys = uintKeys(200000);
    const auto misses = uintMisses(1000, distinct_keys.size());

    std::vector<UInt64> probed_keys;
    std::vector<size_t> probed_indices;
    for (size_t j = 0; j < distinct_keys.size(); j += 2)
    {
        probed_keys.push_back(distinct_keys[j]);
        probed_indices.push_back(j);
    }

    auto expected = expectedRowsForProbe(distinct_keys, duplicates, probed_indices);
    if (kind == JoinKind::Full)
    {
        /// FULL keeps the unmatched probe rows, with default right columns.
        for (size_t m = 0; m < misses.size(); ++m)
            expected.emplace_back(misses[m], probed_keys.size() + m, 0, 0);
        std::sort(expected.begin(), expected.end());
    }

    std::vector<std::pair<UInt64, UInt64>> expected_non_joined;
    for (size_t j = 1; j < distinct_keys.size(); j += 2)
        for (size_t d = 0; d < duplicates; ++d)
            expected_non_joined.emplace_back(distinct_keys[j], j * duplicates + d);
    std::sort(expected_non_joined.begin(), expected_non_joined.end());

    auto ring_built = buildJoin(distinct_keys, duplicates, AmacMode::Force, block_rows, kind);
    const UInt64 ring_probe_before = eventValue(ProfileEvents::ConcurrentHashJoinAmacProbeRows);
    const auto ring_rows = probeAll(ring_built, probed_keys, misses);
    const UInt64 ring_probe_rows = eventValue(ProfileEvents::ConcurrentHashJoinAmacProbeRows) - ring_probe_before;
    EXPECT_EQ(ring_probe_rows, probed_keys.size() + misses.size());
    const auto ring_non_joined = drainNonJoined<UInt64>(ring_built);

    auto sequential_built = buildJoin(distinct_keys, duplicates, AmacMode::Off, block_rows, kind);
    const auto sequential_rows = probeAll(sequential_built, probed_keys, misses);
    const auto sequential_non_joined = drainNonJoined<UInt64>(sequential_built);

    ASSERT_EQ(ring_rows.size(), expected.size());
    ASSERT_TRUE(ring_rows == expected);
    ASSERT_TRUE(sequential_rows == expected);
    ASSERT_EQ(ring_non_joined.size(), expected_non_joined.size());
    ASSERT_TRUE(ring_non_joined == expected_non_joined);
    ASSERT_TRUE(sequential_non_joined == expected_non_joined);
}

}

TEST(ConcurrentHashJoinAmac, ProbeRingRightAllNonJoinedParity)
{
    runFlaggedShapeParity(JoinKind::Right);
}

TEST(ConcurrentHashJoinAmac, ProbeRingFullAllNonJoinedParity)
{
    runFlaggedShapeParity(JoinKind::Full);
}

TEST(ConcurrentHashJoinAmac, ProbeRingRightAnySetUsedOnce)
{
    /// RIGHT ANY pins `setUsedOnce`: probing every key TWICE must attach each key's build rows
    /// to the FIRST probe occurrence only - phase B consumes the find pass's results in row
    /// order, so the winner is deterministic and identical to the sequential loop's.
    constexpr size_t duplicates = 2;
    const auto distinct_keys = uintKeys(100000);
    const auto misses = uintMisses(1000, distinct_keys.size());

    std::vector<UInt64> probed_keys;
    probed_keys.reserve(distinct_keys.size() * 2);
    for (const auto key : distinct_keys)
    {
        probed_keys.push_back(key);
        probed_keys.push_back(key);
    }

    JoinedRows<UInt64> expected;
    expected.reserve(distinct_keys.size() * duplicates);
    for (size_t j = 0; j < distinct_keys.size(); ++j)
        for (size_t d = 0; d < duplicates; ++d)
            expected.emplace_back(distinct_keys[j], 2 * j, distinct_keys[j], j * duplicates + d);
    std::sort(expected.begin(), expected.end());

    auto ring_built = buildJoin(distinct_keys, duplicates, AmacMode::Force, block_rows, JoinKind::Right, JoinStrictness::Any);
    const UInt64 ring_probe_before = eventValue(ProfileEvents::ConcurrentHashJoinAmacProbeRows);
    const auto ring_rows = probeAll(ring_built, probed_keys, misses);
    const UInt64 ring_probe_rows = eventValue(ProfileEvents::ConcurrentHashJoinAmacProbeRows) - ring_probe_before;
    EXPECT_EQ(ring_probe_rows, probed_keys.size() + misses.size());
    const auto ring_non_joined = drainNonJoined<UInt64>(ring_built);

    auto sequential_built = buildJoin(distinct_keys, duplicates, AmacMode::Off, block_rows, JoinKind::Right, JoinStrictness::Any);
    const auto sequential_rows = probeAll(sequential_built, probed_keys, misses);
    const auto sequential_non_joined = drainNonJoined<UInt64>(sequential_built);

    ASSERT_EQ(ring_rows.size(), expected.size());
    ASSERT_TRUE(ring_rows == expected);
    ASSERT_TRUE(sequential_rows == expected);
    EXPECT_TRUE(ring_non_joined.empty()) << "every build row was probed, so nothing may reach the non-joined stream";
    EXPECT_TRUE(sequential_non_joined.empty());
}

TEST(ConcurrentHashJoinAmac, ProbeEmitsInLeftRowOrder)
{
    /// The order-by-construction claim of the routed probe: a multi-block probe with a
    /// monotone tag column must come out with non-decreasing tags inside every drained result
    /// (the scatter probe could not guarantee this). Checked for both hook arms and for the
    /// single-slot plan (null route words).
    constexpr size_t duplicates = 3;
    const auto distinct_keys = uintKeys(150000);
    const auto misses = uintMisses(distinct_keys.size(), distinct_keys.size());

    for (const size_t slots : {1uz, 4uz})
    {
        for (const AmacMode mode : {AmacMode::Force, AmacMode::Off})
        {
            auto built = buildJoin(distinct_keys, duplicates, mode, block_rows, JoinKind::Inner, JoinStrictness::All, slots);

            /// Interleave hits and misses so the filter compaction is exercised too.
            std::vector<UInt64> keys;
            std::vector<UInt64> ids;
            UInt64 probe_id = 0;
            size_t drained_blocks = 0;
            auto probe_block = [&]
            {
                auto result = built.join->joinBlock(makeKeyBlock("k", "probe_id", keys, ids));
                UInt64 prev = 0;
                bool have_prev = false;
                while (true)
                {
                    auto r = result->next();
                    if (r.block.rows())
                    {
                        ++drained_blocks;
                        const ColumnPtr probe_ids = r.block.getByName("probe_id").column->convertToFullColumnIfReplicated();
                        for (size_t i = 0; i < probe_ids->size(); ++i)
                        {
                            const UInt64 tag = columnElement<UInt64>(*probe_ids, i);
                            if (have_prev)
                                ASSERT_LE(prev, tag) << "left row order must be preserved within a drained probe block";
                            prev = tag;
                            have_prev = true;
                        }
                    }
                    if (r.is_last)
                    {
                        EXPECT_EQ(r.next_block, nullptr);
                        break;
                    }
                }
                keys.clear();
                ids.clear();
            };

            for (size_t i = 0; i < distinct_keys.size(); ++i)
            {
                keys.push_back(distinct_keys[i]);
                ids.push_back(probe_id++);
                keys.push_back(misses[i]);
                ids.push_back(probe_id++);
                if (keys.size() >= block_rows)
                    probe_block();
            }
            if (!keys.empty())
                probe_block();

            EXPECT_GT(drained_blocks, 0u);
        }
    }
}

/// The probe-scratch pool: one parked scratch per lane, lock-free acquire/release, with the
/// mutexed pool absorbing collisions and out-of-range lanes.
TEST(ConcurrentHashJoinProbeScratch, PoolParksAndReusesPerLane)
{
    auto built = buildJoin<UInt64>({1, 2, 3}, 1, AmacMode::Auto);

    auto first = built.join->acquireProbeScratch(0);
    ASSERT_NE(first, nullptr);
    first->slot_ids.resize(1000);
    JoinProbeScratch * raw = first.get();
    built.join->releaseProbeScratch(std::move(first), 0);

    /// The same lane gets the same parked scratch back, capacity intact.
    auto second = built.join->acquireProbeScratch(0);
    ASSERT_EQ(second.get(), raw);
    EXPECT_EQ(second->slot_ids.size(), 1000u);

    /// A different lane never steals a parked scratch that belongs to lane 0.
    built.join->releaseProbeScratch(std::move(second), 0);
    auto other_lane = built.join->acquireProbeScratch(1);
    EXPECT_NE(other_lane.get(), raw);
    built.join->releaseProbeScratch(std::move(other_lane), 1);
}

TEST(ConcurrentHashJoinProbeScratch, PoolToleratesLaneCollisions)
{
    auto built = buildJoin<UInt64>({1, 2, 3}, 1, AmacMode::Auto);

    /// Two overlapping acquisitions of the same lane (the totals transform and stream 0 both
    /// use lane 0) must return two distinct live scratches.
    auto first = built.join->acquireProbeScratch(0);
    auto second = built.join->acquireProbeScratch(0);
    ASSERT_NE(first, nullptr);
    ASSERT_NE(second, nullptr);
    ASSERT_NE(first.get(), second.get());

    /// Releasing both parks one under the lane and diverts the other to the pool - neither
    /// is lost: two follow-up acquisitions get both back without allocating.
    JoinProbeScratch * raw_first = first.get();
    JoinProbeScratch * raw_second = second.get();
    built.join->releaseProbeScratch(std::move(first), 0);
    built.join->releaseProbeScratch(std::move(second), 0);
    auto reacquired_a = built.join->acquireProbeScratch(0);
    auto reacquired_b = built.join->acquireProbeScratch(0);
    const bool both_recovered = (reacquired_a.get() == raw_first && reacquired_b.get() == raw_second)
        || (reacquired_a.get() == raw_second && reacquired_b.get() == raw_first);
    EXPECT_TRUE(both_recovered);
    built.join->releaseProbeScratch(std::move(reacquired_a), 0);
    built.join->releaseProbeScratch(std::move(reacquired_b), 0);
}

TEST(ConcurrentHashJoinProbeScratch, PoolToleratesInvalidAndOutOfRangeLanes)
{
    auto built = buildJoin<UInt64>({1, 2, 3}, 1, AmacMode::Auto);

    auto legacy = built.join->acquireProbeScratch(ConcurrentHashJoin::invalid_lane);
    ASSERT_NE(legacy, nullptr);
    built.join->releaseProbeScratch(std::move(legacy), ConcurrentHashJoin::invalid_lane);

    auto out_of_range = built.join->acquireProbeScratch(1000000);
    ASSERT_NE(out_of_range, nullptr);
    built.join->releaseProbeScratch(std::move(out_of_range), 1000000);
}

TEST(ConcurrentHashJoinProbeScratch, ProbeReleasesScratchOnResultDestruction)
{
    auto built = buildJoin<UInt64>({1, 2, 3, 4, 5, 6, 7, 8}, 1, AmacMode::Auto);

    constexpr size_t lane = 2;
    constexpr size_t probe_rows = 6;
    JoinedRows<UInt64> rows;
    {
        auto result = built.join->joinBlock(
            makeKeyBlock("k", "probe_id", std::vector<UInt64>{1, 2, 3, 4, 5, 6}, {0, 1, 2, 3, 4, 5}), lane);
        drainResult(*built.join, *result, rows);
        /// The scratch is still owned by the result here - the lane's entry is empty, so a
        /// fresh acquisition must not observe the in-flight scratch.
        auto while_alive = built.join->acquireProbeScratch(lane);
        EXPECT_EQ(while_alive->slot_ids.size(), 0u);
        built.join->releaseProbeScratch(std::move(while_alive), lane);
        /// The destructor parks the scratch at scope exit - under the lane, or into the pool
        /// if the release above occupied the entry; either way it is recoverable below.
    }
    EXPECT_EQ(rows.size(), probe_rows);

    /// After destruction some recoverable scratch carries the probe's slot ids (sized to the
    /// probed rows - the join has 4 slots, so the routed path filled them).
    auto a = built.join->acquireProbeScratch(lane);
    auto b = built.join->acquireProbeScratch(lane);
    const bool found_probe_scratch = (a && a->slot_ids.size() == probe_rows) || (b && b->slot_ids.size() == probe_rows);
    EXPECT_TRUE(found_probe_scratch);
}
