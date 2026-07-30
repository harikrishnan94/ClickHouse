#include <gtest/gtest.h>

#include <vector>

#include <Columns/ColumnsNumber.h>
#include <Interpreters/HashJoin/AmacProbeImpl.h>
#include <Interpreters/HashJoin/KeyGetter.h>

using namespace DB;

namespace
{

/// Sends every key to one home cell near the end of the buffer, forcing the standard linear
/// probe to wrap. Unreachable through the production hash functions or SQL, which is why this
/// test instantiates the find pass directly.
struct DegenerateHash
{
    size_t operator()(UInt64) const { return 250; }
};

using WrappedMap = WithJoinCursor<HashMap<UInt64, RowRef, DegenerateHash>>::Type;
using WrappedKeyGetter = KeyGetterForType<HashJoin::Type::key64, const WrappedMap>::Type;

}

TEST(AmacWrappedWalk, RingMatchesMapFindOnWrappedChain)
{
    /// 100 keys, one home cell: the chain covers cells 250..255 and wraps into 0..93. Load
    /// factor stays below growth.
    WrappedMap map;
    constexpr UInt64 n_keys = 100;
    for (UInt64 key = 1; key <= n_keys; ++key)
    {
        typename WrappedMap::LookupResult it;
        bool inserted = false;
        map.emplace(key, it, inserted);
        ASSERT_TRUE(inserted);
        it->getMapped() = RowRef(0, key);
    }
    ASSERT_EQ(map.getBufferSizeInCells(), 256);

    /// Probe every built key plus misses that walk the full wrapped chain to its empty end.
    std::vector<UInt64> probe_keys;
    for (UInt64 key = 1; key <= n_keys; ++key)
        probe_keys.push_back(key);
    for (UInt64 key = 1000; key < 1050; ++key)
        probe_keys.push_back(key);

    auto probe_column = ColumnUInt64::create();
    probe_column->getData().assign(probe_keys.begin(), probe_keys.end());
    const ColumnRawPtrs key_columns{probe_column.get()};
    const Sizes key_sizes{sizeof(UInt64)};
    auto key_getter = createKeyGetter<WrappedKeyGetter, false>(key_columns, key_sizes);

    const WrappedMap * slot_maps[1]{&map};
    const SlotMapDesc descs[1]{{map.cursorCells(), map.getBufferSizeInCells() - 1}};
    std::vector<UInt64> found_word(probe_keys.size());

    Arena pool;
    amacFindPass<WrappedKeyGetter, WrappedMap, /*need_flags=*/false, /*selector_is_range=*/true>(
        key_getter,
        slot_maps,
        descs,
        /*route_shift=*/32,
        probe_keys.size(),
        /*range_first=*/0,
        /*selector_indexes=*/nullptr,
        /*skip_data=*/nullptr,
        pool,
        found_word.data(),
        /*found_offset=*/nullptr,
        /*found_slot=*/nullptr);

    for (size_t i = 0; i < probe_keys.size(); ++i)
    {
        const auto * cell = map.find(probe_keys[i]);
        if (cell)
            EXPECT_EQ(found_word[i], mappedWordOf(cell->getMapped())) << "key " << probe_keys[i];
        else
            EXPECT_EQ(found_word[i], 0u) << "key " << probe_keys[i];
    }
}
