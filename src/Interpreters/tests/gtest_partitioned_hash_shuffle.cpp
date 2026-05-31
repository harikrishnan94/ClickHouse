#include <algorithm>
#include <vector>

#include <Columns/ColumnsNumber.h>
#include <Interpreters/PartitionedHashConfig.h>
#include <Interpreters/PartitionedHashShuffle.h>
#include <Common/assert_cast.h>

#include <gtest/gtest.h>

using namespace DB;

namespace
{

PartitionConfig configWithBits(std::vector<UInt8> bits)
{
    PartitionConfig cfg;
    cfg.pass_bits = std::move(bits);
    cfg.total_leaves = size_t{1} << cfg.totalBits();
    return cfg;
}

/// Build a 2-column block: col0 = key, col1 = payload (= key * 1315423911).
Columns makeBlock(size_t n, UInt64 seed_mul)
{
    auto key = ColumnUInt64::create();
    auto payload = ColumnUInt64::create();
    for (size_t i = 0; i < n; ++i)
    {
        const UInt64 k = (i + 1) * seed_mul;
        key->insertValue(k);
        payload->insertValue(k * 1315423911ULL);
    }
    Columns cols;
    cols.push_back(std::move(key));
    cols.push_back(std::move(payload));
    return cols;
}

/// Recompute the routing leaf for a single UInt64 key value, using the same hash kernel as the shuffle.
size_t leafForKey(UInt64 key, const PartitionConfig & cfg)
{
    auto tmp = ColumnUInt64::create();
    tmp->insertValue(key);
    UInt32 h = 0;
    tmp->computeHashInto(0, 1, &h, /*initial=*/true);
    return cfg.leafForHash(h);
}

}

/// Row + byte conservation, partition disjointness, routing identity vs P1, and value preservation.
TEST(PartitionedHashShuffle, ConservationDisjointnessRouting)
{
    const size_t n = 200000;
    PartitionConfig cfg = configWithBits({5, 4, 4}); // 8192 leaves, 3 passes (same as Q7)
    Columns block = makeBlock(n, 2654435761ULL);

    const size_t input_bytes = block[0]->byteSize() + block[1]->byteSize();

    size_t scattered_rows = 0;
    std::vector<Columns> leaves = radixShuffleBlockToLeaves(block, {0}, cfg, scattered_rows);

    ASSERT_EQ(leaves.size(), cfg.total_leaves);
    /// Rows scattered per pass: rows * numPasses.
    EXPECT_EQ(scattered_rows, n * cfg.numPasses());

    size_t total_rows = 0;
    size_t total_bytes = 0;
    std::vector<UInt64> all_keys;
    std::vector<UInt64> all_payloads;
    all_keys.reserve(n);
    all_payloads.reserve(n);

    for (size_t leaf = 0; leaf < leaves.size(); ++leaf)
    {
        const auto & group = leaves[leaf];
        ASSERT_EQ(group.size(), 2u); // hash column dropped
        const size_t leaf_rows = group[0]->size();
        EXPECT_EQ(group[1]->size(), leaf_rows);
        total_rows += leaf_rows;
        total_bytes += group[0]->byteSize() + group[1]->byteSize();

        const auto & keys = assert_cast<const ColumnUInt64 &>(*group[0]).getData();
        const auto & pays = assert_cast<const ColumnUInt64 &>(*group[1]).getData();
        for (size_t i = 0; i < leaf_rows; ++i)
        {
            /// Routing identity: every row in this leaf must route here (spec invariant #6).
            EXPECT_EQ(leafForKey(keys[i], cfg), leaf);
            /// Payload travelled with its key (move-not-corrupt).
            EXPECT_EQ(pays[i], keys[i] * 1315423911ULL);
            all_keys.push_back(keys[i]);
            all_payloads.push_back(pays[i]);
        }
    }

    /// Row conservation.
    EXPECT_EQ(total_rows, n);
    /// Byte conservation (exact-sized scatter; spec invariant #1).
    EXPECT_EQ(total_bytes, input_bytes);

    /// Value preservation: the multiset of keys equals the input multiset.
    std::vector<UInt64> expected_keys;
    expected_keys.reserve(n);
    for (size_t i = 0; i < n; ++i) // NOLINT(modernize-loop-convert): generating values by index
        expected_keys.push_back((i + 1) * 2654435761ULL);
    std::sort(all_keys.begin(), all_keys.end());
    std::sort(expected_keys.begin(), expected_keys.end());
    EXPECT_EQ(all_keys, expected_keys);
}

/// Two-pass schedule yields identical leaf membership as a direct single-hash leaf computation.
TEST(PartitionedHashShuffle, MultiPassEqualsDirectLeaf)
{
    const size_t n = 50000;
    PartitionConfig cfg = configWithBits({6, 6}); // 4096 leaves, 2 passes (same as Q1)
    Columns block = makeBlock(n, 11400714819323198485ULL);

    size_t scattered_rows = 0;
    std::vector<Columns> leaves = radixShuffleBlockToLeaves(block, {0}, cfg, scattered_rows);

    size_t total = 0;
    for (size_t leaf = 0; leaf < leaves.size(); ++leaf)
    {
        const auto & keys = assert_cast<const ColumnUInt64 &>(*leaves[leaf][0]).getData();
        total += keys.size();
        for (size_t i = 0; i < keys.size(); ++i)
            EXPECT_EQ(leafForKey(keys[i], cfg), leaf);
    }
    EXPECT_EQ(total, n);
}

/// Empty input: every leaf group is present and empty, no passes counted.
TEST(PartitionedHashShuffle, EmptyInput)
{
    PartitionConfig cfg = configWithBits({4, 4}); // 256 leaves
    Columns block = makeBlock(0, 7);

    size_t scattered_rows = 0;
    std::vector<Columns> leaves = radixShuffleBlockToLeaves(block, {0}, cfg, scattered_rows);

    ASSERT_EQ(leaves.size(), cfg.total_leaves);
    EXPECT_EQ(scattered_rows, 0u);
    for (const auto & group : leaves)
    {
        ASSERT_EQ(group.size(), 2u);
        EXPECT_EQ(group[0]->size(), 0u);
        EXPECT_EQ(group[1]->size(), 0u);
    }
}
