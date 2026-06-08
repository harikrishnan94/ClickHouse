#include <gtest/gtest.h>

#include <Columns/ColumnFixedString.h>
#include <Columns/ColumnsNumber.h>
#include <Core/Block.h>
#include <DataTypes/DataTypeFixedString.h>
#include <DataTypes/DataTypesNumber.h>
#include <Interpreters/RadixHashJoin/BuildStore.h>
#include <Interpreters/RadixHashJoin/GrowingArena.h>
#include <Interpreters/RadixHashJoin/PartitionConfig.h>
#include <Interpreters/RadixHashJoin/RouteHash.h>

#include <Common/Stopwatch.h>

#include <fmt/format.h>

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iostream>
#include <random>
#include <thread>
#include <vector>

using namespace DB;
using namespace DB::RadixHash;

namespace
{

constexpr size_t l2_bytes = 2 * 1024 * 1024;

/// Drive a cooperative build with real T-thread parallelism.  All threads call coord.run(body);
/// the first becomes the leader (executes body); the rest act as helpers.
void coopRun(CoopPool & coord, size_t threads, std::function<void()> body)
{
    std::vector<std::thread> th;
    th.reserve(threads);
    for (size_t t = 0; t < threads; ++t)
        th.emplace_back([&] { coord.run(body); });
    for (auto & x : th)
        x.join();
}

template <typename T>
ColumnVector<T>::MutablePtr makeColumn(const std::vector<T> & vals)
{
    auto col = ColumnVector<T>::create();
    auto & data = col->getData();
    data.resize(vals.size());
    for (size_t i = 0; i < vals.size(); ++i)
        data[i] = vals[i];
    return col;
}

/// A build block: `key_cols` fixed-width key columns ("k0".. at positions 0..key_cols-1) plus
/// `num_payload` UInt64 payload columns. The payload is never scattered (zero-copy gate).
template <typename Key>
Block makeBlock(const std::vector<std::vector<Key>> & keys, size_t num_payload, UInt64 payload_seed)
{
    const size_t rows = keys.empty() ? 0 : keys[0].size();
    ColumnsWithTypeAndName cols;
    for (size_t c = 0; c < keys.size(); ++c)
        cols.emplace_back(makeColumn<Key>(keys[c]), std::make_shared<DataTypeNumber<Key>>(), fmt::format("k{}", c));
    for (size_t c = 0; c < num_payload; ++c)
    {
        std::vector<UInt64> pv(rows);
        for (size_t i = 0; i < rows; ++i)
            pv[i] = payload_seed * 1000003ull + c * 7919ull + i;
        cols.emplace_back(makeColumn<UInt64>(pv), std::make_shared<DataTypeUInt64>(), fmt::format("p{}", c + 1));
    }
    return Block(std::move(cols));
}

/// Single-key-column block helper.
template <typename Key>
Block makeBlock1(const std::vector<Key> & keys, size_t num_payload, UInt64 payload_seed)
{
    return makeBlock<Key>(std::vector<std::vector<Key>>{keys}, num_payload, payload_seed);
}

/// Raw 32-bit route hash for each row: the byte `routeHash` of the PACKED key (the key columns
/// concatenated at their packed offsets), mirroring the build/probe routing exactly. The leaf id for a
/// given row is `hash >> cfg.shift` (top `total_bits` bits of the hash).
std::vector<UInt32> referenceHashes(
    const Block & block, const std::vector<size_t> & key_positions, const std::vector<size_t> & key_widths, size_t n)
{
    std::vector<size_t> key_offsets(key_widths.size(), 0);
    for (size_t c = 1; c < key_widths.size(); ++c)
        key_offsets[c] = key_offsets[c - 1] + key_widths[c - 1];
    size_t kw = 0;
    for (size_t w : key_widths)
        kw += w;

    std::vector<UInt32> hash(n, 0);
    std::vector<char> packed(kw);
    for (size_t r = 0; r < n; ++r)
    {
        for (size_t c = 0; c < key_positions.size(); ++c)
        {
            const char * col = block.getByPosition(key_positions[c]).column->getRawData().data();
            std::memcpy(packed.data() + key_offsets[c], col + r * key_widths[c], key_widths[c]);
        }
        hash[r] = routeHash(packed.data(), kw);
    }
    return hash;
}

/// Full independent verification of a finished build + scatter against the accumulated source blocks:
/// conservation (every build row exactly once), per-leaf sizes, ref resolution, packed-key pairing and
/// the routing identity (final leaf == pid). Works for single- and multi-pass / multi-column configs.
template <typename Key>
void verifyConservationAndRefs(
    BuildStore & store, const LeafArrays & leaves, const std::vector<size_t> & key_positions, const std::vector<size_t> & key_widths)
{
    const auto & cfg = store.config();
    const auto & blocks = store.blocks();
    const size_t num_blocks = blocks.size();
    const size_t kw = store.packedKeyWidth();

    ASSERT_EQ(leaves.num_leaves, cfg.num_leaves);
    ASSERT_EQ(leaves.leaf_rows.size(), cfg.num_leaves);
    ASSERT_EQ(leaves.key_width, kw);

    std::vector<size_t> key_offsets(key_widths.size(), 0);
    for (size_t c = 1; c < key_widths.size(); ++c)
        key_offsets[c] = key_offsets[c - 1] + key_widths[c - 1];

    std::vector<std::vector<UInt32>> ref_hash(num_blocks);
    std::vector<size_t> block_rows(num_blocks);
    for (size_t b = 0; b < num_blocks; ++b)
    {
        block_rows[b] = blocks[b].rows();
        ref_hash[b] = referenceHashes(blocks[b], key_positions, key_widths, block_rows[b]);
    }

    UInt64 total = 0;
    for (size_t leaf = 0; leaf < cfg.num_leaves; ++leaf)
    {
        EXPECT_EQ(leaves.leaf_rows[leaf], store.globalHistogram()[leaf]) << "leaf=" << leaf;
        total += leaves.leaf_rows[leaf];
    }

    std::vector<std::vector<char>> seen(num_blocks);
    for (size_t b = 0; b < num_blocks; ++b)
        seen[b].assign(block_rows[b], 0);

    UInt64 counted = 0;
    for (size_t leaf = 0; leaf < cfg.num_leaves; ++leaf)
    {
        for (UInt64 i = 0; i < leaves.leaf_rows[leaf]; ++i)
        {
            const RadixShuffle::BuildRef ref = leaves.refAt(leaf, i);
            ASSERT_LT(ref.block_no, num_blocks) << "leaf=" << leaf << " i=" << i;
            /// row_no is 0-based; INVALID_ROW is the empty sentinel and must never appear in a real ref.
            ASSERT_NE(ref.row_no, RadixShuffle::INVALID_ROW) << "INVALID_ROW is the empty sentinel — not a valid ref";
            const UInt32 row = ref.row_no; /// 0-based row index
            ASSERT_LT(row, block_rows[ref.block_no]);

            /// Routing identity: top `total_bits` bits of the stored hash must equal the leaf index.
            /// Guard against total_bits == 0 (cfg.shift == 32): UB for 32-bit shift; leaf == 0 trivially.
            const UInt32 routed_leaf = cfg.total_bits > 0
                ? (ref_hash[ref.block_no][row] >> cfg.shift)
                : 0u;
            EXPECT_EQ(routed_leaf, static_cast<UInt32>(leaf)) << "row routed to the wrong leaf";

            /// Packed-key pairing: the scattered packed key equals the row's key columns concatenated.
            const char * packed = static_cast<const char *>(leaves.keyAt(leaf, i));
            for (size_t c = 0; c < key_positions.size(); ++c)
            {
                const char * col = blocks[ref.block_no].getByPosition(key_positions[c]).column->getRawData().data();
                EXPECT_EQ(0, std::memcmp(packed + key_offsets[c], col + static_cast<size_t>(row) * key_widths[c], key_widths[c]))
                    << "packed key mismatch leaf=" << leaf << " i=" << i << " col=" << c;
            }

            EXPECT_EQ(seen[ref.block_no][row], 0) << "row scattered more than once";
            seen[ref.block_no][row] = 1;
            ++counted;
        }
    }

    EXPECT_EQ(counted, total);

    UInt64 source_rows = 0;
    for (size_t b = 0; b < num_blocks; ++b)
    {
        source_rows += block_rows[b];
        for (size_t r = 0; r < block_rows[b]; ++r)
            EXPECT_EQ(seen[b][r], 1) << "row (" << b << "," << r << ") missing from the scatter output";
    }
    EXPECT_EQ(total, source_rows) << "leaf row total must equal the number of build rows";
}

std::vector<UInt64> randomKeys(size_t n, uint64_t seed)
{
    std::mt19937_64 rng(seed); /// NOLINT(cert-msc32-c,cert-msc51-cpp,bugprone-random-generator-seed) -- deterministic test
    std::vector<UInt64> v(n);
    for (size_t i = 0; i < n; ++i)
        v[i] = rng();
    return v;
}

/// Split `keys` into `num_blocks` contiguous single-key-column blocks and feed them serially.
template <typename Key>
void addBlocksSerial(BuildStore & store, const std::vector<Key> & keys, size_t num_blocks, size_t num_payload)
{
    const size_t n = keys.size();
    const size_t per = (n + num_blocks - 1) / num_blocks;
    for (size_t b = 0; b < num_blocks; ++b)
    {
        const size_t lo = b * per;
        if (lo >= n)
            break;
        const size_t hi = std::min(n, lo + per);
        std::vector<Key> slice(keys.begin() + lo, keys.begin() + hi);
        store.add(makeBlock1<Key>(slice, num_payload, b + 1));
    }
}

/// Key column positions {0, 1, ..., k-1} for a k-column key.
std::vector<size_t> keyPositions(size_t k)
{
    std::vector<size_t> pos(k);
    for (size_t i = 0; i < k; ++i)
        pos[i] = i;
    return pos;
}

/// One FixedString(width) column of `rows` rows filled with deterministic pseudo-random bytes.
/// Bulk-fills the raw char buffer 8 bytes at a time (fast; this is setup, never timed).
ColumnFixedString::MutablePtr makeRandomFixedString(size_t width, size_t rows, std::mt19937_64 & rng)
{
    auto col = ColumnFixedString::create(width);
    col->resize(rows);
    auto & chars = col->getChars();
    const size_t total = chars.size(); /// == width * rows
    size_t i = 0;
    for (; i + 8 <= total; i += 8)
    {
        const UInt64 v = rng();
        std::memcpy(chars.data() + i, &v, 8);
    }
    if (i < total)
    {
        const UInt64 v = rng();
        std::memcpy(chars.data() + i, &v, total - i);
    }
    return col;
}

/// One build block: `widths.size()` FixedString key columns named k0.. of the given byte widths,
/// filled with deterministic pseudo-random bytes. No payload columns (ZC gate — payload is never
/// scattered). `seed` makes each block independently reproducible for parallel generation.
Block makeFixedStringBlock(const std::vector<size_t> & widths, size_t rows, uint64_t seed)
{
    std::mt19937_64 rng(seed); /// NOLINT(cert-msc32-c,cert-msc51-cpp,bugprone-random-generator-seed) -- deterministic test
    ColumnsWithTypeAndName cols;
    cols.reserve(widths.size());
    for (size_t c = 0; c < widths.size(); ++c)
        cols.emplace_back(
            makeRandomFixedString(widths[c], rows, rng),
            std::make_shared<DataTypeFixedString>(widths[c]),
            fmt::format("k{}", c));
    return Block(std::move(cols));
}

/// Generate `ceil(n / block_rows)` blocks of FixedString key columns in parallel across
/// `num_threads` workers (work-stealing over block indices). Setup helper — NOT timed by callers.
std::vector<Block> generateFixedStringBlocksParallel(
    const std::vector<size_t> & widths, size_t n, size_t block_rows, size_t num_threads, uint64_t seed_base)
{
    const size_t num_blocks = (n + block_rows - 1) / block_rows;
    std::vector<Block> blocks(num_blocks);
    std::atomic<size_t> next{0};
    std::vector<std::thread> gen;
    gen.reserve(num_threads);
    for (size_t t = 0; t < num_threads; ++t)
        gen.emplace_back([&]
        {
            for (size_t b = next.fetch_add(1); b < num_blocks; b = next.fetch_add(1))
            {
                const size_t rows = std::min(block_rows, n - b * block_rows);
                blocks[b] = makeFixedStringBlock(widths, rows, seed_base + b);
            }
        });
    for (auto & th : gen)
        th.join();
    return blocks;
}

}

/// Single-pass conservation, sizes, ref resolution, pairing and routing identity (UInt64 key).
TEST(RadixHashBuildScatter, ConservationSinglePassU64)
{
    const size_t n = 1'000'003;
    const auto keys = randomKeys(n, 0xC0FFEE);
    auto cfg = PartitionConfig::make(static_cast<UInt64>(100'000'000), l2_bytes, 8192); /// 2048 leaves, 1 pass
    ASSERT_EQ(cfg.pass_bits.size(), 1u);

    BuildStore store(cfg, {0}, {sizeof(UInt64)}, /*max_threads=*/1);
    addBlocksSerial<UInt64>(store, keys, /*num_blocks=*/17, /*num_payload=*/3);
    store.finishBuild();
    CoopPool coord;
    LeafArrays leaves;
    coopRun(coord, 1, [&] { leaves = store.scatterToLeaves(coord); });

    verifyConservationAndRefs<UInt64>(store, leaves, {0}, {sizeof(UInt64)});
}

/// Single-pass with a UInt32 key (4-byte scatter granularity).
TEST(RadixHashBuildScatter, ConservationSinglePassU32)
{
    const size_t n = 500'009;
    std::vector<UInt32> keys(n);
    std::mt19937_64 rng(0xABCDEF); /// NOLINT(cert-msc32-c,cert-msc51-cpp,bugprone-random-generator-seed) -- deterministic test
    for (size_t i = 0; i < n; ++i)
        keys[i] = static_cast<UInt32>(rng());

    auto cfg = PartitionConfig::make(static_cast<UInt64>(50'000'000), l2_bytes, 8192);
    ASSERT_EQ(cfg.pass_bits.size(), 1u);

    BuildStore store(cfg, {0}, {sizeof(UInt32)}, 1);
    addBlocksSerial<UInt32>(store, keys, 11, 2);
    store.finishBuild();
    CoopPool coord;
    LeafArrays leaves;
    coopRun(coord, 1, [&] { leaves = store.scatterToLeaves(coord); });

    verifyConservationAndRefs<UInt32>(store, leaves, {0}, {sizeof(UInt32)});
}

/// Single-pass scatter through the incremental SWWC/NT path (`scatterColumnIntoSwwc` + drain), engaged
/// at high fanout (num_leaves >= 256) when NT stores are available (x86-64-v3 multitarget). The add()
/// is multi-threaded so several build slots are populated -> several scatter workers write into the
/// SHARED per-leaf arrays at generally-unaligned per-(thread,leaf) starts, exercising the head-peel +
/// line-flush + residual-drain. Conservation / per-leaf membership / routing identity must hold, and
/// the byte-scattered total must stay path-independent (== N*(key_width + 8)).
TEST(RadixHashBuildScatter, ConservationSinglePassSwwc)
{
    const size_t n = 2'000'003;
    const auto keys = randomKeys(n, 0x5EED5);
    const size_t num_threads = 8;
    const size_t num_blocks = 128;
    auto cfg = PartitionConfig::make(static_cast<UInt64>(100'000'000), l2_bytes, 8192); /// 2048 leaves, 1 pass
    ASSERT_EQ(cfg.pass_bits.size(), 1u);
    ASSERT_GE(cfg.num_leaves, 256u) << "need fanout >= 256 so the scatter routes through SWWC";

    BuildStore store(cfg, {0}, {sizeof(UInt64)}, num_threads);

    /// Multi-threaded add so multiple build slots are populated -> multiple scatter workers with
    /// unaligned per-(thread,leaf) write starts (the head-peel path).
    const size_t per = (n + num_blocks - 1) / num_blocks;
    std::vector<Block> blocks;
    for (size_t b = 0; b < num_blocks; ++b)
    {
        const size_t lo = b * per;
        if (lo >= n)
            break;
        const size_t hi = std::min(n, lo + per);
        std::vector<UInt64> slice(keys.begin() + lo, keys.begin() + hi);
        blocks.push_back(makeBlock1<UInt64>(slice, 2, b + 1));
    }
    std::atomic<size_t> next{0};
    std::vector<std::thread> threads;
    for (size_t t = 0; t < num_threads; ++t)
        threads.emplace_back([&]
        {
            for (size_t b = next.fetch_add(1); b < blocks.size(); b = next.fetch_add(1))
                store.add(blocks[b]);
        });
    for (auto & th : threads)
        th.join();

    store.finishBuild();
    CoopPool coord;
    LeafArrays leaves;
    coopRun(coord, num_threads, [&] { leaves = store.scatterToLeaves(coord); });

    /// With NT active (x86-64-v3 multitarget) this fanout routes through the SWWC/NT path under test;
    /// correctness must hold on either path, so this only documents intent (no skip on a v2 build).
    if (RadixShuffle::ntStoresAvailable())
        EXPECT_TRUE(RadixShuffle::shouldUseSwwc(2, static_cast<int>(cfg.num_leaves)));

    /// At least two build slots contributed, so the scatter ran with multiple unaligned workers.
    size_t active_workers = 0;
    for (auto c : leaves.worker_block_counts)
        active_workers += (c > 0);
    EXPECT_GE(active_workers, size_t{2}) << "need multiple scatter workers to exercise head-peeling";

    verifyConservationAndRefs<UInt64>(store, leaves, {0}, {sizeof(UInt64)});

    /// Path-independent byte accounting: key + ref only, exactly once each.
    EXPECT_EQ(leaves.bytes_scattered, UInt64(n) * (sizeof(UInt64) + sizeof(RadixShuffle::BuildRef)));
}

/// Multi-column key (two UInt64 columns, packed width 16): chained hash for routing, row-major packed
/// keys scattered, parallel build + scatter on a pool. Verifies the packed key matches both columns.
TEST(RadixHashBuildScatter, MultiColumnKeyU64x2)
{
    const size_t n = 700'001;
    const size_t num_blocks = 40;
    const size_t num_threads = 4;
    const auto k0 = randomKeys(n, 0x111);
    const auto k1 = randomKeys(n, 0x222);

    auto cfg = PartitionConfig::make(static_cast<UInt64>(100'000'000), l2_bytes, 8192);
    ASSERT_EQ(cfg.pass_bits.size(), 1u);

    const std::vector<size_t> key_pos{0, 1};
    const std::vector<size_t> key_w{sizeof(UInt64), sizeof(UInt64)};
    BuildStore store(cfg, key_pos, key_w, num_threads);

    const size_t per = (n + num_blocks - 1) / num_blocks;
    std::vector<Block> blocks;
    for (size_t b = 0; b < num_blocks; ++b)
    {
        const size_t lo = b * per;
        if (lo >= n)
            break;
        const size_t hi = std::min(n, lo + per);
        std::vector<std::vector<UInt64>> keys{
            std::vector<UInt64>(k0.begin() + lo, k0.begin() + hi), std::vector<UInt64>(k1.begin() + lo, k1.begin() + hi)};
        blocks.push_back(makeBlock<UInt64>(keys, 2, b + 1));
    }

    std::atomic<size_t> next{0};
    std::vector<std::thread> threads;
    for (size_t t = 0; t < num_threads; ++t)
        threads.emplace_back([&]
        {
            for (size_t b = next.fetch_add(1); b < blocks.size(); b = next.fetch_add(1))
                store.add(blocks[b]);
        });
    for (auto & th : threads)
        th.join();

    store.finishBuild();
    CoopPool coord;
    LeafArrays leaves;
    coopRun(coord, num_threads, [&] { leaves = store.scatterToLeaves(coord); });

    EXPECT_EQ(leaves.key_width, 2 * sizeof(UInt64));
    verifyConservationAndRefs<UInt64>(store, leaves, key_pos, key_w);
}

/// Multi-column key with mixed widths (UInt32 + UInt64, packed width 12) exercising the generic
/// (non-power-of-two-tile) direct scatter width.
TEST(RadixHashBuildScatter, MultiColumnMixedWidth)
{
    const size_t n = 333'333;
    auto cfg = PartitionConfig::make(static_cast<UInt64>(20'000'000), l2_bytes, 8192);

    std::mt19937_64 rng(0xDEAD); /// NOLINT(cert-msc32-c,cert-msc51-cpp,bugprone-random-generator-seed) -- deterministic test
    BuildStore store(cfg, {0, 1}, {sizeof(UInt32), sizeof(UInt64)}, 2);

    size_t added = 0;
    for (size_t b = 0; b < 13; ++b)
    {
        const size_t rows = (b == 12) ? (n - added) : n / 13;
        std::vector<UInt32> c0(rows);
        std::vector<UInt64> c1(rows);
        for (size_t i = 0; i < rows; ++i)
        {
            c0[i] = static_cast<UInt32>(rng());
            c1[i] = rng();
        }
        ColumnsWithTypeAndName cols;
        cols.emplace_back(makeColumn<UInt32>(c0), std::make_shared<DataTypeUInt32>(), "k0");
        cols.emplace_back(makeColumn<UInt64>(c1), std::make_shared<DataTypeUInt64>(), "k1");
        store.add(Block(std::move(cols)));
        added += rows;
    }
    ASSERT_EQ(added, n);

    store.finishBuild();
    CoopPool coord;
    LeafArrays leaves;
    coopRun(coord, 2, [&] { leaves = store.scatterToLeaves(coord); });

    EXPECT_EQ(leaves.key_width, sizeof(UInt32) + sizeof(UInt64));
    verifyConservationAndRefs<UInt32>(store, leaves, {0, 1}, {sizeof(UInt32), sizeof(UInt64)});
}

/// Forced two-pass schedule (small per-pass cap) must produce the identical leaf membership as a
/// single-pass / scalar reference (mirrors RadixHashScatter.MultiPassMembership).
TEST(RadixHashBuildScatter, MultiPassMembership)
{
    const size_t n = 800'011;
    const auto keys = randomKeys(n, 0xFACADE);
    auto cfg = PartitionConfig::make(static_cast<UInt64>(100'000'000), l2_bytes, /*max_partitions_per_pass=*/64); /// 2048 leaves -> {6,5}
    ASSERT_EQ(cfg.pass_bits.size(), 2u);
    ASSERT_EQ(cfg.pass_bits[0], 6u);
    ASSERT_EQ(cfg.pass_bits[1], 5u);

    BuildStore store(cfg, {0}, {sizeof(UInt64)}, 4);
    addBlocksSerial<UInt64>(store, keys, 23, 1);
    store.finishBuild();
    CoopPool coord;
    LeafArrays leaves;
    coopRun(coord, 4, [&] { leaves = store.scatterToLeaves(coord); });

    verifyConservationAndRefs<UInt64>(store, leaves, {0}, {sizeof(UInt64)});
}

/// Multi-pass with a multi-column key (two UInt64 columns) -> {6,5} two-pass; each refine pass
/// recomputes the route hash from the scattered packed key (no carried hash column).
TEST(RadixHashBuildScatter, MultiPassMultiColumn)
{
    const size_t n = 600'013;
    const auto k0 = randomKeys(n, 0xA1);
    const auto k1 = randomKeys(n, 0xB2);
    auto cfg = PartitionConfig::make(static_cast<UInt64>(100'000'000), l2_bytes, /*max_partitions_per_pass=*/64);
    ASSERT_EQ(cfg.pass_bits.size(), 2u);

    const std::vector<size_t> key_pos{0, 1};
    const std::vector<size_t> key_w{sizeof(UInt64), sizeof(UInt64)};
    BuildStore store(cfg, key_pos, key_w, 4);

    const size_t num_blocks = 19;
    const size_t per = (n + num_blocks - 1) / num_blocks;
    for (size_t b = 0; b < num_blocks; ++b)
    {
        const size_t lo = b * per;
        if (lo >= n)
            break;
        const size_t hi = std::min(n, lo + per);
        std::vector<std::vector<UInt64>> keys{
            std::vector<UInt64>(k0.begin() + lo, k0.begin() + hi), std::vector<UInt64>(k1.begin() + lo, k1.begin() + hi)};
        store.add(makeBlock<UInt64>(keys, 1, b + 1));
    }

    store.finishBuild();
    CoopPool coord;
    LeafArrays leaves;
    coopRun(coord, 4, [&] { leaves = store.scatterToLeaves(coord); });

    verifyConservationAndRefs<UInt64>(store, leaves, key_pos, key_w);
}

/// num_leaves == 1: all rows in leaf 0, exact conservation.
TEST(RadixHashBuildScatter, SingleLeaf)
{
    const size_t n = 123'457;
    const auto keys = randomKeys(n, 0x1234);
    auto cfg = PartitionConfig::make(static_cast<UInt64>(1), l2_bytes, 8192);
    ASSERT_EQ(cfg.num_leaves, 1u);

    BuildStore store(cfg, {0}, {sizeof(UInt64)}, 1);
    addBlocksSerial<UInt64>(store, keys, 7, 2);
    store.finishBuild();
    CoopPool coord;
    LeafArrays leaves;
    coopRun(coord, 1, [&] { leaves = store.scatterToLeaves(coord); });

    EXPECT_EQ(leaves.leaf_rows[0], n);
    verifyConservationAndRefs<UInt64>(store, leaves, {0}, {sizeof(UInt64)});
}

/// Empty and odd-sized blocks interleaved with normal ones.
TEST(RadixHashBuildScatter, EmptyAndOddBlocks)
{
    auto cfg = PartitionConfig::make(static_cast<UInt64>(10'000'000), l2_bytes, 8192);
    BuildStore store(cfg, {0}, {sizeof(UInt64)}, 1);

    std::mt19937_64 rng(0x9999); /// NOLINT(cert-msc32-c,cert-msc51-cpp,bugprone-random-generator-seed) -- deterministic test
    size_t total = 0;
    for (size_t sizes : {size_t{0}, size_t{1}, size_t{7}, size_t{63}, size_t{64}, size_t{65}, size_t{0}, size_t{4096}})
    {
        std::vector<UInt64> keys(sizes);
        for (size_t i = 0; i < sizes; ++i)
            keys[i] = rng();
        store.add(makeBlock1<UInt64>(keys, 2, total + 1));
        total += sizes;
    }
    store.finishBuild();
    CoopPool coord;
    LeafArrays leaves;
    coopRun(coord, 1, [&] { leaves = store.scatterToLeaves(coord); });

    UInt64 sum = 0;
    for (auto r : leaves.leaf_rows)
        sum += r;
    EXPECT_EQ(sum, total);
    verifyConservationAndRefs<UInt64>(store, leaves, {0}, {sizeof(UInt64)});
}

/// Block larger than one scatter chunk (1024 rows) exercises the persistent-cursor chunked scatter.
TEST(RadixHashBuildScatter, ChunkedScatterLargeBlock)
{
    const size_t n = 50'000; /// one block, ~49 chunks of 1024
    const auto keys = randomKeys(n, 0xC4);
    auto cfg = PartitionConfig::make(static_cast<UInt64>(100'000'000), l2_bytes, 8192);

    BuildStore store(cfg, {0}, {sizeof(UInt64)}, 1);
    store.add(makeBlock1<UInt64>(keys, 1, 1)); /// single big block
    store.finishBuild();
    CoopPool coord;
    LeafArrays leaves;
    coopRun(coord, 1, [&] { leaves = store.scatterToLeaves(coord); });

    EXPECT_EQ(store.numBlocks(), 1u);
    verifyConservationAndRefs<UInt64>(store, leaves, {0}, {sizeof(UInt64)});
}

/// Lock-free parallel build (distinct worker slots) + pool-driven scatter yields the same membership.
TEST(RadixHashBuildScatter, ParallelBuildMatchesSerial)
{
    const size_t n = 2'000'003;
    const auto keys = randomKeys(n, 0x5EED);
    const size_t num_threads = 8;
    const size_t num_blocks = 200;

    auto cfg = PartitionConfig::make(static_cast<UInt64>(100'000'000), l2_bytes, 8192);
    BuildStore store(cfg, {0}, {sizeof(UInt64)}, num_threads);

    const size_t per = (n + num_blocks - 1) / num_blocks;
    std::vector<Block> blocks;
    for (size_t b = 0; b < num_blocks; ++b)
    {
        const size_t lo = b * per;
        if (lo >= n)
            break;
        const size_t hi = std::min(n, lo + per);
        std::vector<UInt64> slice(keys.begin() + lo, keys.begin() + hi);
        blocks.push_back(makeBlock1<UInt64>(slice, 2, b + 1));
    }

    std::atomic<size_t> next{0};
    std::vector<std::thread> threads;
    for (size_t t = 0; t < num_threads; ++t)
        threads.emplace_back([&]
        {
            for (size_t b = next.fetch_add(1); b < blocks.size(); b = next.fetch_add(1))
                store.add(blocks[b]);
        });
    for (auto & th : threads)
        th.join();

    store.finishBuild();
    CoopPool coord;
    LeafArrays leaves;
    coopRun(coord, num_threads, [&] { leaves = store.scatterToLeaves(coord); });

    verifyConservationAndRefs<UInt64>(store, leaves, {0}, {sizeof(UInt64)});
}

/// More distinct build threads than max_threads is a fail-close error (never silent corruption).
TEST(RadixHashBuildScatter, SlotExhaustionThrows)
{
    auto cfg = PartitionConfig::make(static_cast<UInt64>(1'000'000), l2_bytes, 8192);
    BuildStore store(cfg, {0}, {sizeof(UInt64)}, /*max_threads=*/2);

    std::atomic<int> threw{0};
    std::vector<std::thread> threads;
    for (size_t t = 0; t < 4; ++t)
        threads.emplace_back([&]
        {
            try
            {
                std::vector<UInt64> keys(1000, t);
                store.add(makeBlock1<UInt64>(keys, 1, 1));
                std::this_thread::yield();
            }
            catch (...)
            {
                threw.fetch_add(1);
            }
        });
    for (auto & th : threads)
        th.join();

    EXPECT_GT(threw.load(), 0) << "expected at least one slot-exhaustion throw with 4 threads and max_threads=2";
}

/// ZC gate (spec section 9.4): only key + ref bytes are scattered, never payload. The counter must
/// equal exactly N*(key_width + 8) for a single pass regardless of how many payload columns exist.
TEST(RadixHashBuildScatter, ZeroCopyBytesAccounting)
{
    const size_t n = 400'009;
    const auto keys = randomKeys(n, 0x2C0DE);
    auto cfg = PartitionConfig::make(static_cast<UInt64>(100'000'000), l2_bytes, 8192);
    ASSERT_EQ(cfg.pass_bits.size(), 1u);

    BuildStore store(cfg, {0}, {sizeof(UInt64)}, 4);
    addBlocksSerial<UInt64>(store, keys, 40, /*num_payload=*/7); /// 7 payload cols, never scattered
    store.finishBuild();
    CoopPool coord;
    LeafArrays leaves;
    coopRun(coord, 4, [&] { leaves = store.scatterToLeaves(coord); });

    EXPECT_EQ(leaves.bytes_scattered, UInt64(n) * (sizeof(UInt64) + sizeof(RadixShuffle::BuildRef)));
    verifyConservationAndRefs<UInt64>(store, leaves, {0}, {sizeof(UInt64)});
}

/// NC gate (spec section 9.4): runPostBuildPhase allocations are O(num_leaves), independent of the
/// number of build blocks. Same data fed as few-big vs many-small blocks must allocate identically.
TEST(RadixHashBuildScatter, NoAllocatorChurn)
{
    const size_t n = 1'000'003;
    const auto keys = randomKeys(n, 0xA110C);
    auto cfg = PartitionConfig::make(static_cast<UInt64>(100'000'000), l2_bytes, 8192);

    auto run = [&](size_t num_blocks)
    {
        BuildStore store(cfg, {0}, {sizeof(UInt64)}, 1);
        addBlocksSerial<UInt64>(store, keys, num_blocks, 2);
        store.finishBuild();
        CoopPool coord;
        LeafArrays la;
        coopRun(coord, 1, [&] { la = store.scatterToLeaves(coord); });
        return la;
    };

    const LeafArrays few = run(4);
    const LeafArrays many = run(500);

    EXPECT_EQ(few.alloc_count, many.alloc_count);
    EXPECT_LE(few.alloc_count, UInt64(4) * cfg.num_leaves) << "must be O(num_leaves), not O(blocks*leaves)";
    EXPECT_EQ(few.arena.blockCount(), many.arena.blockCount());
}

/// PB gate (spec section 9.3): the scatter uses all build threads. With static per-thread scatter,
/// "engaging workers" means all build threads participated (each block range is non-empty).
/// Must use multi-threaded add() to populate multiple slots.
TEST(RadixHashBuildScatter, ParallelScatterEngagesWorkers)
{
    const size_t n = 4'000'003;
    const auto keys = randomKeys(n, 0xB055);
    const size_t num_threads = 8;
    const size_t num_blocks = 256;
    auto cfg = PartitionConfig::make(static_cast<UInt64>(100'000'000), l2_bytes, 8192);

    BuildStore store(cfg, {0}, {sizeof(UInt64)}, num_threads);

    /// Multi-threaded add so all num_threads slots are populated (= multiple scatter workers).
    const size_t per = (n + num_blocks - 1) / num_blocks;
    std::vector<Block> blocks;
    for (size_t b = 0; b < num_blocks; ++b)
    {
        const size_t lo = b * per;
        if (lo >= n) break;
        const size_t hi = std::min(n, lo + per);
        std::vector<UInt64> slice(keys.begin() + lo, keys.begin() + hi);
        blocks.push_back(makeBlock1<UInt64>(slice, 2, b + 1));
    }
    std::atomic<size_t> next{0};
    std::vector<std::thread> threads;
    for (size_t t = 0; t < num_threads; ++t)
        threads.emplace_back([&]
        {
            for (size_t b = next.fetch_add(1); b < blocks.size(); b = next.fetch_add(1))
                store.add(blocks[b]);
        });
    for (auto & th : threads)
        th.join();

    store.finishBuild();
    CoopPool coord;
    LeafArrays leaves;
    coopRun(coord, num_threads, [&] { leaves = store.scatterToLeaves(coord); });

    /// worker_block_counts has one entry per build thread (used slot).
    ASSERT_EQ(leaves.worker_block_counts.size(), num_threads);
    UInt64 sum = 0;
    UInt64 max_one = 0;
    size_t active = 0;
    for (auto c : leaves.worker_block_counts)
    {
        sum += c;
        max_one = std::max(max_one, c);
        active += (c > 0);
    }
    EXPECT_EQ(sum, store.numBlocks());
    EXPECT_LT(max_one, store.numBlocks()) << "no single worker holds all blocks (static assignment)";
    EXPECT_GE(active, size_t{2}) << "at least two build threads should have contributed blocks";

    verifyConservationAndRefs<UInt64>(store, leaves, {0}, {sizeof(UInt64)});
}

/// PartitionConfig invariants (spec sections 5.2, 5.3): moved here from the now-deleted
/// gtest_radix_hash_selector.cpp to preserve that coverage.
TEST(RadixHashBuildScatter, PartitionConfigInvariants)
{
    constexpr size_t l2 = 2 * 1024 * 1024;
    constexpr UInt64 cap = 8192; /// default max_partitions_per_pass

    /// Spec anchors: absent estimate -> 256 leaves; 100M rows / L2=2 MiB -> 2048 leaves, {11}.
    auto def = PartitionConfig::make(std::nullopt, l2, cap);
    EXPECT_EQ(def.num_leaves, 256u);
    EXPECT_EQ(def.total_bits, 8u);
    ASSERT_EQ(def.pass_bits.size(), 1u);
    EXPECT_EQ(def.pass_bits[0], 8u);

    auto big = PartitionConfig::make(static_cast<UInt64>(100'000'000), l2, cap); /// -> 2048 leaves, single pass
    EXPECT_EQ(big.num_leaves, 2048u);
    EXPECT_EQ(big.total_bits, 11u);
    ASSERT_EQ(big.pass_bits.size(), 1u); /// BITS_PER_PASS=13, so 11 <= 13 -> single pass
    EXPECT_EQ(big.pass_bits[0], 11u);

    /// With the old cap=10 config ({6,5}) still correctly factored via cap=64 -> {6,5}.
    auto two_pass = PartitionConfig::make(static_cast<UInt64>(100'000'000), l2, /*max_partitions_per_pass=*/64);
    EXPECT_EQ(two_pass.num_leaves, 2048u);
    ASSERT_EQ(two_pass.pass_bits.size(), 2u);
    EXPECT_EQ(two_pass.pass_bits[0], 6u);
    EXPECT_EQ(two_pass.pass_bits[1], 5u);

    /// Sweep a range of estimates and assert the structural invariants on every config:
    ///   num_leaves is a power of two in [1, MAX_LEAVES]; total_bits == log2(num_leaves);
    ///   shift == 32 - total_bits; sum(pass_bits) == total_bits; max - min <= 1.
    bool seen_20 = false;
    for (UInt64 e = 1; e <= static_cast<UInt64>(1e11); e = e * 2 + 1)
    {
        auto cfg = PartitionConfig::make(e, l2, cap);
        EXPECT_LE(cfg.num_leaves, PartitionConfig::MAX_LEAVES);
        EXPECT_GE(cfg.num_leaves, 1u);
        EXPECT_EQ(cfg.num_leaves & (cfg.num_leaves - 1), 0u) << "num_leaves must be a power of two";
        EXPECT_EQ(size_t{1} << cfg.total_bits, cfg.num_leaves);
        EXPECT_EQ(cfg.shift, PartitionConfig::HASH_BITS - cfg.total_bits);

        UInt32 sum = 0;
        UInt32 lo = 64;
        UInt32 hi = 0;
        for (auto b : cfg.pass_bits)
        {
            sum += b;
            lo = std::min(lo, b);
            hi = std::max(hi, b);
        }
        EXPECT_EQ(sum, cfg.total_bits);
        if (cfg.total_bits > 0)
            EXPECT_LE(hi - lo, 1u);

        if (cfg.total_bits == 20)
            seen_20 = true;
    }
    EXPECT_TRUE(seen_20) << "sweep should reach total_bits=20 (MAX_LEAVES=2^20)";
}

/// GrowingArena (jemalloc-backed): every alloc is its own aligned jemalloc block, a large allocation is
/// contiguous, and freeBlock() releases one block while the rest stay live.
TEST(RadixHashBuildScatter, GrowingArenaAllocAndFreeBlock)
{
    GrowingArena arena(/*max_block_bytes=*/1024 * 1024); /// cap is retained for API compat but ignored

    /// Each allocation is a separate jemalloc block; all are 64 B-aligned.
    std::vector<char *> ptrs;
    for (int i = 0; i < 100000; ++i)
        ptrs.push_back(static_cast<char *>(arena.alloc(64, 64)));
    EXPECT_EQ(arena.blockCount(), size_t{100000});
    for (char * p : ptrs)
        EXPECT_EQ(reinterpret_cast<uintptr_t>(p) % 64, 0u);

    /// A large allocation is honored contiguously.
    const size_t big = 4 * 1024 * 1024;
    char * b = static_cast<char *>(arena.alloc(big, 64));
    std::memset(b, 0xAB, big); /// fully writable & contiguous
    EXPECT_EQ(static_cast<unsigned char>(b[0]), 0xABu);
    EXPECT_EQ(static_cast<unsigned char>(b[big - 1]), 0xABu);

    /// freeBlock releases exactly one allocation; a live neighbour stays valid and the arena keeps working.
    char * live = static_cast<char *>(arena.alloc(4096, 64));
    std::memset(live, 0xCD, 4096);
    const size_t blocks_before = arena.blockCount();
    arena.freeBlock(b); /// free the 4 MiB block
    EXPECT_EQ(arena.blockCount(), blocks_before - 1) << "freeBlock removes exactly one block";
    EXPECT_EQ(static_cast<unsigned char>(live[0]), 0xCDu) << "freeBlock must not disturb other blocks";
    char * after = static_cast<char *>(arena.alloc(128, 64));
    EXPECT_NE(after, nullptr);
}

/// Large-width composite key: 3x UInt64 = 24 B packed (exercises the generic scatter path — 24 is
/// not in the templated switch 4/8/16/32/64/128, so it uses the 4-byte-lane fallback).
TEST(RadixHashBuildScatter, ConservationThreeColumnU64)
{
    const size_t n = 600'003;
    const auto k0 = randomKeys(n, 0xA01);
    const auto k1 = randomKeys(n, 0xA02);
    const auto k2 = randomKeys(n, 0xA03);
    auto cfg = PartitionConfig::make(static_cast<UInt64>(100'000'000), l2_bytes, 8192);
    ASSERT_EQ(cfg.pass_bits.size(), 1u);

    const std::vector<size_t> kpos{0, 1, 2};
    const std::vector<size_t> kw{8, 8, 8};
    BuildStore store(cfg, kpos, kw, 4);
    const size_t per = (n + 29) / 30;
    for (size_t b = 0; b < 30; ++b)
    {
        const size_t lo = b * per;
        if (lo >= n) break;
        const size_t hi = std::min(n, lo + per);
        std::vector<std::vector<UInt64>> keys{{k0.begin()+lo,k0.begin()+hi},{k1.begin()+lo,k1.begin()+hi},{k2.begin()+lo,k2.begin()+hi}};
        store.add(makeBlock<UInt64>(keys, 1, b));
    }
    store.finishBuild();
    CoopPool coord;
    LeafArrays leaves;
    coopRun(coord, 4, [&] { leaves = store.scatterToLeaves(coord); });

    EXPECT_EQ(leaves.key_width, 3 * sizeof(UInt64));
    verifyConservationAndRefs<UInt64>(store, leaves, kpos, kw);
}

/// Large-width composite key: 4x UInt64 = 32 B packed.
TEST(RadixHashBuildScatter, ConservationFourColumnU64)
{
    const size_t n = 400'007;
    auto gen = [&](uint64_t seed) { return randomKeys(n, seed); };
    auto k0 = gen(0xB01);
    auto k1 = gen(0xB02);
    auto k2 = gen(0xB03);
    auto k3 = gen(0xB04);
    auto cfg = PartitionConfig::make(static_cast<UInt64>(100'000'000), l2_bytes, 8192);

    const std::vector<size_t> kpos{0,1,2,3};
    const std::vector<size_t> kw{8,8,8,8};
    BuildStore store(cfg, kpos, kw, 4);
    const size_t per = (n + 19) / 20;
    for (size_t b = 0; b < 20; ++b)
    {
        const size_t lo = b*per; if (lo>=n) break;
        const size_t hi = std::min(n, lo+per);
        std::vector<std::vector<UInt64>> keys{{k0.begin()+lo,k0.begin()+hi},{k1.begin()+lo,k1.begin()+hi},{k2.begin()+lo,k2.begin()+hi},{k3.begin()+lo,k3.begin()+hi}};
        store.add(makeBlock<UInt64>(keys, 0, b));
    }
    store.finishBuild();
    CoopPool coord;
    LeafArrays leaves;
    coopRun(coord, 4, [&] { leaves = store.scatterToLeaves(coord); });

    EXPECT_EQ(leaves.key_width, 4 * sizeof(UInt64));
    verifyConservationAndRefs<UInt64>(store, leaves, kpos, kw);
}

/// 3x UInt64 (24 B) with a forced 2-pass scatter.
TEST(RadixHashBuildScatter, MultiPassThreeColumnU64)
{
    const size_t n = 500'003;
    const auto k0 = randomKeys(n, 0xC01);
    const auto k1 = randomKeys(n, 0xC02);
    const auto k2 = randomKeys(n, 0xC03);
    auto cfg = PartitionConfig::make(static_cast<UInt64>(100'000'000), l2_bytes, /*max_partitions_per_pass=*/64);
    ASSERT_EQ(cfg.pass_bits.size(), 2u);

    const std::vector<size_t> kpos{0,1,2};
    const std::vector<size_t> kw{8,8,8};
    BuildStore store(cfg, kpos, kw, 4);
    const size_t per = (n+14)/15;
    for (size_t b = 0; b < 15; ++b)
    {
        const size_t lo=b*per; if (lo>=n) break;
        const size_t hi=std::min(n,lo+per);
        std::vector<std::vector<UInt64>> keys{{k0.begin()+lo,k0.begin()+hi},{k1.begin()+lo,k1.begin()+hi},{k2.begin()+lo,k2.begin()+hi}};
        store.add(makeBlock<UInt64>(keys, 1, b));
    }
    store.finishBuild();
    CoopPool coord;
    LeafArrays leaves;
    coopRun(coord, 4, [&] { leaves = store.scatterToLeaves(coord); });
    verifyConservationAndRefs<UInt64>(store, leaves, kpos, kw);
}

/// 3-pass recursion: max_partitions_per_pass=16 -> bits_per_pass=4 -> pass_bits={4,4,3} for 2048 leaves.
/// Exercises refineDepthFirst intermediate levels (the 3-pass path is only reached with a small cap).
TEST(RadixHashBuildScatter, ThreePassRecursion)
{
    const size_t n = 800'011;
    const auto keys = randomKeys(n, 0xD0D0D0);
    auto cfg = PartitionConfig::make(static_cast<UInt64>(100'000'000), l2_bytes, /*max_partitions_per_pass=*/16);
    ASSERT_EQ(cfg.num_leaves, 2048u);
    ASSERT_EQ(cfg.pass_bits.size(), 3u);
    EXPECT_EQ(cfg.pass_bits[0], 4u); /// p0 = 16
    EXPECT_EQ(cfg.pass_bits[1], 4u);
    EXPECT_EQ(cfg.pass_bits[2], 3u);

    BuildStore store(cfg, {0}, {sizeof(UInt64)}, 8);
    addBlocksSerial<UInt64>(store, keys, 64, 1);
    store.finishBuild();
    CoopPool coord;
    LeafArrays leaves;
    coopRun(coord, 8, [&] { leaves = store.scatterToLeaves(coord); });
    verifyConservationAndRefs<UInt64>(store, leaves, {0}, {sizeof(UInt64)});
}

/// 3-pass recursion with a 3-column wide key (24 B), exercising both depth-first and the generic
/// scatter path in the intermediate levels.
TEST(RadixHashBuildScatter, ThreePassWideKey)
{
    const size_t n = 500'007;
    const auto k0 = randomKeys(n, 0xE01);
    const auto k1 = randomKeys(n, 0xE02);
    const auto k2 = randomKeys(n, 0xE03);
    auto cfg = PartitionConfig::make(static_cast<UInt64>(100'000'000), l2_bytes, /*max_partitions_per_pass=*/16);
    ASSERT_EQ(cfg.pass_bits.size(), 3u);

    const std::vector<size_t> kpos{0,1,2};
    const std::vector<size_t> kw{8,8,8};
    BuildStore store(cfg, kpos, kw, 4);
    const size_t per=(n+29)/30;
    for (size_t b=0; b<30; ++b)
    {
        const size_t lo=b*per; if (lo>=n) break;
        const size_t hi=std::min(n,lo+per);
        std::vector<std::vector<UInt64>> keys{{k0.begin()+lo,k0.begin()+hi},{k1.begin()+lo,k1.begin()+hi},{k2.begin()+lo,k2.begin()+hi}};
        store.add(makeBlock<UInt64>(keys, 0, b));
    }
    store.finishBuild();
    CoopPool coord;
    LeafArrays leaves;
    coopRun(coord, 4, [&] { leaves = store.scatterToLeaves(coord); });
    verifyConservationAndRefs<UInt64>(store, leaves, kpos, kw);
}

/// Opt-in 100M-scale ns/row measurement (set RHJ_P3_BENCH=1). Reports the wall-amortized build-scatter
/// ns/row of the parallel region (spec section 9.2 basis), not gated by default.
TEST(RadixHashBuildScatter, ScatterNsPerRowBench)
{
    if (std::getenv("RHJ_P3_BENCH") == nullptr) /// NOLINT(concurrency-mt-unsafe)
        GTEST_SKIP() << "set RHJ_P3_BENCH=1 to run the 100M-scale scatter ns/row benchmark";

    const size_t n = 100'000'000;
    const size_t num_threads = 16;
    const size_t block_rows = 65536;

    auto cfg = PartitionConfig::make(static_cast<UInt64>(n), l2_bytes, 8192);
    BuildStore store(cfg, {0}, {sizeof(UInt64)}, num_threads);

    std::mt19937_64 rng(0xBE0C); /// NOLINT(cert-msc32-c,cert-msc51-cpp,bugprone-random-generator-seed) -- deterministic test
    const size_t num_blocks = (n + block_rows - 1) / block_rows;
    std::vector<Block> blocks;
    blocks.reserve(num_blocks);
    for (size_t b = 0; b < num_blocks; ++b)
    {
        const size_t rows = std::min(block_rows, n - b * block_rows);
        std::vector<UInt64> keys(rows);
        for (size_t i = 0; i < rows; ++i)
            keys[i] = rng();
        blocks.push_back(makeBlock1<UInt64>(keys, 0, b + 1));
    }

    std::atomic<size_t> next{0};
    std::vector<std::thread> threads;
    for (size_t t = 0; t < num_threads; ++t)
        threads.emplace_back([&]
        {
            for (size_t b = next.fetch_add(1); b < blocks.size(); b = next.fetch_add(1))
                store.add(blocks[b]);
        });
    for (auto & th : threads)
        th.join();
    store.finishBuild();

    CoopPool coord;
    LeafArrays leaves;
    Stopwatch sw;
    coopRun(coord, num_threads, [&] { leaves = store.scatterToLeaves(coord); });
    const double wall_ns = static_cast<double>(sw.elapsedNanoseconds());

    const double ns_per_row = wall_ns / static_cast<double>(n);
    std::cout << fmt::format(
        "P3 build scatter: n={} leaves={} passes={} threads={} wall={:.1f}ms ns/row/pass(wall)={:.3f}\n",
        n, cfg.num_leaves, cfg.pass_bits.size(), num_threads, wall_ns / 1e6,
        ns_per_row / static_cast<double>(cfg.pass_bits.size()));

    EXPECT_EQ(leaves.leaf_rows.size(), cfg.num_leaves);
}

/// Opt-in benchmark for the add() / build-select path (set RHJ_P3_BENCH=1).
TEST(RadixHashBuildScatter, AddNsPerRowBench)
{
    if (std::getenv("RHJ_P3_BENCH") == nullptr) /// NOLINT(concurrency-mt-unsafe)
        GTEST_SKIP() << "set RHJ_P3_BENCH=1 to run the add() ns/row benchmark";

    const size_t n = 100'000'000;
    const size_t num_threads = 16;
    const size_t block_rows = 65536;

    auto cfg = PartitionConfig::make(static_cast<UInt64>(n), l2_bytes, 8192);
    BuildStore store(cfg, {0}, {sizeof(UInt64)}, num_threads);

    std::mt19937_64 rng(0xBE0C); /// NOLINT(cert-msc32-c,cert-msc51-cpp,bugprone-random-generator-seed)
    const size_t num_blocks = (n + block_rows - 1) / block_rows;
    std::vector<Block> blocks;
    blocks.reserve(num_blocks);
    for (size_t b = 0; b < num_blocks; ++b)
    {
        const size_t rows = std::min(block_rows, n - b * block_rows);
        std::vector<UInt64> keys(rows);
        for (size_t i = 0; i < rows; ++i)
            keys[i] = rng();
        blocks.push_back(makeBlock1<UInt64>(keys, 0, b + 1));
    }

    std::atomic<size_t> next{0};
    Stopwatch sw;
    std::vector<std::thread> threads;
    for (size_t t = 0; t < num_threads; ++t)
        threads.emplace_back([&]
        {
            for (size_t b = next.fetch_add(1); b < blocks.size(); b = next.fetch_add(1))
                store.add(blocks[b]);
        });
    for (auto & th : threads)
        th.join();
    const double add_ns = static_cast<double>(sw.elapsedNanoseconds());

    std::cout << fmt::format(
        "P3 add (build-select): n={} leaves={} threads={} wall={:.1f}ms ns/row(wall)={:.3f}\n",
        n, cfg.num_leaves, num_threads, add_ns / 1e6, add_ns / static_cast<double>(n));

    /// `numBlocks()` is only populated by `finishBuild()` (the timed add() loop above is done), so
    /// finalise the build before asserting the block count — matching the other benches.
    store.finishBuild();
    EXPECT_EQ(store.numBlocks(), num_blocks);
}

/// Opt-in benchmark for the forced two-pass scatter path (set RHJ_P3_BENCH=1).
TEST(RadixHashBuildScatter, ScatterTwoPassBench)
{
    if (std::getenv("RHJ_P3_BENCH") == nullptr) /// NOLINT(concurrency-mt-unsafe)
        GTEST_SKIP() << "set RHJ_P3_BENCH=1 to run the two-pass scatter benchmark";

    const size_t n = 100'000'000;
    const size_t num_threads = 16;
    const size_t block_rows = 65536;

    /// Force a {6,5} two-pass schedule (cap=64 -> bits_per_pass=6 -> 2 passes for 11 bits).
    auto cfg = PartitionConfig::make(static_cast<UInt64>(n), l2_bytes, /*max_partitions_per_pass=*/64);
    ASSERT_EQ(cfg.pass_bits.size(), 2u);
    BuildStore store(cfg, {0}, {sizeof(UInt64)}, num_threads);

    std::mt19937_64 rng(0xBE1D); /// NOLINT(cert-msc32-c,cert-msc51-cpp,bugprone-random-generator-seed)
    const size_t num_blocks = (n + block_rows - 1) / block_rows;
    std::vector<Block> blocks;
    blocks.reserve(num_blocks);
    for (size_t b = 0; b < num_blocks; ++b)
    {
        const size_t rows = std::min(block_rows, n - b * block_rows);
        std::vector<UInt64> keys(rows);
        for (size_t i = 0; i < rows; ++i)
            keys[i] = rng();
        blocks.push_back(makeBlock1<UInt64>(keys, 0, b + 1));
    }

    std::atomic<size_t> next{0};
    std::vector<std::thread> threads;
    for (size_t t = 0; t < num_threads; ++t)
        threads.emplace_back([&]
        {
            for (size_t b = next.fetch_add(1); b < blocks.size(); b = next.fetch_add(1))
                store.add(blocks[b]);
        });
    for (auto & th : threads)
        th.join();
    store.finishBuild();

    CoopPool coord;
    LeafArrays leaves;
    Stopwatch sw;
    coopRun(coord, num_threads, [&] { leaves = store.scatterToLeaves(coord); });
    const double wall_ns = static_cast<double>(sw.elapsedNanoseconds());

    const double ns_per_row_per_pass = wall_ns / static_cast<double>(n) / static_cast<double>(cfg.pass_bits.size());
    std::cout << fmt::format(
        "P3 scatter (2-pass): n={} leaves={} passes={} threads={} wall={:.1f}ms ns/row/pass(wall)={:.3f}\n",
        n, cfg.num_leaves, cfg.pass_bits.size(), num_threads, wall_ns / 1e6, ns_per_row_per_pass);

    EXPECT_EQ(leaves.leaf_rows.size(), cfg.num_leaves);
}

/// Opt-in bench: wide composite-key (4x UInt64 = 32 B packed, 100M rows, single-pass).
TEST(RadixHashBuildScatter, ScatterWideKeyBench)
{
    if (std::getenv("RHJ_P3_BENCH") == nullptr) /// NOLINT(concurrency-mt-unsafe)
        GTEST_SKIP() << "set RHJ_P3_BENCH=1 to run the wide-key scatter benchmark";

    const size_t n = 100'000'000;
    const size_t num_threads = 16;
    const size_t block_rows = 65536;

    auto cfg = PartitionConfig::make(static_cast<UInt64>(n), l2_bytes, 8192);
    ASSERT_EQ(cfg.pass_bits.size(), 1u);
    const std::vector<size_t> kpos{0, 1, 2, 3};
    const std::vector<size_t> kw_arr{8, 8, 8, 8};
    BuildStore store(cfg, kpos, kw_arr, num_threads);

    std::mt19937_64 rng(0xFACE); /// NOLINT(cert-msc32-c,cert-msc51-cpp,bugprone-random-generator-seed)
    const size_t num_blocks = (n + block_rows - 1) / block_rows;
    std::vector<Block> blocks;
    blocks.reserve(num_blocks);
    for (size_t b = 0; b < num_blocks; ++b)
    {
        const size_t rows = std::min(block_rows, n - b * block_rows);
        std::vector<std::vector<UInt64>> keys(4, std::vector<UInt64>(rows));
        for (size_t c = 0; c < 4; ++c)
            for (size_t i = 0; i < rows; ++i)
                keys[c][i] = rng();
        blocks.push_back(makeBlock<UInt64>(keys, 0, b));
    }

    std::atomic<size_t> next{0};
    std::vector<std::thread> threads;
    for (size_t t = 0; t < num_threads; ++t)
        threads.emplace_back([&]
        {
            for (size_t b = next.fetch_add(1); b < blocks.size(); b = next.fetch_add(1))
                store.add(blocks[b]);
        });
    for (auto & th : threads)
        th.join();
    store.finishBuild();

    CoopPool coord;
    LeafArrays leaves;
    Stopwatch sw;
    coopRun(coord, num_threads, [&] { leaves = store.scatterToLeaves(coord); });
    const double wall_ns = static_cast<double>(sw.elapsedNanoseconds());

    std::cout << fmt::format(
        "P3 scatter wide-key (4x8B=32B): n={} leaves={} passes={} threads={} wall={:.1f}ms ns/row/pass(wall)={:.3f}\n",
        n, cfg.num_leaves, cfg.pass_bits.size(), num_threads, wall_ns / 1e6,
        wall_ns / static_cast<double>(n) / static_cast<double>(cfg.pass_bits.size()));

    EXPECT_EQ(leaves.leaf_rows.size(), cfg.num_leaves);
}

/// Opt-in bench: 3-pass depth-first scatter (cap=16 -> {4,4,3}, 100M rows, 1-column key).
TEST(RadixHashBuildScatter, ScatterThreePassBench)
{
    if (std::getenv("RHJ_P3_BENCH") == nullptr) /// NOLINT(concurrency-mt-unsafe)
        GTEST_SKIP() << "set RHJ_P3_BENCH=1 to run the 3-pass scatter benchmark";

    const size_t n = 100'000'000;
    const size_t num_threads = 16;
    const size_t block_rows = 65536;

    auto cfg = PartitionConfig::make(static_cast<UInt64>(n), l2_bytes, /*max_partitions_per_pass=*/16);
    ASSERT_EQ(cfg.pass_bits.size(), 3u);
    BuildStore store(cfg, {0}, {sizeof(UInt64)}, num_threads);

    std::mt19937_64 rng(0xF3E7); /// NOLINT(cert-msc32-c,cert-msc51-cpp,bugprone-random-generator-seed)
    const size_t num_blocks = (n + block_rows - 1) / block_rows;
    std::vector<Block> blocks;
    blocks.reserve(num_blocks);
    for (size_t b = 0; b < num_blocks; ++b)
    {
        const size_t rows = std::min(block_rows, n - b * block_rows);
        std::vector<UInt64> keys(rows);
        for (size_t i = 0; i < rows; ++i)
            keys[i] = rng();
        blocks.push_back(makeBlock1<UInt64>(keys, 0, b + 1));
    }

    std::atomic<size_t> next{0};
    std::vector<std::thread> threads;
    for (size_t t = 0; t < num_threads; ++t)
        threads.emplace_back([&]
        {
            for (size_t b = next.fetch_add(1); b < blocks.size(); b = next.fetch_add(1))
                store.add(blocks[b]);
        });
    for (auto & th : threads)
        th.join();
    store.finishBuild();

    CoopPool coord;
    LeafArrays leaves;
    Stopwatch sw;
    coopRun(coord, num_threads, [&] { leaves = store.scatterToLeaves(coord); });
    const double wall_ns = static_cast<double>(sw.elapsedNanoseconds());

    std::cout << fmt::format(
        "P3 scatter 3-pass: n={} leaves={} passes={} threads={} wall={:.1f}ms ns/row/pass(wall)={:.3f}\n",
        n, cfg.num_leaves, cfg.pass_bits.size(), num_threads, wall_ns / 1e6,
        wall_ns / static_cast<double>(n) / static_cast<double>(cfg.pass_bits.size()));

    EXPECT_EQ(leaves.leaf_rows.size(), cfg.num_leaves);
}

/// G1 coverage: verifies an end-to-end build + scatter with >65536 leaves (conservation +
/// routing identity). Uses a small per-core L2 to force a large leaf count (2^18 = 262144 leaves)
/// from a billion-row estimate, exercising the 2-pass depth-first scatter at large width.
TEST(RadixHashBuildScatter, ConservationLargeLeafCount)
{
    /// Force num_leaves = 2^18 = 262144 > 65536 by using a tiny l2 and a large row estimate.
    /// table_bytes = 1B * 32 = 32 GB; usable_l2 = 0.8 * 256 KB ≈ 205 KB;
    /// n_leaves = ceil(32 GB / 205 KB) = 152588 -> roundUpToPow2 = 262144 (2^18).
    const size_t l2_small = 256 * 1024;
    auto cfg = PartitionConfig::make(static_cast<UInt64>(1'000'000'000), l2_small, 8192);
    ASSERT_GT(cfg.num_leaves, 65536u) << "config must have >65536 leaves for this test";
    ASSERT_EQ(cfg.pass_bits.size(), 2u) << "expected 2-pass schedule for 18 bits with cap=8192";

    const size_t n = 2'000'003;
    const auto keys = randomKeys(n, 0xBEEF01);
    BuildStore store(cfg, {0}, {sizeof(UInt64)}, 4);
    addBlocksSerial<UInt64>(store, keys, 23, 1);
    store.finishBuild();
    CoopPool coord;
    LeafArrays leaves;
    coopRun(coord, 4, [&] { leaves = store.scatterToLeaves(coord); });

    EXPECT_EQ(leaves.num_leaves, cfg.num_leaves);
    verifyConservationAndRefs<UInt64>(store, leaves, {0}, {sizeof(UInt64)});
}

/// Multi-pass PASS-0 scatter through the incremental SWWC/NT path: multi-threaded add() populates
/// several build slots, so the pass-0 scatter runs with multiple workers writing into the SHARED
/// per-partition arrays at generally-unaligned per-(thread,partition) starts — exercising the head-peel
/// + line-flush + residual-drain for the key and the 8 B ref. This closes the gap left by
/// `ConservationLargeLeafCount` (serial add -> every worker starts at an aligned offset). Reuses the
/// `{9,9}` (fanout 512/512, 2^18 leaves) config so both passes route through SWWC (>= 256).
TEST(RadixHashBuildScatter, ConservationMultiPassSwwc)
{
    const size_t l2_small = 256 * 1024;
    auto cfg = PartitionConfig::make(static_cast<UInt64>(1'000'000'000), l2_small, 8192);
    ASSERT_EQ(cfg.pass_bits.size(), 2u) << "need a multi-pass schedule";
    ASSERT_GE(size_t{1} << cfg.pass_bits[0], 256u) << "pass-0 fanout must be >= 256 to route through SWWC";
    ASSERT_GE(size_t{1} << cfg.pass_bits[1], 256u) << "refine fanout must be >= 256 to route through SWWC";

    const size_t n = 2'000'003;
    const auto keys = randomKeys(n, 0xD15EA5E);
    const size_t num_threads = 8;
    const size_t num_blocks = 128;

    BuildStore store(cfg, {0}, {sizeof(UInt64)}, num_threads);

    /// Multi-threaded add so multiple build slots are populated -> multiple pass-0 scatter workers with
    /// unaligned per-(thread,partition) write starts (the head-peel path, including the hash column).
    const size_t per = (n + num_blocks - 1) / num_blocks;
    std::vector<Block> blocks;
    for (size_t b = 0; b < num_blocks; ++b)
    {
        const size_t lo = b * per;
        if (lo >= n)
            break;
        const size_t hi = std::min(n, lo + per);
        std::vector<UInt64> slice(keys.begin() + lo, keys.begin() + hi);
        blocks.push_back(makeBlock1<UInt64>(slice, 2, b + 1));
    }
    std::atomic<size_t> next{0};
    std::vector<std::thread> threads;
    for (size_t t = 0; t < num_threads; ++t)
        threads.emplace_back([&]
        {
            for (size_t b = next.fetch_add(1); b < blocks.size(); b = next.fetch_add(1))
                store.add(blocks[b]);
        });
    for (auto & th : threads)
        th.join();

    store.finishBuild();
    CoopPool coord;
    LeafArrays leaves;
    coopRun(coord, num_threads, [&] { leaves = store.scatterToLeaves(coord); });

    /// With NT active (x86-64-v3 multitarget) both passes route through the SWWC/NT path under test;
    /// correctness must hold on either path, so this only documents intent (no skip on a v2 build).
    if (RadixShuffle::ntStoresAvailable())
    {
        EXPECT_TRUE(RadixShuffle::shouldUseSwwc(2, static_cast<int>(size_t{1} << cfg.pass_bits[0])));
        EXPECT_TRUE(RadixShuffle::shouldUseSwwc(2, static_cast<int>(size_t{1} << cfg.pass_bits[1])));
    }

    /// At least two build slots contributed, so pass-0 ran with multiple unaligned workers.
    size_t active_workers = 0;
    for (auto c : leaves.worker_block_counts)
        active_workers += (c > 0);
    EXPECT_GE(active_workers, size_t{2}) << "need multiple scatter workers to exercise head-peeling";

    EXPECT_EQ(leaves.num_leaves, cfg.num_leaves);
    verifyConservationAndRefs<UInt64>(store, leaves, {0}, {sizeof(UInt64)});

    /// Path-independent byte accounting across both passes: pass-0 and the (last) refine pass each
    /// scatter key + ref only (the route hash is recomputed, never scattered).
    const UInt64 kw = sizeof(UInt64);
    const UInt64 expected_bytes = 2 * static_cast<UInt64>(n) * (kw + sizeof(RadixShuffle::BuildRef));
    EXPECT_EQ(leaves.bytes_scattered, expected_bytes);
}

/// Unconditional memory-correctness test for the deferred scatter: the output arena holds only
/// ≈ N×(kw+sizeof(BuildRef)) bytes — the scattered key plus an 8 B BuildRef per row, nothing more —
/// and every build row lands exactly once in its correct leaf. Uses a forced multi-pass schedule so the
/// refine path runs and frees each consumed intermediate partition promptly (GrowingArena::freeBlock),
/// which is what keeps the output arena from being inflated by the intermediates. Runs in under 2 s.
TEST(RadixHashBuildScatter, MemoryConsumptionTest)
{
    const size_t n = 10'000'007;
    const size_t num_threads = 4;
    const size_t block_rows = 65536;
    const size_t kw = sizeof(UInt64);

    auto cfg = PartitionConfig::make(static_cast<UInt64>(n), l2_bytes, /*max_partitions_per_pass=*/64);
    ASSERT_GE(cfg.pass_bits.size(), 2u);

    const size_t num_blocks = (n + block_rows - 1) / block_rows;
    const size_t expected_output_bytes = n * (kw + sizeof(RadixShuffle::BuildRef));

    BuildStore store(cfg, {0}, {sizeof(UInt64)}, num_threads);
    std::mt19937_64 rng(0xBEEFDEAD); /// NOLINT(cert-msc32-c,cert-msc51-cpp,bugprone-random-generator-seed)
    std::vector<Block> blks;
    blks.reserve(num_blocks);
    for (size_t b = 0; b < num_blocks; ++b)
    {
        const size_t rows = std::min(block_rows, n - b * block_rows);
        std::vector<UInt64> keys(rows);
        for (size_t i = 0; i < rows; ++i)
            keys[i] = rng();
        blks.push_back(makeBlock1<UInt64>(keys, 0, b + 1));
    }
    std::atomic<size_t> nb{0};
    std::vector<std::thread> threads;
    for (size_t t = 0; t < num_threads; ++t)
        threads.emplace_back([&]
        {
            for (size_t b = nb.fetch_add(1); b < blks.size(); b = nb.fetch_add(1))
                store.add(blks[b]);
        });
    for (auto & th : threads)
        th.join();
    store.finishBuild();

    CoopPool coord;
    LeafArrays leaves;
    coopRun(coord, num_threads, [&] { leaves = store.scatterToLeaves(coord); });

    /// (1) The output arena holds only the scattered output: each non-empty leaf's key + ref sections,
    ///     each roundUpTo64-padded. bytesUsed() therefore lies in
    ///     [N×(kw+sizeof(BuildRef)), N×(kw+sizeof(BuildRef)) + per-leaf alignment slack]. The level0 /
    ///     child intermediates never inflate it — they live in their own arenas and are freed during the
    ///     refine (GrowingArena::freeBlock).
    const size_t alignment_slack = cfg.num_leaves * 128; /// ≤ 2 sections × 64 B padding per leaf
    EXPECT_GE(leaves.arena.bytesUsed(), expected_output_bytes);
    EXPECT_LE(leaves.arena.bytesUsed(), expected_output_bytes + alignment_slack)
        << "scatter output uses more bytes than N×(kw + sizeof(BuildRef)) + per-leaf alignment";

    /// (2) Conservation: every build row exactly once in its correct leaf.
    verifyConservationAndRefs<UInt64>(store, leaves, {0}, {sizeof(UInt64)});
}

/// Always-run end-to-end coverage of the full build path (parallel generate -> work-stolen add ->
/// finishBuild -> scatterToLeaves) for the four mixed-width key configs (K = 1,2,4,8; packed widths
/// 8/16/32/64), at a small row count. Verifies conservation, per-leaf membership and routing identity
/// for every config, so the e2e path is covered independently of the RHJ_P3_BENCH gate. T = 16 for
/// add and scatter. Representative partition config: 2048 leaves, single pass {11}.
TEST(RadixHashBuildScatter, EndToEndBuildSmall)
{
    const size_t n = 300'007;
    const size_t num_threads = 16;
    const size_t block_rows = 2048;

    auto cfg = PartitionConfig::make(static_cast<UInt64>(100'000'000), l2_bytes, 8192); /// 2048 leaves, 1 pass {11}
    ASSERT_EQ(cfg.pass_bits.size(), 1u);

    const std::vector<std::vector<size_t>> configs = {
        {8},
        {4, 12},
        {4, 8, 16, 4},
        {4, 8, 12, 16, 4, 8, 8, 4},
    };

    for (const auto & widths : configs)
    {
        const std::vector<size_t> kpos = keyPositions(widths.size());
        const std::vector<size_t> & kw = widths;
        size_t total_w = 0;
        for (size_t w : widths)
            total_w += w;

        /// (1) Parallel block generation (setup, not timed).
        std::vector<Block> blocks = generateFixedStringBlocksParallel(widths, n, block_rows, num_threads, 0x5A1Eull);

        BuildStore store(cfg, kpos, kw, num_threads);

        /// (2) Work-stolen add() across T threads.
        std::atomic<size_t> next{0};
        std::vector<std::thread> threads;
        threads.reserve(num_threads);
        for (size_t t = 0; t < num_threads; ++t)
            threads.emplace_back([&]
            {
                for (size_t b = next.fetch_add(1); b < blocks.size(); b = next.fetch_add(1))
                    store.add(blocks[b]);
            });
        for (auto & th : threads)
            th.join();

        /// (3) finishBuild().
        store.finishBuild();
        ASSERT_EQ(store.numBlocks(), blocks.size()) << "K=" << widths.size();

        /// (4) scatterToLeaves(coord, T).
        CoopPool coord;
        LeafArrays leaves;
        coopRun(coord, num_threads, [&] { leaves = store.scatterToLeaves(coord); });

        EXPECT_EQ(leaves.num_leaves, cfg.num_leaves);
        EXPECT_EQ(leaves.key_width, total_w) << "K=" << widths.size();

        UInt64 total_rows = 0;
        for (auto r : leaves.leaf_rows)
            total_rows += r;
        ASSERT_EQ(total_rows, UInt64(n)) << "K=" << widths.size();

        /// Conservation / per-leaf membership / routing identity / packed-key pairing.
        verifyConservationAndRefs<UInt64>(store, leaves, kpos, kw);
    }
}

/// Opt-in end-to-end build benchmark (set RHJ_P3_BENCH=1). For each of the four mixed-width key
/// configs (K = 1,2,4,8; packed widths 8/16/32/64) at 100M rows on 16 threads: parallel-generate the
/// blocks (setup, NOT timed), then time work-stolen add(), finishBuild() and scatterToLeaves()
/// separately. Prints one labeled line per config per step (add/finish/scatter/total) with wall time
/// and ns/row (= step_wall_ns / n). Representative partition config: 2048 leaves, single pass {11}.
TEST(RadixHashBuildScatter, EndToEndBuildBench)
{
    if (std::getenv("RHJ_P3_BENCH") == nullptr) /// NOLINT(concurrency-mt-unsafe)
        GTEST_SKIP() << "set RHJ_P3_BENCH=1 to run the end-to-end build benchmark";

    const size_t n = 100'000'000;
    const size_t num_threads = 16;
    const size_t block_rows = 65536;

    auto cfg = PartitionConfig::make(static_cast<UInt64>(n), l2_bytes, 8192); /// 2048 leaves, 1 pass {11}
    ASSERT_EQ(cfg.pass_bits.size(), 1u);

    const std::vector<std::vector<size_t>> configs = {
        {8},
        {4, 12},
        {4, 8, 16, 4},
        {4, 8, 12, 16, 4, 8, 8, 4},
    };

    for (const auto & widths : configs)
    {
        const std::vector<size_t> kpos = keyPositions(widths.size());
        const std::vector<size_t> & kw = widths;

        /// (1) Parallel block generation — setup, NOT timed.
        std::vector<Block> blocks = generateFixedStringBlocksParallel(widths, n, block_rows, num_threads, 0xE2E0000ull);

        BuildStore store(cfg, kpos, kw, num_threads);

        /// (2) Work-stolen add() across T threads — timed.
        std::atomic<size_t> next{0};
        Stopwatch sw_add;
        std::vector<std::thread> threads;
        threads.reserve(num_threads);
        for (size_t t = 0; t < num_threads; ++t)
            threads.emplace_back([&]
            {
                for (size_t b = next.fetch_add(1); b < blocks.size(); b = next.fetch_add(1))
                    store.add(blocks[b]);
            });
        for (auto & th : threads)
            th.join();
        const double add_ns = static_cast<double>(sw_add.elapsedNanoseconds());

        /// (3) finishBuild() — timed.
        Stopwatch sw_finish;
        store.finishBuild();
        const double finish_ns = static_cast<double>(sw_finish.elapsedNanoseconds());

        /// (4) scatterToLeaves(coord, T) — timed. CoopPool construction is setup, outside the timer.
        CoopPool coord;
        Stopwatch sw_scatter;
        LeafArrays leaves;
        coopRun(coord, num_threads, [&] { leaves = store.scatterToLeaves(coord); });
        const double scatter_ns = static_cast<double>(sw_scatter.elapsedNanoseconds());

        const double total_ns = add_ns + finish_ns + scatter_ns;

        auto report = [&](const char * step, double step_ns)
        {
            std::cout << fmt::format(
                "P3 e2e build: K={} key_bytes={} n={} leaves={} passes={} threads={} step={} wall={:.1f}ms ns/row={:.3f}\n",
                widths.size(), store.packedKeyWidth(), n, cfg.num_leaves, cfg.pass_bits.size(), num_threads,
                step, step_ns / 1e6, step_ns / static_cast<double>(n));
        };
        report("add", add_ns);
        report("finish", finish_ns);
        report("scatter", scatter_ns);
        report("total", total_ns);

        EXPECT_EQ(leaves.leaf_rows.size(), cfg.num_leaves);
        EXPECT_EQ(leaves.key_width, store.packedKeyWidth());
    }
}
