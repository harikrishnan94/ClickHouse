#include <gtest/gtest.h>

#include <Columns/ColumnFixedString.h>
#include <Columns/ColumnsNumber.h>
#include <Core/Block.h>
#include <DataTypes/DataTypeFixedString.h>
#include <DataTypes/DataTypesNumber.h>
#include <Interpreters/RadixHashJoin/BuildStore.h>
#include <Interpreters/RadixHashJoin/GrowingArena.h>
#include <Interpreters/RadixHashJoin/LeafHashTable.h>
#include <Interpreters/RadixHashJoin/PartitionConfig.h>
#include <Interpreters/RadixHashJoin/RouteHash.h>

#include <Common/Stopwatch.h>
#include <Common/assert_cast.h>

#include <fmt/format.h>

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iostream>
#include <random>
#include <thread>
#include <unordered_map>
#include <vector>

using namespace DB;
using namespace DB::RadixHash;
using RadixShuffle::BuildRef;

namespace
{

constexpr size_t l2_bytes = 2 * 1024 * 1024;

/// Drive a cooperative build with real T-thread parallelism.
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

Block makeU64Block(const std::vector<UInt64> & keys)
{
    ColumnsWithTypeAndName cols;
    cols.emplace_back(makeColumn<UInt64>(keys), std::make_shared<DataTypeUInt64>(), "k0");
    return Block(std::move(cols));
}

/// Route hashes for a single UInt64 key column: the byte `routeHash` of each 8-byte packed key, the same
/// function the build side routes by, so probe and build agree on the leaf.
std::vector<UInt32> computeHashes(const Block & block, size_t n)
{
    const char * raw = block.getByPosition(0).column->getRawData().data();
    std::vector<UInt32> hash(n);
    for (size_t i = 0; i < n; ++i)
        hash[i] = routeHash(raw + i * sizeof(UInt64), sizeof(UInt64));
    return hash;
}

std::vector<UInt64> randomKeys(size_t n, uint64_t seed)
{
    std::mt19937_64 rng(seed); /// NOLINT(cert-msc32-c,cert-msc51-cpp,bugprone-random-generator-seed)
    std::vector<UInt64> v(n);
    for (size_t i = 0; i < n; ++i)
        v[i] = rng();
    return v;
}

void addBlocksSerial(BuildStore & store, const std::vector<UInt64> & keys, size_t num_blocks)
{
    const size_t n = keys.size();
    const size_t per = (n + num_blocks - 1) / num_blocks;
    for (size_t b = 0; b < num_blocks; ++b)
    {
        const size_t lo = b * per;
        if (lo >= n)
            break;
        const size_t hi = std::min(n, lo + per);
        std::vector<UInt64> slice(keys.begin() + lo, keys.begin() + hi);
        store.add(makeU64Block(slice));
    }
}

/// Read the UInt64 key of build row (block_no, row_no) from the accumulated blocks (row_no 0-based).
UInt64 buildKeyAt(const std::vector<Block> & blocks, BuildRef ref)
{
    const auto & data = assert_cast<const ColumnUInt64 &>(*blocks[ref.block_no].getByPosition(0).column).getData();
    return data[ref.row_no];
}

/// One random FixedString(width) column of `rows` rows.
ColumnFixedString::MutablePtr makeRandomFixedString(size_t width, size_t rows, std::mt19937_64 & rng)
{
    auto col = ColumnFixedString::create(width);
    col->resize(rows);
    auto & chars = col->getChars();
    const size_t total = chars.size();
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

}


/// Gate: insert N rows, look up every key -> all found, refs resolve to a row with the same key.
/// The match succeeding at all proves the bucket is identical on build (insert) and probe (find).
TEST(RadixHashLeafHT, InsertAndFindAll)
{
    const size_t n = 1'000'003;
    const auto keys = randomKeys(n, 0xA11F0);
    auto cfg = PartitionConfig::make(static_cast<UInt64>(100'000'000), l2_bytes, 8192); /// 2048 leaves, single pass
    ASSERT_EQ(cfg.pass_bits.size(), 1u);

    BuildStore store(cfg, {0}, {sizeof(UInt64)}, 4);
    addBlocksSerial(store, keys, 16);
    store.finishBuild();
    CoopPool coord;
    LeafArrays leaves;
    LeafHashTables hts;
    coopRun(coord, 4, [&]
    {
        leaves = store.scatterToLeaves(coord);
        hts = buildLeafHashTables(leaves, store.blockBase(), store.totalRows(), sizeof(UInt64), coord);
    });

    /// Probe with the same keys; every probe row must find at least one match resolving to its key.
    Block probe = makeU64Block(keys);
    const std::vector<UInt32> hashes = computeHashes(probe, n);
    const void * packed = probe.getByPosition(0).column->getRawData().data();

    std::vector<UInt32> left_rows;
    std::vector<BuildRef> refs;
    collectMatches(
        sizeof(UInt64), hts.next_chain != nullptr,
        hts.leaves.data(), cfg.shift, cfg.total_bits, store.blockBase().data(),
        hashes.data(), packed, n, left_rows, refs);

    /// Every probe row must match (100% match, keys present in build).
    std::vector<char> matched(n, 0);
    for (size_t m = 0; m < left_rows.size(); ++m)
    {
        const UInt32 j = left_rows[m];
        EXPECT_EQ(buildKeyAt(store.blocks(), refs[m]), keys[j]) << "matched build row has a different key";
        matched[j] = 1;
    }
    for (size_t j = 0; j < n; ++j)
        EXPECT_EQ(matched[j], 1) << "probe key " << j << " not found";
}

/// Gate: cell conservation + many-to-many JOIN ALL via next_chain. Build with heavy duplicate keys;
/// probing each DISTINCT key once must return every build row exactly once (Σ matches == N), and the
/// per-key match multiset must equal the build multiset for that key.
TEST(RadixHashLeafHT, DuplicateKeysManyToMany)
{
    const size_t n = 500'009;
    /// Few distinct key values -> long chains (many-to-many).
    std::mt19937_64 rng(0xD0D0); /// NOLINT(cert-msc32-c,cert-msc51-cpp,bugprone-random-generator-seed)
    const size_t distinct = 1000;
    std::vector<UInt64> domain(distinct);
    for (auto & d : domain)
        d = rng();
    std::vector<UInt64> keys(n);
    for (size_t i = 0; i < n; ++i)
        keys[i] = domain[rng() % distinct];

    auto cfg = PartitionConfig::make(static_cast<UInt64>(100'000'000), l2_bytes, 8192);
    BuildStore store(cfg, {0}, {sizeof(UInt64)}, 4);
    addBlocksSerial(store, keys, 23);
    store.finishBuild();
    CoopPool coord;
    LeafArrays leaves;
    LeafHashTables hts;
    coopRun(coord, 4, [&]
    {
        leaves = store.scatterToLeaves(coord);
        hts = buildLeafHashTables(leaves, store.blockBase(), store.totalRows(), sizeof(UInt64), coord);
    });

    /// Reference: count of build rows per distinct key value.
    std::unordered_map<UInt64, size_t> ref_count;
    for (auto k : keys)
        ++ref_count[k];

    /// Probe with each distinct key once.
    std::vector<UInt64> distinct_keys;
    distinct_keys.reserve(ref_count.size());
    for (const auto & [k, _] : ref_count)
        distinct_keys.push_back(k);
    const size_t dn = distinct_keys.size();

    Block probe = makeU64Block(distinct_keys);
    const std::vector<UInt32> hashes = computeHashes(probe, dn);
    const void * packed = probe.getByPosition(0).column->getRawData().data();

    std::vector<UInt32> left_rows;
    std::vector<BuildRef> refs;
    collectMatches(
        sizeof(UInt64), hts.next_chain != nullptr,
        hts.leaves.data(), cfg.shift, cfg.total_bits, store.blockBase().data(),
        hashes.data(), packed, dn, left_rows, refs);

    /// Each build row matched exactly once (cell conservation through the chains).
    std::vector<std::vector<char>> seen(store.numBlocks());
    for (size_t b = 0; b < store.numBlocks(); ++b)
        seen[b].assign(store.blocks()[b].rows(), 0);

    std::unordered_map<UInt64, size_t> got_count;
    for (size_t m = 0; m < left_rows.size(); ++m)
    {
        const UInt64 probe_key = distinct_keys[left_rows[m]];
        const BuildRef ref = refs[m];
        EXPECT_EQ(buildKeyAt(store.blocks(), ref), probe_key) << "chain returned a row with a mismatched key";
        EXPECT_EQ(seen[ref.block_no][ref.row_no], 0) << "build row returned more than once";
        seen[ref.block_no][ref.row_no] = 1;
        ++got_count[probe_key];
    }

    EXPECT_EQ(left_rows.size(), n) << "every build row must be returned exactly once across the chains";
    for (const auto & [k, c] : ref_count)
        EXPECT_EQ(got_count[k], c) << "many-to-many fan-out count mismatch for a key";

    for (size_t b = 0; b < store.numBlocks(); ++b)
        for (size_t r = 0; r < seen[b].size(); ++r)
            EXPECT_EQ(seen[b][r], 1) << "build row (" << b << "," << r << ") never returned";
}

/// Gate: an explicitly zeroed cell array is a fully-initialised empty table — every cell reads as the
/// empty sentinel (row_no == 0) and leafFind returns a miss. The jemalloc-backed GrowingArena does NOT
/// zero its memory; `buildLeafHashTables` memsets each leaf's cells before filling, which this test
/// reproduces (zeroed cells are the precondition `leafInsert`/`leafFind` assume).
TEST(RadixHashLeafHT, ZeroedCellsEmptySentinel)
{
    GrowingArena arena;
    constexpr size_t key_width = 8;
    const UInt64 num_buckets = 4096;
    LeafHT ht;
    ht.num_buckets = num_buckets;
    ht.next_chain = nullptr;
    const size_t cell_bytes = num_buckets * leafCellBytes(key_width);
    ht.cells = static_cast<char *>(arena.alloc(cell_bytes, RadixShuffle::LINE_BYTES));
    std::memset(ht.cells, 0xFF, cell_bytes); /// jemalloc arena is not initialised — set the 0xFF empty sentinel like the build path does

    /// Every cell must now be the empty sentinel.
    for (UInt64 b = 0; b < num_buckets; ++b)
    {
        const auto * ref = reinterpret_cast<const BuildRef *>(ht.cells + b * leafCellBytes(key_width));
        ASSERT_EQ(ref->row_no, RadixShuffle::INVALID_ROW) << "freshly carved cell " << b << " is not the empty sentinel";
    }

    /// Any lookup misses cleanly on the empty table.
    for (UInt64 v = 0; v < 1000; ++v)
    {
        const UInt32 h = static_cast<UInt32>(v * 2654435761u);
        const BuildRef r = leafFind<key_width>(ht, h, &v);
        EXPECT_EQ(r.row_no, RadixShuffle::INVALID_ROW) << "found a match in an empty table";
    }
}

/// Gate: exact-value key compare. Two distinct keys that collide into the SAME bucket must remain
/// distinguishable — find must return only the chain for the queried key (linear probing + byte
/// compare), never the colliding neighbour.
TEST(RadixHashLeafHT, ExactKeyCompareOnCollision)
{
    GrowingArena arena;
    constexpr size_t key_width = 8;
    const UInt64 num_buckets = 1024;

    LeafHT ht;
    ht.num_buckets = num_buckets;
    const size_t cell_bytes = num_buckets * leafCellBytes(key_width);
    ht.cells = static_cast<char *>(arena.alloc(cell_bytes, RadixShuffle::LINE_BYTES));
    std::memset(ht.cells, 0xFF, cell_bytes); /// jemalloc arena is not initialised — set the empty sentinel
    /// next_chain: 2 build rows (the two keys); set the 0xFF tails (jemalloc memory is not initialised).
    ht.next_chain = arena.allocArray<BuildRef>(2);
    std::memset(ht.next_chain, 0xFF, 2 * sizeof(BuildRef));
    /// block_base for a single block of 2 rows.
    std::vector<UInt64> block_base{0, 2};

    /// Find two distinct 64-bit values that map to the same bucket under leafBucket.
    UInt64 key_a = 0;
    UInt64 key_b = 0;
    bool found_pair = false;
    std::mt19937_64 rng(0xC0111DE); /// NOLINT(cert-msc32-c,cert-msc51-cpp,bugprone-random-generator-seed)
    std::unordered_map<UInt64, UInt64> by_bucket; /// bucket -> a value mapping there
    for (int tries = 0; tries < 200000 && !found_pair; ++tries)
    {
        const UInt64 v = rng();
        const UInt32 h = static_cast<UInt32>(v * 1099511628211ull); /// arbitrary value->hash for the test
        const UInt64 bkt = leafBucket(h, num_buckets);
        auto [it, inserted] = by_bucket.emplace(bkt, v);
        if (!inserted && it->second != v)
        {
            key_a = it->second;
            key_b = v;
            found_pair = true;
        }
    }
    ASSERT_TRUE(found_pair) << "could not synthesise a bucket collision";

    auto hash_of = [](UInt64 v) { return static_cast<UInt32>(v * 1099511628211ull); };

    /// Insert both keys (block 0, 0-based rows 0 and 1). Both are distinct (no duplicate), so
    /// leafInsert returns the INVALID_ROW sentinel (no chain needed).
    leafInsert<key_width>(ht, hash_of(key_a), &key_a, BuildRef{0, 0});
    leafInsert<key_width>(ht, hash_of(key_b), &key_b, BuildRef{0, 1});
    ASSERT_EQ(leafBucket(hash_of(key_a), num_buckets), leafBucket(hash_of(key_b), num_buckets)) << "test needs a real collision";

    const BuildRef ra = leafFind<key_width>(ht, hash_of(key_a), &key_a);
    const BuildRef rb = leafFind<key_width>(ht, hash_of(key_b), &key_b);
    ASSERT_NE(ra.row_no, RadixShuffle::INVALID_ROW);
    ASSERT_NE(rb.row_no, RadixShuffle::INVALID_ROW);
    EXPECT_EQ(ra.row_no, 0u) << "key_a resolved to the wrong (colliding) row";
    EXPECT_EQ(rb.row_no, 1u) << "key_b resolved to the wrong (colliding) row";
    /// Chains are length 1 (distinct keys): each tail is the INVALID_ROW sentinel.
    EXPECT_EQ(ht.next_chain[0].row_no, RadixShuffle::INVALID_ROW);
    EXPECT_EQ(ht.next_chain[1].row_no, RadixShuffle::INVALID_ROW);
}

/// Gate: all templatized key_width paths (multiples of 4 in [4, 64]). For each width, build a single
/// FixedString(width) key column, build the leaf HTs, probe with the same keys -> all found, exact key.
TEST(RadixHashLeafHT, AllKeyWidthPaths)
{
    const size_t n = 60'013;
    for (size_t width = 4; width <= 64; width += 4)
    {
        std::mt19937_64 rng(0x5150 + width); /// NOLINT(cert-msc32-c,cert-msc51-cpp,bugprone-random-generator-seed)

        /// Build blocks of FixedString(width) keys.
        const size_t num_blocks = 8;
        const size_t per = (n + num_blocks - 1) / num_blocks;
        auto cfg = PartitionConfig::make(static_cast<UInt64>(5'000'000), l2_bytes, 8192);
        BuildStore store(cfg, {0}, {width}, 2);

        std::vector<std::vector<UInt8>> all; /// keep raw bytes for the probe block
        all.reserve(num_blocks);
        for (size_t b = 0; b < num_blocks; ++b)
        {
            const size_t lo = b * per;
            if (lo >= n)
                break;
            const size_t rows = std::min(per, n - lo);
            auto col = makeRandomFixedString(width, rows, rng);
            const auto & chars = col->getChars();
            all.emplace_back(chars.begin(), chars.end());
            ColumnsWithTypeAndName cols;
            cols.emplace_back(std::move(col), std::make_shared<DataTypeFixedString>(width), "k0");
            store.add(Block(std::move(cols)));
        }

        store.finishBuild();
        CoopPool coord;
        LeafArrays leaves;
        LeafHashTables hts;
        coopRun(coord, 2, [&]
        {
            leaves = store.scatterToLeaves(coord);
            hts = buildLeafHashTables(leaves, store.blockBase(), store.totalRows(), width, coord);
        });

        /// Probe with every build key (reassembled in block order from `all`).
        auto probe_col = ColumnFixedString::create(width);
        for (const auto & bytes : all)
            probe_col->getChars().insert(bytes.begin(), bytes.end());
        const size_t total = probe_col->size();
        ASSERT_EQ(total, store.totalRows());

        const void * packed = probe_col->getRawData().data();
        std::vector<UInt32> hashes(total);
        for (size_t i = 0; i < total; ++i)
            hashes[i] = routeHash(static_cast<const char *>(packed) + i * width, width);

        std::vector<UInt32> left_rows;
        std::vector<BuildRef> refs;
        collectMatches(
            width, hts.next_chain != nullptr,
            hts.leaves.data(), cfg.shift, cfg.total_bits, store.blockBase().data(),
            hashes.data(), packed, total, left_rows, refs);

        std::vector<char> matched(total, 0);
        for (size_t m = 0; m < left_rows.size(); ++m)
        {
            const UInt32 j = left_rows[m];
            const char * pk = static_cast<const char *>(packed) + static_cast<size_t>(j) * width;
            /// Verify the matched build row's key bytes equal the probe key bytes.
            const char * build_key = static_cast<const char *>(
                store.blocks()[refs[m].block_no].getByPosition(0).column->getRawData().data())
                + static_cast<size_t>(refs[m].row_no) * width;
            EXPECT_EQ(0, std::memcmp(pk, build_key, width)) << "width=" << width << " key compare mismatch";
            matched[j] = 1;
        }
        for (size_t j = 0; j < total; ++j)
            ASSERT_EQ(matched[j], 1) << "width=" << width << " probe row " << j << " not found";
    }
}

/// Gate: the FULL build-then-probe round-trip through a forced MULTI-PASS scatter. Every functional gate
/// above (`InsertAndFindAll`, `DuplicateKeysManyToMany`, `AllKeyWidthPaths`) uses
/// `max_partitions_per_pass=8192` -> a single pass, so the multi-pass refine cascade
/// (`BuildStore::scatterMultiPass`/`refineDepthFirst`, which recomputes `routeHash` from the scattered
/// packed key at every pass) was never exercised on the leaf-HT build + probe path. Force {6,5} (two
/// passes) with a small per-pass cap, build every key, then probe every key -> all found, exact key.
TEST(RadixHashLeafHT, MultiPassInsertAndFindAll)
{
    const size_t n = 1'000'003;
    const auto keys = randomKeys(n, 0xA11F0);
    auto cfg = PartitionConfig::make(static_cast<UInt64>(100'000'000), l2_bytes, /*max_partitions_per_pass=*/64); /// 2048 leaves -> {6,5}
    ASSERT_EQ(cfg.pass_bits.size(), 2u) << "config must force a multi-pass scatter or this test is vacuous";
    ASSERT_EQ(cfg.pass_bits[0], 6u);
    ASSERT_EQ(cfg.pass_bits[1], 5u);

    BuildStore store(cfg, {0}, {sizeof(UInt64)}, 4);
    addBlocksSerial(store, keys, 16);
    store.finishBuild();
    CoopPool coord;
    LeafArrays leaves;
    LeafHashTables hts;
    coopRun(coord, 4, [&]
    {
        leaves = store.scatterToLeaves(coord);
        hts = buildLeafHashTables(leaves, store.blockBase(), store.totalRows(), sizeof(UInt64), coord);
    });

    /// Probe with the same keys; every probe row must find at least one match resolving to its key.
    Block probe = makeU64Block(keys);
    const std::vector<UInt32> hashes = computeHashes(probe, n);
    const void * packed = probe.getByPosition(0).column->getRawData().data();

    std::vector<UInt32> left_rows;
    std::vector<BuildRef> refs;
    collectMatches(
        sizeof(UInt64), hts.next_chain != nullptr,
        hts.leaves.data(), cfg.shift, cfg.total_bits, store.blockBase().data(),
        hashes.data(), packed, n, left_rows, refs);

    std::vector<char> matched(n, 0);
    for (size_t m = 0; m < left_rows.size(); ++m)
    {
        const UInt32 j = left_rows[m];
        EXPECT_EQ(buildKeyAt(store.blocks(), refs[m]), keys[j]) << "matched build row has a different key";
        matched[j] = 1;
    }
    for (size_t j = 0; j < n; ++j)
        EXPECT_EQ(matched[j], 1) << "probe key " << j << " not found";
}

/// Gate: many-to-many JOIN ALL via `next_chain` through a forced MULTI-PASS scatter. Heavy duplicate keys
/// are routed through pass-0 partitions and refined in pass-1; probing each DISTINCT key once must still
/// return every build row exactly once (cell conservation through the chains AND the refine cascade), and
/// the per-key fan-out must equal the build multiset for that key.
TEST(RadixHashLeafHT, MultiPassDuplicateKeysManyToMany)
{
    const size_t n = 500'009;
    /// Few distinct key values -> long chains (many-to-many).
    std::mt19937_64 rng(0xD0D0); /// NOLINT(cert-msc32-c,cert-msc51-cpp,bugprone-random-generator-seed)
    const size_t distinct = 1000;
    std::vector<UInt64> domain(distinct);
    for (auto & d : domain)
        d = rng();
    std::vector<UInt64> keys(n);
    for (size_t i = 0; i < n; ++i)
        keys[i] = domain[rng() % distinct];

    auto cfg = PartitionConfig::make(static_cast<UInt64>(100'000'000), l2_bytes, /*max_partitions_per_pass=*/64);
    ASSERT_EQ(cfg.pass_bits.size(), 2u) << "config must force a multi-pass scatter or this test is vacuous";
    BuildStore store(cfg, {0}, {sizeof(UInt64)}, 4);
    addBlocksSerial(store, keys, 23);
    store.finishBuild();
    CoopPool coord;
    LeafArrays leaves;
    LeafHashTables hts;
    coopRun(coord, 4, [&]
    {
        leaves = store.scatterToLeaves(coord);
        hts = buildLeafHashTables(leaves, store.blockBase(), store.totalRows(), sizeof(UInt64), coord);
    });

    /// Reference: count of build rows per distinct key value.
    std::unordered_map<UInt64, size_t> ref_count;
    for (auto k : keys)
        ++ref_count[k];

    /// Probe with each distinct key once.
    std::vector<UInt64> distinct_keys;
    distinct_keys.reserve(ref_count.size());
    for (const auto & [k, _] : ref_count)
        distinct_keys.push_back(k);
    const size_t dn = distinct_keys.size();

    Block probe = makeU64Block(distinct_keys);
    const std::vector<UInt32> hashes = computeHashes(probe, dn);
    const void * packed = probe.getByPosition(0).column->getRawData().data();

    std::vector<UInt32> left_rows;
    std::vector<BuildRef> refs;
    collectMatches(
        sizeof(UInt64), hts.next_chain != nullptr,
        hts.leaves.data(), cfg.shift, cfg.total_bits, store.blockBase().data(),
        hashes.data(), packed, dn, left_rows, refs);

    /// Each build row matched exactly once (cell conservation through the chains).
    std::vector<std::vector<char>> seen(store.numBlocks());
    for (size_t b = 0; b < store.numBlocks(); ++b)
        seen[b].assign(store.blocks()[b].rows(), 0);

    std::unordered_map<UInt64, size_t> got_count;
    for (size_t m = 0; m < left_rows.size(); ++m)
    {
        const UInt64 probe_key = distinct_keys[left_rows[m]];
        const BuildRef ref = refs[m];
        EXPECT_EQ(buildKeyAt(store.blocks(), ref), probe_key) << "chain returned a row with a mismatched key";
        EXPECT_EQ(seen[ref.block_no][ref.row_no], 0) << "build row returned more than once";
        seen[ref.block_no][ref.row_no] = 1;
        ++got_count[probe_key];
    }

    EXPECT_EQ(left_rows.size(), n) << "every build row must be returned exactly once across the chains";
    for (const auto & [k, c] : ref_count)
        EXPECT_EQ(got_count[k], c) << "many-to-many fan-out count mismatch for a key";

    for (size_t b = 0; b < store.numBlocks(); ++b)
        for (size_t r = 0; r < seen[b].size(); ++r)
            EXPECT_EQ(seen[b][r], 1) << "build row (" << b << "," << r << ") never returned";
}

/// Opt-in cell-conservation at 100M rows (set RHJ_P4_BENCH=1). Builds the full leaf-HT set at scale and
/// asserts Σ over leaves of inserted distinct-key cells + chain length == N (every build row reachable).
TEST(RadixHashLeafHT, CellConservation100M)
{
    if (std::getenv("RHJ_P4_BENCH") == nullptr) /// NOLINT(concurrency-mt-unsafe)
        GTEST_SKIP() << "set RHJ_P4_BENCH=1 to run the 100M-scale cell-conservation test";

    const size_t n = 100'000'000;
    const size_t num_threads = 16;
    const size_t block_rows = 65536;

    auto cfg = PartitionConfig::make(static_cast<UInt64>(n), l2_bytes, 8192);
    BuildStore store(cfg, {0}, {sizeof(UInt64)}, num_threads);

    std::mt19937_64 rng(0xCE11); /// NOLINT(cert-msc32-c,cert-msc51-cpp,bugprone-random-generator-seed)
    const size_t num_blocks = (n + block_rows - 1) / block_rows;
    std::vector<Block> blocks;
    blocks.reserve(num_blocks);
    for (size_t b = 0; b < num_blocks; ++b)
    {
        const size_t rows = std::min(block_rows, n - b * block_rows);
        std::vector<UInt64> keys(rows);
        for (size_t i = 0; i < rows; ++i)
            keys[i] = rng();
        blocks.push_back(makeU64Block(keys));
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
    LeafHashTables hts;
    Stopwatch sw;
    coopRun(coord, num_threads, [&]
    {
        leaves = store.scatterToLeaves(coord);
        hts = buildLeafHashTables(leaves, store.blockBase(), store.totalRows(), sizeof(UInt64), coord);
    });
    const double ht_ms = static_cast<double>(sw.elapsedNanoseconds()) / 1e6;

    /// Conservation: walk every occupied cell's chain across all leaves; total visited == N.
    /// next_chain may be nullptr for all-unique builds (hts.next_chain == nullptr ↔ all-unique).
    UInt64 visited = 0;
    for (const LeafHT & ht : hts.leaves)
    {
        for (UInt64 b = 0; b < ht.num_buckets; ++b)
        {
            BuildRef cur = *reinterpret_cast<const BuildRef *>(ht.cells + b * leafCellBytes(sizeof(UInt64)));
            while (cur.row_no != RadixShuffle::INVALID_ROW)
            {
                ++visited;
                if (hts.next_chain == nullptr)
                    break; /// all-unique: every occupied cell is a length-1 chain, no next_chain
                cur = ht.next_chain[leafFlat(cur, store.blockBase().data())];
            }
        }
    }
    std::cout << fmt::format(
        "P4 cell conservation: n={} leaves={} ht_build={:.1f}ms visited={}\n", n, cfg.num_leaves, ht_ms, visited);
    EXPECT_EQ(visited, UInt64(n)) << "every build row must be reachable exactly once via the chains";
}

/// Open question Q1 (set RHJ_P4_BENCH=1): leaf-HT build time with THP vs without, at 100M rows. The
/// random inserts are fault/TLB-bound, so the THP arena is expected to help materially.
TEST(RadixHashLeafHT, ThpVsNonThpBuildTime)
{
    if (std::getenv("RHJ_P4_BENCH") == nullptr) /// NOLINT(concurrency-mt-unsafe)
        GTEST_SKIP() << "set RHJ_P4_BENCH=1 to run the THP vs non-THP leaf-HT build benchmark";

    const size_t n = 100'000'000;
    const size_t num_threads = 16;
    const size_t block_rows = 65536;

    auto cfg = PartitionConfig::make(static_cast<UInt64>(n), l2_bytes, 8192);
    BuildStore store(cfg, {0}, {sizeof(UInt64)}, num_threads);

    std::mt19937_64 rng(0x7777); /// NOLINT(cert-msc32-c,cert-msc51-cpp,bugprone-random-generator-seed)
    const size_t num_blocks = (n + block_rows - 1) / block_rows;
    std::vector<Block> blocks;
    blocks.reserve(num_blocks);
    for (size_t b = 0; b < num_blocks; ++b)
    {
        const size_t rows = std::min(block_rows, n - b * block_rows);
        std::vector<UInt64> keys(rows);
        for (size_t i = 0; i < rows; ++i)
            keys[i] = rng();
        blocks.push_back(makeU64Block(keys));
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

    {
        /// A fresh scatter per run so each build faults its own leaf arrays + HT cells.
        CoopPool coord;
        LeafArrays la;
        LeafHashTables hts;
        Stopwatch sw;
        coopRun(coord, num_threads, [&]
        {
            la = store.scatterToLeaves(coord);
            hts = buildLeafHashTables(la, store.blockBase(), store.totalRows(), sizeof(UInt64), coord);
        });
        const double ms = static_cast<double>(sw.elapsedNanoseconds()) / 1e6;
        std::cout << fmt::format(
            "P4 leaf-HT build jemalloc-parallel: n={} leaves={} wall={:.1f}ms ns/row={:.3f}\n",
            n, cfg.num_leaves, ms, ms * 1e6 / static_cast<double>(n));
        EXPECT_EQ(hts.leaves.size(), cfg.num_leaves);
    }
}
