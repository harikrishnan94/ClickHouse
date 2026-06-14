#include <Interpreters/RadixHashJoin/Arena.h>
#include <Interpreters/RadixHashJoin/BuildSide.h>
#include <Interpreters/RadixHashJoin/KeyRefScatter.h>
#include <Interpreters/RadixHashJoin/LeafTable.h>
#include <Interpreters/RadixHashJoin/PackedKeyHash.h>
#include <Interpreters/RadixHashJoin/ParallelFor.h>
#include <Interpreters/RadixHashJoin/PartitionPlan.h>

#include <Columns/ColumnsNumber.h>
#include <Core/Block.h>
#include <DataTypes/DataTypesNumber.h>
#include <Common/typeid_cast.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <exception>
#include <map>
#include <mutex>
#include <random>
#include <thread>
#include <vector>

using namespace DB;
using namespace DB::RadixJoin;

namespace
{

Block makeU64Block(const std::vector<UInt64> & values, const std::string & name)
{
    auto col = ColumnUInt64::create();
    col->getData().assign(values.begin(), values.end());
    Block block;
    block.insert(ColumnWithTypeAndName(std::move(col), std::make_shared<DataTypeUInt64>(), name));
    return block;
}

/// A `ParallelFor` for the tests: sequential when `num_workers <= 1`, otherwise `num_workers`
/// std::threads doing dynamic work-stealing on an atomic cursor, each carrying a fixed dense worker id.
/// Mirrors the production pool-backed runner's contract (dense worker ids, dynamic balancing, exception
/// propagation), so it exercises the same `BuildSide`/`buildLeafTables` code paths.
ParallelFor makeParallelFor(size_t num_workers)
{
    return [num_workers](size_t total, const UnitFn & fn)
    {
        if (total == 0)
            return;
        if (num_workers <= 1)
        {
            for (size_t unit = 0; unit < total; ++unit)
                fn(unit, 0);
            return;
        }

        const size_t workers = std::min(num_workers, total);
        std::atomic<size_t> next{0};
        std::mutex exc_mutex;
        std::exception_ptr first_exc;
        std::vector<std::thread> ts;
        ts.reserve(workers);
        for (size_t w = 0; w < workers; ++w)
            ts.emplace_back([&, w]
            {
                while (true)
                {
                    const size_t unit = next.fetch_add(1);
                    if (unit >= total)
                        break;
                    try
                    {
                        fn(unit, w);
                    }
                    catch (...)
                    {
                        std::lock_guard lock(exc_mutex);
                        if (!first_exc)
                            first_exc = std::current_exception();
                        next.store(total);
                        break;
                    }
                }
            });
        for (auto & t : ts)
            t.join();
        if (first_exc)
            std::rethrow_exception(first_exc);
    };
}

/// Build the leaf tables from a single-column UInt64 build side, probe every probe key, and assert the
/// (probe_row -> matched build flat-rows) result equals the brute-force reference.
void checkBuildAndProbe(
    const std::vector<UInt64> & build_keys,
    const std::vector<UInt64> & probe_keys,
    PartitionPlan plan,
    size_t build_threads,
    size_t post_build_threads)
{
    constexpr size_t key_width = 8;

    BuildSide build_side(plan, {0}, {key_width}, std::max<size_t>(build_threads, 1));

    /// Feed the build keys as several blocks (exercise multi-block accumulation), from `build_threads`
    /// concurrent adders if requested.
    const size_t block_rows = 257; /// deliberately not a power of two
    std::vector<Block> build_blocks;
    for (size_t begin = 0; begin < build_keys.size(); begin += block_rows)
    {
        const size_t end = std::min(begin + block_rows, build_keys.size());
        build_blocks.push_back(makeU64Block({build_keys.begin() + begin, build_keys.begin() + end}, "k0"));
    }
    if (build_threads <= 1)
    {
        for (const auto & b : build_blocks)
            build_side.add(b, 0);
    }
    else
    {
        std::atomic<size_t> next{0};
        std::vector<std::thread> ts;
        for (size_t t = 0; t < build_threads; ++t)
            ts.emplace_back([&, lane = t]
            {
                for (size_t i = next.fetch_add(1); i < build_blocks.size(); i = next.fetch_add(1))
                    build_side.add(build_blocks[i], lane);
            });
        for (auto & t : ts)
            t.join();
    }

    build_side.finishBuild();
    ASSERT_EQ(build_side.totalRows(), build_keys.size());

    const ParallelFor parallel_for = makeParallelFor(post_build_threads);
    LeafArrays leaves = build_side.scatterToLeaves(parallel_for);
    /// No-churn: one output allocation per non-empty leaf, never per (block x leaf).
    UInt64 non_empty = 0;
    for (UInt64 r : leaves.leaf_rows)
        non_empty += (r != 0);
    EXPECT_EQ(leaves.alloc_count, non_empty);

    LeafTables tables = buildLeafTables(leaves, build_side.totalRows(), key_width, post_build_threads, parallel_for);

    /// The build key stored at a matched BuildRef, read back from the (possibly reordered, under
    /// parallel add) accumulated blocks. The flat-row order need not match the original vector — only
    /// the matched KEY VALUES and per-probe-row match COUNTS are part of the join contract.
    const auto & blocks = build_side.blocks();
    auto build_key_at = [&](BuildRef ref) -> UInt64
    {
        const auto & col = typeid_cast<const ColumnUInt64 &>(*blocks[ref.blockNo()].getByPosition(0).column);
        return col.getData()[ref.rowNo()];
    };

    /// Expected per-key build-row count (every build row with that key value).
    std::map<UInt64, size_t> expected_count;
    for (UInt64 k : build_keys)
        ++expected_count[k];

    std::vector<UInt32> out_rows;
    std::vector<BuildRef> out_refs;
    collectMatches(
        key_width, tables.leaves.data(), plan.leaf_shift, plan.total_bits,
        probe_keys.data(), probe_keys.size(), tables.max_bucket_bits <= 31, out_rows, out_refs);

    /// Every emitted match must resolve to a build row whose key equals the probe key, and the number
    /// of matches per probe row must equal the count of build rows with that key.
    std::map<UInt32, size_t> got_count;
    for (size_t m = 0; m < out_rows.size(); ++m)
    {
        ASSERT_EQ(build_key_at(out_refs[m]), probe_keys[out_rows[m]]) << "match " << m;
        ++got_count[out_rows[m]];
    }
    for (size_t i = 0; i < probe_keys.size(); ++i)
    {
        auto it = expected_count.find(probe_keys[i]);
        const size_t expected = it == expected_count.end() ? 0 : it->second;
        const size_t actual = got_count.contains(static_cast<UInt32>(i)) ? got_count[static_cast<UInt32>(i)] : 0;
        ASSERT_EQ(actual, expected) << "probe row " << i << " key " << probe_keys[i];
    }
}

}

TEST(RadixHashJoin, PackedKeyHashDeterministicAndIndependentHalves)
{
    /// Same bytes -> same hash; both 32-bit halves vary across keys (a single hash drives leaf + bucket).
    std::map<UInt32, int> high_counts;
    std::map<UInt32, int> low_counts;
    for (UInt64 v = 0; v < 100000; ++v)
    {
        const UInt64 h1 = hashPackedKey<8>(&v);
        const UInt64 h2 = hashPackedKey(&v, 8);
        ASSERT_EQ(h1, h2);
        ++high_counts[routeBits(h1)];
        ++low_counts[bucketBits(h1)];
    }
    /// Sequential integers must not collapse the high or low words into a few buckets.
    EXPECT_GT(high_counts.size(), 99000u);
    EXPECT_GT(low_counts.size(), 99000u);
}

TEST(RadixHashJoin, PartitionPlanSizingAndPasses)
{
    /// Unknown estimate -> default leaves.
    auto p0 = PartitionPlan::choose(std::nullopt, 2 << 20, 8192);
    EXPECT_EQ(p0.num_leaves, PartitionPlan::DEFAULT_LEAVES);

    /// 100M rows, 2 MiB L2 -> 2048 leaves, single pass at the default 8192 cap.
    auto p1 = PartitionPlan::choose(100'000'000, 2u << 20, 8192);
    EXPECT_EQ(p1.num_leaves, 2048u);
    EXPECT_EQ(p1.total_bits, 11u);
    EXPECT_EQ(p1.pass_bits.size(), 1u);

    /// A small per-pass cap forces multiple, evenly-spread passes.
    auto p2 = PartitionPlan::choose(100'000'000, 2u << 20, 4); /// bits_per_pass = 2 -> ceil(11/2)=6 passes
    UInt32 sum = 0;
    for (UInt32 b : p2.pass_bits)
        sum += b;
    EXPECT_EQ(sum, p2.total_bits);
    EXPECT_LE(*std::max_element(p2.pass_bits.begin(), p2.pass_bits.end())
                  - *std::min_element(p2.pass_bits.begin(), p2.pass_bits.end()),
              1u);
}

TEST(RadixHashJoin, ScatterColumnRoundTripDirect)
{
    /// A direct (small fanout) scatter must place every element into the partition its route selects,
    /// in arrival order, and the per-partition counts must match the histogram.
    constexpr size_t partitions = 16;
    constexpr UInt32 shift = 0;
    constexpr UInt32 mask = partitions - 1;
    const size_t n = 5000;

    std::vector<UInt32> route(n);
    std::vector<UInt64> src(n);
    std::vector<size_t> counts(partitions, 0);
    std::mt19937 rng(123); // NOLINT(bugprone-random-generator-seed, cert-msc32-c, cert-msc51-cpp)
    for (size_t i = 0; i < n; ++i)
    {
        route[i] = static_cast<UInt32>(rng());
        src[i] = i;
        ++counts[route[i] & mask];
    }

    RadixJoin::Arena arena;
    std::vector<UInt64 *> bases(partitions);
    std::vector<void *> cursors(partitions);
    for (size_t p = 0; p < partitions; ++p)
    {
        bases[p] = arena.allocateArray<UInt64>(std::max<size_t>(counts[p], 1));
        cursors[p] = bases[p];
    }
    appendColumnDirect(route.data(), shift, mask, n, src.data(), sizeof(UInt64), cursors.data());

    /// Reconstruct: each partition holds exactly its routed source values in arrival order.
    std::vector<std::vector<UInt64>> expected(partitions);
    for (size_t i = 0; i < n; ++i)
        expected[route[i] & mask].push_back(src[i]);
    for (size_t p = 0; p < partitions; ++p)
    {
        const size_t written = static_cast<UInt64 *>(cursors[p]) - bases[p];
        ASSERT_EQ(written, counts[p]);
        std::vector<UInt64> got(bases[p], bases[p] + written);
        ASSERT_EQ(got, expected[p]);
    }
}

TEST(RadixHashJoin, BuildProbeUniqueKeys)
{
    std::vector<UInt64> build_keys;
    std::vector<UInt64> probe_keys;
    for (UInt64 i = 0; i < 5000; ++i)
        build_keys.push_back(i * 2654435761ULL);
    for (UInt64 i = 0; i < 4000; ++i)
        probe_keys.push_back((i + 1000) * 2654435761ULL); /// 3000 hits, 1000 misses
    checkBuildAndProbe(build_keys, probe_keys, PartitionPlan::choose(5000, 2u << 20, 8192), 1, 1);
}

TEST(RadixHashJoin, BuildProbeManyToManyParallel)
{
    /// Heavy duplicates on both sides -> exercises the chain and the singleton fast path.
    std::vector<UInt64> build_keys;
    std::vector<UInt64> probe_keys;
    std::mt19937 rng(7); // NOLINT(bugprone-random-generator-seed, cert-msc32-c, cert-msc51-cpp)
    for (size_t i = 0; i < 20000; ++i)
        build_keys.push_back(rng() % 500); /// ~40 build rows per key
    for (size_t i = 0; i < 8000; ++i)
        probe_keys.push_back(rng() % 700); /// some keys never in build
    checkBuildAndProbe(build_keys, probe_keys, PartitionPlan::choose(20000, 2u << 20, 8192), 4, 4);
}

TEST(RadixHashJoin, BuildProbeHeavyDuplicatesFewKeys)
{
    /// Very few distinct keys with hundreds of rows each: all rows of a key share one hash, so they land in
    /// ONE leaf homing to the SAME bucket, and up to `ring_size` of these same-key rows are in flight in the
    /// AMAC build ring simultaneously. This is the maximal stress for the ring's fused read->act duplicate
    /// coalescing — a stale read (reading a batch of cells before acting) would either drop rows or split a
    /// key across two cells. The per-probe-row match counts must still equal the brute-force counts.
    std::vector<UInt64> build_keys;
    std::vector<UInt64> probe_keys;
    for (UInt64 k = 0; k < 8; ++k)
        for (size_t r = 0; r < 600; ++r)
            build_keys.push_back(k * 2654435761ULL); /// 8 keys x 600 rows, each key in a single leaf/bucket
    for (UInt64 k = 0; k < 12; ++k)
        probe_keys.push_back(k * 2654435761ULL); /// keys 0..7 hit (600 each), 8..11 miss
    /// Single-threaded build + post-build keeps every same-key row of a leaf in the one ring, in flight.
    checkBuildAndProbe(build_keys, probe_keys, PartitionPlan::choose(4800, 2u << 20, 8192), 1, 1);
}

TEST(RadixHashJoin, BuildProbeForcedMultiPass)
{
    /// Force several scatter passes with a tiny per-pass cap; results must be identical to single-pass.
    std::vector<UInt64> build_keys;
    std::vector<UInt64> probe_keys;
    std::mt19937 rng(99); // NOLINT(bugprone-random-generator-seed, cert-msc32-c, cert-msc51-cpp)
    for (size_t i = 0; i < 30000; ++i)
        build_keys.push_back(rng());
    for (size_t i = 0; i < 30000; ++i)
        probe_keys.push_back(i % 2 == 0 ? build_keys[i % build_keys.size()] : rng());
    /// A tiny L2 forces many leaves; the per-pass cap of 4 (2 bits) then forces several passes. The
    /// result must be identical to a single-pass run.
    auto plan = PartitionPlan::choose(30000, 8u << 10, 4);
    ASSERT_GT(plan.pass_bits.size(), 1u);
    checkBuildAndProbe(build_keys, probe_keys, plan, 4, 4);
}
