/// bench_build_bandwidth — isolate and compare the three "build-side data movement" pipelines of the
/// RadixHashJoin / ConcurrentHashJoin build, over the SAME pre-generated right-side blocks, so that two
/// headline numbers fall out as deltas where the shared per-block ingest work cancels:
///
///   CHJ zero-copy reference   per block: materialize (Alloc) + COW refcount retain + dispatch
///   RHJ scatter               per block: Alloc + hist (BuildSide::add); then scatterToLeaves (HLL off)
///   RHJ memcpy                per block: Alloc + hist;             then parallel key+ref copy
///
///   True RHJ scatter overhead  = T_scatter - T_memcpy   (shared ingest cancels; pure routing cost over
///                                                         a sequential copy of the SAME key+ref bytes)
///   True CHJ zero-copy advantage = RHJ_memcpy_total - CHJ_total = (T_hist + T_memcpy) - T_dispatch
///                                                         (the hist + the key+ref copy CHJ never pays)
///
/// This is a research/reproduction harness only. It drives the genuine `DB::RadixJoin::BuildSide` scatter
/// (src/Interpreters/RadixHashJoin/BuildSide.h) unchanged, and replicates `ConcurrentHashJoin`'s zero-copy
/// dispatch in-bench (its `dispatchBlock` is private; production is left untouched). The replica mirrors,
/// for the single fixed-width UInt64 key (HashMap<UInt64, Mapped, HashCRC32<UInt64>>, the non-two-level
/// `key64` path):
///   - selectDispatchBlock + hashToSelector   src/Interpreters/ConcurrentHashJoin.cpp:782 / :718
///     (per-row hash = HashCRC32<UInt64>; selector[i] = hash & (num_shards - 1))
///   - scatterBlocksWithSelector              src/Interpreters/ConcurrentHashJoin.cpp:848
///     (one ScatteredBlock::Indexes per shard; emit the real DB::ScatteredBlock COW view, no column copy)
/// The 8-byte key takes the zero-copy selector path (>4-byte threshold in dispatchBlock).

#include <Columns/ColumnsNumber.h>
#include <Columns/IColumn.h>
#include <Core/Block.h>
#include <DataTypes/DataTypesNumber.h>

#include <Interpreters/HashJoin/ScatteredBlock.h>
#include <Interpreters/RadixHashJoin/Arena.h>
#include <Interpreters/RadixHashJoin/BuildSide.h>
#include <Interpreters/RadixHashJoin/KeyRefScatter.h>
#include <Interpreters/RadixHashJoin/ParallelFor.h>
#include <Interpreters/RadixHashJoin/PartitionPlan.h>

#include <Common/HashTable/Hash.h>
#include <Common/ThreadStatus.h>

#include <fmt/format.h>

#include <algorithm>
#include <atomic>
#include <bit>
#include <chrono>
#include <cstring>
#include <exception>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <sched.h>
#include <unistd.h>

namespace
{

using namespace DB;
using namespace DB::RadixJoin;

/// One generated block carries this many rows by default (close to the server's default max_block_size).
constexpr UInt64 DEFAULT_BLOCK_ROWS = 65536;
/// The two narrow build columns the scatter and the memcpy baseline both move: the packed key and the
/// 8-byte BuildRef. Single UInt64 key => key_width == 8.
constexpr size_t KEY_WIDTH = 8;

/// Distinct pseudo-random UInt64 build key for global row `i` (a bijection: intHash64 is the Murmur
/// finalizer in src/Common/HashTable/Hash.h), so the build holds distinct keys and the route/dispatch
/// distributions are well mixed.
UInt64 buildKey(UInt64 i)
{
    return intHash64(i);
}

double nowMs(std::chrono::steady_clock::time_point a, std::chrono::steady_clock::time_point b)
{
    return std::chrono::duration<double, std::milli>(b - a).count();
}

double minOf(const std::vector<double> & v)
{
    return v.empty() ? 0.0 : *std::min_element(v.begin(), v.end());
}

double medianOf(std::vector<double> v)
{
    if (v.empty())
        return 0.0;
    std::sort(v.begin(), v.end());
    const size_t n = v.size();
    return (n % 2) ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

/// GiB/s (1024^3) for `bytes` moved in `ms` milliseconds.
double gibPerSec(UInt64 bytes, double ms)
{
    return ms > 0.0 ? (static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0)) / (ms / 1000.0) : 0.0;
}

/// A dynamically-balanced `ParallelFor` honoring the production contract: `fn(unit, worker)` once
/// per unit, dense `worker` id in [0, workers) stable per unit and owned by one thread, work-stealing on
/// an atomic cursor (leaf sizes are skewed), exception propagation after all workers stop. Mirrors
/// gtest's makeParallelFor.
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

/// Build one `{ k0 : UInt64, p0..p(payload_cols-1) : UInt64 }` block for global rows [begin, begin+count).
/// The key column is at position 0; payload columns only inflate T_gen and cancel out of both headline
/// deltas (they are not moved by scatter/memcpy and not part of the dispatch selector).
Block makeBlock(UInt64 begin, UInt64 count, size_t payload_cols)
{
    Block block;
    {
        auto col = ColumnUInt64::create();
        auto & data = col->getData();
        data.resize(count);
        for (UInt64 r = 0; r < count; ++r)
            data[r] = buildKey(begin + r);
        block.insert(ColumnWithTypeAndName(std::move(col), std::make_shared<DataTypeUInt64>(), "k0"));
    }
    for (size_t c = 0; c < payload_cols; ++c)
    {
        auto col = ColumnUInt64::create();
        auto & data = col->getData();
        data.resize(count);
        for (UInt64 r = 0; r < count; ++r)
            data[r] = begin + r + c; /// cheap fill; content irrelevant
        block.insert(ColumnWithTypeAndName(std::move(col), std::make_shared<DataTypeUInt64>(), fmt::format("p{}", c)));
    }
    return block;
}

/// ── CHJ zero-copy dispatch replica (production untouched) ─────────────────────────────────────────────

/// Mirrors selectDispatchBlock + hashToSelector for a single UInt64 key over the non-two-level key64 map:
/// per-row hash = HashCRC32<UInt64>, shard = hash & (num_shards - 1) (num_shards is a power of two).
IColumn::Selector computeDispatchSelector(const ColumnUInt64 & key_col, size_t num_shards)
{
    const auto & keys = key_col.getData();
    const size_t n = keys.size();
    IColumn::Selector selector(n);
    HashCRC32<UInt64> hasher;
    const size_t mask = num_shards - 1;
    for (size_t i = 0; i < n; ++i)
        selector[i] = hasher(keys[i]) & mask;
    return selector;
}

/// Mirrors scatterBlocksWithSelector: one ScatteredBlock::Indexes per shard, then emit the real
/// DB::ScatteredBlock (COW view over `from_block` + its selector) — no column bytes copied.
ScatteredBlocks scatterWithSelector(size_t num_shards, const IColumn::Selector & selector, const Block & from_block)
{
    std::vector<ScatteredBlock::IndexesPtr> selectors(num_shards);
    for (size_t i = 0; i < num_shards; ++i)
    {
        selectors[i] = ScatteredBlock::Indexes::create();
        selectors[i]->reserve(selector.size() / num_shards + 1);
    }
    for (size_t i = 0; i < selector.size(); ++i)
        selectors[selector[i]]->getData().push_back(i);

    ScatteredBlocks result;
    result.reserve(num_shards);
    for (size_t i = 0; i < num_shards; ++i)
        result.emplace_back(from_block, std::move(selectors[i]));
    return result;
}

/// ── The two RHJ final batches (the only operations of interest) ──────────────────────────────────────

/// One interface over the two RHJ final batches; both run over the same finished BuildSide and return the
/// key+ref bytes moved (the common numerator = total_rows * (key_width + sizeof(BuildRef))).
struct Batch
{
    virtual ~Batch() = default;
    virtual const char * name() const = 0;
    virtual UInt64 run(const ParallelFor & parallel_for, size_t num_workers) = 0;
};

/// RHJ scatter: the genuine BuildSide::scatterToLeaves with HLL disabled. The returned LeafArrays (owning
/// its Arena) is dropped each run, so jemalloc hands back warm pages exactly as in production; repeatable
/// after one finishBuild. Records bytes_scattered to flag a multi-pass build (physical > key+ref volume).
struct ScatterBatch : Batch
{
    BuildSide & build_side;
    UInt64 key_ref_volume;
    UInt64 last_bytes_scattered = 0;

    ScatterBatch(BuildSide & bs, UInt64 vol) : build_side(bs), key_ref_volume(vol) {}

    const char * name() const override { return "scatter"; }

    UInt64 run(const ParallelFor & parallel_for, size_t num_workers) override
    {
        LeafArrays leaves = build_side.scatterToLeaves(parallel_for, num_workers, /*estimate_distinct_keys=*/false);
        last_bytes_scattered = leaves.bytes_scattered;
        return key_ref_volume;
    }
};

/// RHJ memcpy baseline: carve one fused-record buffer per block exactly like allocExactPartitions
/// (BuildSide.cpp) — line-aligned, line-padded, ref-first `[ BuildRef | key ]` — and memcpy each
/// block's keys and generated BuildRef scratch into it. No routing: the lower-bound sequential copy the
/// scatter is measured against.
struct MemcpyBatch : Batch
{
    BuildSide & build_side;
    UInt64 key_ref_volume;
    std::vector<BuildRef> ref_scratch; /// read-only; content irrelevant for a copy-bandwidth measurement

    MemcpyBatch(BuildSide & bs, UInt64 vol, size_t max_block_rows) : build_side(bs), key_ref_volume(vol)
    {
        ref_scratch.assign(max_block_rows, BuildRef(0, 0));
    }

    const char * name() const override { return "memcpy"; }

    UInt64 run(const ParallelFor & parallel_for, size_t /*num_workers*/) override
    {
        const auto & blocks = build_side.blocks();
        const size_t num_blocks = blocks.size();
        const size_t record_width = KEY_WIDTH + sizeof(BuildRef);

        RadixJoin::Arena arena;
        std::vector<char *> bases(num_blocks, nullptr);

        parallel_for(num_blocks, [&](size_t b, size_t /*worker*/)
        {
            const size_t n = blocks[b].rows();
            if (n == 0)
                return;
            const size_t record_bytes = roundUpToLine(n * record_width);
            bases[b] = static_cast<char *>(arena.allocate(record_bytes, LINE_BYTES));
        });

        std::atomic<UInt64> total_bytes{0};
        parallel_for(num_blocks, [&](size_t b, size_t /*worker*/)
        {
            const size_t n = blocks[b].rows();
            if (n == 0)
                return;
            char * base = bases[b];
            const char * key_src = blocks[b].getByPosition(0).column->getRawData().data();
            for (size_t row = 0; row < n; ++row)
            {
                char * rec = base + row * record_width;
                std::memcpy(rec, &ref_scratch[row], sizeof(BuildRef));
                std::memcpy(rec + sizeof(BuildRef), key_src + row * KEY_WIDTH, KEY_WIDTH);
            }
            total_bytes.fetch_add(n * record_width, std::memory_order_relaxed);
        });

        chassert(total_bytes.load() == key_ref_volume);
        return key_ref_volume;
    }
};

struct Args
{
    UInt64 build_rows = 100'000'000ULL;
    size_t payload_cols = 0;
    int threads = 48;
    size_t chj_shards = 0; /// 0 => ceilPowerOfTwo(threads)
    UInt64 block_rows = DEFAULT_BLOCK_ROWS;
    size_t leaves = 0; /// 0 => PartitionPlan::choose; else fixed power-of-two single-pass plan
    int passes = 0; /// 0 => PartitionPlan::choose; else fixed single-pass plan
    int repeats = 5;
};

UInt64 parseU64(const char * s) { return std::strtoull(s, nullptr, 10); }

/// A fixed plan with `leaves` (rounded up to a power of two) leaves, splitting its `total_bits` across
/// `passes` radix passes exactly the way PartitionPlan::choose does — spread evenly with the remainder
/// on the first passes (max - min <= 1) so no pass is a degenerate low-fanout scatter. A `passes <= 0`
/// (or larger than `total_bits`) request is clamped; `total_bits == 0` yields a single pass.
PartitionPlan makeFixedPlan(size_t leaves, int passes = 1)
{
    PartitionPlan plan;
    leaves = ceilPowerOfTwo(std::max<size_t>(leaves, 1));
    plan.num_leaves = leaves;
    plan.total_bits = static_cast<UInt32>(std::countr_zero(leaves));
    plan.leaf_shift = PartitionPlan::ROUTE_BITS - plan.total_bits;

    UInt32 num_passes = 1;
    if (plan.total_bits > 0)
        num_passes = std::clamp<UInt32>(static_cast<UInt32>(std::max(passes, 1)), 1, plan.total_bits);

    /// Spread the bits evenly and put the remainder on the first passes, so the per-pass fanout is
    /// balanced and there is no degenerate trailing low-fanout pass (mirrors PartitionPlan::choose).
    const UInt32 base = plan.total_bits / num_passes;
    const UInt32 rem = plan.total_bits % num_passes;
    plan.pass_bits.assign(num_passes, base);
    for (UInt32 i = 0; i < rem; ++i)
        plan.pass_bits[i] += 1;

    return plan;
}

/// Run a repeatable phase `--repeats` times (after one warmup), returning min (peak) and median ms.
struct RepeatStat
{
    double min_ms = 0.0;
    double median_ms = 0.0;
    UInt64 numerator = 0;
};

RepeatStat repeatPhase(int repeats, const std::function<UInt64()> & once)
{
    once(); /// warmup (warms destination pages / allocator)
    std::vector<double> ms;
    ms.reserve(repeats);
    UInt64 num = 0;
    for (int r = 0; r < repeats; ++r)
    {
        const auto t0 = std::chrono::steady_clock::now();
        num = once();
        const auto t1 = std::chrono::steady_clock::now();
        ms.push_back(nowMs(t0, t1));
    }
    return {minOf(ms), medianOf(ms), num};
}

}

int main(int argc, char ** argv)
{
    Args args;
    for (int i = 1; i < argc; ++i)
    {
        std::string a = argv[i];
        auto next = [&]() -> const char * { return (i + 1 < argc) ? argv[++i] : ""; };
        if (a == "--build") args.build_rows = parseU64(next());
        else if (a == "--payload-cols") args.payload_cols = static_cast<size_t>(parseU64(next()));
        else if (a == "--threads") args.threads = static_cast<int>(parseU64(next()));
        else if (a == "--chj-shards") args.chj_shards = static_cast<size_t>(parseU64(next()));
        else if (a == "--block-rows") args.block_rows = parseU64(next());
        else if (a == "--leaves") args.leaves = static_cast<size_t>(parseU64(next()));
        else if (a == "--passes") args.passes = static_cast<int>(parseU64(next()));
        else if (a == "--repeats") args.repeats = static_cast<int>(parseU64(next()));
        else if (a == "--help")
        {
            fmt::print(
                "Usage: bench_build_bandwidth [--build N] [--payload-cols C] [--threads T] [--chj-shards S]\n"
                "       [--block-rows B] [--leaves L] [--passes P] [--repeats R]\n"
                "\n"
                "Compares three build-side data-movement pipelines over the same pre-generated blocks:\n"
                "  CHJ zero-copy reference = T_gen + T_dispatch (materialize + COW refcount + dispatch)\n"
                "  RHJ scatter             = T_gen + T_hist + T_scatter (BuildSide::scatterToLeaves, HLL off)\n"
                "  RHJ memcpy              = T_gen + T_hist + T_memcpy  (sequential key+ref copy)\n"
                "\n"
                "  RHJ scatter overhead     = T_scatter - T_memcpy (and the ratio)\n"
                "  CHJ zero-copy advantage  = RHJ_memcpy_total - CHJ_total (and the ratio)\n"
                "\n"
                "NOTE: bandwidth is DRAM-faithful only when the dataset >> LLC; with a small --build the\n"
                "      retained key column reads hot from cache and inflates the GiB/s.\n");
            return 0;
        }
        else
        {
            fmt::print(stderr, "unknown arg: {}\n", a);
            return 2;
        }
    }

    args.threads = std::max(args.threads, 1);
    args.repeats = std::max(args.repeats, 1);
    args.block_rows = std::max<UInt64>(args.block_rows, 1);
    if (args.chj_shards == 0)
        args.chj_shards = ceilPowerOfTwo(static_cast<size_t>(args.threads));
    else
        args.chj_shards = ceilPowerOfTwo(args.chj_shards);

    /// The engines' internal use of `ThreadStatus`-aware machinery is satisfied by a main ThreadStatus.
    DB::MainThreadStatus::getInstance();

    const size_t num_workers = static_cast<size_t>(args.threads);
    const ParallelFor parallel_for = makeParallelFor(num_workers);

    const size_t l2_raw = []() -> size_t
    {
        const Int64 v = sysconf(_SC_LEVEL2_CACHE_SIZE);
        return v > 0 ? static_cast<size_t>(v) : 0; /// 0 => PartitionPlan falls back to L2_FALLBACK_BYTES
    }();

    PartitionPlan plan = args.leaves != 0
        ? makeFixedPlan(args.leaves, args.passes)
        : PartitionPlan::choose(std::optional<UInt64>(args.build_rows), l2_raw, /*max_partitions_per_pass=*/8192);

    const UInt64 num_blocks = (args.build_rows + args.block_rows - 1) / args.block_rows;

    fmt::print(
        stderr,
        "build={} payload_cols={} threads={} chj_shards={} block_rows={} leaves={} passes={} repeats={}\n",
        args.build_rows, args.payload_cols, args.threads, args.chj_shards, args.block_rows,
        plan.num_leaves, plan.pass_bits.size(), args.repeats);

    /// ── T_gen: pre-generate all blocks once (materialize: alloc + fill, retained by COW into the vector).
    std::vector<Block> blocks(num_blocks);
    const auto gen_t0 = std::chrono::steady_clock::now();
    parallel_for(num_blocks, [&](size_t b, size_t /*worker*/)
    {
        const UInt64 begin = b * args.block_rows;
        const UInt64 count = std::min<UInt64>(args.block_rows, args.build_rows - begin);
        blocks[b] = makeBlock(begin, count, args.payload_cols);
    });
    const double t_gen = nowMs(gen_t0, std::chrono::steady_clock::now());

    /// ── T_hist: build one BuildSide by add-ing every block (route hashing + replicated histogram + a
    /// second COW retain), then finishBuild. Measured once. add(block, lane) is lock-free per lane, so the
    /// dense worker id is used directly as the lane.
    BuildSide build_side(plan, /*key_positions=*/{0}, /*key_widths=*/{KEY_WIDTH}, num_workers);
    const auto hist_t0 = std::chrono::steady_clock::now();
    parallel_for(num_blocks, [&](size_t b, size_t worker)
    {
        build_side.add(blocks[b], worker);
    });
    build_side.finishBuild();
    const double t_hist = nowMs(hist_t0, std::chrono::steady_clock::now());

    const UInt64 total_rows = build_side.totalRows();
    const UInt64 key_ref_volume = total_rows * (KEY_WIDTH + sizeof(BuildRef));

    /// ── T_dispatch: the CHJ zero-copy dispatch replica over all retained blocks (COW retain + selector +
    /// ScatteredBlocks), with a sink to defeat DCE. Repeatable.
    std::atomic<UInt64> dispatch_sink{0};
    auto dispatch_once = [&]() -> UInt64
    {
        std::atomic<UInt64> rows_seen{0};
        parallel_for(num_blocks, [&](size_t b, size_t /*worker*/)
        {
            const auto & kept = blocks[b]; /// COW refcount retain (what CHJ holds per block)
            const auto & key_col = typeid_cast<const ColumnUInt64 &>(*kept.getByPosition(0).column);
            IColumn::Selector selector = computeDispatchSelector(key_col, args.chj_shards);
            ScatteredBlocks scattered = scatterWithSelector(args.chj_shards, selector, kept);
            UInt64 local = 0;
            for (const auto & sb : scattered)
                local += sb.rows();
            rows_seen.fetch_add(local, std::memory_order_relaxed);
        });
        dispatch_sink.fetch_add(rows_seen.load(), std::memory_order_relaxed);
        return key_ref_volume;
    };

    ScatterBatch scatter_batch(build_side, key_ref_volume);
    MemcpyBatch memcpy_batch(build_side, key_ref_volume, static_cast<size_t>(args.block_rows));

    auto scatter_once = [&]() -> UInt64 { return scatter_batch.run(parallel_for, num_workers); };
    auto memcpy_once = [&]() -> UInt64 { return memcpy_batch.run(parallel_for, num_workers); };

    const RepeatStat disp = repeatPhase(args.repeats, dispatch_once);
    const RepeatStat scat = repeatPhase(args.repeats, scatter_once);
    const RepeatStat memc = repeatPhase(args.repeats, memcpy_once);

    if (scat.numerator != memc.numerator)
        fmt::print(stderr, "WARNING: scatter/memcpy numerators differ ({} vs {})\n", scat.numerator, memc.numerator);
    const bool multipass = scatter_batch.last_bytes_scattered > key_ref_volume;
    if (multipass)
        fmt::print(
            stderr,
            "WARNING: multi-pass scatter physically moved {} bytes (> key+ref volume {}); T_scatter is not a\n"
            "         single-pass key+ref move. Use --leaves to force a single pass for a like-for-like delta.\n",
            scatter_batch.last_bytes_scattered, key_ref_volume);

    /// ── Composed totals (deltas isolate the operation of interest; the shared ingest cancels).
    const double chj_total_min = t_gen + disp.min_ms;
    const double rhj_scatter_total_min = t_gen + t_hist + scat.min_ms;
    const double rhj_memcpy_total_min = t_gen + t_hist + memc.min_ms;

    const double scatter_overhead_ms = scat.min_ms - memc.min_ms;
    const double scatter_overhead_ratio = memc.min_ms > 0 ? scat.min_ms / memc.min_ms : 0.0;
    const double chj_advantage_ms = rhj_memcpy_total_min - chj_total_min;
    const double chj_advantage_ratio = chj_total_min > 0 ? rhj_memcpy_total_min / chj_total_min : 0.0;

    fmt::print("\n==================== bench_build_bandwidth ====================\n");
    fmt::print(
        "build={} rows total_rows={} payload_cols={} threads={} chj_shards={} leaves={} passes={}\n",
        args.build_rows, total_rows, args.payload_cols, args.threads, args.chj_shards,
        plan.num_leaves, plan.pass_bits.size());
    fmt::print("key+ref volume = {} bytes ({:.2f} GiB)\n",
               key_ref_volume, static_cast<double>(key_ref_volume) / (1024.0 * 1024.0 * 1024.0));

    fmt::print("\nPHASES (ms)\n");
    fmt::print("  T_gen      (materialize, once)     = {:8.2f}\n", t_gen);
    fmt::print("  T_hist     (BuildSide::add, once)  = {:8.2f}\n", t_hist);
    fmt::print("  T_dispatch (CHJ replica)  min/med  = {:8.2f} / {:8.2f}\n", disp.min_ms, disp.median_ms);
    fmt::print("  T_scatter  (scatterToLeaves) min/med = {:8.2f} / {:8.2f}   peak {:6.2f} GiB/s (med {:6.2f})\n",
               scat.min_ms, scat.median_ms, gibPerSec(key_ref_volume, scat.min_ms), gibPerSec(key_ref_volume, scat.median_ms));
    fmt::print("  T_memcpy   (key+ref copy)  min/med = {:8.2f} / {:8.2f}   peak {:6.2f} GiB/s (med {:6.2f})\n",
               memc.min_ms, memc.median_ms, gibPerSec(key_ref_volume, memc.min_ms), gibPerSec(key_ref_volume, memc.median_ms));

    fmt::print("\nCOMPOSED TOTALS (min ms)\n");
    fmt::print("  CHJ zero-copy reference  = T_gen + T_dispatch          = {:8.2f}\n", chj_total_min);
    fmt::print("  RHJ scatter total        = T_gen + T_hist + T_scatter  = {:8.2f}\n", rhj_scatter_total_min);
    fmt::print("  RHJ memcpy total         = T_gen + T_hist + T_memcpy   = {:8.2f}\n", rhj_memcpy_total_min);

    fmt::print("\nHEADLINES\n");
    fmt::print("  RHJ scatter overhead    = T_scatter - T_memcpy      = {:8.2f} ms   (ratio {:.2f}x)\n",
               scatter_overhead_ms, scatter_overhead_ratio);
    fmt::print("  CHJ zero-copy advantage = RHJ_memcpy_total - CHJ    = {:8.2f} ms   (ratio {:.2f}x)\n",
               chj_advantage_ms, chj_advantage_ratio);
    fmt::print("===============================================================\n");

    /// Keep the sink observable so the dispatch work cannot be optimized away.
    fmt::print(stderr, "(dispatch sink: {})\n", dispatch_sink.load());
    return 0;
}
