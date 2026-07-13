/// Microbenchmark for scatter kernels — Gate-1 instrument of the ColumnsScatter work.
///
/// U0 reference arm: a VERBATIM copy of the in-tree radix scatter kernels from
/// src/Interpreters/RadixHashJoin/RadixHashJoin.cpp (namespace `ScatterReference` below). The copied
/// block is delimited by the two marker comment lines further down and is kept byte-identical to the
/// source line range named in the opening marker: to verify, extract the lines strictly between the
/// markers and diff them against that line range of the source file. It measures the byte-bandwidth
/// of `scatterOne<8>` in both routing modes at representative fanouts covering both regimes (direct
/// below `SWWC_MIN_FANOUT`, SWWC + NT stores at or above it).
///
/// Timed region per iteration = seed(all cursors) + scatterOne<8>(n rows) + drain — exactly the
/// per-(batch, column) work of the in-tree barrier-3 scatter. Histogram, prefix sum and destination
/// allocation are untimed setup. Bytes counted = payload bytes written (n * 8); the UInt16 pids
/// emitted by the key mode are a by-product and are not counted. Destinations are warm steady-state
/// (rewritten every iteration): the reference is a RELATIVE kernel gate, not an end-to-end claim.
///
/// Run (single-threaded, pinned):
///   taskset -c 8 ./benchmark_columns_scatter --benchmark_repetitions=7 \
///       --benchmark_report_aggregates_only=true --benchmark_format=json

#include <benchmark/benchmark.h>

#include <Columns/ColumnsNumber.h>
#include <Columns/ColumnsScatter.h>
#include <Common/PODArray.h>

#include <base/defines.h>
#include <base/types.h>

#include <pcg_random.hpp>

#include <bit>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(__aarch64__) && defined(__ARM_FEATURE_CRC32)
#include <arm_acle.h>
#endif

namespace DB::ScatterReference
{

/// Prototypes for the verbatim block below. The block is kept byte-identical to the in-tree source;
/// there it lives in an anonymous namespace and needs no prototypes, here the namespace is named
/// (so the copied kernels stay visible to the driver) and `-Wmissing-prototypes` requires these.
size_t scatterBatchRowsTarget(size_t fanout);
bool widthSupportsSwwc(size_t w);
ALWAYS_INLINE UInt32 routeWord(UInt64 key);
ALWAYS_INLINE UInt64 mixStep(UInt64 h, UInt64 x);
ALWAYS_INLINE UInt32 finalizeRoute(UInt64 h);
ALWAYS_INLINE UInt64 foldBytes(UInt64 h, const char * p, size_t w);
ALWAYS_INLINE UInt32 routeWordBytes(const char * p, size_t w);

/// ---- BEGIN verbatim copy of src/Interpreters/RadixHashJoin/RadixHashJoin.cpp lines 70-345 ----
constexpr size_t LINE_BYTES = 64;
constexpr size_t ELEMS_PER_LINE = LINE_BYTES / sizeof(UInt64);
/// Fanout from which the SWWC + non-temporal path wins over plain per-partition cursors.
constexpr size_t SWWC_MIN_FANOUT = 256;
/// Below this fanout the histogram uses 4 interleaved lanes to break the load-increment-store chain.
constexpr size_t HIST_INTERLEAVE_MAX_FANOUT = 2048;
/// First-pass batch sizing: the boundary cost (cursor sweeps, partial-line flushes) stays a small
/// fraction of the lines written in between.
constexpr size_t SCATTER_BATCH_MIN_ROWS = 256 << 10;
constexpr size_t SCATTER_BATCH_LINES_PER_PARTITION = 64;

/// Partition-plan constants (5.1): the target leaf working set (~L2), the per-pass fanout ceiling
/// (the benchmark's SWWC staging cache ceiling, MAX_FANOUT_PER_PASS), and the per-entry hash-table
/// byte estimate (a cell at 0.5 load factor, matching the bench bandwidth model).
constexpr size_t LEAF_TARGET_BYTES = 1 << 20;
constexpr size_t MAX_FANOUT_PER_PASS = 8192;
constexpr size_t HT_CELL_BYTES = 16;

using NtLine = char __attribute__((vector_size(LINE_BYTES)));

size_t scatterBatchRowsTarget(size_t fanout)
{
    return std::max(SCATTER_BATCH_MIN_ROWS, fanout * SCATTER_BATCH_LINES_PER_PARTITION * ELEMS_PER_LINE);
}

/// SWWC is enabled only for widths that divide the 64-byte line and are covered by the 16-byte
/// minimum alignment of column data (so the per-partition staging line fills to exactly 64 bytes).
bool widthSupportsSwwc(size_t w)
{
    return w == 1 || w == 2 || w == 4 || w == 8 || w == 16;
}

/// Route hashes are deliberately independent of the CRC32C the leaf hash tables use for bucketing:
/// otherwise partition assignment would correlate with in-table bucket placement and each leaf
/// table would see a skewed hash space. The hot single-UInt64 path exactly matches the benchmark:
/// ISO-polynomial CRC32 on aarch64, golden-ratio multiply-shift elsewhere. Wider and composite keys
/// retain the width-generic multiply-shift fold.
ALWAYS_INLINE UInt32 routeWord(UInt64 key)
{
#if defined(__aarch64__) && defined(__ARM_FEATURE_CRC32)
    return __crc32d(-1U, key);
#else
    return static_cast<UInt32>((key * 0x9E3779B97F4A7C15ULL) >> 32);
#endif
}

ALWAYS_INLINE UInt64 mixStep(UInt64 h, UInt64 x)
{
    return (h ^ x) * 0x9E3779B97F4A7C15ULL;
}

ALWAYS_INLINE UInt32 finalizeRoute(UInt64 h)
{
    return static_cast<UInt32>(h >> 32);
}

/// Fold `w` bytes at `p` into the accumulator, 8 bytes at a time with a zero-padded tail.
ALWAYS_INLINE UInt64 foldBytes(UInt64 h, const char * p, size_t w)
{
    size_t i = 0;
    for (; i + 8 <= w; i += 8)
    {
        UInt64 x = 0;
        memcpy(&x, p + i, sizeof(x));
        h = mixStep(h, x);
    }
    if (i < w)
    {
        UInt64 x = 0;
        memcpy(&x, p + i, w - i);
        h = mixStep(h, x);
    }
    return h;
}

/// Compile-time width variant for the hot single-key path (the loop unrolls fully).
template <size_t width>
ALWAYS_INLINE UInt32 routeWordFixed(const char * p)
{
    if constexpr (width == sizeof(UInt64))
    {
        UInt64 key{};
        __builtin_memcpy_inline(&key, p, sizeof(key));
        return routeWord(key);
    }
    else
    {
        return finalizeRoute(foldBytes(0, p, width));
    }
}

ALWAYS_INLINE UInt32 routeWordBytes(const char * p, size_t w)
{
    return finalizeRoute(foldBytes(0, p, w));
}

/// Per-worker scatter state: write cursors (byte-granular), and for the SWWC path one 64-byte
/// staging line per partition plus a byte fill counter. Ported from the benchmark's ScatterScratch,
/// generalized from 8-byte elements to arbitrary fixed widths.
///
/// Invariant: staged bytes for partition p live at staging + p*64 + [m, fill), where
/// m = (uintptr)cursors[p] & 63. seed() seeds `fill` with the cursor misalignment; before the first
/// flush the cursor has not advanced (m == fill start), after the first flush the cursor is
/// line-aligned (m == 0). Column-data bases are >= 16-byte aligned and per-worker start offsets are
/// multiples of the element width, so for the SWWC-enabled widths (1,2,4,8,16) m is a multiple of the
/// width and the staging line fills to exactly 64 bytes.
struct ScatterScratch
{
    size_t fanout = 0;
    bool use_swwc = false;
    PaddedPODArray<char> staging_mem;
    char * staging = nullptr;
    PaddedPODArray<char *> cursors;
    PaddedPODArray<UInt32> fill;

    void init(size_t fanout_, bool use_swwc_)
    {
        fanout = fanout_;
        use_swwc = use_swwc_;
        cursors.resize(fanout);
        if (use_swwc)
        {
            staging_mem.resize(fanout * LINE_BYTES + LINE_BYTES);
            staging = reinterpret_cast<char *>(
                (reinterpret_cast<uintptr_t>(staging_mem.data()) + LINE_BYTES - 1) & ~static_cast<uintptr_t>(LINE_BYTES - 1));
            fill.resize(fanout);
        }
    }

    void setUseSwwc(bool use_swwc_)
    {
        chassert(!use_swwc_ || staging);
        use_swwc = use_swwc_;
    }

    void seed(size_t p, char * cursor)
    {
        cursors[p] = cursor;
        if (use_swwc)
            fill[p] = static_cast<UInt32>(reinterpret_cast<uintptr_t>(cursor) & (LINE_BYTES - 1));
    }

    /// Flush residual staged bytes of every partition and publish the non-temporal stores.
    void drain()
    {
        if (!use_swwc)
            return;
        for (size_t p = 0; p < fanout; ++p)
        {
            const UInt32 f = fill[p];
            if (!f)
                continue;
            char * cur = cursors[p];
            const UInt32 m = static_cast<UInt32>(reinterpret_cast<uintptr_t>(cur) & (LINE_BYTES - 1));
            if (f > m)
            {
                memcpy(cur, staging + p * LINE_BYTES + m, f - m);
                cursors[p] = cur + (f - m);
            }
            fill[p] = 0;
        }
        /// NT stores are weakly ordered; make them visible before the outputs are read.
        std::atomic_thread_fence(std::memory_order_seq_cst);
    }
};

/// The routing source per row. The single-column key kernel computes the partition from the key (and
/// optionally emits it as a 2-byte pid); the payload kernels reload the emitted pid.
template <size_t width>
struct RouteFromKey
{
    const char * keys;
    UInt32 shift;
    UInt32 mask;
    UInt16 * pids; /// null when there are no columns to consume the ids

    ALWAYS_INLINE UInt32 partition(size_t i) const
    {
        const UInt32 p = (routeWordFixed<width>(keys + i * width) >> shift) & mask;
        if (pids)
            pids[i] = static_cast<UInt16>(p);
        return p;
    }
};

struct RouteFromKeyGeneric
{
    const char * keys;
    size_t width;
    UInt32 shift;
    UInt32 mask;
    UInt16 * pids;

    ALWAYS_INLINE UInt32 partition(size_t i) const
    {
        const UInt32 p = (routeWordBytes(keys + i * width, width) >> shift) & mask;
        if (pids)
            pids[i] = static_cast<UInt16>(p);
        return p;
    }
};

struct RouteFromPids
{
    const UInt16 * pids;
    ALWAYS_INLINE UInt32 partition(size_t i) const { return pids[i]; }
};

template <size_t width, typename Route>
void scatterDirect(Route route, const char * data, size_t n, char ** cursors)
{
    for (size_t i = 0; i < n; ++i)
    {
        const UInt32 p = route.partition(i);
        char * dst = cursors[p];
        __builtin_memcpy_inline(dst, data + i * width, width);
        cursors[p] = dst + width;
    }
}

template <typename Route>
void scatterDirectGeneric(Route route, const char * data, size_t n, size_t w, char ** cursors)
{
    for (size_t i = 0; i < n; ++i)
    {
        const UInt32 p = route.partition(i);
        char * dst = cursors[p];
        memcpy(dst, data + i * w, w);
        cursors[p] = dst + w;
    }
}

template <size_t width, typename Route>
void scatterSwwc(Route route, const char * data, size_t n, ScatterScratch & scratch)
{
    /// Hoisted like `staging`: the char*/vector NT store defeats TBAA hoisting, so without this the
    /// compiler reloads scratch.cursors/fill.data() every row.
    char * const staging = scratch.staging;
    char ** const cursors = scratch.cursors.data();
    UInt32 * const fill = scratch.fill.data();

    for (size_t i = 0; i < n; ++i)
    {
        const UInt32 p = route.partition(i);
        char * line = staging + static_cast<size_t>(p) * LINE_BYTES;
        UInt32 f = fill[p];
        __builtin_memcpy_inline(line + f, data + i * width, width);
        f += width;
        if (f == LINE_BYTES)
        {
            char * cur = cursors[p];
            const UInt32 m = static_cast<UInt32>(reinterpret_cast<uintptr_t>(cur) & (LINE_BYTES - 1));
            if (m) /// first flush of a misaligned stream: emit the partial head line with regular stores
            {
                __builtin_memcpy(cur, line + m, LINE_BYTES - m);
                cursors[p] = cur + (LINE_BYTES - m);
            }
            else
            {
                __builtin_nontemporal_store(*reinterpret_cast<const NtLine *>(line), reinterpret_cast<NtLine *>(cur));
                cursors[p] = cur + LINE_BYTES;
            }
            f = 0;
        }
        fill[p] = f;
    }
}

template <size_t width, typename Route>
ALWAYS_INLINE void scatterOne(Route route, const char * data, size_t n, bool use_swwc, ScatterScratch & scratch)
{
    if (use_swwc)
        scatterSwwc<width>(route, data, n, scratch);
    else
        scatterDirect<width>(route, data, n, scratch.cursors.data());
}
/// ---- END verbatim copy ----

}

namespace
{

using namespace DB;
using namespace DB::ScatterReference;

/// Pin the copied constants: a drift here means the copy no longer matches the in-tree kernels.
static_assert(LINE_BYTES == 64);
static_assert(ELEMS_PER_LINE == 8);
static_assert(SWWC_MIN_FANOUT == 256);
static_assert(HIST_INTERLEAVE_MAX_FANOUT == 2048);
static_assert(SCATTER_BATCH_MIN_ROWS == 256 << 10);
static_assert(SCATTER_BATCH_LINES_PER_PARTITION == 64);
static_assert(LEAF_TARGET_BYTES == 1 << 20);
static_assert(MAX_FANOUT_PER_PASS == 8192);
static_assert(HT_CELL_BYTES == 16);

/// One benchmark cell: UInt64 payload scattered to `fanout` exact-sized partitions, in-tree policy
/// (SWWC iff fanout >= SWWC_MIN_FANOUT; 8 bytes support SWWC), batch sized per the in-tree constants.
struct ReferenceFixture
{
    size_t fanout;
    size_t n;
    bool use_swwc;
    UInt32 shift;
    UInt32 mask;
    PaddedPODArray<UInt64> keys;
    PaddedPODArray<UInt16> pids;
    PaddedPODArray<UInt16> pids_out;
    std::vector<PaddedPODArray<char>> parts;
    std::vector<char *> bases;
    ScatterScratch scratch;

    explicit ReferenceFixture(size_t fanout_)
        : fanout(fanout_)
        , n(scatterBatchRowsTarget(fanout_))
        , use_swwc(fanout_ >= SWWC_MIN_FANOUT && widthSupportsSwwc(8))
        , shift(static_cast<UInt32>(32 - std::countr_zero(fanout_)))
        , mask(static_cast<UInt32>(fanout_ - 1))
    {
        keys.resize(n);
        pids.resize(n);
        pids_out.resize(n);

        /// Fixed seed: identical inputs across repetitions, process runs, and (in U4) arms.
        pcg64 rng(42);
        for (size_t i = 0; i < n; ++i)
            keys[i] = rng();
        for (size_t i = 0; i < n; ++i)
            pids[i] = static_cast<UInt16>((routeWord(keys[i]) >> shift) & mask);

        /// Untimed: histogram + exact-sized per-partition allocation (the in-tree barriers 1-2).
        std::vector<size_t> counts(fanout, 0);
        for (size_t i = 0; i < n; ++i)
            ++counts[pids[i]];
        parts.resize(fanout);
        bases.resize(fanout);
        for (size_t p = 0; p < fanout; ++p)
        {
            parts[p].resize(counts[p] * sizeof(UInt64));
            bases[p] = parts[p].data();
        }

        scratch.init(fanout, use_swwc);
        verify();
    }

    void seedAll()
    {
        for (size_t p = 0; p < fanout; ++p)
            scratch.seed(p, bases[p]);
    }

    void runPidMode()
    {
        seedAll();
        scatterOne<8>(RouteFromPids{pids.data()}, reinterpret_cast<const char *>(keys.data()), n, use_swwc, scratch);
        scratch.drain();
    }

    void runKeyMode()
    {
        seedAll();
        scatterOne<8>(
            RouteFromKey<8>{reinterpret_cast<const char *>(keys.data()), shift, mask, pids_out.data()},
            reinterpret_cast<const char *>(keys.data()),
            n,
            use_swwc,
            scratch);
        scratch.drain();
    }

    /// Correctness oracle (pre-registered soundness check S4): per-partition count and value-sum of
    /// the scattered output must match a scalar reference; both routing modes must agree; the key
    /// mode must emit exactly the pids the fixture derived from routeWord.
    void verify()
    {
        std::vector<size_t> expected_count(fanout, 0);
        std::vector<UInt64> expected_sum(fanout, 0);
        for (size_t i = 0; i < n; ++i)
        {
            ++expected_count[pids[i]];
            expected_sum[pids[i]] += keys[i];
        }

        auto check = [&](const char * mode)
        {
            for (size_t p = 0; p < fanout; ++p)
            {
                const size_t count = parts[p].size() / sizeof(UInt64);
                if (count != expected_count[p])
                    throw std::runtime_error(std::string("scatter reference oracle: bad count in mode ") + mode);
                UInt64 sum = 0;
                for (size_t i = 0; i < count; ++i)
                {
                    UInt64 v = 0;
                    std::memcpy(&v, parts[p].data() + i * sizeof(UInt64), sizeof(v));
                    sum += v;
                }
                if (sum != expected_sum[p])
                    throw std::runtime_error(std::string("scatter reference oracle: bad content sum in mode ") + mode);
            }
        };

        runPidMode();
        check("pid8");
        runKeyMode();
        check("key8");
        for (size_t i = 0; i < n; ++i)
            if (pids_out[i] != pids[i])
                throw std::runtime_error("scatter reference oracle: key mode emitted wrong pids");
    }
};

/// Module arms (U1+): the same cells driven through DB::ColumnsScatter. The Layer-0 arms write into
/// the SAME preallocated partition buffers as the reference arm (identical timed-region definition:
/// seed + scatter + drain, allocation untimed) — a kernel-parity measurement. The Layer-1 arm times
/// the full one-shot `scatter` call (dispatch + normalization gate + exact allocation + kernel +
/// result teardown) — a definitionally different, informational cell (PREREG P-U1-2).
struct ModuleFixture
{
    ReferenceFixture & ref; /// borrowed; the registering lambda co-captures the owning shared_ptr
    PaddedPODArray<UInt32> pids32;
    ColumnsScatter::ScatterScratch scratch;
    /// Layer-1 inputs: a real column mirroring the reference keys + precomputed shard counts.
    MutableColumnPtr source_column;
    std::vector<UInt32> counts32;

    explicit ModuleFixture(ReferenceFixture & ref_) : ref(ref_)
    {
        pids32.resize(ref.n);
        for (size_t i = 0; i < ref.n; ++i)
            pids32[i] = ref.pids[i];
        scratch.init(ref.fanout, ref.use_swwc);

        auto column = ColumnUInt64::create();
        auto raw = column->insertRawUninitialized(ref.n);
        std::memcpy(raw.data(), ref.keys.data(), ref.n * sizeof(UInt64));
        source_column = std::move(column);
        counts32.assign(ref.fanout, 0);
        for (size_t i = 0; i < ref.n; ++i)
            ++counts32[ref.pids[i]];

        verify();
    }

    void seedAll()
    {
        for (size_t p = 0; p < ref.fanout; ++p)
            scratch.seed(p, ref.bases[p]);
    }

    void runPid16()
    {
        seedAll();
        ColumnsScatter::scatterPidChunk(8, ref.pids.data(), reinterpret_cast<const char *>(ref.keys.data()), ref.n, ref.use_swwc, scratch);
        scratch.drain();
    }

    void runPid32()
    {
        seedAll();
        ColumnsScatter::scatterPidChunk(8, pids32.data(), reinterpret_cast<const char *>(ref.keys.data()), ref.n, ref.use_swwc, scratch);
        scratch.drain();
    }

    void runKey16()
    {
        seedAll();
        ColumnsScatter::scatterKeyChunk(
            8, reinterpret_cast<const char *>(ref.keys.data()), ref.n, ref.shift, ref.mask, ref.pids_out.data(), ref.use_swwc, scratch);
        scratch.drain();
    }

    MutableColumns runLayer1()
    {
        const IColumn * source = source_column.get();
        std::span<const UInt16> pid_span(ref.pids.data(), ref.n);
        return ColumnsScatter::scatter(
            std::span<const IColumn * const>(&source, 1),
            std::span<const std::span<const UInt16>>(&pid_span, 1),
            ref.fanout,
            std::span<const UInt32>(counts32.data(), counts32.size()));
    }

    /// Same oracle as the reference arm, applied to every module path.
    void verify()
    {
        std::vector<size_t> expected_count(ref.fanout, 0);
        std::vector<UInt64> expected_sum(ref.fanout, 0);
        for (size_t i = 0; i < ref.n; ++i)
        {
            ++expected_count[ref.pids[i]];
            expected_sum[ref.pids[i]] += ref.keys[i];
        }

        auto check_parts = [&](const char * mode)
        {
            for (size_t p = 0; p < ref.fanout; ++p)
            {
                const size_t count = ref.parts[p].size() / sizeof(UInt64);
                UInt64 sum = 0;
                for (size_t i = 0; i < count; ++i)
                {
                    UInt64 v = 0;
                    std::memcpy(&v, ref.parts[p].data() + i * sizeof(UInt64), sizeof(v));
                    sum += v;
                }
                if (count != expected_count[p] || sum != expected_sum[p])
                    throw std::runtime_error(std::string("scatter module oracle: bad partition in mode ") + mode);
            }
        };

        runPid16();
        check_parts("mod0_pid16");
        runPid32();
        check_parts("mod0_pid32");
        runKey16();
        check_parts("mod0_key16");
        for (size_t i = 0; i < ref.n; ++i)
            if (ref.pids_out[i] != ref.pids[i])
                throw std::runtime_error("scatter module oracle: key mode emitted wrong pids");

        auto shards = runLayer1();
        for (size_t p = 0; p < ref.fanout; ++p)
        {
            const auto raw = shards[p]->getRawData();
            const size_t count = raw.size() / sizeof(UInt64);
            UInt64 sum = 0;
            for (size_t i = 0; i < count; ++i)
            {
                UInt64 v = 0;
                std::memcpy(&v, raw.data() + i * sizeof(UInt64), sizeof(v));
                sum += v;
            }
            if (count != expected_count[p] || sum != expected_sum[p])
                throw std::runtime_error("scatter module oracle: bad shard in mode mod1_layer1");
        }
    }
};

std::vector<std::shared_ptr<ReferenceFixture>> registerReferenceBenchmarks()
{
    static constexpr size_t fanouts[] = {64, 256, 2048, 8192};
    std::vector<std::shared_ptr<ReferenceFixture>> fixtures;
    for (size_t fanout : fanouts)
    {
        auto fixture = std::make_shared<ReferenceFixture>(fanout);
        fixtures.push_back(fixture);
        benchmark::RegisterBenchmark(
            ("BM_ref_pid8/F" + std::to_string(fanout)).c_str(),
            [fixture](benchmark::State & state)
            {
                for (auto _ : state)
                {
                    fixture->runPidMode();
                    benchmark::ClobberMemory();
                }
                state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) * fixture->n * sizeof(UInt64));
                state.counters["rows"] = static_cast<double>(fixture->n);
                state.counters["swwc"] = fixture->use_swwc ? 1 : 0;
            });
        benchmark::RegisterBenchmark(
            ("BM_ref_key8/F" + std::to_string(fanout)).c_str(),
            [fixture](benchmark::State & state)
            {
                for (auto _ : state)
                {
                    fixture->runKeyMode();
                    benchmark::ClobberMemory();
                }
                state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) * fixture->n * sizeof(UInt64));
                state.counters["rows"] = static_cast<double>(fixture->n);
                state.counters["swwc"] = fixture->use_swwc ? 1 : 0;
            });

        auto module_fixture = std::make_shared<ModuleFixture>(*fixture);
        auto register_module_cell = [&](const char * name, auto run)
        {
            benchmark::RegisterBenchmark(
                (std::string(name) + "/F" + std::to_string(fanout)).c_str(),
                [fixture, module_fixture, run](benchmark::State & state)
                {
                    for (auto _ : state)
                    {
                        run(*module_fixture);
                        benchmark::ClobberMemory();
                    }
                    state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) * fixture->n * sizeof(UInt64));
                    state.counters["rows"] = static_cast<double>(fixture->n);
                    state.counters["swwc"] = fixture->use_swwc ? 1 : 0;
                });
        };
        register_module_cell("BM_mod0_pid16", [](ModuleFixture & f) { f.runPid16(); });
        register_module_cell("BM_mod0_pid32", [](ModuleFixture & f) { f.runPid32(); });
        register_module_cell("BM_mod0_key16", [](ModuleFixture & f) { f.runKey16(); });
        register_module_cell(
            "BM_mod1_full",
            [](ModuleFixture & f)
            {
                auto shards = f.runLayer1();
                benchmark::DoNotOptimize(shards.data());
            });
    }
    return fixtures;
}

/// U2 String cells: the variable-length Layer-0 kernel at fixed row length L, same fanouts and
/// batch-row sizing as the reference. Bytes counted = chars + offsets actually written
/// ((L + 8) per row). Timed region = seed + scatterStringChunk (no staging to drain); byte
/// histogram and destination allocation are untimed, mirroring the fixed-width cells.
struct StringFixture
{
    size_t fanout;
    size_t n;
    size_t length;
    PaddedPODArray<char> chars;
    PaddedPODArray<UInt64> offsets;
    PaddedPODArray<UInt16> pids;
    std::vector<PaddedPODArray<char>> chars_destination; /// one per shard: overflow-15 tolerant each
    PaddedPODArray<UInt64> offsets_destination;
    std::vector<char *> chars_bases;
    std::vector<UInt64 *> offsets_bases;
    ColumnsScatter::StringScatterState state;

    StringFixture(size_t fanout_, size_t length_)
        : fanout(fanout_)
        , n(ScatterReference::scatterBatchRowsTarget(fanout_))
        , length(length_)
    {
        chars.resize(n * length);
        offsets.resize(n);
        pids.resize(n);
        pcg64 rng(42);
        for (auto & byte : chars)
            byte = static_cast<char>(rng());
        for (size_t i = 0; i < n; ++i)
            offsets[i] = (i + 1) * length;
        const UInt32 shift = static_cast<UInt32>(32 - std::countr_zero(fanout));
        const UInt32 mask = static_cast<UInt32>(fanout - 1);
        for (size_t i = 0; i < n; ++i)
            pids[i] = static_cast<UInt16>((ScatterReference::routeWord(rng()) >> shift) & mask);

        /// Untimed: byte/row histograms + exact destination carving at per-shard offsets.
        std::vector<UInt64> byte_counts(fanout, 0);
        std::vector<UInt64> row_counts(fanout, 0);
        for (size_t i = 0; i < n; ++i)
        {
            byte_counts[pids[i]] += length;
            ++row_counts[pids[i]];
        }
        chars_destination.resize(fanout);
        offsets_destination.resize(n);
        chars_bases.resize(fanout);
        offsets_bases.resize(fanout);
        UInt64 row_prefix = 0;
        for (size_t p = 0; p < fanout; ++p)
        {
            /// Per-shard chars allocations: the kernel's overflow-15 copies require each shard's
            /// region to be independently overflow-tolerant (see the Layer-0 contract).
            chars_destination[p].resize(byte_counts[p]);
            chars_bases[p] = chars_destination[p].data();
            offsets_bases[p] = offsets_destination.data() + row_prefix;
            row_prefix += row_counts[p];
        }
        state.init(fanout);
        verify();
    }

    void run()
    {
        for (size_t p = 0; p < fanout; ++p)
            state.seed(p, chars_bases[p], offsets_bases[p], 0);
        ColumnsScatter::scatterStringChunk(chars.data(), offsets.data(), pids.data(), n, state);
    }

    /// Scalar-reference oracle over both output streams.
    void verify()
    {
        run();
        std::vector<UInt64> row_cursor(fanout, 0);
        std::vector<UInt64> byte_cursor(fanout, 0);
        for (size_t i = 0; i < n; ++i)
        {
            const size_t p = pids[i];
            const char * expected = chars.data() + i * length;
            const char * actual = chars_bases[p] + byte_cursor[p];
            if (std::memcmp(expected, actual, length) != 0)
                throw std::runtime_error("string scatter oracle: bad chars");
            byte_cursor[p] += length;
            if (offsets_bases[p][row_cursor[p]] != byte_cursor[p])
                throw std::runtime_error("string scatter oracle: bad rebased offset");
            ++row_cursor[p];
        }
    }
};

/// U2 Nullable(UInt64) cell: two fixed streams (width-1 null map + width-8 payload) through the
/// module chunk kernel, driven by one pid stream. Bytes counted = 9 per row.
struct NullableFixture
{
    ReferenceFixture & ref; /// borrowed; the registering lambda co-captures the owning shared_ptr
    PaddedPODArray<char> null_bytes;
    PaddedPODArray<char> null_destination;
    std::vector<char *> null_bases;
    ColumnsScatter::ScatterScratch null_scratch;
    ColumnsScatter::ScatterScratch payload_scratch;

    explicit NullableFixture(ReferenceFixture & ref_) : ref(ref_)
    {
        pcg64 rng(43);
        null_bytes.resize(ref.n);
        for (auto & byte : null_bytes)
            byte = (rng() % 4) == 0;
        null_destination.resize(ref.n);
        null_bases.resize(ref.fanout);
        std::vector<size_t> counts(ref.fanout, 0);
        for (size_t i = 0; i < ref.n; ++i)
            ++counts[ref.pids[i]];
        size_t prefix = 0;
        for (size_t p = 0; p < ref.fanout; ++p)
        {
            null_bases[p] = null_destination.data() + prefix;
            prefix += counts[p];
        }
        null_scratch.init(ref.fanout, ref.use_swwc);
        payload_scratch.init(ref.fanout, ref.use_swwc);
    }

    void run()
    {
        for (size_t p = 0; p < ref.fanout; ++p)
        {
            null_scratch.seed(p, null_bases[p]);
            payload_scratch.seed(p, ref.bases[p]);
        }
        ColumnsScatter::scatterPidChunk(1, ref.pids.data(), null_bytes.data(), ref.n, ref.use_swwc, null_scratch);
        ColumnsScatter::scatterPidChunk(
            8, ref.pids.data(), reinterpret_cast<const char *>(ref.keys.data()), ref.n, ref.use_swwc, payload_scratch);
        null_scratch.drain();
        payload_scratch.drain();
    }
};

void registerStringBenchmarks()
{
    static constexpr size_t fanouts[] = {64, 256, 2048, 8192};
    for (size_t fanout : fanouts)
    {
        for (size_t length : {8uz, 32uz})
        {
            auto fixture = std::make_shared<StringFixture>(fanout, length);
            benchmark::RegisterBenchmark(
                ("BM_mod0_str_L" + std::to_string(length) + "/F" + std::to_string(fanout)).c_str(),
                [fixture](benchmark::State & state)
                {
                    for (auto _ : state)
                    {
                        fixture->run();
                        benchmark::ClobberMemory();
                    }
                    state.SetBytesProcessed(
                        static_cast<int64_t>(state.iterations()) * fixture->n * (fixture->length + sizeof(UInt64)));
                    state.counters["rows"] = static_cast<double>(fixture->n);
                });
        }
    }
}

void registerNullableBenchmarks(const std::vector<std::shared_ptr<ReferenceFixture>> & ref_fixtures)
{
    for (const auto & ref : ref_fixtures)
    {
        auto fixture = std::make_shared<NullableFixture>(*ref);
        benchmark::RegisterBenchmark(
            ("BM_mod0_null8/F" + std::to_string(ref->fanout)).c_str(),
            [fixture, ref](benchmark::State & state)
            {
                for (auto _ : state)
                {
                    fixture->run();
                    benchmark::ClobberMemory();
                }
                state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) * ref->n * 9);
                state.counters["rows"] = static_cast<double>(ref->n);
                state.counters["swwc"] = ref->use_swwc ? 1 : 0;
            });
    }
}

}

int main(int argc, char ** argv)
{
    auto reference_fixtures = registerReferenceBenchmarks();
    registerStringBenchmarks();
    registerNullableBenchmarks(reference_fixtures);
    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
    return 0;
}
