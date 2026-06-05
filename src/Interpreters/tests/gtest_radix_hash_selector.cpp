#include <gtest/gtest.h>

#include <Columns/ColumnsNumber.h>
#include <Interpreters/RadixHashJoin/HugeArena.h>
#include <Interpreters/RadixHashJoin/PartitionConfig.h>
#include <Interpreters/RadixHashJoin/Selector.h>

#include <Common/Stopwatch.h>

#include <fmt/format.h>

#include <atomic>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <iterator>
#include <map>
#include <random>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

using namespace DB;
using namespace DB::RadixHash;

namespace
{

ColumnUInt64::MutablePtr makeKeyColumn(const std::vector<UInt64> & vals)
{
    auto col = ColumnUInt64::create();
    auto & data = col->getData();
    data.resize(vals.size());
    for (size_t i = 0; i < vals.size(); ++i)
        data[i] = vals[i];
    return col;
}

std::vector<UInt64> sequentialKeys(size_t n)
{
    std::vector<UInt64> v(n);
    for (size_t i = 0; i < n; ++i)
        v[i] = i;
    return v;
}

std::vector<UInt64> randomKeys(size_t n, uint64_t seed)
{
    std::mt19937_64 rng(seed);
    std::vector<UInt64> v(n);
    for (size_t i = 0; i < n; ++i)
        v[i] = rng();
    return v;
}

/// Read /proc/self/smaps and return AnonHugePages (in bytes) of the VMA containing `ptr`, or -1.
Int64 anonHugePagesForPtr(const void * ptr)
{
    const auto addr = reinterpret_cast<uintptr_t>(ptr);
    std::ifstream smaps("/proc/self/smaps");
    if (!smaps)
        return -1;

    auto parse = [](std::string_view sv, auto & out, int base)
    {
        while (!sv.empty() && sv.front() == ' ')
            sv.remove_prefix(1);
        return std::from_chars(sv.data(), sv.data() + sv.size(), out, base).ec == std::errc{};
    };

    static constexpr std::string_view anon_key = "AnonHugePages:";
    std::string line;
    bool in_range = false;
    while (std::getline(smaps, line))
    {
        const std::string_view sv{line};
        const size_t dash = sv.find('-');
        const size_t space = sv.find(' ');
        if (dash != std::string_view::npos && space != std::string_view::npos && dash < space)
        {
            // VMA header line: "7f...-7f... rw-p ..."
            uintptr_t start = 0;
            uintptr_t end = 0;
            if (parse(sv.substr(0, dash), start, 16) && parse(sv.substr(dash + 1, space - dash - 1), end, 16))
                in_range = (addr >= start && addr < end);
        }
        else if (in_range && sv.starts_with(anon_key))
        {
            // "AnonHugePages:        2048 kB"
            UInt64 kb = 0;
            if (parse(sv.substr(anon_key.size()), kb, 10))
                return static_cast<Int64>(kb) * 1024;
        }
    }
    return -1;
}

/// True if the system transparent-hugepage mode is `madvise` or `always` (so `madvise` can back THP).
bool thpModeIsActive()
{
    std::ifstream f("/sys/kernel/mm/transparent_hugepage/enabled");
    if (!f)
        return false;
    const std::string s{std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>()};
    return s.contains("[madvise]") || s.contains("[always]");
}

}

/// (v) pass_bits even-split invariant and the spec anchors (sections 5.2, 5.3).
TEST(RadixHashSelector, PartitionConfigInvariants)
{
    constexpr UInt64 cap = 1024;
    constexpr size_t l2 = 2 * 1024 * 1024;

    // Spec anchors.
    auto def = PartitionConfig::make(std::nullopt, l2, cap);
    EXPECT_EQ(def.num_leaves, 256u);
    EXPECT_EQ(def.total_bits, 8u);
    ASSERT_EQ(def.pass_bits.size(), 1u);
    EXPECT_EQ(def.pass_bits[0], 8u);

    auto big = PartitionConfig::make(UInt64(100'000'000), l2, cap); // spec section 5.5 -> 2048 leaves
    EXPECT_EQ(big.num_leaves, 2048u);
    EXPECT_EQ(big.total_bits, 11u);
    ASSERT_EQ(big.pass_bits.size(), 2u);
    EXPECT_EQ(big.pass_bits[0], 6u);
    EXPECT_EQ(big.pass_bits[1], 5u);

    // Sweep estimates to cover total_bits 0..16 and assert the invariants on each.
    std::vector<std::optional<UInt64>> estimates = {std::nullopt};
    for (UInt64 e = 1; e <= UInt64(1e11); e = e * 2 + 1)
        estimates.emplace_back(e);

    const UInt32 bits_per_pass = 10; // floor(log2(1024))
    bool seen_16 = false;
    std::map<UInt32, std::vector<UInt32>> splits; // total_bits -> pass_bits
    for (auto est : estimates)
    {
        auto cfg = PartitionConfig::make(est, l2, cap);
        // num_leaves is a power of two in [1, MAX_LEAVES].
        EXPECT_LE(cfg.num_leaves, PartitionConfig::MAX_LEAVES);
        EXPECT_GE(cfg.num_leaves, 1u);
        EXPECT_EQ(cfg.num_leaves & (cfg.num_leaves - 1), 0u) << "num_leaves must be a power of two";
        EXPECT_EQ(size_t(1) << cfg.total_bits, cfg.num_leaves);
        EXPECT_EQ(cfg.shift, PartitionConfig::HASH_BITS - cfg.total_bits);

        // sum(pass_bits) == total_bits, max - min <= 1, each <= bits_per_pass, minimal #passes.
        UInt32 sum = 0;
        UInt32 lo = 64;
        UInt32 hi = 0;
        for (auto b : cfg.pass_bits)
        {
            sum += b;
            lo = std::min(lo, b);
            hi = std::max(hi, b);
            EXPECT_LE(b, bits_per_pass);
        }
        EXPECT_EQ(sum, cfg.total_bits);
        if (!cfg.pass_bits.empty() && cfg.total_bits > 0)
            EXPECT_LE(hi - lo, 1u);
        const UInt32 expected_passes = cfg.total_bits == 0 ? 1 : (cfg.total_bits + bits_per_pass - 1) / bits_per_pass;
        EXPECT_EQ(cfg.pass_bits.size(), expected_passes);
        splits.emplace(cfg.total_bits, cfg.pass_bits);
        if (cfg.total_bits == 16)
            seen_16 = true;
    }
    EXPECT_TRUE(seen_16) << "sweep should reach total_bits = 16 (MAX_LEAVES)";

    // The documented multi-pass factorisations from the spec (section 5.3).
    using Bits = std::vector<UInt32>;
    EXPECT_EQ(splits[8], (Bits{8}));
    EXPECT_EQ(splits[10], (Bits{10}));
    EXPECT_EQ(splits[11], (Bits{6, 5}));
    EXPECT_EQ(splits[13], (Bits{7, 6}));
    EXPECT_EQ(splits[16], (Bits{8, 8}));
}

/// (i)+(ii) pid is the top total_bits of computeHashInto; histogram sums to N and matches a reference.
TEST(RadixHashSelector, PidAndHistogram)
{
    const size_t n = 1'000'000;
    auto col = makeKeyColumn(sequentialKeys(n));
    auto cfg = PartitionConfig::make(UInt64(n), 2 * 1024 * 1024, 1024);

    Selector sel(cfg);
    std::vector<UInt32> hashes(n);
    std::vector<UInt16> pid(n);
    sel.process(*col, n, hashes.data(), pid.data());

    // Independent reference hash via a fresh computeHashInto call.
    std::vector<UInt32> ref_hash(n);
    col->computeHashInto(0, n, ref_hash.data(), /*initial=*/true);

    std::vector<UInt64> ref_hist(cfg.num_leaves, 0);
    for (size_t j = 0; j < n; ++j)
    {
        ASSERT_EQ(hashes[j], ref_hash[j]) << "selector hash must equal computeHashInto";
        const UInt16 expect_pid = static_cast<UInt16>(static_cast<UInt64>(ref_hash[j]) >> cfg.shift);
        ASSERT_EQ(pid[j], expect_pid) << "pid must be the top total_bits of the hash at row " << j;
        ASSERT_LT(pid[j], cfg.num_leaves);
        ++ref_hist[pid[j]];
    }

    std::vector<UInt32> merged;
    const UInt64 total = sel.mergedHistogram(merged);
    ASSERT_EQ(total, n);
    ASSERT_EQ(merged.size(), cfg.num_leaves);
    for (size_t p = 0; p < cfg.num_leaves; ++p)
        ASSERT_EQ(merged[p], ref_hist[p]) << "histogram bin " << p << " mismatch";
}

/// (iii) merge across threads + exclusive prefix sum is exact.
TEST(RadixHashSelector, MergeAndPrefixSum)
{
    const size_t n = 2'000'000;
    const size_t num_threads = 8;
    auto col = makeKeyColumn(sequentialKeys(n));
    auto cfg = PartitionConfig::make(UInt64(n), 2 * 1024 * 1024, 1024);

    std::vector<std::vector<UInt32>> per_thread(num_threads);
    std::vector<std::thread> threads;
    for (size_t t = 0; t < num_threads; ++t)
    {
        threads.emplace_back([&, t]
        {
            const size_t start = n * t / num_threads;
            const size_t end = n * (t + 1) / num_threads;
            const size_t len = end - start;
            auto sub = col->cut(start, len); // disjoint slice, own column
            Selector sel(cfg);
            std::vector<UInt32> h(len);
            std::vector<UInt16> p(len);
            sel.process(*sub, len, h.data(), p.data());
            sel.mergedHistogram(per_thread[t]);
        });
    }
    for (auto & th : threads)
        th.join();

    std::vector<UInt64> global_hist;
    std::vector<UInt64> offset;
    const UInt64 total = mergeHistograms(per_thread, cfg.num_leaves, global_hist, offset);
    ASSERT_EQ(total, n);

    // Reference: single-thread histogram over all rows.
    Selector ref_sel(cfg);
    std::vector<UInt32> rh(n);
    std::vector<UInt16> rp(n);
    ref_sel.process(*col, n, rh.data(), rp.data());
    std::vector<UInt32> ref_hist;
    ref_sel.mergedHistogram(ref_hist);

    UInt64 acc = 0;
    for (size_t p = 0; p < cfg.num_leaves; ++p)
    {
        EXPECT_EQ(global_hist[p], ref_hist[p]) << "merged global_hist mismatch at " << p;
        EXPECT_EQ(offset[p], acc) << "offset must be the exclusive prefix sum at " << p;
        acc += global_hist[p];
    }
    EXPECT_EQ(acc, n);
}

/// (iv) build and probe selectors are bit-identical for the same keys (spec invariant 15.5).
TEST(RadixHashSelector, BuildProbeBitIdentical)
{
    const size_t n = 500'000;
    auto keys = randomKeys(n, 0xC0FFEE);
    auto build_col = makeKeyColumn(keys);
    auto probe_col = makeKeyColumn(keys); // physically distinct, logically equal
    auto cfg = PartitionConfig::make(UInt64(n), 2 * 1024 * 1024, 1024);

    Selector build_sel(cfg);
    Selector probe_sel(cfg);
    std::vector<UInt32> bh(n);
    std::vector<UInt16> bp(n);
    std::vector<UInt32> ph(n);
    std::vector<UInt16> pp(n);
    build_sel.process(*build_col, n, bh.data(), bp.data());
    probe_sel.process(*probe_col, n, ph.data(), pp.data());

    for (size_t j = 0; j < n; ++j)
    {
        ASSERT_EQ(bh[j], ph[j]) << "build/probe hash diverge at " << j;
        ASSERT_EQ(bp[j], pp[j]) << "build/probe pid diverge at " << j;
    }
}

/// Degenerate single-leaf config (num_leaves=1, shift=32): all pids 0, hist[0]==n. Exercises the
/// 64-bit-shift UB-avoidance for total_bits==0 at runtime.
TEST(RadixHashSelector, SinglePartition)
{
    PartitionConfig cfg;
    cfg.num_leaves = 1;
    cfg.total_bits = 0;
    cfg.shift = PartitionConfig::HASH_BITS; // 32
    cfg.pass_bits = {0};

    const size_t n = 100'000;
    auto col = makeKeyColumn(randomKeys(n, 0xBEEF));
    Selector sel(cfg);
    std::vector<UInt32> h(n);
    std::vector<UInt16> p(n);
    sel.process(*col, n, h.data(), p.data());
    for (size_t j = 0; j < n; ++j)
        ASSERT_EQ(p[j], 0u);

    std::vector<UInt32> hist;
    const UInt64 total = sel.mergedHistogram(hist);
    ASSERT_EQ(hist.size(), 1u);
    ASSERT_EQ(hist[0], n);
    ASSERT_EQ(total, n);
}

/// (vi) HugeArena returns 2 MiB-aligned (first) pointers, respects per-alloc alignment, fail-open.
TEST(RadixHashSelector, HugeArenaAlignment)
{
    HugeArena arena;
    void * first = arena.alloc(100, alignof(UInt16));
    EXPECT_EQ(reinterpret_cast<uintptr_t>(first) % HugeArena::SLAB, 0u) << "first alloc must be 2 MiB-aligned";

    auto * a = arena.allocArray<UInt16>(1000);
    EXPECT_EQ(reinterpret_cast<uintptr_t>(a) % alignof(UInt16), 0u);
    auto * b = arena.allocArray<UInt64>(1000);
    EXPECT_EQ(reinterpret_cast<uintptr_t>(b) % alignof(UInt64), 0u);

    // A multi-slab allocation: each new slab base is 2 MiB-aligned.
    void * big = arena.alloc(3 * HugeArena::SLAB, 64);
    EXPECT_EQ(reinterpret_cast<uintptr_t>(big) % HugeArena::SLAB, 0u);

    EXPECT_EQ(arena.hugePagesUsed() + arena.hugePagesFailed(), arena.slabCount());
    EXPECT_GT(arena.slabCount(), 0u);

    // Memory is actually usable.
    auto * pids = arena.allocArray<UInt16>(1024);
    for (size_t i = 0; i < 1024; ++i)
        pids[i] = static_cast<UInt16>(i);
    UInt64 s = 0;
    for (size_t i = 0; i < 1024; ++i)
        s += pids[i];
    EXPECT_EQ(s, (1023ull * 1024ull) / 2ull);
}

/// (Q2) top-bit uniformity of computeHashInto at total_bits = 16 (no gross leaf skew).
TEST(RadixHashSelector, TopBitUniformity)
{
    // Force MAX_LEAVES (total_bits = 16) with a large estimate.
    auto cfg = PartitionConfig::make(UInt64(4'000'000'000ull), 2 * 1024 * 1024, 1024);
    ASSERT_EQ(cfg.num_leaves, PartitionConfig::MAX_LEAVES);
    ASSERT_EQ(cfg.total_bits, 16u);

    const size_t n = 16'000'000; // mean ~244 rows/leaf
    auto col = makeKeyColumn(sequentialKeys(n)); // bench key pattern: k0 = number

    Selector sel(cfg);
    std::vector<UInt32> h(n);
    std::vector<UInt16> p(n);
    sel.process(*col, n, h.data(), p.data());
    std::vector<UInt32> hist;
    sel.mergedHistogram(hist);

    const double mean = static_cast<double>(n) / static_cast<double>(cfg.num_leaves);
    UInt32 lo = std::numeric_limits<UInt32>::max();
    UInt32 hi = 0;
    size_t empty = 0;
    double var = 0;
    for (UInt32 c : hist)
    {
        lo = std::min(lo, c);
        hi = std::max(hi, c);
        if (c == 0)
            ++empty;
        const double d = c - mean;
        var += d * d;
    }
    var /= static_cast<double>(cfg.num_leaves);
    const double cv = std::sqrt(var) / mean;

    std::cout << fmt::format(
        "[uniformity] total_bits=16 leaves={} n={} mean={:.1f} min={} max={} empty={} cv={:.4f}\n",
        cfg.num_leaves, n, mean, lo, hi, empty, cv);

    EXPECT_EQ(empty, 0u) << "sequential keys must not leave empty leaves at total_bits=16";
    EXPECT_LT(cv, 0.20) << "coefficient of variation indicates leaf skew";
    EXPECT_LT(static_cast<double>(hi), 3.0 * mean) << "max leaf grossly above mean";
}

/// Perf report (selector ns/row, single + 16-thread) and THP backing check. Not a hard perf gate.
TEST(RadixHashSelector, PerfAndThp)
{
    const size_t n = 16'000'000;
    auto keys = sequentialKeys(n);
    auto col = makeKeyColumn(keys);
    auto cfg = PartitionConfig::make(UInt64(100'000'000), 2 * 1024 * 1024, 1024); // 2048 leaves
    auto per_row = [&](UInt64 ns) { return static_cast<double>(ns) / static_cast<double>(n); };

    // Single-thread breakdown: hash, pid (shift), pid+histogram, fused total. A few reps; keep best.
    {
        std::vector<UInt32> h(n);
        std::vector<UInt16> p(n);
        UInt64 best_hash = ~0ull;
        UInt64 best_pid = ~0ull;
        UInt64 best_pidhist = ~0ull;
        UInt64 best_total = ~0ull;
        for (int rep = 0; rep < 3; ++rep)
        {
            { Stopwatch sw; col->computeHashInto(0, n, h.data(), true); best_hash = std::min(best_hash, sw.elapsedNanoseconds()); }
            { Stopwatch sw; const UInt32 sh = cfg.shift; for (size_t j = 0; j < n; ++j) p[j] = static_cast<UInt16>(static_cast<UInt64>(h[j]) >> sh); best_pid = std::min(best_pid, sw.elapsedNanoseconds()); }
            { Selector s(cfg); Stopwatch sw; s.pidsFromHashes(h.data(), n, p.data()); best_pidhist = std::min(best_pidhist, sw.elapsedNanoseconds()); }
            { Selector s(cfg); Stopwatch sw; s.process(*col, n, h.data(), p.data()); best_total = std::min(best_total, sw.elapsedNanoseconds()); }
        }
        std::cout << fmt::format(
            "[perf] single-thread ns/row (best of 3, leaves={}): hash={:.3f}  pid={:.3f}  pid+hist={:.3f}  (hist={:.3f})  fused_total={:.3f}\n",
            cfg.num_leaves, per_row(best_hash), per_row(best_pid), per_row(best_pidhist),
            per_row(best_pidhist) - per_row(best_pid), per_row(best_total));
    }

    // 16 threads, disjoint slices, each its own selector (lock-free). Time only sel.process per
    // thread (cut/alloc excluded); report the slowest thread (the wall-relevant cost) + throughput.
    {
        const size_t num_threads = 16;
        const size_t len = n / num_threads;
        std::vector<ColumnPtr> subs(num_threads);
        for (size_t t = 0; t < num_threads; ++t)
            subs[t] = col->cut(t * len, len); // pre-cut outside the timed region
        std::vector<std::thread> threads;
        std::atomic<uint64_t> max_ns{0};
        Stopwatch wall;
        for (size_t t = 0; t < num_threads; ++t)
        {
            threads.emplace_back([&, t]
            {
                Selector sel(cfg);
                std::vector<UInt32> h(len);
                std::vector<UInt16> p(len);
                Stopwatch sw;
                sel.process(*subs[t], len, h.data(), p.data());
                const auto ns = sw.elapsedNanoseconds();
                uint64_t prev = max_ns.load();
                while (ns > prev && !max_ns.compare_exchange_weak(prev, ns)) { }
            });
        }
        for (auto & th : threads)
            th.join();
        const double slowest_ns_per_row = static_cast<double>(max_ns.load()) / static_cast<double>(len);
        const double throughput_mrows = static_cast<double>(n) / (static_cast<double>(wall.elapsedNanoseconds()) / 1e3);
        std::cout << fmt::format(
            "[perf] 16-thread: slowest-thread {:.3f} ns/row (fused, leaves={}); aggregate throughput {:.1f} Mrows/s\n",
            slowest_ns_per_row, cfg.num_leaves, throughput_mrows);
    }

    // THP backing: fault a large pid arena and read AnonHugePages for its VMA.
    {
        HugeArena arena;
        const size_t pid_count = 100'000'000; // ~190 MiB of uint16 pid (100M-row scale)
        auto * pids = arena.allocArray<UInt16>(pid_count);
        for (size_t i = 0; i < pid_count; i += 4096 / sizeof(UInt16))
            pids[i] = 1; // fault every page
        const Int64 anon_huge = anonHugePagesForPtr(pids);
        std::cout << fmt::format(
            "[thp] pid arena: slabs={} used={} failed={} AnonHugePages={} bytes ({:.1f} MiB)\n",
            arena.slabCount(), arena.hugePagesUsed(), arena.hugePagesFailed(),
            anon_huge, anon_huge < 0 ? 0.0 : static_cast<double>(anon_huge) / 1024.0 / 1024.0);

        // Only assert THP backing when the system mode actually allows madvise-backed huge pages;
        // otherwise the arena's fail-open path (4 KiB pages) is the correct behaviour.
        if (thpModeIsActive())
        {
            EXPECT_GT(arena.hugePagesUsed(), 0u);
            // AnonHugePages is -1 if smaps is unavailable; when present it should be > 0 here.
            if (anon_huge >= 0)
                EXPECT_GT(anon_huge, 0) << "arena range is not THP-backed";
        }
    }
}
