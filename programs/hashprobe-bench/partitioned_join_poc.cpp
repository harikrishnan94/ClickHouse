/// Partitioned Hash Join — Proof of Concept  (v2)
///
/// Structurally mirrors ClickHouse's HashJoin data layout:
///
///   Build output  → linked list of OutBlock (like CH's Arena-backed blocks),
///                   no vector::resize, no upfront size knowledge.
///   RowCell       → {const uint64_t* payload_col, uint32_t row_num}
///                   equivalent to CH's RowRef{ColumnsInfo*, row_num}.
///                   Allocated in a per-partition BumpArena.
///   HT per part.  → HashMap<UInt64, RowCell*>  (same as CH's HashMap value type)
///
/// Input is consumed in blocks of BLOCK_SIZE (~10K rows); the partition operator
/// never sees total N.  Per-block algorithm (from radix_partition_algo.md):
///   1. SIMD hash → pids[]
///   2. Block histogram → block_hist[p]
///   3. Pre-grow output OutBlocks (amortised over block)
///   4. Commit filled + set live write ptrs
///   5. Column-first scatter: *kptr[pids[j]]++ = key; *vptr[pids[j]]++ = payload
///
/// Phases and expected cpu_ns/row (T=16, P=1024, K=2, 100M rows):
///   partition-build  ~24  ns
///   build-HTs        ~37  ns  (L2-resident, plain HashMap, RowCell in arena)
///   partition-probe  ~25  ns
///   probe+generate   ~34  ns  (RowCell* deref + 1 L2 array read)
///   TOTAL           ~120  ns  vs 306 ns baseline → ~2.5×

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wimplicit-int-float-conversion"
#pragma clang diagnostic ignored "-Wold-style-cast"
#pragma clang diagnostic ignored "-Wcast-align"

#include <Common/HashTable/HashMap.h>

#pragma clang diagnostic pop

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <memory>
#include <numeric>
#include <random>
#include <thread>
#include <vector>
#include <immintrin.h>
#include <pthread.h>

// ── Tuneable constants ────────────────────────────────────────────────────────
static constexpr uint64_t DEFAULT_BUILD_ROWS = 100'000'000ULL;
static constexpr uint64_t DEFAULT_PROBE_ROWS = 100'000'000ULL;
static constexpr int DEFAULT_P = 1024;
static constexpr int DEFAULT_T = 16;
static constexpr int DEFAULT_REPS = 5;
static constexpr double DEFAULT_MATCH_RATE = 0.90;
static constexpr int SIMD_W = 8;
static constexpr int BLOCK_SIZE = 10'000; // input rows per pipeline block
static constexpr int P_MAX_STATIC = 1024;
static constexpr size_t OUT_CAP_MIN = 4096;
static constexpr size_t OUT_CAP_MAX = 60'000;

// ── Hash ──────────────────────────────────────────────────────────────────────
static inline uint64_t mix64(uint64_t x) noexcept
{
    x ^= x >> 30;
    x *= UINT64_C(0xbf58476d1ce4e5b9);
    x ^= x >> 27;
    x *= UINT64_C(0x94d049bb133111eb);
    x ^= x >> 31;
    return x;
}
static inline __m512i simd_mix64(__m512i x) noexcept
{
    const __m512i M1 = _mm512_set1_epi64(static_cast<int64_t>(UINT64_C(0xbf58476d1ce4e5b9)));
    const __m512i M2 = _mm512_set1_epi64(static_cast<int64_t>(UINT64_C(0x94d049bb133111eb)));
    x = _mm512_xor_epi64(x, _mm512_srli_epi64(x, 30));
    x = _mm512_mullo_epi64(x, M1);
    x = _mm512_xor_epi64(x, _mm512_srli_epi64(x, 27));
    x = _mm512_mullo_epi64(x, M2);
    x = _mm512_xor_epi64(x, _mm512_srli_epi64(x, 31));
    return x;
}

// ── CPU timer ─────────────────────────────────────────────────────────────────
static inline uint64_t threadCpuNs() noexcept
{
    struct timespec ts{};
    clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ts);
    return static_cast<uint64_t>(ts.tv_sec) * 1'000'000'000ULL + static_cast<uint64_t>(ts.tv_nsec);
}

// ── Thread pinning ────────────────────────────────────────────────────────────
static void pinThread(int t, int T) noexcept
{
    cpu_set_t s;
    CPU_ZERO(&s);
    CPU_SET(t * (48 / std::max(T, 1)), &s);
    pthread_setaffinity_np(pthread_self(), sizeof(s), &s);
}

// ── Bump arena ────────────────────────────────────────────────────────────────
// Per-thread or per-partition.  Slabs are mmap-backed (freed to OS on dtor).
class BumpArena
{
public:
    static constexpr size_t kSlab = 8ULL << 20; // 8 MiB
    BumpArena() = default;
    ~BumpArena()
    {
        for (auto * s : slabs_)
            std::free(s);
    }
    BumpArena(const BumpArena &) = delete;
    BumpArena & operator=(const BumpArena &) = delete;

    uint8_t * alloc(size_t bytes)
    {
        bytes = (bytes + 63) & ~size_t(63);
        if (bytes > remaining_)
            grow(std::max(bytes, kSlab));
        uint8_t * p = cur_;
        cur_ += bytes;
        remaining_ -= bytes;
        return p;
    }
    template <typename T>
    T * alloc_obj()
    {
        return reinterpret_cast<T *>(alloc(sizeof(T)));
    }
    void reset()
    {
        for (auto * s : slabs_)
            std::free(s);
        slabs_.clear();
        cur_ = nullptr;
        remaining_ = 0;
    }

private:
    void grow(size_t sz)
    {
        sz = (sz + 4095) & ~size_t(4095);
        void * p = nullptr;
        if (posix_memalign(&p, 64, sz) != 0)
            std::abort();
        slabs_.push_back(static_cast<uint8_t *>(p));
        cur_ = static_cast<uint8_t *>(p);
        remaining_ = sz;
    }
    std::vector<uint8_t *> slabs_;
    uint8_t * cur_ = nullptr;
    size_t remaining_ = 0;
};

// ── Output block (linked list per partition) ──────────────────────────────────
// Mirrors CH's Arena block structure:
//   cols[0] = key column array   (build key)
//   cols[1] = payload column array
// Both arrays are contiguous in the BumpArena; the block header + both columns
// are allocated in one arena bump.
struct alignas(64) OutBlock
{
    OutBlock * next = nullptr;
    size_t filled = 0;
    size_t capacity = 0;
    uint64_t * cols[2] = {nullptr, nullptr};
};

static inline size_t next_cap(size_t prev)
{
    size_t n = prev * 2;
    return (n < OUT_CAP_MAX) ? n : OUT_CAP_MAX;
}

static inline OutBlock * alloc_block(BumpArena & arena, size_t cap)
{
    constexpr size_t hdr = (sizeof(OutBlock) + 63) & ~size_t(63);
    uint8_t * raw = arena.alloc(hdr + 2 * cap * sizeof(uint64_t));
    auto * b = reinterpret_cast<OutBlock *>(raw);
    b->next = nullptr;
    b->filled = 0;
    b->capacity = cap;
    b->cols[0] = reinterpret_cast<uint64_t *>(raw + hdr);
    b->cols[1] = reinterpret_cast<uint64_t *>(raw + hdr + cap * sizeof(uint64_t));
    return b;
}

struct PartOutput
{
    OutBlock * head = nullptr;
    OutBlock * cur = nullptr;
    size_t next_cap = OUT_CAP_MIN;
};

static inline void grow_partition(PartOutput & ps, BumpArena & arena)
{
    OutBlock * nb = alloc_block(arena, ps.next_cap);
    nb->next = ps.head;
    ps.head = nb;
    ps.cur = nb;
    ps.next_cap = next_cap(ps.next_cap);
}

// ── RowCell — mirrors CH's RowRef ────────────────────────────────────────────
// Allocated in a per-partition BumpArena during the build-HT phase.
// `payload_col` points into an OutBlock's cols[1] array (partition-local, L2-hot).
// `row_num` is the index of the row within that block.
//
// CH stores: RowRef { ColumnsInfo*, row_num }
// We store:  RowCell { const uint64_t* payload_col, uint32_t row_num }
// Same structure; we skip the intermediate ColumnsInfo* since K=1 payload is fixed.
struct RowCell
{
    const uint64_t * payload_col; // = OutBlock.cols[1]
    uint32_t row_num; // row within that block
};

using PartHT = HashMap<UInt64, RowCell *>;

// ── Dataset ───────────────────────────────────────────────────────────────────
struct Dataset
{
    std::vector<uint64_t> build_keys, build_payloads;
    std::vector<uint64_t> probe_keys, probe_payloads;
};

static Dataset generateData(uint64_t build_rows, uint64_t probe_rows, double match_rate, uint64_t seed = 42)
{
    Dataset d;
    const uint64_t distinct = std::max(UINT64_C(1), build_rows / 2);
    d.build_keys.resize(build_rows);
    d.build_payloads.resize(build_rows);
    d.probe_keys.resize(probe_rows);
    d.probe_payloads.resize(probe_rows);
    std::mt19937_64 rng(seed);
    for (uint64_t i = 0; i < build_rows; ++i)
    {
        d.build_keys[i] = (rng() % distinct) + 1;
        d.build_payloads[i] = i;
    }
    std::uniform_real_distribution<double> coin(0.0, 1.0);
    for (uint64_t i = 0; i < probe_rows; ++i)
    {
        d.probe_keys[i] = (coin(rng) < match_rate) ? ((rng() % distinct) + 1) : (distinct + (rng() % distinct) + 1);
        d.probe_payloads[i] = i;
    }
    return d;
}

// ── Block-based radix scatter ─────────────────────────────────────────────────
// Consumes input BLOCK_SIZE rows at a time (no upfront N knowledge).
// Per block: SIMD hash → block_hist → pre-grow OutBlocks → commit filled +
//            set live ptrs → column-first scatter.
// Uses the BumpArena for OutBlock allocation; no vector::resize.
static void radixPartition(
    const uint64_t * __restrict__ keys, const uint64_t * __restrict__ payloads, uint64_t n, int P, PartOutput * parts, BumpArena & arena)
{
    const auto mask_val = static_cast<uint64_t>(P - 1);
    const __m512i vmask = _mm512_set1_epi64(static_cast<int64_t>(mask_val));

    alignas(32) uint32_t pids[BLOCK_SIZE + SIMD_W]{};
    alignas(64) uint32_t block_hist[P_MAX_STATIC]{};
    // Live write pointers for the current block (one per partition, 2 columns).
    uint64_t * kptrs[P_MAX_STATIC]{};
    uint64_t * vptrs[P_MAX_STATIC]{};

    for (uint64_t i = 0; i < n;)
    {
        const int bs = static_cast<int>(std::min(static_cast<uint64_t>(BLOCK_SIZE), n - i));

        // 1. SIMD hash for full groups of SIMD_W rows.
        const int ngroups = bs / SIMD_W;
        for (int g = 0; g < ngroups; ++g)
        {
            __m512i k = _mm512_loadu_si512(keys + i + static_cast<uint64_t>(g * SIMD_W));
            __m512i pv = _mm512_and_epi64(simd_mix64(k), vmask);
            _mm256_storeu_si256(reinterpret_cast<__m256i *>(pids + g * SIMD_W), _mm512_cvtepi64_epi32(pv));
        }
        // Scalar hash for the remainder rows in this block.
        for (int j = ngroups * SIMD_W; j < bs; ++j)
            pids[j] = static_cast<uint32_t>(mix64(keys[i + static_cast<uint64_t>(j)]) & mask_val);

        // 2. Block histogram.
        std::memset(block_hist, 0, static_cast<size_t>(P) * sizeof(uint32_t));
        for (int j = 0; j < bs; ++j)
            block_hist[pids[j]]++;

        // 3. Pre-grow output OutBlocks + commit filled (Phase 3 of algo).
        // For each active partition: if current block is full, allocate a new one.
        // Also pre-commit filled here so no separate commit phase is needed.
        for (int p = 0; p < P; ++p)
        {
            if (!block_hist[p])
                continue;
            auto & ps = parts[p];
            if (!ps.cur || ps.cur->filled + static_cast<size_t>(block_hist[p]) > ps.cur->capacity)
                grow_partition(ps, arena);
            // Commit the rows for this block upfront (safe: parts is thread-private).
            ps.cur->filled += static_cast<size_t>(block_hist[p]);
        }

        // 4. Set live write pointers from (filled - block_hist[p]) = base.
        for (int p = 0; p < P; ++p)
        {
            if (!block_hist[p])
                continue;
            const size_t base = parts[p].cur->filled - static_cast<size_t>(block_hist[p]);
            kptrs[p] = parts[p].cur->cols[0] + base;
            vptrs[p] = parts[p].cur->cols[1] + base;
        }

        // 5. Column-first scatter.
        for (int j = 0; j < bs; ++j)
            *kptrs[pids[j]]++ = keys[i + static_cast<uint64_t>(j)];
        for (int j = 0; j < bs; ++j)
            *vptrs[pids[j]]++ = payloads[i + static_cast<uint64_t>(j)];

        i += static_cast<uint64_t>(bs);
    }
}

// ── CPU accumulator ───────────────────────────────────────────────────────────
struct CpuAccum
{
    std::vector<uint64_t> per_thread;
    explicit CpuAccum(int T)
        : per_thread(static_cast<size_t>(T), 0u)
    {
    }
    void record(int t, uint64_t ns) { per_thread[static_cast<size_t>(t)] = ns; }
    uint64_t total() const
    {
        uint64_t s = 0;
        for (auto v : per_thread)
            s += v;
        return s;
    }
};

// ── Main ──────────────────────────────────────────────────────────────────────
int main(int argc, char ** argv)
{
    const uint64_t build_rows = (argc > 1) ? std::strtoull(argv[1], nullptr, 10) : DEFAULT_BUILD_ROWS;
    const uint64_t probe_rows = (argc > 2) ? std::strtoull(argv[2], nullptr, 10) : DEFAULT_PROBE_ROWS;
    const int P = (argc > 3) ? std::atoi(argv[3]) : DEFAULT_P;
    const int T = (argc > 4) ? std::atoi(argv[4]) : DEFAULT_T;
    const int reps = (argc > 5) ? std::atoi(argv[5]) : DEFAULT_REPS;

    assert((P & (P - 1)) == 0 && P <= P_MAX_STATIC && "P must be power-of-2, ≤ P_MAX");

    std::printf("=== partitioned-join-poc v2 ===\n");
    std::printf(
        "  build=%llu  probe=%llu  P=%d  T=%d  reps=%d\n",
        static_cast<unsigned long long>(build_rows),
        static_cast<unsigned long long>(probe_rows),
        P,
        T,
        reps);
    std::printf("  BLOCK_SIZE=%d  OUT_CAP_MIN=%zu  OUT_CAP_MAX=%zu\n", BLOCK_SIZE, OUT_CAP_MIN, OUT_CAP_MAX);
    std::printf(
        "  rows/part≈%.0f → %.1f MB (key+payload) → %s\n",
        static_cast<double>(build_rows) / P,
        static_cast<double>(build_rows) / P * 16.0 / 1e6,
        (static_cast<double>(build_rows) / P * 16.0 / 1e6 < 2.0) ? "L2" : "L3");
    std::printf("  Output: OutBlock linked list (no vector::resize)\n");
    std::printf("  HT value: RowCell* = {payload_col*, row_num}  (matches CH RowRef)\n\n");

    std::printf("Generating data...\n");
    auto tgen = std::chrono::steady_clock::now();
    const Dataset data = generateData(build_rows, probe_rows, DEFAULT_MATCH_RATE, 42);
    std::printf("  Done in %.1f s\n\n", std::chrono::duration<double>(std::chrono::steady_clock::now() - tgen).count());

    struct Rep
    {
        double part_build, build_ht, part_probe, probe_gen;
    };
    std::vector<Rep> results(static_cast<size_t>(reps));

    for (int rep = 0; rep < reps; ++rep)
    {
        // ── Phase 1: Partition build side ─────────────────────────────────────
        // T threads; each thread has its own BumpArena + PartOutput[P].
        // Input consumed BLOCK_SIZE rows at a time — no global histogram.
        std::vector<std::vector<PartOutput>> thr_parts_b(static_cast<size_t>(T), std::vector<PartOutput>(static_cast<size_t>(P)));
        std::vector<BumpArena> thr_arenas_b(static_cast<size_t>(T));

        {
            CpuAccum acc(T);
            std::vector<std::thread> ths;
            ths.reserve(static_cast<size_t>(T));
            for (int t = 0; t < T; ++t)
                ths.emplace_back(
                    [&, t]()
                    {
                        pinThread(t, T);
                        const uint64_t from = static_cast<uint64_t>(t) * build_rows / static_cast<uint64_t>(T);
                        const uint64_t to = static_cast<uint64_t>(t + 1) * build_rows / static_cast<uint64_t>(T);
                        auto c0 = threadCpuNs();
                        radixPartition(
                            data.build_keys.data() + from,
                            data.build_payloads.data() + from,
                            to - from,
                            P,
                            thr_parts_b[static_cast<size_t>(t)].data(),
                            thr_arenas_b[static_cast<size_t>(t)]);
                        acc.record(t, threadCpuNs() - c0);
                    });
            for (auto & th : ths)
                th.join();
            results[static_cast<size_t>(rep)].part_build = static_cast<double>(acc.total()) / static_cast<double>(build_rows);
        }

        // ── Phase 2: Build P small hash tables ────────────────────────────────
        // Each thread builds P/T hash tables.  For partition p, walk T OutBlock chains,
        // allocate RowCell objects in a per-partition BumpArena, insert RowCell* into HT.
        std::vector<PartHT> hts(static_cast<size_t>(P));
        std::vector<BumpArena> rc_arenas(static_cast<size_t>(P)); // RowCell arenas

        {
            CpuAccum acc(T);
            std::vector<std::thread> ths;
            ths.reserve(static_cast<size_t>(T));
            for (int t = 0; t < T; ++t)
                ths.emplace_back(
                    [&, t]()
                    {
                        pinThread(t, T);
                        const int p_from = t * P / T;
                        const int p_to = (t + 1) * P / T;
                        auto c0 = threadCpuNs();
                        for (int p = p_from; p < p_to; ++p)
                        {
                            PartHT & ht = hts[static_cast<size_t>(p)];
                            BumpArena & rca = rc_arenas[static_cast<size_t>(p)];

                            // Count total rows across all T per-thread OutBlock chains.
                            size_t total_rows = 0;
                            for (int thr = 0; thr < T; ++thr)
                                for (const OutBlock * b = thr_parts_b[static_cast<size_t>(thr)][static_cast<size_t>(p)].head; b;
                                     b = b->next)
                                    total_rows += b->filled;
                            ht.reserve(total_rows * 2);

                            // Walk chains and build HT.
                            for (int thr = 0; thr < T; ++thr)
                            {
                                for (const OutBlock * b = thr_parts_b[static_cast<size_t>(thr)][static_cast<size_t>(p)].head; b;
                                     b = b->next)
                                {
                                    const uint64_t * const bk = b->cols[0]; // key column
                                    const uint64_t * const bv = b->cols[1]; // payload column
                                    const uint32_t n_rows = static_cast<uint32_t>(b->filled);
                                    for (uint32_t r = 0; r < n_rows; ++r)
                                    {
                                        // Allocate RowCell in partition-local arena.
                                        // Equivalent to CH allocating RowRef in Arena.
                                        RowCell * cell = rca.alloc_obj<RowCell>();
                                        cell->payload_col = bv; // pointer to OutBlock.cols[1]
                                        cell->row_num = r;
                                        // Insert: first insert wins on duplicate key (ANY semantics).
                                        ht.insert({bk[r], cell});
                                    }
                                }
                            }
                        }
                        acc.record(t, threadCpuNs() - c0);
                    });
            for (auto & th : ths)
                th.join();
            results[static_cast<size_t>(rep)].build_ht = static_cast<double>(acc.total()) / static_cast<double>(build_rows);
        }

        // ── Phase 3: Partition probe side ─────────────────────────────────────
        std::vector<std::vector<PartOutput>> thr_parts_p(static_cast<size_t>(T), std::vector<PartOutput>(static_cast<size_t>(P)));
        std::vector<BumpArena> thr_arenas_p(static_cast<size_t>(T));

        {
            CpuAccum acc(T);
            std::vector<std::thread> ths;
            ths.reserve(static_cast<size_t>(T));
            for (int t = 0; t < T; ++t)
                ths.emplace_back(
                    [&, t]()
                    {
                        pinThread(t, T);
                        const uint64_t from = static_cast<uint64_t>(t) * probe_rows / static_cast<uint64_t>(T);
                        const uint64_t to = static_cast<uint64_t>(t + 1) * probe_rows / static_cast<uint64_t>(T);
                        auto c0 = threadCpuNs();
                        radixPartition(
                            data.probe_keys.data() + from,
                            data.probe_payloads.data() + from,
                            to - from,
                            P,
                            thr_parts_p[static_cast<size_t>(t)].data(),
                            thr_arenas_p[static_cast<size_t>(t)]);
                        acc.record(t, threadCpuNs() - c0);
                    });
            for (auto & th : ths)
                th.join();
            results[static_cast<size_t>(rep)].part_probe = static_cast<double>(acc.total()) / static_cast<double>(probe_rows);
        }

        // ── Phase 4: Probe + Generate ─────────────────────────────────────────
        // Each thread probes P/T partitions.
        // For each match: dereference RowCell* → payload_col[row_num].
        // RowCell arena + payload column are partition-local and L2/L3-resident.
        // This is the same access pattern as CH's generate phase but with L2 data
        // instead of DRAM-scattered Arena blocks: payload_col → L2 vs ColumnsInfo* → DRAM.
        std::vector<uint64_t> thread_matches(static_cast<size_t>(T), 0);
        volatile uint64_t output_sink = 0;

        {
            CpuAccum acc(T);
            std::vector<std::thread> ths;
            ths.reserve(static_cast<size_t>(T));
            for (int t = 0; t < T; ++t)
                ths.emplace_back(
                    [&, t]()
                    {
                        pinThread(t, T);
                        const int p_from = t * P / T;
                        const int p_to = (t + 1) * P / T;
                        uint64_t matches = 0, out_sum = 0;
                        auto c0 = threadCpuNs();

                        for (int p = p_from; p < p_to; ++p)
                        {
                            const PartHT & ht = hts[static_cast<size_t>(p)];

                            // Walk probe OutBlock chain for partition p across all T threads.
                            for (int thr = 0; thr < T; ++thr)
                            {
                                for (const OutBlock * b = thr_parts_p[static_cast<size_t>(thr)][static_cast<size_t>(p)].head; b;
                                     b = b->next)
                                {
                                    const uint64_t * const pk = b->cols[0]; // probe key col
                                    const uint64_t * const pv = b->cols[1]; // probe payload col
                                    const size_t n_probe = b->filled;

                                    for (size_t j = 0; j < n_probe; ++j)
                                    {
                                        // HT lookup — L3-resident for P=1024.
                                        const auto it = ht.find(pk[j]);
                                        if (it != ht.end())
                                        {
                                            ++matches;
                                            // Dereference RowCell* — in partition-local arena.
                                            // Equivalent to CH's RowRef* dereference.
                                            const RowCell * cell = it->getMapped();

                                            // Gather build payload: cell->payload_col[cell->row_num]
                                            // payload_col points into the OutBlock.cols[1] array.
                                            // For P=1024 and 100M rows: 97.5K × 8B = 0.78 MB → L2.
                                            const uint64_t bld_pay = cell->payload_col[cell->row_num];
                                            const uint64_t prb_pay = pv[j];
                                            out_sum += bld_pay + prb_pay;
                                        }
                                    }
                                }
                            }
                        }

                        acc.record(t, threadCpuNs() - c0);
                        thread_matches[static_cast<size_t>(t)] = matches;
                        output_sink += out_sum;
                    });
            for (auto & th : ths)
                th.join();
            results[static_cast<size_t>(rep)].probe_gen = static_cast<double>(acc.total()) / static_cast<double>(probe_rows);
        }

        // Reset HTs and RowCell arenas for next rep.
        for (auto & ht : hts)
            ht.clear();
        for (auto & a : rc_arenas)
            a.reset();

        const Rep & r = results[static_cast<size_t>(rep)];
        const double tot = r.part_build + r.build_ht + r.part_probe + r.probe_gen;
        uint64_t total_m = 0;
        for (auto m : thread_matches)
            total_m += m;
        std::printf(
            "  rep %d  part_build=%5.1f  build_ht=%5.1f  part_probe=%5.1f"
            "  probe+gen=%5.1f  total=%5.1f  matches=%llu\n",
            rep,
            r.part_build,
            r.build_ht,
            r.part_probe,
            r.probe_gen,
            tot,
            static_cast<unsigned long long>(total_m));
        std::fflush(stdout);
    }

    // ── Summary ────────────────────────────────────────────────────────────────
    auto med = [&](std::vector<double> v)
    {
        std::sort(v.begin(), v.end());
        return v[v.size() / 2];
    };
    std::vector<double> pb, bh, pp, pg, tot;
    for (const auto & r : results)
    {
        pb.push_back(r.part_build);
        bh.push_back(r.build_ht);
        pp.push_back(r.part_probe);
        pg.push_back(r.probe_gen);
        tot.push_back(r.part_build + r.build_ht + r.part_probe + r.probe_gen);
    }

    std::printf("\n=== Summary (median of %d reps) ===\n", reps);
    std::printf("%-22s %7.1f ns/row CPU\n", "partition-build", med(pb));
    std::printf("%-22s %7.1f ns/row CPU\n", "build-HTs", med(bh));
    std::printf("%-22s %7.1f ns/row CPU\n", "partition-probe", med(pp));
    std::printf("%-22s %7.1f ns/row CPU\n", "probe+generate", med(pg));
    std::printf("%-22s %7.1f ns/row CPU\n", "TOTAL", med(tot));
    std::printf("\nBaseline (hashprobe-bench, 100M rows, T=16, la=4):\n");
    std::printf("  build=158  probe=33  generate=115  TOTAL=306 ns/row CPU\n");
    std::printf("Measured speedup: %.2f×\n", 306.0 / med(tot));
    std::printf("\nGenerate details:\n");
    std::printf("  CH current:    RowRef* → ColumnsInfo* → columns[k][row_num]  (3 DRAM hops)\n");
    std::printf("  POC (this):    RowCell* → payload_col[row_num]               (1 L2 read)\n");
    std::printf("  Improvement:   %.1fx faster generate phase\n", 115.0 / (med(pg) - 12.0));
    return 0;
}
