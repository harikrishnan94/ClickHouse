/// P1 de-risk benchmark of the "Key-Only Scatter + Partitioned HT Build" (AHJ Mode 2) plan.
///
/// Measures the BUILD PHASE ONLY, end to end, over identical synthetic input (fixed-width UInt64
/// keys pre-split into 65536-row blocks) for three competitors:
///
///  - BM_build_baseline: a `parallel_hash`-shaped build modeled on `ConcurrentHashJoin`
///    (src/Interpreters/ConcurrentHashJoin.cpp): slots = toPowerOfTwo(min(threads, 256)); per block
///    a first map-hash pass feeding `hashToSelector` whose result is then discarded; zero-copy
///    dispatch into per-slot row-index lists (`scatterBlocksWithSelector` shape); per-slot mutex
///    with try-lock work stealing and yield; inserts into real two-level maps
///    (`TwoLevelHashMap<UInt64, UInt64, HashCRC32<UInt64>>`, the `two_level_key64` `MapsOne` shape)
///    where the map hash is recomputed inside `emplace`; the adaptive look-ahead software prefetch
///    of `insertFromBlockImplTypeCase` once the slot map outgrows L2; and the per-inserted-block
///    accounting walks of `updateTotalRowsAndBytesUnlocked` (`getTotalRowCount` +
///    `getTotalByteCount`, each a 256-bucket sweep). At slots == 1 the real dispatch shortcut is
///    taken (no hash pass, no selector).
///
///  - BM_build_pipe_seq / BM_build_pipe_amac: the Mode-2 build pipeline: (1) per-lane fill —
///    `routeWord` per row, 2-byte route saved (top 16 bits), per-lane dense HLL updated;
///    (2) barrier — merge HLLs, pick partition bits by the L2 rule; (3) the 3-barrier cooperative
///    scatter of the key column plus an 8-byte locator column, pids derived from the saved routes
///    by shift/mask (MSB-first), on the imported ColumnsScatter Layer-0 kernels; (4) ONE contiguous
///    unzeroed HT allocation right before the leaf builds, sized from the HLL estimate clamped by
///    the histogram; (5) leaf inserts into real `HashMap<UInt64, UInt64, HashCRC32<UInt64>>` maps
///    carved from the slab by a fixed-region allocator (each worker zeroes exactly its leaf region
///    right before filling it), leaves claimed dynamically largest-first — with either a plain
///    sequential `emplace` loop (seq) or the facts-B AMAC ring of 32 slots (amac).
///
/// Duplicate keys append to a plain arena list through the same helper on all three sides
/// (symmetric treatment standing in for `RowRefList`). Both competitors must produce identical
/// totals (inserted rows, distinct keys, multiplicity-weighted key sum) — checked on the first
/// iteration of every cell against the shape's shared expected values and thrown on mismatch. The
/// pipeline additionally asserts exactly ONE hash-table slab allocation per build and zero heap
/// fallbacks out of it at these shapes.
///
/// Timing: `UseManualTime` with one build per iteration — the timed region is exactly the build
/// (fill through last leaf insert); input reset, map destruction and result verification are
/// untimed. Run with `--benchmark_repetitions=N` and read medians; per-stage wall times (pipeline)
/// and per-stage thread-time sums (baseline) are exported as counters, averaged over iterations.

#include <benchmark/benchmark.h>

#include <Columns/ColumnsScatter.h>
#include <Common/Allocator.h>
#include <Common/Arena.h>
#include <Common/HashTable/Hash.h>
#include <Common/HashTable/HashMap.h>
#include <Common/HashTable/HashTableAllocator.h>
#include <Common/HashTable/Prefetching.h>
#include <Common/HashTable/TwoLevelHashMap.h>
#include <Common/PODArray.h>

#include <base/defines.h>
#include <base/getL2CacheSize.h>
#include <base/types.h>

#include <pcg_random.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <barrier>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstring>
#include <deque>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace
{

using namespace DB;

constexpr size_t block_rows = 65536; /// upstream pipeline block granularity (~DEFAULT_BLOCK_SIZE)

using Clock = std::chrono::steady_clock;

double secondsBetween(Clock::time_point start, Clock::time_point end)
{
    return std::chrono::duration<double>(end - start).count();
}

/// ------------------------------------------------------------------------------------------------
/// Shared plumbing: fork-join pool (same shape as benchmark_columns_scatter's), locators, duplicate
/// lists, result totals.
/// ------------------------------------------------------------------------------------------------

class ForkJoinPool
{
public:
    explicit ForkJoinPool(size_t threads_) : threads(threads_), start_barrier(threads_ + 1), end_barrier(threads_ + 1)
    {
        for (size_t t = 0; t < threads; ++t)
            workers.emplace_back(
                [this, t]
                {
                    while (true)
                    {
                        start_barrier.arrive_and_wait();
                        if (stop.load(std::memory_order_acquire))
                            return;
                        job(t);
                        end_barrier.arrive_and_wait();
                    }
                });
    }

    ~ForkJoinPool()
    {
        stop.store(true, std::memory_order_release);
        start_barrier.arrive_and_wait();
        for (auto & worker : workers)
            worker.join();
    }

    void run(const std::function<void(size_t)> & job_)
    {
        job = job_;
        start_barrier.arrive_and_wait();
        end_barrier.arrive_and_wait();
    }

private:
    size_t threads;
    std::barrier<> start_barrier;
    std::barrier<> end_barrier;
    std::function<void(size_t)> job;
    std::atomic<bool> stop{false};
    std::vector<std::thread> workers;
};

/// The 8-byte row locator both competitors store as the mapped value — the stand-in for
/// RowRef{row_no, block_no}. Block counts here keep the top bit clear, which the duplicate-list
/// tag below relies on.
ALWAYS_INLINE UInt64 makeLocator(UInt64 block_no, UInt64 row_in_block)
{
    return (block_no << 32) | row_in_block;
}

/// Duplicate handling, symmetric on all sides (the RowRefList stand-in): the 8-byte mapped word
/// holds the locator inline for single-row keys and a tagged pointer to an arena list once a key
/// repeats. Locators and user-space pointers both fit in 63 bits.
struct DupNode
{
    UInt64 locator;
    DupNode * next;
};

constexpr UInt64 dup_list_tag = 1ULL << 63;

ALWAYS_INLINE void addRowToMapped(UInt64 & mapped, UInt64 locator, bool inserted, Arena & arena)
{
    if (inserted)
    {
        mapped = locator;
        return;
    }
    auto * node = reinterpret_cast<DupNode *>(arena.alignedAlloc(sizeof(DupNode), alignof(DupNode)));
    if (mapped & dup_list_tag)
    {
        node->locator = locator;
        node->next = reinterpret_cast<DupNode *>(mapped & ~dup_list_tag);
    }
    else
    {
        /// First duplicate of this key: move the inline locator into the list head.
        auto * head = reinterpret_cast<DupNode *>(arena.alignedAlloc(sizeof(DupNode), alignof(DupNode)));
        head->locator = mapped;
        head->next = nullptr;
        node->locator = locator;
        node->next = head;
    }
    mapped = reinterpret_cast<UInt64>(node) | dup_list_tag;
}

ALWAYS_INLINE UInt64 mappedRowCount(UInt64 mapped)
{
    if (!(mapped & dup_list_tag))
        return 1;
    UInt64 rows = 0;
    for (const auto * node = reinterpret_cast<const DupNode *>(mapped & ~dup_list_tag); node; node = node->next)
        ++rows;
    return rows;
}

/// Correctness fingerprint of a finished build; must agree between competitors.
struct BuildTotals
{
    UInt64 rows = 0;
    UInt64 distinct = 0;
    UInt64 weighted_key_sum = 0; /// sum of key * multiplicity, mod 2^64

    bool operator==(const BuildTotals &) const = default;
};

template <typename Map>
void accumulateMapTotals(const Map & map, BuildTotals & totals)
{
    for (const auto & cell : map)
    {
        const UInt64 rows = mappedRowCount(cell.getMapped());
        totals.rows += rows;
        totals.distinct += 1;
        totals.weighted_key_sum += cell.getKey() * rows;
    }
}

/// ------------------------------------------------------------------------------------------------
/// Synthetic dataset, shared by every competitor of one shape. Also holds the expected totals: the
/// first competitor to finish a build registers them, everyone else must match them exactly.
/// ------------------------------------------------------------------------------------------------

struct Dataset
{
    size_t rows;
    size_t rows_per_key;
    PaddedPODArray<UInt64> keys;
    std::optional<BuildTotals> expected;

    Dataset(size_t rows_, size_t rows_per_key_) : rows(rows_), rows_per_key(rows_per_key_)
    {
        if (rows % block_rows != 0 || rows % rows_per_key != 0)
            throw std::runtime_error("Dataset: rows must be divisible by the block size and the multiplicity");
        keys.resize(rows);
        pcg64 rng(20260717);
        /// Zero keys are avoided so that the map's zero-value special case never fires; the code
        /// under test still handles it.
        auto next_key = [&rng]
        {
            UInt64 key;
            do
                key = rng();
            while (key == 0);
            return key;
        };
        if (rows_per_key == 1)
        {
            for (auto & key : keys)
                key = next_key();
        }
        else
        {
            const size_t distinct = rows / rows_per_key;
            PODArray<UInt64> base(distinct);
            for (auto & key : base)
                key = next_key();
            for (size_t i = 0; i < rows; ++i)
                keys[i] = base[i / rows_per_key];
            std::shuffle(keys.begin(), keys.end(), rng);
        }
    }

    size_t numBlocks() const { return rows / block_rows; }
    const UInt64 * blockKeys(size_t block_no) const { return keys.data() + block_no * block_rows; }

    void checkTotals(const BuildTotals & totals, const char * who)
    {
        if (totals.rows != rows)
            throw std::runtime_error(std::string(who) + ": inserted-row count mismatch: got " + std::to_string(totals.rows)
                + ", expected " + std::to_string(rows));
        if (!expected)
        {
            expected = totals;
            return;
        }
        if (!(totals == *expected))
            throw std::runtime_error(std::string(who) + ": build totals mismatch vs the first competitor: distinct "
                + std::to_string(totals.distinct) + " vs " + std::to_string(expected->distinct) + ", weighted key sum "
                + std::to_string(totals.weighted_key_sum) + " vs " + std::to_string(expected->weighted_key_sum));
    }
};

std::shared_ptr<Dataset> getDataset(size_t rows, size_t rows_per_key)
{
    static std::map<std::pair<size_t, size_t>, std::shared_ptr<Dataset>> cache;
    auto & slot = cache[{rows, rows_per_key}];
    if (!slot)
        slot = std::make_shared<Dataset>(rows, rows_per_key);
    return slot;
}

/// ================================================================================================
/// BASELINE — the `parallel_hash`-shaped build
/// ================================================================================================

using SlotMap = TwoLevelHashMap<UInt64, UInt64, HashCRC32<UInt64>>; /// two_level_key64 with an 8-byte mapped value

UInt32 toPowerOfTwo(UInt32 x)
{
    if (x <= 1)
        return 1;
    return static_cast<UInt32>(1) << (32 - std::countl_zero(x - 1));
}

/// The insert loop of insertFromBlockImplTypeCase: adaptive look-ahead software prefetch (when the
/// map outgrew L2), map hash recomputed inside `emplace`, duplicates appended under the slot lock.
/// `get_row` mirrors `selectorIndexAt`'s two shapes: an index list (dispatched shard) or a
/// continuous range (the slots == 1 shortcut).
template <typename GetRow>
void insertRowsIntoSlot(
    SlotMap & map, const UInt64 * keys, UInt64 block_no, size_t rows, GetRow && get_row, Arena & arena, bool use_prefetch)
{
    PrefetchingHelper prefetching;
    size_t prefetch_look_ahead = PrefetchingHelper::getInitialLookAheadValue();
    for (size_t i = 0; i < rows; ++i)
    {
        if (use_prefetch)
        {
            if (i == PrefetchingHelper::iterationsToMeasure())
                prefetch_look_ahead = prefetching.calcPrefetchLookAhead();
            if (i + prefetch_look_ahead < rows)
                map.prefetch(keys[get_row(i + prefetch_look_ahead)]);
        }
        const size_t row = get_row(i);
        SlotMap::LookupResult it;
        bool inserted;
        map.emplace(keys[row], it, inserted);
        addRowToMapped(it->getMapped(), makeLocator(block_no, row), inserted, arena);
    }
}

class BaselineBench
{
public:
    BaselineBench(Dataset & data_, size_t threads_)
        : data(data_)
        , slots(toPowerOfTwo(static_cast<UInt32>(std::min<size_t>(threads_, 256))))
        , min_bytes_for_prefetch(std::max<size_t>(getL2CacheSize(), 1 << 20)) /// getMinBytesForPrefetchInJoin
        , mutexes(slots)
        , slot_states(slots)
        , pool(threads_)
    {
    }

    /// Untimed, mirroring the ConcurrentHashJoin constructor: per-slot HashJoin instances (maps)
    /// exist before the first block arrives.
    void resetIteration()
    {
        maps.clear();
        maps.resize(slots);
        arenas.clear();
        global_rows.store(0, std::memory_order_relaxed);
        global_bytes.store(0, std::memory_order_relaxed);
        for (size_t s = 0; s < slots; ++s)
        {
            maps[s] = std::make_unique<SlotMap>();
            arenas.emplace_back();
            slot_states[s].local_rows = 0;
            slot_states[s].local_bytes = maps[s]->getBufferSizeInBytes();
            global_bytes.fetch_add(slot_states[s].local_bytes, std::memory_order_relaxed);
        }
        next_block.store(0, std::memory_order_relaxed);
    }

    double runBuild()
    {
        const auto start = Clock::now();
        pool.run([this](size_t) { laneJob(); });
        const auto end = Clock::now();
        return secondsBetween(start, end);
    }

    void verify()
    {
        BuildTotals totals;
        for (const auto & map : maps)
            accumulateMapTotals(*map, totals);
        data.checkTotals(totals, "baseline");
        distinct = totals.distinct;
    }

    void exportCounters(benchmark::State & state, size_t iterations) const
    {
        const double to_mean_ms = 1e-6 / static_cast<double>(iterations);
        state.counters["slots"] = static_cast<double>(slots);
        state.counters["distinct_keys"] = static_cast<double>(distinct);
        state.counters["dispatch_thread_ms"] = static_cast<double>(dispatch_ns.load()) * to_mean_ms;
        state.counters["insert_thread_ms"] = static_cast<double>(insert_ns.load()) * to_mean_ms;
        state.counters["account_thread_ms"] = static_cast<double>(account_ns.load()) * to_mean_ms;
    }

private:
    struct SlotState
    {
        UInt64 local_rows = 0;
        UInt64 local_bytes = 0;
    };

    Dataset & data;
    size_t slots;
    size_t min_bytes_for_prefetch;
    std::vector<std::unique_ptr<SlotMap>> maps;
    std::deque<std::mutex> mutexes;
    std::deque<Arena> arenas;
    std::vector<SlotState> slot_states;
    std::atomic<UInt64> next_block{0};
    std::atomic<UInt64> global_rows{0};
    std::atomic<UInt64> global_bytes{0};
    /// Per-stage thread-time totals across all lanes and iterations (exported as per-iteration means).
    std::atomic<UInt64> dispatch_ns{0};
    std::atomic<UInt64> insert_ns{0};
    std::atomic<UInt64> account_ns{0};
    UInt64 distinct = 0;
    ForkJoinPool pool;

    /// The per-inserted-block accounting of updateTotalRowsAndBytesUnlocked: getTotalRowCount and
    /// getTotalByteCount, each a full 256-bucket sweep of the slot's two-level map, plus the global
    /// atomics. Called under the slot lock, exactly like the original.
    void updateTotalsUnlocked(size_t s)
    {
        SlotState & slot = slot_states[s];
        const UInt64 rows = maps[s]->size();
        const UInt64 rows_delta = rows - slot.local_rows;
        global_rows.fetch_add(rows_delta, std::memory_order_relaxed);
        slot.local_rows = rows;

        const UInt64 bytes = maps[s]->getBufferSizeInBytes();
        if (bytes >= slot.local_bytes)
            global_bytes.fetch_add(bytes - slot.local_bytes, std::memory_order_relaxed);
        else
            global_bytes.fetch_sub(slot.local_bytes - bytes, std::memory_order_relaxed);
        slot.local_bytes = bytes;
    }

    void laneJob()
    {
        /// Reused per lane; the original allocates fresh hash/selector arrays per block, so reuse
        /// slightly favors the baseline (conservative for the gate).
        PODArray<UInt64> hashes;
        PODArray<UInt64> selector;
        std::vector<PODArray<UInt64>> shard_rows(slots);
        std::vector<UInt8> shard_done(slots);
        UInt64 lane_dispatch_ns = 0;
        UInt64 lane_insert_ns = 0;
        UInt64 lane_account_ns = 0;

        const size_t num_blocks = data.numBlocks();
        while (true)
        {
            const UInt64 b = next_block.fetch_add(1, std::memory_order_relaxed);
            if (b >= num_blocks)
                break;
            const UInt64 * keys = data.blockKeys(b);

            if (slots == 1)
            {
                /// dispatchBlock's num_shards == 1 shortcut: no hash pass, no selector.
                std::lock_guard lock(mutexes[0]);
                const auto t0 = Clock::now();
                const bool use_prefetch = maps[0]->getBufferSizeInBytes() > min_bytes_for_prefetch;
                insertRowsIntoSlot(*maps[0], keys, b, block_rows, [](size_t i) { return i; }, arenas[0], use_prefetch);
                const auto t1 = Clock::now();
                updateTotalsUnlocked(0);
                const auto t2 = Clock::now();
                lane_insert_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
                lane_account_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
                continue;
            }

            /// Pass 1: the map hash per row, used only to compute the dispatch selector and then
            /// discarded (calculateHashes + hashToSelector); then the zero-copy per-shard row-index
            /// lists (scatterBlocksWithSelector).
            const auto t0 = Clock::now();
            hashes.resize(block_rows);
            for (size_t i = 0; i < block_rows; ++i)
                hashes[i] = HashCRC32<UInt64>()(keys[i]);
            selector.resize(block_rows);
            for (size_t i = 0; i < block_rows; ++i)
                selector[i] = SlotMap::getBucketFromHash(hashes[i]) & (slots - 1);
            for (size_t s = 0; s < slots; ++s)
            {
                shard_rows[s].clear();
                shard_rows[s].reserve(block_rows / slots + 1);
            }
            for (size_t i = 0; i < block_rows; ++i)
                shard_rows[selector[i]].push_back(i);
            const auto t1 = Clock::now();
            lane_dispatch_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();

            /// Try-lock work stealing over the slots, with yield when no slot is free.
            size_t blocks_left = 0;
            for (size_t s = 0; s < slots; ++s)
            {
                shard_done[s] = shard_rows[s].empty();
                blocks_left += !shard_done[s];
            }
            while (blocks_left > 0)
            {
                bool made_progress = false;
                for (size_t s = 0; s < slots; ++s)
                {
                    if (shard_done[s])
                        continue;
                    std::unique_lock lock(mutexes[s], std::try_to_lock);
                    if (!lock.owns_lock())
                        continue;
                    made_progress = true;
                    const auto t2 = Clock::now();
                    const bool use_prefetch = maps[s]->getBufferSizeInBytes() > min_bytes_for_prefetch;
                    const auto & rows = shard_rows[s];
                    insertRowsIntoSlot(
                        *maps[s], keys, b, rows.size(), [&rows](size_t i) { return rows[i]; }, arenas[s], use_prefetch);
                    const auto t3 = Clock::now();
                    updateTotalsUnlocked(s);
                    const auto t4 = Clock::now();
                    lane_insert_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(t3 - t2).count();
                    lane_account_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(t4 - t3).count();
                    shard_done[s] = 1;
                    --blocks_left;
                }
                if (!made_progress)
                    std::this_thread::yield();
            }
        }

        dispatch_ns.fetch_add(lane_dispatch_ns, std::memory_order_relaxed);
        insert_ns.fetch_add(lane_insert_ns, std::memory_order_relaxed);
        account_ns.fetch_add(lane_account_ns, std::memory_order_relaxed);
    }
};

/// ================================================================================================
/// PIPELINE — the Mode-2 build
/// ================================================================================================

/// Dense-only HyperLogLog per the facts-B math: 2^p one-byte registers fed a 32-bit hash (the route
/// word); top p bits pick the register, the remaining field gives rank = leftmost-set-bit position
/// + 1; estimate = alpha_m * m^2 / sum(2^-reg) with the linear-counting correction.
struct DenseHll
{
    static constexpr UInt32 precision = 13;
    static constexpr UInt32 register_count = 1u << precision;

    std::array<UInt8, register_count> registers;

    void reset() { registers.fill(0); }

    ALWAYS_INLINE void add(UInt32 hash)
    {
        const UInt32 index = hash >> (32 - precision);
        const UInt32 field = hash & ((1u << (32 - precision)) - 1);
        const UInt8 rank
            = field ? static_cast<UInt8>(std::countl_zero(field) - precision + 1) : static_cast<UInt8>(32 - precision + 1);
        registers[index] = std::max(registers[index], rank);
    }

    void merge(const DenseHll & other)
    {
        for (size_t i = 0; i < register_count; ++i)
            registers[i] = std::max(registers[i], other.registers[i]);
    }

    double estimate() const
    {
        static const std::array<double, 33> inverse_powers = []
        {
            std::array<double, 33> result{};
            for (size_t rank = 0; rank < result.size(); ++rank)
                result[rank] = std::ldexp(1.0, -static_cast<int>(rank));
            return result;
        }();
        const double m = register_count;
        const double alpha = register_count == 16 ? 0.673 : 0.7213 / (1.0 + 1.079 / m);
        double inverse_sum = 0;
        size_t zeros = 0;
        for (const UInt8 rank : registers)
        {
            inverse_sum += inverse_powers[rank];
            zeros += rank == 0;
        }
        const double raw = alpha * m * m / inverse_sum;
        if (raw <= 2.5 * m && zeros > 0)
            return m * std::log(m / static_cast<double>(zeros));
        return raw;
    }
};

/// HashTable-compatible allocator that carves each leaf map's buffer out of the single contiguous
/// slab (R1). The worker points `current_region` at its claimed leaf region right before
/// constructing the map; the carve zeroes exactly that region (the slab itself is allocated
/// unzeroed). Growth past the region (an HLL underestimate) falls back to the heap — correct,
/// counted, never silent. Statics are fine here: the benchmark builds one join at a time, and the
/// stage barriers order slab registration against the carves.
struct FixedRegionAllocator
{
    struct Region
    {
        char * ptr = nullptr;
        size_t bytes_left = 0;
    };

    static inline thread_local Region * current_region = nullptr;
    static inline char * slab_begin = nullptr;
    static inline char * slab_end = nullptr;
    static inline std::atomic<UInt64> region_carves{0};
    static inline std::atomic<UInt64> heap_fallbacks{0};

    static bool inSlab(const void * ptr) { return ptr >= slab_begin && ptr < slab_end; }

    void * alloc(size_t size)
    {
        if (Region * region = current_region; region && size <= region->bytes_left)
        {
            char * ptr = region->ptr;
            region->ptr += size;
            region->bytes_left -= size;
            /// The worker zeroes exactly its leaf region, right before the map fills it.
            memset(ptr, 0, size);
            region_carves.fetch_add(1, std::memory_order_relaxed);
            return ptr;
        }
        heap_fallbacks.fetch_add(1, std::memory_order_relaxed);
        return heap.alloc(size);
    }

    void free(void * buf, size_t size)
    {
        if (inSlab(buf))
            return; /// carved from the slab, which is released as a whole
        heap.free(buf, size);
    }

    void * realloc(void * buf, size_t old_size, size_t new_size)
    {
        if (!inSlab(buf))
            return heap.realloc(buf, old_size, new_size);
        /// Growth beyond the carved region: move to the heap (zero-filled by clear_memory).
        heap_fallbacks.fetch_add(1, std::memory_order_relaxed);
        void * new_buf = heap.alloc(new_size);
        memcpy(new_buf, buf, old_size);
        return new_buf;
    }

    HashTableAllocator heap;
};

/// The key64 MapsOne shape over the fixed-region allocator, with the protected internals the AMAC
/// insert ring needs (cell walk, size bump, growth) exposed to the policy below.
using LeafMapBase = HashMap<UInt64, UInt64, HashCRC32<UInt64>, HashTableGrowerWithPrecalculation<8>, FixedRegionAllocator>;

struct LeafMap : public LeafMapBase
{
    using Cell = LeafMapBase::cell_type;

    explicit LeafMap(size_t reserve_for_num_elements) : LeafMapBase(reserve_for_num_elements) { }

    Cell * cells() { return this->buf; }
    const auto & growerRef() const { return this->grower; }
    void bumpSize() { ++this->m_size; }
    bool overflowed() const { return this->grower.overflow(this->m_size); }
    void growOnce() { this->resize(); }
};

static_assert(sizeof(LeafMap::Cell) == 16);

/// Replicates the grower math of HashTable(reserve_for_num_elements) so the carved region size
/// matches the map's single buffer allocation exactly.
size_t leafBucketCount(size_t reserve)
{
    HashTableGrowerWithPrecalculation<8> grower;
    grower.set(reserve);
    return grower.bufSize();
}

const HashTableNoState leaf_cell_state{};

/// ---- AMAC insert ring (facts-B contract) -------------------------------------------------------

struct AmacSlot
{
    UInt64 pos = 0;
    UInt32 row = 0;
    UInt32 active = 0;
};

static_assert(sizeof(AmacSlot) == 16); /// slot minimalism: per-leaf invariants live in the policy

enum class AmacStep : UInt8
{
    Advance,
    Done,
    DoneGrew,
};

/// Build-insert policy. `start` computes the map hash (its latency overlaps the other slots'
/// outstanding misses) and issues the home-cell prefetch with write intent, locality 3. `step` is
/// the ONE fused read-then-act — the load-bearing correctness invariant: two in-flight rows with
/// the same key (or colliding on one cell) must never both observe an empty cell.
struct BuildInsertPolicy
{
    LeafMap & map;
    const UInt64 * keys;
    const UInt64 * locators;
    Arena & arena;
    UInt64 & grow_events;
    LeafMap::Cell * buf; /// hoisted; refreshed by grow

    BuildInsertPolicy(LeafMap & map_, const UInt64 * keys_, const UInt64 * locators_, Arena & arena_, UInt64 & grow_events_)
        : map(map_), keys(keys_), locators(locators_), arena(arena_), grow_events(grow_events_), buf(map_.cells())
    {
    }

    /// Seed a slot with a row: hash, home position, home prefetch. Returns false when the row was
    /// handled synchronously (the zero key lives in the map's dedicated zero cell) and the slot
    /// stays free — such rows never enter the ring, so re-seeding after growth cannot fail.
    ALWAYS_INLINE bool start(AmacSlot & slot, UInt32 row)
    {
        const UInt64 key = keys[row];
        if (unlikely(key == 0))
        {
            LeafMap::LookupResult it;
            bool inserted;
            map.emplace(key, it, inserted);
            addRowToMapped(it->getMapped(), locators[row], inserted, arena);
            return false;
        }
        const size_t hash = map.hash(key);
        slot.pos = map.growerRef().place(hash);
        slot.row = row;
        slot.active = 1;
        __builtin_prefetch(&buf[slot.pos], 1, 3);
        return true;
    }

    ALWAYS_INLINE AmacStep step(AmacSlot & slot)
    {
        LeafMap::Cell * cell = &buf[slot.pos];
        const UInt64 key = keys[slot.row];
        if (cell->isZero(leaf_cell_state))
        {
            /// Claim the empty cell: insert the key and the locator, exactly what emplaceNonZeroImpl
            /// plus the Inserter would do.
            new (cell) LeafMap::Cell(key, leaf_cell_state);
            cell->getMapped() = locators[slot.row];
            map.bumpSize();
            if (unlikely(map.overflowed()))
                return AmacStep::DoneGrew;
            return AmacStep::Done;
        }
        if (cell->keyEquals(key))
        {
            addRowToMapped(cell->getMapped(), locators[slot.row], false, arena);
            return AmacStep::Done;
        }
        slot.pos = map.growerRef().next(slot.pos);
        __builtin_prefetch(&buf[slot.pos], 1, 3);
        return AmacStep::Advance;
    }

    /// Map growth: the ring was drained by the driver; resize (rehashes the just-claimed row too),
    /// then refresh the hoisted buffer pointer.
    void grow()
    {
        ++grow_events;
        map.growOnce();
        buf = map.cells();
    }
};

constexpr UInt32 amac_ring_size = 32; /// power of two per facts-B; ~8-10 in-flight rows saturate the MSHRs

/// Growth is a cancellation point: in-flight positions index the old buffer, so collect the other
/// active rows, let the map resize, and re-seed them in place. The current slot's row is already
/// inserted and is rehashed by the resize.
template <typename Policy>
void drainRingAndGrow(Policy & policy, AmacSlot * ring, UInt32 skip)
{
    UInt32 pending_rows[amac_ring_size];
    UInt32 pending_count = 0;
    for (UInt32 j = 0; j < amac_ring_size; ++j)
    {
        if (j == skip || !ring[j].active)
            continue;
        pending_rows[pending_count++] = ring[j].row;
        ring[j].active = 0;
    }
    policy.grow();
    UInt32 k = 0;
    for (UInt32 j = 0; j < amac_ring_size && k < pending_count; ++j)
    {
        if (j == skip)
            continue;
        const bool started = policy.start(ring[j], pending_rows[k++]);
        chassert(started); /// zero-key rows never enter the ring
        (void)started;
    }
}

template <typename Policy>
void amacBuildRun(Policy & policy, UInt32 rows)
{
    AmacSlot ring[amac_ring_size]{};
    UInt32 next = 0;
    UInt32 active = 0;

    /// Prime the ring. After this loop either the ring is full or the rows are exhausted.
    for (UInt32 s = 0; s < amac_ring_size; ++s)
    {
        while (next < rows && !policy.start(ring[s], next++))
            ;
        active += ring[s].active;
    }

    /// Steady phase: while rows remain every slot is provably active, so sweep the ring with a
    /// plain array for, no per-visit active check (facts-B probe-loop structure).
    if (active == amac_ring_size)
    {
        bool full = true;
        while (full && next < rows)
        {
            for (UInt32 s = 0; s < amac_ring_size; ++s)
            {
                const AmacStep result = policy.step(ring[s]);
                if (result == AmacStep::Advance)
                    continue;
                if (result == AmacStep::DoneGrew)
                    drainRingAndGrow(policy, ring, s);
                ring[s].active = 0;
                while (next < rows && !policy.start(ring[s], next++))
                    ;
                if (!ring[s].active)
                {
                    --active;
                    full = false;
                }
            }
        }
    }

    /// Drain phase: no refills left; finish the in-flight rows.
    while (active > 0)
    {
        for (UInt32 s = 0; s < amac_ring_size; ++s)
        {
            if (!ring[s].active)
                continue;
            const AmacStep result = policy.step(ring[s]);
            if (result == AmacStep::Advance)
                continue;
            if (result == AmacStep::DoneGrew)
                drainRingAndGrow(policy, ring, s);
            ring[s].active = 0;
            --active;
        }
    }
}

/// ---- The pipeline itself -----------------------------------------------------------------------

class PipelineBench
{
public:
    PipelineBench(Dataset & data_, size_t threads_, bool use_amac_)
        : data(data_)
        , threads(threads_)
        , use_amac(use_amac_)
        , routes(data_.numBlocks())
        , hlls(threads_)
        , worker_scratch(threads_)
        , stage_sync(static_cast<std::ptrdiff_t>(threads_))
        , pool(threads_)
    {
        for (auto & block_routes : routes)
            block_routes.resize(block_rows);
        for (auto & scratch : worker_scratch)
        {
            scratch.pid_buf.resize(block_rows);
            scratch.loc_buf.resize(locator_piece_rows);
        }
    }

    ~PipelineBench() { releaseBuildState(); }

    void resetIteration()
    {
        releaseBuildState();
        arenas.clear();
        for (size_t t = 0; t < threads; ++t)
            arenas.emplace_back();
        next_fill_block.store(0, std::memory_order_relaxed);
        leaf_claim.store(0, std::memory_order_relaxed);
        filled_blocks.clear();
        accumulated_rows.store(0, std::memory_order_relaxed);
        grow_events_total = 0;
        carves_before = FixedRegionAllocator::region_carves.load(std::memory_order_relaxed);
        fallbacks_before = FixedRegionAllocator::heap_fallbacks.load(std::memory_order_relaxed);
        slab_allocations_this_build = 0;
    }

    double runBuild()
    {
        marks[0] = Clock::now();
        pool.run([this](size_t t) { buildJob(t); });
        marks[7] = Clock::now();

        const char * stage_names[] = {"fill_ms", "plan_ms", "hist_ms", "chunk_alloc_ms", "scatter_ms", "ht_alloc_ms", "insert_ms"};
        for (size_t stage = 0; stage < 7; ++stage)
            stage_ms_total[stage_names[stage]] += secondsBetween(marks[stage], marks[stage + 1]) * 1e3;
        for (const auto & scratch : worker_scratch)
        {
            grow_events_total += scratch.grow_events;
            stage_ms_total["leaf_setup_thread_ms"] += static_cast<double>(scratch.setup_ns) * 1e-6;
            stage_ms_total["leaf_insert_thread_ms"] += static_cast<double>(scratch.insert_ns) * 1e-6;
            stage_ms_total["leaf_release_thread_ms"] += static_cast<double>(scratch.release_ns) * 1e-6;
        }

        /// The one-allocation property (R1), asserted per build: exactly one slab allocation, every
        /// leaf buffer carved out of it, no heap fallbacks at these shapes.
        if (slab_allocations_this_build != 1)
            throw std::runtime_error("pipeline: expected exactly ONE hash-table allocation, got "
                + std::to_string(slab_allocations_this_build));
        const UInt64 carves = FixedRegionAllocator::region_carves.load(std::memory_order_relaxed) - carves_before;
        const UInt64 fallbacks = FixedRegionAllocator::heap_fallbacks.load(std::memory_order_relaxed) - fallbacks_before;
        if (carves != fanout)
            throw std::runtime_error("pipeline: expected " + std::to_string(fanout) + " leaf carves from the slab, got "
                + std::to_string(carves));
        if (fallbacks != 0)
            throw std::runtime_error("pipeline: unexpected heap fallback out of the contiguous slab ("
                + std::to_string(fallbacks) + " allocations)");

        return secondsBetween(marks[0], marks[7]);
    }

    void verify()
    {
        BuildTotals totals;
        for (size_t leaf = 0; leaf < fanout; ++leaf)
        {
            const LeafMap & map = *leaf_maps[leaf];
            accumulateMapTotals(map, totals);
            /// Scatter integrity: every key in a leaf must route there.
            for (const auto & cell : map)
            {
                const UInt32 route = ColumnsScatter::routeWord(cell.getKey()) >> 16;
                if ((route >> (16 - bits)) != leaf)
                    throw std::runtime_error("pipeline: key routed to the wrong leaf");
            }
        }
        data.checkTotals(totals, use_amac ? "pipe_amac" : "pipe_seq");
        distinct = totals.distinct;
        const double relative_error = std::abs(hll_estimate - static_cast<double>(distinct)) / static_cast<double>(distinct);
        if (relative_error > 0.05)
            throw std::runtime_error("pipeline: HLL estimate " + std::to_string(hll_estimate) + " is off by more than 5% vs "
                + std::to_string(distinct));
    }

    void exportCounters(benchmark::State & state, size_t iterations)
    {
        const double inv_iterations = 1.0 / static_cast<double>(iterations);
        for (const auto & [name, total] : stage_ms_total)
            state.counters[name] = total * inv_iterations;
        state.counters["bits"] = static_cast<double>(bits);
        state.counters["partitions"] = static_cast<double>(fanout);
        state.counters["ht_slab_mb"] = static_cast<double>(slab_bytes) / (1 << 20);
        state.counters["ht_allocations"] = 1;
        state.counters["grow_events"] = static_cast<double>(grow_events_total) * inv_iterations;
        state.counters["hll_estimate"] = hll_estimate;
        state.counters["distinct_keys"] = static_cast<double>(distinct);
    }

private:
    static constexpr size_t locator_piece_rows = 32768; /// locator synthesis scratch stays L2-resident

    struct WorkerScratch
    {
        PODArray<UInt16> pid_buf; /// per-chunk pids derived from the saved routes
        PODArray<UInt64> loc_buf; /// synthesized locator pieces
        PODArray<UInt32> hist_lanes;
        ColumnsScatter::ScatterScratch key_scratch;
        ColumnsScatter::ScatterScratch loc_scratch;
        UInt64 grow_events = 0;
        /// Thread-time split of the leaf-build stage, for the per-stage breakdown.
        UInt64 setup_ns = 0; /// region carve + memset + map construction
        UInt64 insert_ns = 0; /// the actual inserts
        UInt64 release_ns = 0; /// freeing the consumed scatter chunks
    };

    Dataset & data;
    size_t threads;
    bool use_amac;

    /// Fill-phase state (reused across iterations; the fill overwrites it).
    std::vector<PODArray<UInt16>> routes; /// 2 bytes per row (R4)
    std::vector<DenseHll> hlls;
    std::atomic<UInt64> next_fill_block{0};
    std::mutex fill_mutex;
    std::vector<UInt32> filled_blocks; /// models the cheap mutexed per-lane block append
    std::atomic<UInt64> accumulated_rows{0}; /// models the fill-phase size-limit accounting

    /// Plan (decided at the barrier).
    double hll_estimate = 0;
    UInt32 bits = 0;
    size_t fanout = 0;
    bool use_swwc = false;

    /// Scatter state.
    std::vector<UInt64> hist_total; /// per partition
    PODArray<UInt32> worker_hist; /// threads x fanout
    PODArray<UInt64> starts; /// fanout x threads: row start of worker w inside partition p
    std::vector<PaddedPODArray<char>> part_keys;
    std::vector<PaddedPODArray<char>> part_locs;
    std::vector<WorkerScratch> worker_scratch;

    /// Hash-table state.
    std::vector<UInt64> leaf_reserve;
    std::vector<UInt64> leaf_bytes;
    std::vector<UInt64> leaf_offset;
    std::vector<UInt32> leaf_order; /// largest first
    char * slab = nullptr;
    size_t slab_bytes = 0;
    UInt64 slab_allocations_this_build = 0;
    std::vector<std::unique_ptr<LeafMap>> leaf_maps;
    std::atomic<UInt32> leaf_claim{0};
    std::deque<Arena> arenas;

    /// Instrumentation.
    std::array<Clock::time_point, 8> marks;
    std::map<std::string, double> stage_ms_total;
    UInt64 grow_events_total = 0;
    UInt64 carves_before = 0;
    UInt64 fallbacks_before = 0;
    UInt64 distinct = 0;

    std::barrier<> stage_sync;
    ForkJoinPool pool;

    Allocator<false, false> slab_allocator; /// not zeroed, not pre-faulted (facts-B)

    void releaseBuildState()
    {
        leaf_maps.clear(); /// before the slab: destructors must still see the slab bounds
        if (slab)
        {
            slab_allocator.free(slab, slab_bytes);
            slab = nullptr;
            FixedRegionAllocator::slab_begin = nullptr;
            FixedRegionAllocator::slab_end = nullptr;
        }
        part_keys.clear();
        part_locs.clear();
    }

    void buildJob(size_t t)
    {
        fillLane(t);
        stage_sync.arrive_and_wait();
        if (t == 0)
        {
            marks[1] = Clock::now();
            planPartitions();
        }
        stage_sync.arrive_and_wait();
        if (t == 0)
            marks[2] = Clock::now();
        histogramWorker(t);
        stage_sync.arrive_and_wait();
        if (t == 0)
            marks[3] = Clock::now();
        prefixAndAllocateWorker(t);
        stage_sync.arrive_and_wait();
        if (t == 0)
            marks[4] = Clock::now();
        scatterWorker(t);
        stage_sync.arrive_and_wait();
        if (t == 0)
        {
            marks[5] = Clock::now();
            planAndAllocateHashTables();
        }
        stage_sync.arrive_and_wait();
        if (t == 0)
            marks[6] = Clock::now();
        leafBuildWorker(t);
    }

    /// Stage 1 — fill (R3, R4, R6): per lane, over the blocks as they arrive: one route word per
    /// row, the 2-byte route saved, the lane HLL updated; the block reference appended under a
    /// cheap mutex; accumulated size tracked for limit checks.
    void fillLane(size_t t)
    {
        DenseHll & hll = hlls[t];
        hll.reset();
        const size_t num_blocks = data.numBlocks();
        while (true)
        {
            const UInt64 b = next_fill_block.fetch_add(1, std::memory_order_relaxed);
            if (b >= num_blocks)
                break;
            const UInt64 * keys = data.blockKeys(b);
            UInt16 * block_routes = routes[b].data();
            for (size_t i = 0; i < block_rows; ++i)
            {
                const UInt32 word = ColumnsScatter::routeWord(keys[i]);
                block_routes[i] = static_cast<UInt16>(word >> 16);
                hll.add(word);
            }
            {
                std::lock_guard lock(fill_mutex);
                filled_blocks.push_back(static_cast<UInt32>(b));
            }
            accumulated_rows.fetch_add(block_rows, std::memory_order_relaxed);
        }
    }

    /// Stage 2 — the cheap barrier (R5): merge the lane HLLs, pick the partition bits by the L2
    /// rule, size the shared scatter state.
    void planPartitions()
    {
        DenseHll merged = hlls[0];
        for (size_t t = 1; t < threads; ++t)
            merged.merge(hlls[t]);
        hll_estimate = merged.estimate();
        const UInt64 estimate = std::max<UInt64>(1, static_cast<UInt64>(std::llround(hll_estimate)));

        /// bits = min b such that ceilPow2((U >> b) / LOAD) * sizeof(Cell) fits the leaf budget
        /// (LOAD = 0.5, the HashMap grower's max fill), floored for parallelism per facts-A.
        const size_t l2_bytes = std::max<size_t>(getL2CacheSize(), 1 << 20);
        const size_t leaf_budget = static_cast<size_t>(0.8 * static_cast<double>(l2_bytes));
        bits = 0;
        while (bits < 16 && std::bit_ceil(std::max<UInt64>(2 * (estimate >> bits), 1)) * sizeof(LeafMap::Cell) > leaf_budget)
            ++bits;
        const UInt32 parallelism_floor_bits = static_cast<UInt32>(std::bit_width(std::bit_ceil(threads) - 1));
        bits = std::max(bits, parallelism_floor_bits);
        /// This benchmark implements the single-pass 3-barrier driver only, so the per-pass fanout
        /// ceiling caps the bits; the gate shapes stay well below it (the plan's full cap is 16
        /// with multi-pass refine).
        bits = std::min(bits, static_cast<UInt32>(std::countr_zero(ColumnsScatter::MAX_FANOUT_PER_PASS)));

        fanout = 1uz << bits;
        use_swwc = fanout >= ColumnsScatter::SWWC_MIN_FANOUT && ColumnsScatter::widthSupportsSwwc(sizeof(UInt64));

        hist_total.assign(fanout, 0);
        worker_hist.resize(threads * fanout);
        starts.resize(fanout * threads);
        part_keys.resize(fanout);
        part_locs.resize(fanout);
        leaf_reserve.assign(fanout, 0);
        leaf_bytes.assign(fanout, 0);
        leaf_offset.assign(fanout, 0);
        leaf_order.resize(fanout);
        leaf_maps.clear();
        leaf_maps.resize(fanout);
    }

    /// Contiguous stripe of blocks owned by a worker across scatter barriers 1 and 3 (the
    /// histogram and the writes must agree on the per-worker row sets).
    std::pair<size_t, size_t> blockStripe(size_t t) const
    {
        const size_t num_blocks = data.numBlocks();
        return {t * num_blocks / threads, (t + 1) * num_blocks / threads};
    }

    ALWAYS_INLINE void convertRoutesToPids(const PODArray<UInt16> & block_routes, UInt16 * pids) const
    {
        const UInt32 shift = 16 - bits;
        for (size_t i = 0; i < block_rows; ++i)
            pids[i] = static_cast<UInt16>(block_routes[i] >> shift);
    }

    /// Scatter barrier 1: per-worker histograms over the chunk stripe, from the saved routes
    /// (interleaved lanes iff fanout <= HIST_INTERLEAVE_MAX_FANOUT).
    void histogramWorker(size_t t)
    {
        WorkerScratch & scratch = worker_scratch[t];
        UInt32 * hist = worker_hist.data() + t * fanout;
        memset(hist, 0, fanout * sizeof(UInt32));
        UInt32 * lanes = nullptr;
        if (fanout <= ColumnsScatter::HIST_INTERLEAVE_MAX_FANOUT)
        {
            scratch.hist_lanes.resize(4 * fanout);
            memset(scratch.hist_lanes.data(), 0, 4 * fanout * sizeof(UInt32));
            lanes = scratch.hist_lanes.data();
        }
        const auto [begin, end] = blockStripe(t);
        for (size_t b = begin; b < end; ++b)
        {
            convertRoutesToPids(routes[b], scratch.pid_buf.data());
            ColumnsScatter::histogramPidChunk(scratch.pid_buf.data(), block_rows, hist, lanes, fanout);
        }
        if (lanes)
            ColumnsScatter::reduceHistogramLanes(hist, lanes, fanout);
    }

    /// Scatter barrier 2: fused parallel prefix sum over the per-worker histograms plus ONE exact
    /// uninitialized allocation per (partition, column) — pages are first-touched by the scatter
    /// writes themselves.
    void prefixAndAllocateWorker(size_t t)
    {
        const size_t partitions_begin = t * fanout / threads;
        const size_t partitions_end = (t + 1) * fanout / threads;
        for (size_t p = partitions_begin; p < partitions_end; ++p)
        {
            UInt64 running = 0;
            for (size_t w = 0; w < threads; ++w)
            {
                starts[p * threads + w] = running;
                running += worker_hist[w * fanout + p];
            }
            hist_total[p] = running;
            part_keys[p].resize(running * sizeof(UInt64));
            part_locs[p].resize(running * sizeof(UInt64));
        }
    }

    /// Scatter barrier 3: batched scatter of the key column and the synthesized locator column,
    /// pids derived from the saved routes. Per-(column, partition) cursors persist across the whole
    /// stripe (the original's batches of whole chunks only bound when input chunks may be dropped —
    /// this benchmark keeps the inputs for the next iteration, see the P1 report), one drain at the
    /// end publishes the non-temporal stores.
    void scatterWorker(size_t t)
    {
        WorkerScratch & scratch = worker_scratch[t];
        scratch.key_scratch.init(fanout, use_swwc);
        scratch.loc_scratch.init(fanout, use_swwc);
        for (size_t p = 0; p < fanout; ++p)
        {
            const UInt64 start = starts[p * threads + t] * sizeof(UInt64);
            scratch.key_scratch.seed(p, part_keys[p].data() + start);
            scratch.loc_scratch.seed(p, part_locs[p].data() + start);
        }
        const auto [begin, end] = blockStripe(t);
        for (size_t b = begin; b < end; ++b)
        {
            convertRoutesToPids(routes[b], scratch.pid_buf.data());
            ColumnsScatter::scatterPidChunk(
                sizeof(UInt64),
                scratch.pid_buf.data(),
                reinterpret_cast<const char *>(data.blockKeys(b)),
                block_rows,
                use_swwc,
                scratch.key_scratch);
            /// The locator column does not exist upstream; synthesize it in L2-resident pieces.
            for (size_t offset = 0; offset < block_rows; offset += locator_piece_rows)
            {
                const size_t piece = std::min(locator_piece_rows, block_rows - offset);
                for (size_t j = 0; j < piece; ++j)
                    scratch.loc_buf[j] = makeLocator(b, offset + j);
                ColumnsScatter::scatterPidChunk(
                    sizeof(UInt64),
                    scratch.pid_buf.data() + offset,
                    reinterpret_cast<const char *>(scratch.loc_buf.data()),
                    piece,
                    use_swwc,
                    scratch.loc_scratch);
            }
        }
        scratch.key_scratch.drain();
        scratch.loc_scratch.drain();
    }

    /// Right before the leaf builds (R1): per-leaf reserves from the HLL estimate clamped by the
    /// histogram, exact bucket counts via the map's own grower math, 64-byte-aligned offsets, ONE
    /// contiguous unzeroed allocation, and the largest-first leaf order.
    void planAndAllocateHashTables()
    {
        constexpr double safety = 1.2; /// covers the HLL error (~1.15% at p = 13) and per-leaf spread
        const UInt64 per_leaf_estimate = std::max<UInt64>(1, static_cast<UInt64>(hll_estimate * safety / static_cast<double>(fanout)));
        UInt64 running = 0;
        for (size_t leaf = 0; leaf < fanout; ++leaf)
        {
            leaf_reserve[leaf] = std::clamp<UInt64>(per_leaf_estimate, 1, std::max<UInt64>(hist_total[leaf], 1));
            leaf_bytes[leaf] = leafBucketCount(leaf_reserve[leaf]) * sizeof(LeafMap::Cell);
            running = (running + ColumnsScatter::LINE_BYTES - 1) & ~static_cast<UInt64>(ColumnsScatter::LINE_BYTES - 1);
            leaf_offset[leaf] = running;
            running += leaf_bytes[leaf];
        }
        slab_bytes = running;
        slab = reinterpret_cast<char *>(slab_allocator.alloc(slab_bytes, ColumnsScatter::LINE_BYTES));
        ++slab_allocations_this_build;
        FixedRegionAllocator::slab_begin = slab;
        FixedRegionAllocator::slab_end = slab + slab_bytes;

        for (size_t leaf = 0; leaf < fanout; ++leaf)
            leaf_order[leaf] = static_cast<UInt32>(leaf);
        std::sort(leaf_order.begin(), leaf_order.end(), [this](UInt32 a, UInt32 b) { return hist_total[a] > hist_total[b]; });
    }

    /// Leaf builds: dynamically claimed largest-first (G6). Each worker carves the leaf region
    /// (zeroing it just before the fill), constructs the map with the exact reserve, inserts the
    /// leaf's compact scattered chunk sequentially or through the AMAC ring, and releases the
    /// consumed chunk.
    void leafBuildWorker(size_t t)
    {
        Arena & arena = arenas[t];
        WorkerScratch & scratch = worker_scratch[t];
        scratch.grow_events = 0;
        scratch.setup_ns = 0;
        scratch.insert_ns = 0;
        scratch.release_ns = 0;
        while (true)
        {
            const UInt32 claim = leaf_claim.fetch_add(1, std::memory_order_relaxed);
            if (claim >= fanout)
                break;
            const UInt32 leaf = leaf_order[claim];
            const auto t0 = Clock::now();
            FixedRegionAllocator::Region region{slab + leaf_offset[leaf], leaf_bytes[leaf]};
            FixedRegionAllocator::current_region = &region;
            leaf_maps[leaf] = std::make_unique<LeafMap>(leaf_reserve[leaf]);
            FixedRegionAllocator::current_region = nullptr;
            const auto t1 = Clock::now();

            LeafMap & map = *leaf_maps[leaf];
            const auto * keys = reinterpret_cast<const UInt64 *>(part_keys[leaf].data());
            const auto * locators = reinterpret_cast<const UInt64 *>(part_locs[leaf].data());
            const UInt64 rows = hist_total[leaf];
            if (use_amac)
            {
                BuildInsertPolicy policy(map, keys, locators, arena, scratch.grow_events);
                amacBuildRun(policy, static_cast<UInt32>(rows));
            }
            else
            {
                for (UInt64 i = 0; i < rows; ++i)
                {
                    LeafMap::LookupResult it;
                    bool inserted;
                    map.emplace(keys[i], it, inserted); /// map hash computed here, once per build row
                    addRowToMapped(it->getMapped(), locators[i], inserted, arena);
                }
            }
            const auto t2 = Clock::now();
            /// Release the consumed scatter chunk (the transient-memory contract of the plan).
            part_keys[leaf] = {};
            part_locs[leaf] = {};
            const auto t3 = Clock::now();
            scratch.setup_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
            scratch.insert_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
            scratch.release_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(t3 - t2).count();
        }
    }
};

/// ================================================================================================
/// Registration
/// ================================================================================================

void runBaselineCell(benchmark::State & state, size_t rows, size_t rows_per_key, size_t threads)
{
    auto data = getDataset(rows, rows_per_key);
    BaselineBench bench(*data, threads);
    bool verified = false;
    size_t iterations = 0;
    for (auto _ : state)
    {
        bench.resetIteration();
        state.SetIterationTime(bench.runBuild());
        ++iterations;
        if (!verified)
        {
            bench.verify();
            verified = true;
        }
    }
    bench.exportCounters(state, iterations);
    state.SetItemsProcessed(static_cast<int64_t>(iterations * rows));
}

void runPipelineCell(benchmark::State & state, size_t rows, size_t rows_per_key, size_t threads, bool use_amac)
{
    auto data = getDataset(rows, rows_per_key);
    PipelineBench bench(*data, threads, use_amac);
    bool verified = false;
    size_t iterations = 0;
    for (auto _ : state)
    {
        bench.resetIteration();
        state.SetIterationTime(bench.runBuild());
        ++iterations;
        if (!verified)
        {
            bench.verify();
            verified = true;
        }
    }
    bench.exportCounters(state, iterations);
    state.SetItemsProcessed(static_cast<int64_t>(iterations * rows));
}

void registerBuildBenchmarks()
{
    struct Shape
    {
        size_t rows;
        size_t rows_per_key;
        const char * tag;
    };
    static constexpr Shape shapes[] = {
        {32 << 20, 1, "D32M_unique"},
        {32 << 20, 8, "D32M_dup8"},
        /// Diagnostic shape: a non-power-of-two D whose per-leaf distinct count does NOT land on
        /// the grower's maxFill boundary (contrast with D = 2^26, where half the leaves round up
        /// to double-sized bucket arrays and outgrow the L2 budget).
        {48 << 20, 1, "D48M_unique"},
        {64 << 20, 1, "D64M_unique"},
        {64 << 20, 8, "D64M_dup8"},
    };
    static constexpr size_t thread_counts[] = {1, 16, 32, 64, 96};

    for (const auto & shape : shapes)
    {
        for (const size_t threads : thread_counts)
        {
            const std::string suffix = "/" + std::string(shape.tag) + "/T" + std::to_string(threads);
            benchmark::RegisterBenchmark(
                ("BM_build_baseline" + suffix).c_str(),
                [shape, threads](benchmark::State & state) { runBaselineCell(state, shape.rows, shape.rows_per_key, threads); })
                ->UseManualTime()
                ->Iterations(1)
                ->Unit(benchmark::kMillisecond);
            benchmark::RegisterBenchmark(
                ("BM_build_pipe_seq" + suffix).c_str(),
                [shape, threads](benchmark::State & state)
                { runPipelineCell(state, shape.rows, shape.rows_per_key, threads, /*use_amac*/ false); })
                ->UseManualTime()
                ->Iterations(1)
                ->Unit(benchmark::kMillisecond);
            benchmark::RegisterBenchmark(
                ("BM_build_pipe_amac" + suffix).c_str(),
                [shape, threads](benchmark::State & state)
                { runPipelineCell(state, shape.rows, shape.rows_per_key, threads, /*use_amac*/ true); })
                ->UseManualTime()
                ->Iterations(1)
                ->Unit(benchmark::kMillisecond);
        }
    }
}

}

int main(int argc, char ** argv)
{
    registerBuildBenchmarks();
    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
    return 0;
}
