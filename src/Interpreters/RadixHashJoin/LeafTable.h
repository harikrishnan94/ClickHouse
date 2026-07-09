#pragma once

#include <Interpreters/RadixHashJoin/Arena.h>
#include <Interpreters/RadixHashJoin/BuildSide.h>
#include <Interpreters/RadixHashJoin/KeyRefScatter.h>
#include <Interpreters/RadixHashJoin/ParallelFor.h>

#include <Common/Arena.h>

#include <atomic>
#include <base/types.h>

#include <bit>
#include <cstddef>
#include <memory>
#include <mutex>
#include <vector>

namespace DB::RadixJoin
{

/** One open-addressing, linear-probe hash table per leaf. A cell is laid out as
  *
  *     [ RowRefList word (8 B) | packed key (key_width B) ]
  *
  * so the cell head is a `DB::RowRefList` (see Interpreters/RowRefs.h): the unified ALL-join row-ref
  * value. A unique key holds the build row inline (the word IS the encoded singleton `RowRef`); the
  * first duplicate of a key allocates a `RowRefList::Batch` node from the owning build worker's arena
  * and the word becomes a count-tagged pointer to it. A probe match reads the cell word and either
  * emits the one inline ref directly (the common single-row fast path, NO extra load) or iterates the
  * list.
  *
  * Empty sentinel: the cell word is 0 (`RowRefList::word == 0`). The cell array is carved from a
  * (non-zeroed) Arena and `memset` to 0 by the worker that owns the leaf, so every cell starts empty
  * with no separate init pass. (A valid inline ref always has bit 63 set — the singleton flag — so it
  * can never collide with the all-zero empty word.)
  *
  * Singleton fast path: a unique key's cell word is the encoded singleton `RowRef` (bit 63 set), so a
  * probe emits it with NO node access — saving a likely LLC/DRAM miss for the common single-row keys
  * even in a build that also contains duplicates. The Batch node is allocated only when the first
  * duplicate of a key arrives.
  *
  * Leaf = top `total_bits` of the key's 32-bit `HashT` hash; bucket = low bits masked by
  * `num_buckets - 1`. Routing and bucketing share one word, so the partition plan must keep
  * `total_bits + log2(buckets) <= 32`. Build and probe recompute the identical hash, so the bucket
  * matches on both sides; the leaves carry key + ref only (no stored hash).
  */
static constexpr size_t MAX_UNIQUE_BUCKET_SIZES = 256;
static constexpr UInt32 MAX_GROUP_BITS = 8;

/** Per-group leaf hash-table descriptor: cell-array base and bucket count stored separately (16 B).
  *
  * Grouped use: one entry per homogeneous leaf group (the group base and the shared bucket count).
  * `cells == nullptr` is the empty sentinel (no allocation in this group). `num_buckets` is a power
  * of two when non-empty.
  */
struct LeafHT
{
    char * cells = nullptr;
    UInt64 num_buckets = 0;

    LeafHT() = default;

    /// `cells_` may be written during the build; the pointer is mutable on purpose.
    LeafHT(char * cells_, UInt64 num_buckets_) noexcept /// NOLINT(readability-non-const-parameter)
        : cells(cells_)
        , num_buckets(num_buckets_)
    {
    }

    /// True for an empty table (no rows): null `cells`.
    bool empty() const noexcept { return cells == nullptr; }

    /// The cell array base (stride = leafCellBytes(key_width); memset to 0). Null for an empty table.
    char * cellsPtr() const noexcept { return cells; }

    /// log2(num_buckets). Only meaningful for a non-empty table.
    UInt8 bits() const noexcept { return num_buckets ? static_cast<UInt8>(std::countr_zero(num_buckets)) : 0; }

    /// Number of buckets (a power of two). Only meaningful for a non-empty table.
    UInt64 numBuckets() const noexcept { return num_buckets; }

    /// The probe mask `num_buckets - 1`. Only meaningful for a non-empty table.
    UInt64 mask() const noexcept { return num_buckets - 1; }
};

static_assert(sizeof(LeafHT) == 16, "LeafHT must be 16 bytes (cells pointer + num_buckets)");

/** Homogeneous leaf groups: consecutive leaf-id ranges share one bucket count and a single arena
  * allocation, so the probe-side metadata is at most 256 entries (4 KB, one page, L1-resident)
  * instead of an up-to-8 MB per-leaf descriptor vector.
  */
struct GroupedLeaves
{
    UInt32 group_bits = 0;   /// g = route >> (32 - group_bits)
    UInt32 local_shift = 0;  /// total_bits - group_bits; local = (route >> leaf_shift) & ((1<<local_shift)-1)
    std::vector<LeafHT> groups; /// one descriptor per group; LeafHT{} (cells null) == empty group
};

/// Cell stride for a key of `key_width` bytes (RowRefList head word + key).
constexpr size_t leafCellBytes(size_t key_width) noexcept
{
    return sizeof(DB::RowRefList) + key_width;
}

/// Bucket index in [0, num_buckets): the low bits of the hash's low word. `num_buckets` is a non-zero
/// power of two, so this is a single AND.
inline UInt64 leafBucket(UInt32 low_hash, UInt64 num_buckets) noexcept
{
    return static_cast<UInt64>(low_hash) & (num_buckets - 1);
}

/// Lifecycle of one leaf group's tables under the lazy first-touch build: EMPTY (prepared, no cell
/// allocation yet) -> BUILDING (one owner is allocating and filling the group's cells) -> READY (the
/// group's `LeafHT` is published and immutable for the rest of the probe). Rowless groups start READY.
enum class GroupBuildState : UInt8
{
    EMPTY = 0,
    BUILDING = 1,
    READY = 2,
};

/// All the leaf tables; owns the arena backing the cells plus the per-worker arenas backing the
/// RowRefList Batch nodes (a READY group is read-only for the whole probe).
///
/// Groups are built at GROUP granularity, either eagerly (`buildLeafTables`) or lazily at first probe
/// touch (`prepareLeafTables` + `ensureLeafGroupBuilt`): a group shares one cell allocation, so a leaf
/// is built exactly when its group is. The prepared per-group sizing (`group_bucket_bits`,
/// `group_leaf_stride`) is fixed by `prepareLeafTables` and only ever changed by the owning builder
/// during an overflow rebuild, before the group turns READY.
struct LeafTables
{
    GroupedLeaves grouped;
    /// Per-build-worker arenas holding the RowRefList Batch nodes. One per worker (single-writer
    /// during build); their stable addresses must outlive the probe, hence they live here.
    std::vector<std::unique_ptr<DB::Arena>> build_arenas;
    /// Set during build the first time any key gets a duplicate row. Selects the grouped probe path.
    std::atomic<bool> any_duplicates{false};
    UInt64 num_rows = 0;
    /// Upper bound on log2(num_buckets) over groups, fixed at prepare time from row-count sizing (the
    /// overflow-rebuild sizing, always >= the estimate sizing a group is actually built with). The
    /// probe uses a dense 16-byte (UInt32 bucket-index) slot iff this is <= 31 — i.e. no group can ever
    /// have more than 2^31 buckets (always, in practice). A prepare-time constant so the probe's slot
    /// choice needs no synchronization with concurrent lazy group builds.
    UInt8 max_bucket_bits = 0;
    /// Number of cell-array allocations from `arena` (one per built non-empty group plus rare overflow
    /// rebuilds). Atomic: lazy group builds run concurrently. Diagnostic mirror of `LeafArrays::alloc_count`.
    std::atomic<UInt64> cell_alloc_count{0};
    Arena arena;                     /// owns the cells (a READY group's range is read-only)

    /// ── Prepared group layout (set once by `prepareLeafTables`, consumed by `buildLeafGroup`) ──
    size_t key_width = 0;
    UInt32 total_bits = 0;           /// log2(num_leaves)
    size_t group_size = 1;           /// leaves per group == 1 << grouped.local_shift
    /// Per-group log2(bucket count); 0 == rowless group (no allocation, READY at prepare). Written by
    /// the group's owning builder during an overflow rebuild only.
    std::vector<UInt8> group_bucket_bits;
    /// Per-group padded per-leaf stride in bytes (builder-private bookkeeping; the probe recomputes it
    /// from the published `LeafHT::num_buckets`).
    std::vector<size_t> group_leaf_stride;
    /// Per-group lazy-build state (see `GroupBuildState`); a heap array keeps `LeafTables` movable.
    std::unique_ptr<std::atomic<GroupBuildState>[]> group_state;
    /// Serializes lazy group builds. Two jobs: it upholds the `build_arenas[worker]` single-writer
    /// invariant when several probe threads win different groups' CAS at once (each drives its own
    /// `ParallelFor`, whose dense worker ids would otherwise collide across invocations), and it
    /// bounds the transient build memory to one group at a time. unique_ptr keeps the struct movable.
    std::unique_ptr<std::mutex> lazy_build_mutex;

    LeafTables() = default;
    LeafTables(const LeafTables &) = delete;
    LeafTables & operator=(const LeafTables &) = delete;
    /// Move-only: `buildLeafTables`/`prepareLeafTables` return by value into `State::leaf_tables`.
    /// `std::atomic` is neither copyable nor movable, so the moves are spelled out (relaxed load/store;
    /// the build barrier already orders the concurrent build-time writes against the post-build read).
    LeafTables(LeafTables && other) noexcept
        : grouped(std::move(other.grouped))
        , build_arenas(std::move(other.build_arenas))
        , any_duplicates(other.any_duplicates.load(std::memory_order_relaxed))
        , num_rows(other.num_rows)
        , max_bucket_bits(other.max_bucket_bits)
        , cell_alloc_count(other.cell_alloc_count.load(std::memory_order_relaxed))
        , arena(std::move(other.arena))
        , key_width(other.key_width)
        , total_bits(other.total_bits)
        , group_size(other.group_size)
        , group_bucket_bits(std::move(other.group_bucket_bits))
        , group_leaf_stride(std::move(other.group_leaf_stride))
        , group_state(std::move(other.group_state))
        , lazy_build_mutex(std::move(other.lazy_build_mutex))
    {
    }
    LeafTables & operator=(LeafTables && other) noexcept
    {
        grouped = std::move(other.grouped);
        build_arenas = std::move(other.build_arenas);
        any_duplicates.store(other.any_duplicates.load(std::memory_order_relaxed), std::memory_order_relaxed);
        num_rows = other.num_rows;
        max_bucket_bits = other.max_bucket_bits;
        cell_alloc_count.store(other.cell_alloc_count.load(std::memory_order_relaxed), std::memory_order_relaxed);
        arena = std::move(other.arena);
        key_width = other.key_width;
        total_bits = other.total_bits;
        group_size = other.group_size;
        group_bucket_bits = std::move(other.group_bucket_bits);
        group_leaf_stride = std::move(other.group_leaf_stride);
        group_state = std::move(other.group_state);
        lazy_build_mutex = std::move(other.lazy_build_mutex);
        return *this;
    }
};

/// Prepare the group layout WITHOUT building any leaf table (the lazy-build entry, one cheap
/// serial pass): group split, per-group sizing (distinct estimate when present, else row count),
/// per-worker build arenas, and the per-group EMPTY/READY state (rowless groups start READY). No cell
/// memory is allocated. Groups are then built either lazily (`ensureLeafGroupBuilt` at first probe
/// touch) or all at once (`buildLeafTables`). `num_workers` must match the worker-id space of every
/// `ParallelFor` later passed to the build entries.
LeafTables prepareLeafTables(const LeafArrays & leaf_arrays, UInt64 num_rows, size_t key_width, size_t num_workers);

/// Build ONE group's leaf tables now: allocate the group's cell block, 0-initialise and fill its
/// leaves (parallel over the group's leaves via `parallel_for`), rebuild the group with safe row-count
/// sizing if a leaf overflows its distinct-estimate sizing, release the group's fused-record blocks
/// back to the `LeafArrays` arena (keeping build memory ~flat as groups are consumed), and publish the
/// group READY (release store — the probe's acquire on the state makes the cells visible).
/// The caller must own the group's build exclusively: either hold its BUILDING transition
/// (`ensureLeafGroupBuilt`) or be the single eager builder (`buildLeafTables`).
void buildLeafGroup(LeafTables & tables, LeafArrays & leaf_arrays, size_t gpos, const ParallelFor & parallel_for);

/// First-touch, at-most-once lazy group build (D-0004). READY -> immediate false. EMPTY -> try to CAS
/// to BUILDING: the winner builds the group via `buildLeafGroup` (under `lazy_build_mutex`) and returns
/// true; on failure the state is reset to EMPTY (a later toucher retries) and the exception propagates.
/// BUILDING (someone else won) -> spin+yield until READY, then false. Deadlock-free: a builder never
/// waits on another group's state.
bool ensureLeafGroupBuilt(LeafTables & tables, LeafArrays & leaf_arrays, size_t gpos, const ParallelFor & parallel_for);

/// Build every leaf table from a finished `LeafArrays`: `prepareLeafTables` + build all groups (the
/// eager path, used by the tests and by callers that do not probe lazily). Groups are built in
/// parallel via `parallel_for` — one unit per group, each group filled serially by its unit's worker
/// (whose dense id keys `build_arenas[worker]`, keeping the single-writer invariant). The RowRefList
/// Batch nodes are allocated from the per-worker `build_arenas` (`num_workers` of them). `num_workers`
/// must match the worker-id space of `parallel_for`.
LeafTables buildLeafTables(
    LeafArrays & leaf_arrays,
    UInt64 num_rows,
    size_t key_width,
    size_t num_workers,
    const ParallelFor & parallel_for);

/// Probe `n` left rows (their packed keys) against the leaf tables, appending one `(left_row, RowRef)`
/// per match to the output buffers (singleton keys emit one ref; multi-row keys iterate the whole
/// RowRefList). The 32-bit `HashT` hash of each key is computed internally, inside the probe pipeline, so its
/// latency overlaps with the in-flight cell misses (no precomputed hash array is passed in).
/// The probe is the AMAC ring pipeline for every key width and both duplicate-free and duplicate builds
/// (singletons emit inline; multi-row keys iterate their RowRefList). `pos_fits_u32` selects the UInt32
/// bucket-index ring slot (pass `LeafTables::max_bucket_bits <= 31`); when false a UInt64-index slot keeps a
/// >2^31-bucket group correct.
void collectMatches(
    size_t key_width,
    const GroupedLeaves & grouped,
    UInt32 leaf_shift,
    UInt32 total_bits,
    const void * packed_keys,
    size_t n,
    bool pos_fits_u32,
    std::vector<UInt32> & out_left_rows,
    std::vector<RowRef> & out_refs);

}
