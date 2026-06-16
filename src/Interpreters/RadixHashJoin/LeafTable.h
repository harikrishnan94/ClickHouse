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
#include <vector>

namespace DB::RadixJoin
{

/** One open-addressing, linear-probe hash table per leaf. A cell is laid out as
  *
  *     [ BuildRefList word (8 B) | packed key (key_width B) ]
  *
  * so the cell head is a `DB::BuildRefList` (see Interpreters/RowRefs.h): the unified ALL-join row-ref
  * value. A unique key holds the build row inline (the word IS the encoded singleton `BuildRef`); the
  * first duplicate of a key allocates a `BuildRefList::Batch` node from the owning build worker's arena
  * and the word becomes a count-tagged pointer to it. A probe match reads the cell word and either
  * emits the one inline ref directly (the common single-row fast path, NO extra load) or iterates the
  * list.
  *
  * Empty sentinel: the cell word is 0 (`BuildRefList::word == 0`). The cell array is carved from a
  * (non-zeroed) Arena and `memset` to 0 by the worker that owns the leaf, so every cell starts empty
  * with no separate init pass. (A valid inline ref always has bit 63 set — the singleton flag — so it
  * can never collide with the all-zero empty word.)
  *
  * Singleton fast path: a unique key's cell word is the encoded singleton `BuildRef` (bit 63 set), so a
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

/// Cell stride for a key of `key_width` bytes (BuildRefList head word + key).
constexpr size_t leafCellBytes(size_t key_width) noexcept
{
    return sizeof(DB::BuildRefList) + key_width;
}

/// Bucket index in [0, num_buckets): the low bits of the hash's low word. `num_buckets` is a non-zero
/// power of two, so this is a single AND.
inline UInt64 leafBucket(UInt32 low_hash, UInt64 num_buckets) noexcept
{
    return static_cast<UInt64>(low_hash) & (num_buckets - 1);
}

/// All the built leaf tables; owns the arena backing the cells plus the per-worker arenas backing the
/// BuildRefList Batch nodes (all read-only for the whole probe).
struct LeafTables
{
    GroupedLeaves grouped;
    /// Per-build-worker arenas holding the BuildRefList Batch nodes. One per worker (single-writer
    /// during build); their stable addresses must outlive the probe, hence they live here.
    std::vector<std::unique_ptr<DB::Arena>> build_arenas;
    /// Set during build the first time any key gets a duplicate row. Selects the grouped probe path.
    std::atomic<bool> any_duplicates{false};
    UInt64 num_rows = 0;
    /// max log2(num_buckets) over groups. The probe uses a dense 16-byte (UInt32 bucket-index) slot iff
    /// this is <= 31 — i.e. no group has more than 2^31 buckets (always, in practice).
    UInt8 max_bucket_bits = 0;
    /// Number of cell-array allocations from `arena` during `buildLeafTables` (one per non-empty group plus
    /// rare overflow rebuilds). Diagnostic mirror of `LeafArrays::alloc_count`.
    UInt64 cell_alloc_count = 0;
    Arena arena;                     /// owns the cells (read-only for the whole probe)

    LeafTables() = default;
    LeafTables(const LeafTables &) = delete;
    LeafTables & operator=(const LeafTables &) = delete;
    /// Move-only: `buildLeafTables` returns by value into `State::leaf_tables`. `std::atomic` is neither
    /// copyable nor movable, so the moves are spelled out (relaxed load/store; the build barrier already
    /// orders the concurrent build-time writes against the post-build read).
    LeafTables(LeafTables && other) noexcept
        : grouped(std::move(other.grouped))
        , build_arenas(std::move(other.build_arenas))
        , any_duplicates(other.any_duplicates.load(std::memory_order_relaxed))
        , num_rows(other.num_rows)
        , max_bucket_bits(other.max_bucket_bits)
        , cell_alloc_count(other.cell_alloc_count)
        , arena(std::move(other.arena))
    {
    }
    LeafTables & operator=(LeafTables && other) noexcept
    {
        grouped = std::move(other.grouped);
        build_arenas = std::move(other.build_arenas);
        any_duplicates.store(other.any_duplicates.load(std::memory_order_relaxed), std::memory_order_relaxed);
        num_rows = other.num_rows;
        max_bucket_bits = other.max_bucket_bits;
        cell_alloc_count = other.cell_alloc_count;
        arena = std::move(other.arena);
        return *this;
    }
};

/// Build every leaf table from a finished `LeafArrays`. Cell arrays are allocated, 0-initialised and
/// filled in parallel via `parallel_for`; the BuildRefList Batch nodes are allocated from the per-worker
/// `build_arenas` (`num_workers` of them, indexed by the unit callback's dense `worker` id under a
/// single-writer invariant). `num_workers` must match the worker-id space of `parallel_for`.
LeafTables buildLeafTables(
    const LeafArrays & leaf_arrays,
    UInt64 num_rows,
    size_t key_width,
    size_t num_workers,
    const ParallelFor & parallel_for);

/// Probe `n` left rows (their packed keys) against the leaf tables, appending one `(left_row, BuildRef)`
/// per match to the output buffers (singleton keys emit one ref; multi-row keys iterate the whole
/// BuildRefList). The 32-bit `HashT` hash of each key is computed internally, inside the probe pipeline, so its
/// latency overlaps with the in-flight cell misses (no precomputed hash array is passed in).
/// The probe is the AMAC ring pipeline for every key width and both duplicate-free and duplicate builds
/// (singletons emit inline; multi-row keys iterate their BuildRefList). `pos_fits_u32` selects the UInt32
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
    std::vector<BuildRef> & out_refs);

}
