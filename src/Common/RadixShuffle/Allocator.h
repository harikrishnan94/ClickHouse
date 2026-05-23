#pragma once

#include <Common/RadixShuffle/PartitionTypes.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <vector>


namespace DB::RadixShuffle
{

/// Default per-(column × partition) minimum chunk row floor for fixed-width
/// columns (§3.1 constraint 3). Reservations smaller than this slice within
/// the same allocated chunk; only a reservation that does not fit triggers
/// a new chunk, and that new chunk is sized to
/// `max(MIN_CHUNK_FLOOR_ROWS, requested_rows)`.
inline constexpr size_t DEFAULT_MIN_CHUNK_FLOOR_ROWS = 256;


/// Equivalent variable-length byte budget. Picked so a `ColumnString` chunk
/// holds roughly the same number of "average-sized" rows (~64-byte strings)
/// as a fixed-width chunk holds elements. The floor only applies to
/// individual chunk **allocations**; small reservations slice within an
/// existing chunk's remaining capacity.
///
/// Note: variable-length chunks impose BOTH the row floor and this byte
/// floor on every allocation. The spec describes only an "equivalent byte
/// budget" for variable-length floor sizing (§3.1 constraint 3); applying
/// both bounds is more conservative than the spec strictly requires, but
/// it lets us treat variable-length and fixed-width chains uniformly in
/// the waste-bound bookkeeping (the row capacity is what `Chunk` exposes
/// to callers; the byte capacity bounds the chars region).
inline constexpr size_t DEFAULT_MIN_CHUNK_FLOOR_BYTES = 16 * 1024;


/// Public knobs (defaulted to the constants above).
struct AllocatorOptions
{
    size_t min_chunk_floor_rows = DEFAULT_MIN_CHUNK_FLOOR_ROWS;
    size_t min_chunk_floor_bytes = DEFAULT_MIN_CHUNK_FLOOR_BYTES;
};


class Allocator;


/// Opaque per-producer-thread cursor over all (column × partition) chains.
/// Owned by `Allocator`; producers acquire one through `Allocator::acquire`
/// and release it through `Allocator::release`. While held, a handle's
/// reservation operations are entirely contention-independent: each
/// reservation reads/writes per-handle state only, and chunk allocations
/// happen through a private arena owned by the handle (the global
/// allocator only sees a per-handle commit at chunk-boundary time, which
/// is contention-free in the common case where each handle pulls fresh
/// memory from its own jemalloc arena).
class Handle
{
public:
    Handle(const Handle &) = delete;
    Handle & operator=(const Handle &) = delete;
    Handle(Handle &&) = delete;
    Handle & operator=(Handle &&) = delete;
    ~Handle();

    /// Per-batch reservation across all P partitions for one column. For
    /// every `p ∈ [0, P)` the caller asks for `requests[p]` and the handle
    /// hands back `output[p]` with at least the requested capacity. The
    /// reservation IS the commit (§3.1): cursors advance atomically with
    /// the call, and the slot bytes are considered "spent" from the
    /// allocator's accounting perspective even if the caller under-fills
    /// them. Reservation MAY allocate (when the current chunk does not
    /// have room); the allocation goes through the handle's private arena
    /// and does not contend with other handles.
    void reserve(
        size_t col_idx,
        const ReservationRequest * requests,
        Reservation * output);

private:
    friend class Allocator;
    struct PerChain;
    struct ArenaPage;

    Handle(Allocator & parent_, size_t num_columns, size_t partitions);

    /// Internal: ensure the per-handle (column × partition) chain has a
    /// chunk with room for one more reservation of `rows` rows and (for
    /// variable-length) `bytes` bytes. Allocates a new chunk if needed.
    /// Returns a pointer to the chain's writable tail.
    PerChain & ensureChunk(size_t col_idx, size_t part_idx, size_t rows, size_t bytes);

    /// Internal: allocate `bytes` bytes from the handle's private arena
    /// with the requested `align`. The arena is bump-allocated; new arena
    /// pages are pulled from the global allocator only at page-boundary
    /// time (cold path).
    void * arenaAllocate(size_t bytes, size_t align);

    Allocator & parent;

    /// Per-(column, partition) writable tail. `num_columns * partitions`
    /// entries, indexed `col * partitions + part`.
    std::vector<PerChain> chains;

    /// Per-handle bump-allocated arena page chain. Owned by the handle;
    /// pages live until the parent allocator is destroyed (released
    /// handles do not free their pages — that would break the
    /// "monotonic, no per-chunk dealloc" rule of §3.1).
    ArenaPage * arena_head = nullptr;
    char * arena_cursor = nullptr;
    char * arena_end = nullptr;

    /// Sharded counters: each handle accumulates its own reserved /
    /// allocated bytes and chunk / active-chain counts in non-shared
    /// atomics. `Allocator::total*Bytes()` sums across all handles. This
    /// avoids the cache-line bouncing that a single shared counter would
    /// incur on the hot path at high T. The atomics are still needed
    /// because tests read the running totals from a different thread
    /// while producers are writing them; `memory_order_relaxed` is
    /// sufficient because we don't depend on any ordering between
    /// counter reads and other allocator state.
    alignas(64) std::atomic<uint64_t> local_reserved_bytes{0};
    alignas(64) std::atomic<uint64_t> local_allocated_bytes{0};
    alignas(64) std::atomic<uint64_t> local_chunks{0};
    alignas(64) std::atomic<uint64_t> local_active_chains{0};

    /// True between `acquire` and `release`; once released the handle is
    /// inert (further reservations would be a contract violation). The
    /// flag is single-threaded by contract (§3.1: "A thread holds at most
    /// one handle. Handles are not transferable across threads."); both
    /// the writer (the owning thread, via `Allocator::release`) and the
    /// reader (`Handle::reserve`'s `chassert`) operate on the same thread,
    /// so no atomic or fence is needed.
    bool live = true;
};


/// Append-only, monotonic, type-aware allocator that hands out writable
/// chunks for scatter (§3.1).
class Allocator
{
public:
    Allocator(
        std::vector<ColumnDesc> column_descs,
        size_t partitions,
        size_t expected_total_rows,
        AllocatorOptions options = {});

    Allocator(const Allocator &) = delete;
    Allocator & operator=(const Allocator &) = delete;
    Allocator(Allocator &&) = delete;
    Allocator & operator=(Allocator &&) = delete;

    ~Allocator();

    [[nodiscard]] size_t numColumns() const noexcept { return column_descs.size(); }
    [[nodiscard]] size_t partitions() const noexcept { return num_partitions; }
    [[nodiscard]] const ColumnDesc & columnDesc(size_t col_idx) const { return column_descs[col_idx]; }
    [[nodiscard]] const AllocatorOptions & options() const noexcept { return opts; }

    /// Cold-path: a producer thread acquires a handle. Thread owns the
    /// handle and MUST release it via `release` before the allocator is
    /// destroyed. A thread holds at most one handle. Handles are not
    /// transferable.
    Handle * acquire();

    /// Cold-path: producer hands back a handle. Subsequent reservations
    /// through the released handle are a contract violation. The handle's
    /// allocated chunks remain readable until the allocator is destroyed.
    void release(Handle * handle);

    /// Global accounting (post-condition of the waste bound). Both values
    /// are computed by summing per-handle counters; the per-handle
    /// counters are incremented on the hot path (uncontended within a
    /// thread). Tests that read these values during execution see an
    /// eventually-consistent total — `memory_order_relaxed` on the
    /// per-handle counters is sufficient because no ordering with other
    /// allocator state is required.
    [[nodiscard]] uint64_t totalAllocatedBytes() const noexcept;
    [[nodiscard]] uint64_t totalReservedBytes() const noexcept;

    /// Number of (col, part) chains that have at least one chunk across
    /// all handles. Summed lazily from per-handle counters.
    [[nodiscard]] uint64_t activeChains() const noexcept;

    /// Number of chunks the allocator has handed out across all handles.
    [[nodiscard]] uint64_t totalChunks() const noexcept;

private:
    const std::vector<ColumnDesc> column_descs;
    const size_t num_partitions;
    const AllocatorOptions opts;

    /// Cold-path mutex protecting the handle pool. Acquire/release are not
    /// on the hot path. Reads of the aggregated totals (test code) lock
    /// it to walk the handles vector.
    mutable std::mutex handle_pool_mutex;
    std::vector<std::unique_ptr<Handle>> handles;
};

}
