#pragma once

#include <Common/RadixShuffle/PartSchema.h>
#include <Common/RadixShuffle/PartitionTypes.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <vector>


namespace DB::RadixShuffle
{

/// Default per-partition minimum fixed-chunk row floor (constraint 3).
/// Small reservations slice within an existing chunk; only a reservation
/// that does not fit triggers a new chunk, sized to at least this many rows.
inline constexpr size_t DEFAULT_MIN_CHUNK_FLOOR_ROWS = 256;

/// Minimum byte budget for newly allocated data chunks.
inline constexpr size_t DEFAULT_MIN_CHUNK_FLOOR_BYTES = 16 * 1024;


struct AllocatorOptions
{
    size_t min_chunk_floor_rows = DEFAULT_MIN_CHUNK_FLOOR_ROWS;
    size_t min_chunk_floor_bytes_data = DEFAULT_MIN_CHUNK_FLOOR_BYTES;
};


class Allocator;


/// Opaque per-producer-thread cursor over P per-partition chains.
/// Owned by Allocator; acquire/release are cold-path; reserve is hot-path
/// and entirely contention-free (reads/writes only per-handle state, chunk
/// allocations go through a private arena).
class Handle
{
public:
    Handle(const Handle &) = delete;
    Handle & operator=(const Handle &) = delete;
    Handle(Handle &&) = delete;
    Handle & operator=(Handle &&) = delete;
    ~Handle();

    /// Per-batch SOA reservation across all P partitions.
    ///
    /// rows[p]          — row count to reserve for partition p.
    /// varlen_bytes[p]  — total varlen byte payload for partition p
    ///                    (caller-computed; 0 for fixed-only schemas).
    /// grants[p]        — output: reservation result for partition p.
    /// stale_fixed_bitset — caller-zeroed array of ceil(P/64) uint64_t
    ///                    words; bit p is set iff partition p's FixedChunk
    ///                    was newly allocated during this call.  Callers
    ///                    may cache FixedChunk* across batches and consult
    ///                    the bitset to detect when to reload.
    void reserve(
        const size_t * rows,
        const size_t * varlen_bytes,
        PartReserveGrant * grants,
        uint64_t * stale_fixed_bitset);

private:
    friend class Allocator;
    struct PerPartition;
    struct ArenaPage;

    Handle(Allocator & parent_, size_t partitions);

    void * arenaAllocate(size_t bytes, size_t align);

    /// Ensure partition p's fixed chunk has room for `rows` rows.
    /// Returns true if a new chunk was allocated (stale-pointer event).
    bool ensureFixed(size_t p, size_t rows);

    /// Ensure partition p's data chunk has room for `varlen_bytes` bytes.
    void ensureData(size_t p, size_t varlen_bytes);

    Allocator & parent;
    std::vector<PerPartition> parts;

    ArenaPage * arena_head = nullptr;
    char * arena_cursor = nullptr;
    char * arena_end = nullptr;

    /// Sharded counters — per-handle atomics avoid cache-line bouncing
    /// with other threads' counters at high T.  memory_order_relaxed
    /// suffices because there is no ordering dependency with other state.
    alignas(64) std::atomic<uint64_t> local_reserved_bytes{0};
    alignas(64) std::atomic<uint64_t> local_allocated_bytes{0};
    alignas(64) std::atomic<uint64_t> local_chunks{0};
    alignas(64) std::atomic<uint64_t> local_active_partitions{0};

    bool live = true;
};


/// Append-only, monotonic allocator that hands out per-partition fixed and
/// data chunks for scatter.
class Allocator
{
public:
    Allocator(
        PartSchema schema,
        size_t partitions,
        size_t expected_total_rows,
        AllocatorOptions options = {});

    Allocator(const Allocator &) = delete;
    Allocator & operator=(const Allocator &) = delete;
    Allocator(Allocator &&) = delete;
    Allocator & operator=(Allocator &&) = delete;

    ~Allocator();

    [[nodiscard]] size_t partitions() const noexcept { return num_partitions; }
    [[nodiscard]] const PartSchema & schema() const noexcept { return part_schema; }
    [[nodiscard]] const AllocatorOptions & options() const noexcept { return opts; }

    Handle * acquire();
    void release(Handle * handle);

    [[nodiscard]] uint64_t totalAllocatedBytes() const noexcept;
    [[nodiscard]] uint64_t totalReservedBytes() const noexcept;
    [[nodiscard]] uint64_t activePartitions() const noexcept;
    [[nodiscard]] uint64_t totalChunks() const noexcept;

private:
    const PartSchema part_schema;
    const size_t num_partitions;
    const AllocatorOptions opts;

    mutable std::mutex handle_pool_mutex;
    std::vector<std::unique_ptr<Handle>> handles;
};

}
