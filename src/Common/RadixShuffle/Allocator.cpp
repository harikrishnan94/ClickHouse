#include <Common/RadixShuffle/Allocator.h>

#include <Common/Exception.h>

#include <algorithm>
#include <cstdlib>
#include <new>


namespace DB
{

namespace ErrorCodes
{
extern const int LOGICAL_ERROR;
extern const int BAD_ARGUMENTS;
}

}


namespace DB::RadixShuffle
{

namespace
{

/// Bump-allocated arena page. Each handle owns its own page chain; pages
/// stay allocated until the parent `Allocator` is destroyed (§3.1: no
/// per-chunk deallocation during the allocator's lifetime).
constexpr size_t DEFAULT_ARENA_PAGE_BYTES = 64 * 1024;

/// Round `n` up to the next multiple of `align` (which must be a power of two).
[[nodiscard]] constexpr size_t alignUp(size_t n, size_t align) noexcept
{
    return (n + (align - 1)) & ~(align - 1);
}

}


/// Per-handle (column × partition) writable tail. The "current chunk" plus
/// how many rows/bytes of its capacity remain unsliced. When `remaining_rows
/// < requested_rows` (or, for variable-length, `remaining_bytes <
/// requested_bytes`) we allocate a fresh chunk from the handle's arena.
struct Handle::PerChain
{
    Chunk * chunk = nullptr;
    size_t next_row = 0; /// next free row index inside `chunk`
    size_t next_byte = 0; /// variable-length only: next free byte index inside chunk->primary
    size_t reserved_rows = 0; /// cumulative rows reserved into this chain
    size_t reserved_bytes = 0; /// cumulative bytes reserved into this chain (variable-length)
};


struct Handle::ArenaPage
{
    ArenaPage * next = nullptr;
    /// Bytes follow immediately after this header. The page allocates
    /// `header_size + payload_bytes`; we cast `this` + sizeof(ArenaPage)
    /// to a `char *` to obtain the payload region.
};


Handle::Handle(Allocator & parent_, size_t num_columns, size_t partitions_)
    : parent(parent_)
    , chains(num_columns * partitions_)
{
}


Handle::~Handle() = default;


/// Bump-allocate a buffer from the handle's arena. Allocates a new arena
/// page if the current page does not have room; new pages are sized to the
/// max of the requested amount (rounded up to the default page size) and
/// the default page size.
void * Handle::arenaAllocate(size_t bytes, size_t align)
{
    /// All pages produced by the underlying allocator are aligned to at
    /// least 16 bytes; if the caller asks for higher we round the cursor
    /// up. `align` must be a power of two.
    ///
    /// Arena-page allocations are amortized over many chunk allocations
    /// and are NOT counted toward `total_allocated_bytes` — that counter
    /// tracks the chunks the allocator hands out (primary, offsets, null
    /// map), which is what the waste bound (§3.1 constraint 2) is defined
    /// over.
    if (arena_cursor == nullptr
        || alignUp(reinterpret_cast<uintptr_t>(arena_cursor), align) + bytes > reinterpret_cast<uintptr_t>(arena_end))
    {
        const size_t header = alignUp(sizeof(ArenaPage), alignof(std::max_align_t));
        const size_t needed = header + bytes + (align > alignof(std::max_align_t) ? align : 0);
        const size_t page_bytes = std::max(needed, DEFAULT_ARENA_PAGE_BYTES);

        auto * page = static_cast<ArenaPage *>(std::malloc(page_bytes));
        if (page == nullptr)
            throw std::bad_alloc();
        page->next = arena_head;
        arena_head = page;

        char * payload = reinterpret_cast<char *>(page) + header;
        arena_cursor = payload;
        arena_end = reinterpret_cast<char *>(page) + page_bytes;
    }

    /// Align the cursor up to `align`.
    auto cursor_int = reinterpret_cast<uintptr_t>(arena_cursor);
    const auto aligned = alignUp(cursor_int, align);
    arena_cursor = reinterpret_cast<char *>(aligned + bytes);
    return reinterpret_cast<void *>(aligned);
}


Handle::PerChain & Handle::ensureChunk(size_t col_idx, size_t part_idx, size_t rows, size_t bytes)
{
    PerChain & ch = chains[col_idx * parent.partitions() + part_idx];
    const ColumnDesc & desc = parent.columnDesc(col_idx);

    /// Fits in current chunk?
    if (ch.chunk != nullptr)
    {
        if (ch.next_row + rows <= ch.chunk->row_capacity && (!desc.variable_length || ch.next_byte + bytes <= ch.chunk->byte_capacity))
        {
            return ch;
        }
    }

    /// Need a fresh chunk. The size has to satisfy three constraints:
    ///   1. The floor (constraint 3): at least `MIN_FLOOR_ROWS`.
    ///   2. The current request: at least `rows`.
    ///   3. The waste bound (constraint 2): `total_allocated_bytes <=
    ///      active_chains * MIN_CHUNK_FLOOR_BYTES + 1.10 * total_reserved_bytes`
    ///      (per-chain accounting, where each active chain may carry one
    ///      trailing chunk undersized by up to `MIN_CHUNK_FLOOR_BYTES`).
    ///      The simplest way to achieve this without predicting future
    ///      request sizes is:
    ///        (a) round the chunk to a MULTIPLE of the current request
    ///            (this request fits exactly K times into the chunk,
    ///            leaving zero in-batch tail waste), AND
    ///        (b) grow chunks proportionally (10% of the chain's
    ///            already-reserved bytes) so the 10% term takes over from
    ///            the per-chain MIN_FLOOR term once the chain is well past
    ///            the floor.
    /// In aggregate this keeps the chain's allocated bytes at
    /// `MIN_FLOOR + 1.10 * reserved`, matching the spec's bound.
    const size_t floor_rows = parent.options().min_chunk_floor_rows;
    const size_t growth_rows = ch.reserved_rows / 10;
    size_t chunk_rows = std::max({floor_rows, rows, growth_rows});

    /// Round up to a multiple of `rows` to eliminate per-chunk tail waste
    /// caused by `chunk_rows % rows != 0`. This is the difference between
    /// "fits 2 batches with 56 rows wasted" and "fits 3 batches exactly".
    if (rows > 0)
    {
        const size_t k = (chunk_rows + rows - 1) / rows;
        chunk_rows = k * rows;
    }

    size_t chunk_bytes = 0;
    if (desc.variable_length)
    {
        const size_t floor_bytes = parent.options().min_chunk_floor_bytes;
        const size_t growth_bytes = ch.reserved_bytes / 10;
        chunk_bytes = std::max({floor_bytes, bytes, growth_bytes});
        if (bytes > 0)
        {
            const size_t kb = (chunk_bytes + bytes - 1) / bytes;
            chunk_bytes = kb * bytes;
        }
    }
    else
    {
        chunk_bytes = chunk_rows * desc.element_size;
    }

    /// Allocate the chunk header.
    auto * chunk = static_cast<Chunk *>(arenaAllocate(sizeof(Chunk), alignof(Chunk)));
    new (chunk) Chunk{};
    chunk->row_capacity = chunk_rows;
    chunk->byte_capacity = chunk_bytes;

    /// Primary buffer.
    const size_t primary_bytes = chunk_bytes;
    if (primary_bytes > 0)
        chunk->primary = arenaAllocate(primary_bytes, std::max<size_t>(desc.alignment, 1));

    /// Offsets buffer (variable-length only).
    const size_t offsets_bytes = desc.has_offsets ? sizeof(uint64_t) * chunk_rows : 0;
    if (desc.has_offsets)
        chunk->offsets = static_cast<uint64_t *>(arenaAllocate(offsets_bytes, alignof(uint64_t)));

    /// Null map (nullable only).
    const size_t null_map_bytes = desc.has_null_map ? sizeof(uint8_t) * chunk_rows : 0;
    if (desc.has_null_map)
        chunk->null_map = static_cast<uint8_t *>(arenaAllocate(null_map_bytes, alignof(uint8_t)));

    const bool was_empty = (ch.chunk == nullptr);

    ch.chunk = chunk;
    ch.next_row = 0;
    ch.next_byte = 0;

    /// Sharded counters: write to this handle's local atomic counters.
    /// Tests that read the totals from another thread observe the sum.
    local_chunks.fetch_add(1, std::memory_order_relaxed);
    local_allocated_bytes.fetch_add(primary_bytes + offsets_bytes + null_map_bytes, std::memory_order_relaxed);
    if (was_empty)
        local_active_chains.fetch_add(1, std::memory_order_relaxed);

    return ch;
}


void Handle::reserve(size_t col_idx, const ReservationRequest * requests, Reservation * output)
{
    chassert(live);
    const ColumnDesc & desc = parent.columnDesc(col_idx);
    const size_t partitions = parent.partitions();
    uint64_t reserved_delta = 0;
    for (size_t p = 0; p < partitions; ++p)
    {
        const size_t rows = requests[p].rows;
        const size_t bytes_req = desc.variable_length ? requests[p].bytes : rows * desc.element_size;
        if (rows == 0 && bytes_req == 0)
        {
            output[p] = Reservation{};
            continue;
        }

        PerChain & chain = ensureChunk(col_idx, p, rows, bytes_req);

        Reservation r;
        r.chunk = chain.chunk;
        r.begin_row = chain.next_row;
        r.reserved_rows = rows;
        r.begin_byte = chain.next_byte;
        r.reserved_bytes = bytes_req;
        output[p] = r;

        chain.next_row += rows;
        chain.reserved_rows += rows;
        if (desc.variable_length)
            chain.next_byte += bytes_req;
        chain.reserved_bytes += bytes_req;

        reserved_delta += bytes_req;
    }

    /// Sharded counter: this handle's own atomic, no cache-line bouncing
    /// with other threads' counters.
    local_reserved_bytes.fetch_add(reserved_delta, std::memory_order_relaxed);
}


Allocator::Allocator(std::vector<ColumnDesc> column_descs_, size_t partitions_, size_t /*expected_total_rows*/, AllocatorOptions options_)
    : column_descs(std::move(column_descs_))
    , num_partitions(partitions_)
    , opts(options_)
{
    if (num_partitions == 0)
        throw Exception(ErrorCodes::BAD_ARGUMENTS, "RadixShuffle::Allocator: partitions must be > 0");

    /// All alignments must be powers of two.
    for (const auto & d : column_descs)
    {
        if (d.alignment == 0 || (d.alignment & (d.alignment - 1)) != 0)
            throw Exception(ErrorCodes::BAD_ARGUMENTS, "RadixShuffle::Allocator: column alignment must be a power of two");
    }
}


Allocator::~Allocator()
{
    /// Free all arena pages across all handles. We do this in destructor,
    /// not on `release`: §3.1 mandates the chunks remain valid for read
    /// until allocator destruction.
    std::lock_guard lk(handle_pool_mutex);
    for (auto & h : handles)
    {
        auto * page = h->arena_head;
        while (page != nullptr)
        {
            auto * next = page->next;
            std::free(page);
            page = next;
        }
    }
}


Handle * Allocator::acquire()
{
    std::lock_guard lk(handle_pool_mutex);
    auto handle = std::unique_ptr<Handle>(new Handle(*this, column_descs.size(), num_partitions));
    auto * raw = handle.get();
    handles.push_back(std::move(handle));
    return raw;
}


void Allocator::release(Handle * handle)
{
    /// Mark as released. We keep the handle's chunks alive (via the
    /// allocator's `handles` vector) so reconstruct can still read from
    /// them. The handle itself becomes inert: any further `reserve` calls
    /// hit the `chassert(live)` in `Handle::reserve`.
    if (handle != nullptr)
        handle->live = false;
}


uint64_t Allocator::totalAllocatedBytes() const noexcept
{
    std::lock_guard lk(handle_pool_mutex);
    uint64_t sum = 0;
    for (const auto & h : handles)
        sum += h->local_allocated_bytes.load(std::memory_order_relaxed);
    return sum;
}


uint64_t Allocator::totalReservedBytes() const noexcept
{
    std::lock_guard lk(handle_pool_mutex);
    uint64_t sum = 0;
    for (const auto & h : handles)
        sum += h->local_reserved_bytes.load(std::memory_order_relaxed);
    return sum;
}


uint64_t Allocator::activeChains() const noexcept
{
    std::lock_guard lk(handle_pool_mutex);
    uint64_t sum = 0;
    for (const auto & h : handles)
        sum += h->local_active_chains.load(std::memory_order_relaxed);
    return sum;
}


uint64_t Allocator::totalChunks() const noexcept
{
    std::lock_guard lk(handle_pool_mutex);
    uint64_t sum = 0;
    for (const auto & h : handles)
        sum += h->local_chunks.load(std::memory_order_relaxed);
    return sum;
}

}
