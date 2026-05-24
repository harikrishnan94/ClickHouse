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

constexpr size_t DEFAULT_ARENA_PAGE_BYTES = 64 * 1024;

[[nodiscard]] constexpr size_t alignUp(size_t n, size_t align) noexcept
{
    return (n + (align - 1)) & ~(align - 1);
}

} // namespace


/// Per-partition writable tail for one Handle.
struct Handle::PerPartition
{
    FixedChunk * fixed_tail = nullptr;
    size_t fixed_next_row = 0;
    size_t fixed_remaining_rows = 0;
    size_t reserved_rows = 0; ///< Cumulative, for growth-factor computation.

    DataChunk * data_tail = nullptr;
    size_t data_next_byte = 0;
    size_t data_remaining_bytes = 0;
    size_t reserved_bytes = 0; ///< Cumulative varlen, for growth-factor computation.
};


struct Handle::ArenaPage
{
    ArenaPage * next = nullptr;
    /// Payload bytes follow immediately after this header.
};


Handle::Handle(Allocator & parent_, size_t partitions_)
    : parent(parent_)
    , parts(partitions_)
{
}


Handle::~Handle() = default;


void * Handle::arenaAllocate(size_t bytes, size_t align)
{
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

    const auto aligned = alignUp(reinterpret_cast<uintptr_t>(arena_cursor), align);
    arena_cursor = reinterpret_cast<char *>(aligned + bytes);
    return reinterpret_cast<void *>(aligned);
}


bool Handle::ensureFixed(size_t p, size_t rows)
{
    PerPartition & pc = parts[p];
    if (pc.fixed_tail != nullptr && pc.fixed_remaining_rows >= rows)
        return false;

    const PartSchema & sc = parent.schema();
    const AllocatorOptions & opts = parent.options();

    const size_t floor_rows = opts.min_chunk_floor_rows;
    const size_t growth_rows = pc.reserved_rows / 10;
    size_t chunk_rows = std::max({floor_rows, rows, growth_rows});
    if (rows > 0)
    {
        const size_t k = (chunk_rows + rows - 1) / rows;
        chunk_rows = k * rows;
    }
    /// Keep column slot starts 64-byte aligned for 8-byte values.  SWWC uses
    /// `_mm512_stream_si512`, whose destination must be 64-byte aligned.
    chunk_rows = (chunk_rows + 7) & ~size_t(7);

    /// Compute the column-major layout for this chunk and the total byte size.
    const size_t num_slots = sc.fixed_slots.size();
    size_t * chunk_offsets = nullptr;
    size_t total_data_bytes = 0;

    if (num_slots > 0)
    {
        chunk_offsets = static_cast<size_t *>(arenaAllocate(sizeof(size_t) * num_slots, alignof(size_t)));
        size_t off = 0;
        for (size_t s = 0; s < num_slots; ++s)
        {
            off = alignUp(off, sc.fixed_slots[s].alignment);
            chunk_offsets[s] = off;
            off += chunk_rows * sc.fixed_slots[s].element_size;
        }
        total_data_bytes = alignUp(off, 64); // cache-line pad
    }

    const bool was_empty = (pc.fixed_tail == nullptr);

    auto * fc = static_cast<FixedChunk *>(arenaAllocate(sizeof(FixedChunk), alignof(FixedChunk)));
    new (fc) FixedChunk{};
    fc->row_capacity = chunk_rows;
    fc->slot_byte_offsets = chunk_offsets;
    if (total_data_bytes > 0)
        fc->data = arenaAllocate(total_data_bytes, 64);

    pc.fixed_tail = fc;
    pc.fixed_next_row = 0;
    pc.fixed_remaining_rows = chunk_rows;

    local_chunks.fetch_add(1, std::memory_order_relaxed);
    local_allocated_bytes.fetch_add(total_data_bytes, std::memory_order_relaxed);
    if (was_empty)
        local_active_partitions.fetch_add(1, std::memory_order_relaxed);

    return true; // new chunk allocated → stale-pointer event
}


void Handle::ensureData(size_t p, size_t varlen_bytes)
{
    PerPartition & pc = parts[p];
    if (pc.data_tail != nullptr && pc.data_remaining_bytes >= varlen_bytes)
        return;

    const AllocatorOptions & opts = parent.options();

    const size_t floor_bytes = opts.min_chunk_floor_bytes_data;
    const size_t growth_bytes = pc.reserved_bytes / 10;
    size_t chunk_bytes = std::max({floor_bytes, varlen_bytes, growth_bytes});
    if (varlen_bytes > 0)
    {
        const size_t kb = (chunk_bytes + varlen_bytes - 1) / varlen_bytes;
        chunk_bytes = kb * varlen_bytes;
    }

    auto * dc = static_cast<DataChunk *>(arenaAllocate(sizeof(DataChunk), alignof(DataChunk)));
    new (dc) DataChunk{};
    dc->byte_capacity = chunk_bytes;
    dc->bytes = static_cast<unsigned char *>(arenaAllocate(chunk_bytes, 1));

    pc.data_tail = dc;
    pc.data_next_byte = 0;
    pc.data_remaining_bytes = chunk_bytes;

    local_chunks.fetch_add(1, std::memory_order_relaxed);
    local_allocated_bytes.fetch_add(chunk_bytes, std::memory_order_relaxed);
}


void Handle::reserve(const size_t * rows, const size_t * varlen_bytes, PartReserveGrant * grants, uint64_t * stale_fixed_bitset)
{
    chassert(live);

    const PartSchema & sc = parent.schema();
    const size_t partitions = parent.partitions();
    uint64_t reserved_delta = 0;

    for (size_t p = 0; p < partitions; ++p)
    {
        const size_t row_req = rows[p];
        const size_t byte_req = varlen_bytes[p];

        if (row_req == 0 && byte_req == 0)
        {
            grants[p] = PartReserveGrant{};
            grants[p].fully_satisfied = true;
            continue;
        }

        /// Ensure fixed chunk capacity.
        if (row_req > 0)
        {
            const bool new_chunk = ensureFixed(p, row_req);
            if (new_chunk)
            {
                const size_t word = p / 64;
                const size_t bit = p % 64;
                stale_fixed_bitset[word] |= (uint64_t{1} << bit);
            }
        }

        /// Ensure data chunk capacity (varlen schemas only).
        if (sc.has_varlen_portion)
            ensureData(p, byte_req);

        PerPartition & pc = parts[p];

        PartReserveGrant & g = grants[p];
        g.granted_rows = row_req;
        g.granted_varlen_bytes = byte_req;
        g.slice.fixed = pc.fixed_tail;
        g.slice.begin_row = pc.fixed_next_row;
        g.slice.reserved_rows = row_req;
        g.slice.data = (sc.has_varlen_portion && (pc.data_tail != nullptr)) ? pc.data_tail : nullptr;
        g.slice.begin_byte = pc.data_next_byte;
        g.slice.reserved_bytes = byte_req;
        g.fully_satisfied = true;

        pc.fixed_next_row += row_req;
        pc.fixed_remaining_rows -= row_req;
        pc.reserved_rows += row_req;

        if (sc.has_varlen_portion)
        {
            pc.data_next_byte += byte_req;
            pc.data_remaining_bytes = (byte_req <= pc.data_remaining_bytes) ? pc.data_remaining_bytes - byte_req : 0;
            pc.reserved_bytes += byte_req;
        }

        reserved_delta += row_req * sc.fixed_bytes_per_row + byte_req;
    }

    local_reserved_bytes.fetch_add(reserved_delta, std::memory_order_relaxed);
}


Allocator::Allocator(PartSchema schema_, size_t partitions_, size_t /*expected_total_rows*/, AllocatorOptions options_)
    : part_schema(std::move(schema_))
    , num_partitions(partitions_)
    , opts(options_)
{
    if (num_partitions == 0)
        throw Exception(ErrorCodes::BAD_ARGUMENTS, "RadixShuffle::Allocator: partitions must be > 0");
    for (const auto & slot : part_schema.fixed_slots)
    {
        if (slot.alignment == 0 || (slot.alignment & (slot.alignment - 1)) != 0)
            throw Exception(ErrorCodes::BAD_ARGUMENTS, "RadixShuffle::Allocator: slot alignment must be a power of two");
    }
}


Allocator::~Allocator()
{
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
    auto handle = std::unique_ptr<Handle>(new Handle(*this, num_partitions));
    auto * raw = handle.get();
    handles.push_back(std::move(handle));
    return raw;
}


void Allocator::release(Handle * handle)
{
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


uint64_t Allocator::activePartitions() const noexcept
{
    std::lock_guard lk(handle_pool_mutex);
    uint64_t sum = 0;
    for (const auto & h : handles)
        sum += h->local_active_partitions.load(std::memory_order_relaxed);
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
