#include <Storages/SharedMemorySource/Wire/SharedMemoryRegion.h>
#include <Storages/SharedMemorySource/Wire/Layout.h>

#include <Common/Exception.h>
#include <Common/ErrnoException.h>

#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <fcntl.h>
#include <unistd.h>

#include <atomic>
#include <cerrno>
#include <utility>


namespace DB
{

namespace ErrorCodes
{
    extern const int SHM_ATTACH_FAILED;
    extern const int SHM_HANDSHAKE_INVALID;
}

using SharedMemoryWire::HandshakeRegion;
using SharedMemoryWire::SlotEntry;
using SharedMemoryWire::SchemaEntry;
using SharedMemoryWire::SHM_MAGIC;
using SharedMemoryWire::SHM_ABI_VERSION_1;
using SharedMemoryWire::IMPL_MAX_K;
using SharedMemoryWire::IMPL_MAX_COLUMNS;

namespace
{

struct ByteRange
{
    const char * label;
    uint64_t begin;
    uint64_t end;
};

bool addOverflows(uint64_t a, uint64_t b, uint64_t & out) noexcept
{
    out = a + b;
    return out < a;
}

bool mulOverflows(uint64_t a, uint64_t b, uint64_t & out) noexcept
{
    if (a == 0 || b == 0) { out = 0; return false; }
    out = a * b;
    return out / a != b;
}

[[noreturn]] void throwHandshakeInvalid(int fd, void * mapping, size_t size, String msg)
{
    if (mapping != nullptr && mapping != MAP_FAILED)
        ::munmap(mapping, size);
    if (fd >= 0)
        ::close(fd);
    throw Exception(ErrorCodes::SHM_HANDSHAKE_INVALID, "{}", msg);
}

void validateHandshake(const HandshakeRegion * hs, size_t size, int fd, void * mapping, const String & name)
{
    /// First read is acquire-ordered per `shm-block-stream.md` §ABI version negotiation;
    /// acquire on magic implies acquire of every other field the producer wrote first.
    const uint64_t observed_magic = hs->magic.load(std::memory_order_acquire);
    if (observed_magic != SHM_MAGIC)
        throwHandshakeInvalid(fd, mapping, size, fmt::format(
            "SHM '{}' wrong magic 0x{:016x}, expected 0x{:016x}", name, observed_magic, SHM_MAGIC));

    if (hs->abi_version != SHM_ABI_VERSION_1)
        throwHandshakeInvalid(fd, mapping, size, fmt::format(
            "SHM '{}' unsupported abi_version={} (consumer supports {{{}}})",
            name, hs->abi_version, SHM_ABI_VERSION_1));

    if (hs->ring_depth_k == 0 || hs->ring_depth_k > IMPL_MAX_K)
        throwHandshakeInvalid(fd, mapping, size, fmt::format(
            "SHM '{}' ring_depth_k={} out of range [1, {}]", name, hs->ring_depth_k, IMPL_MAX_K));

    if (hs->schema_count == 0 || hs->schema_count > IMPL_MAX_COLUMNS)
        throwHandshakeInvalid(fd, mapping, size, fmt::format(
            "SHM '{}' schema_count={} out of range [1, {}]", name, hs->schema_count, IMPL_MAX_COLUMNS));

    if (hs->slot_table_stride < sizeof(SlotEntry))
        throwHandshakeInvalid(fd, mapping, size, fmt::format(
            "SHM '{}' slot_table_stride={} < sizeof(SlotEntry)={}",
            name, hs->slot_table_stride, sizeof(SlotEntry)));

    /// F9 (handshake validity, precondition 7 extension): alignment + size-
    /// exactness checks on the table descriptors. The wire ABI v1
    /// (Wire/Layout.h) pins the alignment of every region the consumer
    /// reinterpret_casts into; a stride or offset that violates `alignof` for
    /// the target type yields a misaligned `std::atomic<...>` access inside
    /// `SlotEntry` / `SchemaEntry`, which is UB on architectures that require
    /// natural alignment (ARM strict-align). Schema-table size must be
    /// *exactly* `schema_count * sizeof(SchemaEntry)` — a truncated value
    /// hides entries past the declared size; an inflated value lets a
    /// non-conforming producer hide unused bytes (or other regions) inside
    /// the declared schema span. Both classes of handshake corruption are
    /// SHM_HANDSHAKE_INVALID (the consumer cannot safely interpret the
    /// payload).

    if (hs->slot_table_stride % alignof(SlotEntry) != 0)
        throwHandshakeInvalid(fd, mapping, size, fmt::format(
            "SHM '{}' slot_table_stride={} is not a multiple of alignof(SlotEntry)={}",
            name, hs->slot_table_stride, alignof(SlotEntry)));

    if (hs->slot_table_offset % alignof(SlotEntry) != 0)
        throwHandshakeInvalid(fd, mapping, size, fmt::format(
            "SHM '{}' slot_table_offset={} is not a multiple of alignof(SlotEntry)={}",
            name, hs->slot_table_offset, alignof(SlotEntry)));

    if (hs->schema_table_offset % alignof(SchemaEntry) != 0)
        throwHandshakeInvalid(fd, mapping, size, fmt::format(
            "SHM '{}' schema_table_offset={} is not a multiple of alignof(SchemaEntry)={}",
            name, hs->schema_table_offset, alignof(SchemaEntry)));

    /// Exact-size check uses checked arithmetic: schema_count is bounded by
    /// IMPL_MAX_COLUMNS (already validated above), and sizeof(SchemaEntry) is
    /// a compile-time small constant, so the product never overflows uint64_t
    /// — but use `mulOverflows` regardless to keep the safety idiom uniform
    /// with the surrounding range checks.
    uint64_t expected_schema_table_size = 0;
    if (mulOverflows(static_cast<uint64_t>(hs->schema_count),
                     static_cast<uint64_t>(sizeof(SchemaEntry)),
                     expected_schema_table_size))
        throwHandshakeInvalid(fd, mapping, size, fmt::format(
            "SHM '{}' schema_count={} * sizeof(SchemaEntry)={} overflows uint64_t",
            name, hs->schema_count, sizeof(SchemaEntry)));
    if (hs->schema_table_size != expected_schema_table_size)
        throwHandshakeInvalid(fd, mapping, size, fmt::format(
            "SHM '{}' schema_table_size={} != schema_count*sizeof(SchemaEntry)={} "
            "(schema_count={})",
            name, hs->schema_table_size, expected_schema_table_size, hs->schema_count));

    /// The data region holds column descriptor arrays and value/offsets
    /// buffers; the strictest alignment the descriptor parsers expect from a
    /// region-base pointer is `alignof(uint64_t)` (every per-column buffer
    /// offset is then validated separately at adopt() time). A misaligned
    /// `data_region_offset` would propagate misalignment into every
    /// descriptor parse, with no salvage path.
    if (hs->data_region_offset % alignof(uint64_t) != 0)
        throwHandshakeInvalid(fd, mapping, size, fmt::format(
            "SHM '{}' data_region_offset={} is not a multiple of alignof(uint64_t)={}",
            name, hs->data_region_offset, alignof(uint64_t)));

    /// Per `pollable-shm-source.md` §Producer-side preconditions row 7: regions
    /// must fit within the SHM and not overlap each other or the handshake.
    uint64_t slot_table_extent = 0;
    uint64_t slot_table_end = 0;
    uint64_t data_region_end = 0;
    uint64_t schema_table_end = 0;

    if (mulOverflows(hs->ring_depth_k, hs->slot_table_stride, slot_table_extent)
        || addOverflows(hs->slot_table_offset, slot_table_extent, slot_table_end)
        || slot_table_end > size)
        throwHandshakeInvalid(fd, mapping, size, fmt::format(
            "SHM '{}' slot table [offset={}, K*stride={}] does not fit in size={}",
            name, hs->slot_table_offset, slot_table_extent, size));

    if (addOverflows(hs->data_region_offset, hs->data_region_size, data_region_end)
        || data_region_end > size)
        throwHandshakeInvalid(fd, mapping, size, fmt::format(
            "SHM '{}' data region [offset={}, size={}] does not fit in size={}",
            name, hs->data_region_offset, hs->data_region_size, size));

    if (addOverflows(hs->schema_table_offset, hs->schema_table_size, schema_table_end)
        || schema_table_end > size)
        throwHandshakeInvalid(fd, mapping, size, fmt::format(
            "SHM '{}' schema table [offset={}, size={}] does not fit in size={}",
            name, hs->schema_table_offset, hs->schema_table_size, size));

    const ByteRange regions[] = {
        {"handshake", 0, sizeof(HandshakeRegion)},
        {"slot_table", hs->slot_table_offset, slot_table_end},
        {"schema_table", hs->schema_table_offset, schema_table_end},
        {"data_region", hs->data_region_offset, data_region_end},
    };
    for (size_t i = 0; i < std::size(regions); ++i)
        for (size_t j = i + 1; j < std::size(regions); ++j)
            if (!(regions[i].end <= regions[j].begin || regions[j].end <= regions[i].begin))
                throwHandshakeInvalid(fd, mapping, size, fmt::format(
                    "SHM '{}' regions overlap: {} [{}, {}) vs {} [{}, {})", name,
                    regions[i].label, regions[i].begin, regions[i].end,
                    regions[j].label, regions[j].begin, regions[j].end));
}

}

std::unique_ptr<SharedMemoryRegion> SharedMemoryRegion::attach(const String & name)
{
    /// O_RDWR (not O_RDONLY): the consumer release-stores into the slot
    /// table's `retain_refcount`, `state`, and `transition_counter` per
    /// `shm-block-stream.md` §Retain/release contract + §Publication state
    /// machine + precondition 24. PAYLOAD bytes remain read-only by
    /// convention (system.md N2: no ClickHouse-writes-to-SHM on the data
    /// plane); the kernel can't enforce that split, so PROT_READ|PROT_WRITE
    /// applies to the whole mapping and the no-payload-write rule is
    /// enforced by code review.
    const int fd = ::shm_open(name.c_str(), O_RDWR, 0);
    if (fd < 0)
    {
        const int saved_errno = errno;
        if (saved_errno == ENOENT)
            throw Exception(ErrorCodes::SHM_ATTACH_FAILED, "SHM object '{}' does not exist", name);
        if (saved_errno == EACCES)
            throw Exception(ErrorCodes::SHM_ATTACH_FAILED, "permission denied for SHM object '{}'", name);
        ErrnoException::throwWithErrno(ErrorCodes::SHM_ATTACH_FAILED, saved_errno,
            "shm_open failed for SHM object '{}'", name);
    }

    struct stat stat_buf{};
    if (::fstat(fd, &stat_buf) != 0)
    {
        const int saved_errno = errno;
        ::close(fd);
        ErrnoException::throwWithErrno(ErrorCodes::SHM_ATTACH_FAILED, saved_errno,
            "fstat failed for SHM object '{}'", name);
    }

    if (stat_buf.st_size < 0
        || static_cast<size_t>(stat_buf.st_size) < sizeof(HandshakeRegion))
    {
        const auto observed = stat_buf.st_size;
        ::close(fd);
        throw Exception(ErrorCodes::SHM_HANDSHAKE_INVALID,
            "SHM object '{}' size={} is smaller than sizeof(HandshakeRegion)={}",
            name, observed, sizeof(HandshakeRegion));
    }

    const size_t size = static_cast<size_t>(stat_buf.st_size);
    void * mapping = ::mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (mapping == MAP_FAILED)
    {
        const int saved_errno = errno;
        ::close(fd);
        ErrnoException::throwWithErrno(ErrorCodes::SHM_ATTACH_FAILED, saved_errno,
            "mmap failed for SHM object '{}' (size={})", name, size);
    }

    validateHandshake(static_cast<const HandshakeRegion *>(mapping), size, fd, mapping, name);

    return std::unique_ptr<SharedMemoryRegion>(new SharedMemoryRegion(name, fd, mapping, size));
}

SharedMemoryRegion::SharedMemoryRegion(String name_, int fd_, void * mapping_, size_t size_)
    : shm_name(std::move(name_))
    , shm_fd(fd_)
    , mapping(mapping_)
    , mapping_size(size_)
{
}

SharedMemoryRegion::~SharedMemoryRegion()
{
    /// Destructor must not unlink: the consumer does not own the SHM name
    /// per `shm-block-stream.md` §SHM primitive (producer creates and
    /// unlinks; the consumer detaches by unmap + close only). The mapping
    /// was opened RW for control-plane writes (slot state, refcount,
    /// transition counter) but payload bytes were never written by the
    /// consumer per `system.md` N2; unmap is symmetric regardless.
    if (mapping != nullptr && mapping != MAP_FAILED)
        ::munmap(mapping, mapping_size);
    if (shm_fd >= 0)
        ::close(shm_fd);
}

const SharedMemoryWire::HandshakeRegion & SharedMemoryRegion::handshake() const noexcept
{
    return *reinterpret_cast<const HandshakeRegion *>(mapping);
}

}
