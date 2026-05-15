#pragma once

#include <base/types.h>

#include <cstddef>
#include <memory>


namespace DB
{

namespace SharedMemoryWire
{
struct HandshakeRegion;
}

/// RAII wrapper around a POSIX shared-memory object the consumer attaches to.
/// Mirrors `MMappedFileDescriptor` (closest in-tree precedent for `mmap` +
/// RAII fd management), but opens via `shm_open` and maps with `MAP_SHARED`
/// so the producer's release-stores become visible under the memory-ordering
/// contract pinned by `shm-block-stream.md` §Memory ordering.
///
/// The mapping is opened with `O_RDWR` / `PROT_READ | PROT_WRITE`. The
/// PAYLOAD bytes (column buffers + descriptors) are still read-only from
/// the consumer's perspective per `system.md` §Non-goals N2 ("no
/// ClickHouse-writes-to-SHM" applies to the data plane). The control-plane
/// `SlotEntry` fields, however, ARE written by the consumer: it increments
/// `retain_refcount` when adopting a block and release-stores `EMPTY` into
/// `state` on the last RetainToken drop per `shm-block-stream.md` §Retain/
/// release contract + §Publication state machine. The same writable mapping
/// also carries the `transition_counter` increment driven by the consumer's
/// P→E transition (precondition 24 deterministic detection).
///
/// The consumer does NOT own the SHM name: the producer creates and
/// `shm_unlink`s it. The destructor unmaps and closes the fd only.
///
/// `attach` discharges the attach-time failures from
/// `pollable-shm-source.md` §Attach-time observable failures that are
/// localizable to the SHM object itself: object-missing / permission-denied
/// (`SHM_ATTACH_FAILED`), plus the magic / version / region-offset checks
/// (`SHM_HANDSHAKE_INVALID`, preconditions 1, 2, 3, 7). Schema cross-
/// validation (preconditions 4–6) lives in the source, since this class
/// knows nothing about the SQL-declared schema.
class SharedMemoryRegion
{
public:
    /// Opens the named POSIX shared memory object read-write (the control
    /// plane requires consumer writes; see class doc), mmaps the whole
    /// region with `MAP_SHARED | PROT_READ | PROT_WRITE`, and validates the
    /// handshake region (magic via acquire-load, ABI version, ring depth,
    /// region offsets).
    static std::unique_ptr<SharedMemoryRegion> attach(const String & name);

    ~SharedMemoryRegion();

    SharedMemoryRegion(const SharedMemoryRegion &) = delete;
    SharedMemoryRegion & operator=(const SharedMemoryRegion &) = delete;
    SharedMemoryRegion(SharedMemoryRegion &&) = delete;
    SharedMemoryRegion & operator=(SharedMemoryRegion &&) = delete;

    const void * data() const noexcept { return mapping; }
    size_t size() const noexcept { return mapping_size; }
    const SharedMemoryWire::HandshakeRegion & handshake() const noexcept;
    const String & name() const noexcept { return shm_name; }

    /// Exposed so the source can `poll(POLLHUP)` for producer-side unlink
    /// detection per `auto_click/specs/pollable-shm-source.md` §Failure
    /// classes row `producer-death-before-eos` (precondition 25).
    int fd() const noexcept { return shm_fd; }

private:
    SharedMemoryRegion(String name_, int fd_, void * mapping_, size_t size_);

    String shm_name;
    int shm_fd;
    void * mapping;
    size_t mapping_size;
};

}
