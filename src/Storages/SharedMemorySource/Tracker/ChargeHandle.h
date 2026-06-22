#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>

namespace DB
{

class ThreadGroup;

struct AdoptedByteState
{
    std::atomic<int64_t> charged_current{0};
    std::atomic<int64_t> logical_current{0};
};

/// RAII wrapper around an `Adopted byte count` charge against the active query-level
/// MemoryTracker. The destructor releases the charge against the same query group that was
/// active at charge time (captured as an owning `ThreadGroup` handle when one exists),
/// atomically decrements the per-source shared counter state the caller registered, and updates
/// the process-wide current gauges plus the `ShmRetainsReleased` cumulative event.
///
/// The constructor is `noexcept` and only stores the byte counts, shared counter state, and
/// captured query group; the `CurrentMemoryTracker::alloc` call (which may throw
/// `MEMORY_LIMIT_EXCEEDED`) is performed by `AdoptedByteCharger::charge()` *before* this
/// handle is constructed, so a limit-rejection failure path never produces a
/// partially-constructed handle.
///
/// Why capture the query group at charge time (H2): the thread-level MemoryTracker is owned by
/// `ThreadStatus` and has no owning public handle, so storing its raw pointer would be unsafe if
/// an adopted column outlived the producer thread. The query group's `MemoryTracker` is owned by
/// `ThreadGroup`; holding a `std::shared_ptr<ThreadGroup>` keeps that query-level tracker alive
/// until the last handle releases. When destruction still runs in the captured query group, the
/// release uses `CurrentMemoryTracker::free` and flushes the thread-local cushion immediately,
/// mirroring the public allocation path. If destruction runs after detaching from the query, it
/// releases directly against the captured query tracker; this preserves exact query/process
/// accounting without dereferencing the producer thread's raw tracker.
///
/// A moved-from or default-constructed handle is a no-op on destruction. If
/// `query_group_to_release_` is nullptr, the destructor falls back to `CurrentMemoryTracker::free`,
/// mirroring the charge path for total-only or tracker-less tests. The adopted-byte state is
/// a shared object owned by handles and the source charger, so a chunk may outlive the charger
/// without leaving the destructor with a dangling counter pointer.
///
/// Spec authority: memory-tracker-integration spec §Charge entry point, §Release semantics,
/// AC5 (no charged-then-rolled-back transient), I7 (Adopted-byte accounting is exact at the
/// feature boundary); system spec I5, I10.
class ChargeHandle
{
public:
    /// Default-constructed handle is a no-op sentinel.
    ChargeHandle() noexcept = default;

    /// Internal constructor used by `AdoptedByteCharger::charge()`. Public so the charger
    /// can instantiate without friendship gymnastics; not intended for direct user use.
    ///
    /// `charged_bytes` is the value previously passed to `CurrentMemoryTracker::alloc`
    /// (logical payload plus safe-read padding); the destructor releases the same value
    /// against `query_group_at_charge`. `logical_bytes` is the safe-read-padding-excluded
    /// payload, retained for observability via `logicalBytes()` (see system glossary
    /// "Adopted byte count"). `state` is the feature-local per-source counter state; the
    /// destructor atomically subtracts the charged and logical byte counts from it.
    /// `query_group_at_charge` is the query group captured at charge time (see the
    /// class-level doc for the lifetime contract); may be nullptr in test binaries where
    /// no query group was set up.
    ChargeHandle(
        size_t charged_bytes,
        size_t logical_bytes,
        std::shared_ptr<AdoptedByteState> state,
        std::shared_ptr<ThreadGroup> query_group_at_charge = {}) noexcept;

    /// Releases the charge if active.
    ~ChargeHandle();

    ChargeHandle(const ChargeHandle &) = delete;
    ChargeHandle & operator=(const ChargeHandle &) = delete;

    /// Moveable; the source handle becomes a no-op sentinel.
    ChargeHandle(ChargeHandle && other) noexcept;
    ChargeHandle & operator=(ChargeHandle && other) noexcept;

    /// True iff this handle currently holds a charge.
    bool isActive() const noexcept { return bytes_ != 0 || logical_bytes_ != 0; }

    /// The charged byte count (zero if moved-from / default-constructed).
    size_t bytes() const noexcept { return bytes_; }

    /// The logical payload byte count (zero if moved-from / default-constructed).
    size_t logicalBytes() const noexcept { return logical_bytes_; }

private:
    void release() noexcept;

    /// Trailing underscores disambiguate from the same-named accessor methods above.
    /// `readability-identifier-naming` is disabled project-wide (.clang-tidy), so this
    /// convention is permitted; NOLINTs silence the IDE-only warning.
    size_t bytes_ = 0;             // NOLINT(readability-identifier-naming)
    size_t logical_bytes_ = 0;     // NOLINT(readability-identifier-naming)
    std::shared_ptr<AdoptedByteState> state_;          // NOLINT(readability-identifier-naming)
    std::shared_ptr<ThreadGroup> query_group_to_release_; // NOLINT(readability-identifier-naming)
};

}
