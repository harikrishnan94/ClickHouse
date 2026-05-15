#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>

class MemoryTracker;

namespace DB
{

/// RAII wrapper around an `Adopted byte count` charge against the active query-level
/// MemoryTracker. The destructor releases the charge against the SAME MemoryTracker chain
/// that was decremented at construction (captured at charge time as a raw pointer,
/// `tracker_to_release_`), atomically decrements the per-source feature-local counter the
/// caller registered, and updates the process-wide `ShmAdoptedBytesCurrent` gauge plus the
/// `ShmRetainsReleased` cumulative event.
///
/// The constructor is `noexcept` and only stores the byte count, counter pointer, and
/// captured tracker pointer; the `CurrentMemoryTracker::alloc` call (which may throw
/// `MEMORY_LIMIT_EXCEEDED`) is performed by `AdoptedByteCharger::charge()` *before* this
/// handle is constructed, so a limit-rejection failure path never produces a
/// partially-constructed handle.
///
/// Why capture the tracker at charge time (Finding 5): the calling thread's MemoryTracker
/// chain (`CurrentThread::getMemoryTracker()`) is captured at charge time and passed in as
/// `tracker_at_charge`. The destructor calls `tracker_at_charge->adjustWithUntrackedMemory
/// (-bytes)` directly — bypassing CurrentMemoryTracker's thread-local lookup — so a handle
/// dropped on a pipeline thread different from the producer thread still decrements the
/// right tracker chain. Without this pinning, a Block emitted by the source and consumed
/// downstream on a different thread would release through the destruction thread's chain,
/// silently double-counting against the wrong query/user tracker.
///
/// Lifetime contract: `tracker_at_charge` must outlive every ChargeHandle issued against it.
/// In phase 1 the source's lifetime equals the query's lifetime, and adopted columns
/// outlive the source only within the same query (chunks emitted by the source are consumed
/// by downstream operators of the same query). The thread-level MemoryTracker captured here
/// is owned by the executor thread's ThreadStatus; its parent is the query group's tracker;
/// both are alive for the duration of any chunk derived from the source. Once the query
/// completes, the pipeline shuts down BEFORE the query group tracker is torn down — every
/// ChargeHandle is destroyed before the captured pointer would dangle.
///
/// A moved-from or default-constructed handle is a no-op on destruction. If
/// `tracker_to_release_` is nullptr (gtest binaries without MainThreadStatus, where
/// `CurrentMemoryTracker::alloc` was itself a no-op), the destructor falls back to
/// `CurrentMemoryTracker::free` — symmetrical with the no-tracker alloc path.
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
    /// against `tracker_at_charge`. `logical_bytes` is the safe-read-padding-excluded
    /// payload, retained for observability via `logicalBytes()` (see system glossary
    /// "Adopted byte count"). `counter_to_decrement` is the feature-local per-source counter
    /// the charger owns; the destructor atomically subtracts `charged_bytes` from it.
    /// `tracker_at_charge` is the MemoryTracker chain captured at charge time (see the
    /// class-level doc for the lifetime contract); may be nullptr in test binaries where
    /// no thread-local or total tracker was set up.
    ChargeHandle(
        size_t charged_bytes,
        size_t logical_bytes,
        std::atomic<int64_t> * counter_to_decrement,
        MemoryTracker * tracker_at_charge) noexcept;

    /// Releases the charge if active.
    ~ChargeHandle();

    ChargeHandle(const ChargeHandle &) = delete;
    ChargeHandle & operator=(const ChargeHandle &) = delete;

    /// Moveable; the source handle becomes a no-op sentinel.
    ChargeHandle(ChargeHandle && other) noexcept;
    ChargeHandle & operator=(ChargeHandle && other) noexcept;

    /// True iff this handle currently holds a charge.
    bool isActive() const noexcept { return bytes_ != 0; }

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
    std::atomic<int64_t> * counter_to_decrement_ = nullptr; // NOLINT(readability-identifier-naming)
    MemoryTracker * tracker_to_release_ = nullptr;          // NOLINT(readability-identifier-naming)
};

}
