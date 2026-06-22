#pragma once

#include <Storages/SharedMemorySource/Tracker/ChargeHandle.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>


namespace DB
{

/// Per-source-instance entry point that translates a per-block adopted-byte count into a
/// MemoryTracker charge and hands back a `ChargeHandle` that releases the charge on destruction.
/// Holds feature-local exact counters required by I7 (per-source, test-visible, returns to
/// zero when every `ChargeHandle` issued by this charger is destroyed).
///
/// Lifecycle:
///   - One `AdoptedByteCharger` is owned by the `PollableShmSource` (one per source instance).
///   - For each per-block retain acquisition the source calls `charge(adopted_bytes)`:
///       on success — returns a `ChargeHandle` the source threads into the adopted columns;
///       on failure — throws `MEMORY_LIMIT_EXCEEDED` with the feature-local counter unchanged.
///   - The returned `ChargeHandle`'s destructor releases the tracker charge, decrements the
///     shared feature-local counters, decrements the current metrics, and increments
///     `ShmRetainsReleased` (release logic lives in `ChargeHandle::release()`).
///
/// Concurrency: `charge()` is callable from a single thread (the source's executor thread). The
/// counters are atomic so external observability (gauge, gtest) is safe under concurrent
/// reads, and the `ChargeHandle` destructor — which may run on a different thread once a Block
/// has been handed downstream — uses shared state that remains alive even if the source-owned
/// charger is gone.
///
/// Spec authority: memory-tracker-integration §Charge entry point + §Release semantics + I7 + I8;
/// pollable-shm-source §Per-block adoption call.
class AdoptedByteCharger
{
public:
    AdoptedByteCharger();
    AdoptedByteCharger(const AdoptedByteCharger &) = delete;
    AdoptedByteCharger & operator=(const AdoptedByteCharger &) = delete;
    AdoptedByteCharger(AdoptedByteCharger &&) = delete;
    AdoptedByteCharger & operator=(AdoptedByteCharger &&) = delete;

    /// Charge `adopted_bytes` of producer-owned memory against the active query MemoryTracker
    /// chain. `adopted_bytes` is the charged amount (includes safe-read padding per the system
    /// glossary entry "Adopted byte count"); `logical_bytes` is the payload-only byte count
    /// (data without padding) reported through `ShmAdoptedBytesLogical`.
    ///
    /// Operation order (matters for AC5 "no charged-then-rolled-back transient"):
    ///   1. Snapshot `CurrentThread::getGroup()` so the inverse release on the
    ///      returned handle pins to the same query-level tracker even if the handle is
    ///      dropped on a different pipeline thread.
    ///   2. `CurrentMemoryTracker::alloc(adopted_bytes)` — may throw `MEMORY_LIMIT_EXCEEDED`;
    ///      if it throws, the charger returns without touching any observable surface
    ///      (no counter increment to roll back, no metric/event to undo). This is the
    ///      structural enforcement of AC5.
    ///   3. After the tracker accepts the charge: flush thread-local untracked memory when
    ///      a query group was captured, making the query-level charge visible immediately
    ///      and bounding the release path's untracked-cushion slack to zero for adopted bytes.
    ///   4. Increment the feature-local counter by
    ///      `adopted_bytes`, increment `logical_bytes`, update current metrics, and bump
    ///      `ProfileEvents::ShmAdoptedBytesCharged += adopted_bytes`,
    ///      `ShmAdoptedBytesLogical += logical_bytes`, `ShmAdoptedBlocks += 1`,
    ///      `ShmRetainsAcquired += 1`.
    ///
    /// On tracker rejection (`MEMORY_LIMIT_EXCEEDED` propagates out): the feature-local
    /// counter is unchanged from its pre-call value, the gauge is untouched, and no
    /// ProfileEvent is incremented. There is NO transient interval where any of these is
    /// non-zero relative to its committed state (this is what AC5 prohibits, and what the
    /// pre-Finding-5 try/catch rollback path did NOT guarantee — a concurrent reader could
    /// observe the in-flight pre-increment before the rollback ran).
    ///
    /// Returns a `ChargeHandle` whose destruction performs the inverse operations against
    /// the snapshotted query group (H2: cross-thread-safe release without retaining a raw
    /// thread-tracker pointer).
    [[nodiscard]] ChargeHandle charge(size_t adopted_bytes, size_t logical_bytes);

    /// Convenience overload for the common case where logical == charged (no safe-read padding
    /// to subtract; e.g. fixed-width primitive columns).
    [[nodiscard]] ChargeHandle charge(size_t adopted_bytes) { return charge(adopted_bytes, adopted_bytes); }

    /// Test-visible exact counter. Returns the sum of `charged_bytes` for every live
    /// `ChargeHandle` this charger has issued. Safe to call concurrently with `charge()` and
    /// with handle destructors. Returns 0 when the source is freshly constructed and after all
    /// handles issued by this charger have been destroyed.
    ///
    /// Spec authority: memory-tracker-integration §Observability of the feature-local counter.
    int64_t currentChargedBytes() const noexcept { return state_->charged_current.load(std::memory_order_acquire); }

    /// Test-visible exact logical payload counter. Returns the sum of `logical_bytes` for
    /// every live `ChargeHandle` this charger has issued.
    int64_t currentLogicalBytes() const noexcept { return state_->logical_current.load(std::memory_order_acquire); }

private:
    std::shared_ptr<AdoptedByteState> state_; // NOLINT(readability-identifier-naming)
};

}
