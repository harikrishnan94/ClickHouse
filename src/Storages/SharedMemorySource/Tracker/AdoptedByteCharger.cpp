#include <Storages/SharedMemorySource/Tracker/AdoptedByteCharger.h>

#include <Common/CurrentMemoryTracker.h>
#include <Common/CurrentMetrics.h>
#include <Common/CurrentThread.h>
#include <Common/ProfileEvents.h>

#include <base/types.h>

#include <utility>


namespace CurrentMetrics
{
    extern const Metric ShmAdoptedBytesCurrent;
    extern const Metric ShmAdoptedBytesLogicalCurrent;
}

namespace ProfileEvents
{
    extern const Event ShmAdoptedBlocks;
    extern const Event ShmAdoptedBytesCharged;
    extern const Event ShmAdoptedBytesLogical;
    extern const Event ShmRetainsAcquired;
}


namespace DB
{

AdoptedByteCharger::AdoptedByteCharger()
    : state_(std::make_shared<AdoptedByteState>())
{
}

ChargeHandle AdoptedByteCharger::charge(size_t adopted_bytes, size_t logical_bytes)
{
    /// Snapshot the calling thread's query group BEFORE we charge against it. The returned
    /// ChargeHandle keeps this group alive and releases against its query-level tracker even
    /// if the handle is dropped by a different or detached pipeline thread (H2). Null is
    /// legal: ChargeHandle::release falls back to the CurrentMemoryTracker path for
    /// total-only or tracker-less tests.
    auto query_group_at_charge = CurrentThread::getGroup();

    /// Order matters (Finding 5 + AC5 "no charged-then-rolled-back transient"): the alloc
    /// comes FIRST and may throw `MEMORY_LIMIT_EXCEEDED`. We do not touch the feature-local
    /// counters, current gauges, or any ProfileEvent until the tracker has accepted the charge,
    /// so a thrown alloc leaves every observable surface at its pre-call value WITHOUT a
    /// rollback try/catch (no rollback path = no transient interval where a concurrent
    /// reader could observe a not-yet-rolled-back counter spike).
    [[maybe_unused]] auto trace = CurrentMemoryTracker::alloc(static_cast<Int64>(adopted_bytes));

    /// CurrentMemoryTracker batches small alloc/free deltas in ThreadStatus::untracked_memory.
    /// Once a query group is captured, flush accepted adopted-byte charges immediately so
    /// cross-thread or post-detach release can subtract the exact same bytes from the query
    /// tracker without depending on the producer thread's raw MemoryTracker lifetime.
    if (query_group_at_charge)
        CurrentThread::flushUntrackedMemory();

    /// Tracker accepted the charge — now publish the success-path observability surfaces
    /// and bump the feature-local counters. From here on the operation is committed; the
    /// returned ChargeHandle owns the inverse operations and runs them in its destructor
    /// (I7 exact at the feature boundary; I8 enforcement contract).
    state_->charged_current.fetch_add(static_cast<int64_t>(adopted_bytes), std::memory_order_acq_rel);
    state_->logical_current.fetch_add(static_cast<int64_t>(logical_bytes), std::memory_order_acq_rel);
    CurrentMetrics::add(CurrentMetrics::ShmAdoptedBytesCurrent, static_cast<CurrentMetrics::Value>(adopted_bytes));
    CurrentMetrics::add(CurrentMetrics::ShmAdoptedBytesLogicalCurrent, static_cast<CurrentMetrics::Value>(logical_bytes));
    ProfileEvents::increment(ProfileEvents::ShmAdoptedBytesCharged, adopted_bytes);
    ProfileEvents::increment(ProfileEvents::ShmAdoptedBytesLogical, logical_bytes);
    ProfileEvents::increment(ProfileEvents::ShmAdoptedBlocks);
    ProfileEvents::increment(ProfileEvents::ShmRetainsAcquired);

    return ChargeHandle(adopted_bytes, logical_bytes, state_, std::move(query_group_at_charge));
}

}
