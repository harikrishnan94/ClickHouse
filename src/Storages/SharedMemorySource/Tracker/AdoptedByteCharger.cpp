#include <Storages/SharedMemorySource/Tracker/AdoptedByteCharger.h>

#include <Common/CurrentMemoryTracker.h>
#include <Common/CurrentMetrics.h>
#include <Common/CurrentThread.h>
#include <Common/ProfileEvents.h>

#include <base/types.h>


namespace CurrentMetrics
{
    extern const Metric ShmAdoptedBytesCurrent;
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

ChargeHandle AdoptedByteCharger::charge(size_t adopted_bytes, size_t logical_bytes)
{
    /// Snapshot the calling thread's MemoryTracker chain BEFORE we charge against it. The
    /// returned ChargeHandle stores this pointer and releases against the same chain on
    /// destruction, regardless of which pipeline thread drops the handle (Finding 5).
    /// Null is legal: gtest binaries without MainThreadStatus / total_memory_tracker setup
    /// see no tracker on either side of the alloc/free pair (see ChargeHandle::release for
    /// the symmetric fallback).
    auto * tracker_at_charge = CurrentThread::getMemoryTracker();

    /// Order matters (Finding 5 + AC5 "no charged-then-rolled-back transient"): the alloc
    /// comes FIRST and may throw `MEMORY_LIMIT_EXCEEDED`. We do not touch the feature-local
    /// counter, the gauge, or any ProfileEvent until the tracker has accepted the charge,
    /// so a thrown alloc leaves every observable surface at its pre-call value WITHOUT a
    /// rollback try/catch (no rollback path = no transient interval where a concurrent
    /// reader could observe a not-yet-rolled-back counter spike).
    [[maybe_unused]] auto trace = CurrentMemoryTracker::alloc(static_cast<Int64>(adopted_bytes));

    /// Tracker accepted the charge — now publish the success-path observability surfaces
    /// and bump the feature-local counter. From here on the operation is committed; the
    /// returned ChargeHandle owns the inverse operations and runs them in its destructor
    /// (I7 exact at the feature boundary; I8 enforcement contract).
    feature_local_counter_.fetch_add(static_cast<int64_t>(adopted_bytes), std::memory_order_acq_rel);
    CurrentMetrics::add(CurrentMetrics::ShmAdoptedBytesCurrent, static_cast<CurrentMetrics::Value>(adopted_bytes));
    ProfileEvents::increment(ProfileEvents::ShmAdoptedBytesCharged, adopted_bytes);
    ProfileEvents::increment(ProfileEvents::ShmAdoptedBytesLogical, logical_bytes);
    ProfileEvents::increment(ProfileEvents::ShmAdoptedBlocks);
    ProfileEvents::increment(ProfileEvents::ShmRetainsAcquired);

    return ChargeHandle(adopted_bytes, logical_bytes, &feature_local_counter_, tracker_at_charge);
}

}
