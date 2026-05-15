#include <Storages/SharedMemorySource/Tracker/ChargeHandle.h>

#include <Common/CurrentMemoryTracker.h>
#include <Common/CurrentMetrics.h>
#include <Common/MemoryTracker.h>
#include <Common/ProfileEvents.h>

#include <base/types.h>

#include <utility>


namespace CurrentMetrics
{
    extern const Metric ShmAdoptedBytesCurrent;
}

namespace ProfileEvents
{
    extern const Event ShmRetainsReleased;
}


namespace DB
{

ChargeHandle::ChargeHandle(
    size_t charged_bytes,
    size_t logical_bytes,
    std::atomic<int64_t> * counter,
    MemoryTracker * tracker_at_charge) noexcept
    : bytes_(charged_bytes)
    , logical_bytes_(logical_bytes)
    , counter_to_decrement_(counter)
    , tracker_to_release_(tracker_at_charge)
{
}

ChargeHandle::ChargeHandle(ChargeHandle && other) noexcept
    : bytes_(std::exchange(other.bytes_, 0))
    , logical_bytes_(std::exchange(other.logical_bytes_, 0))
    , counter_to_decrement_(std::exchange(other.counter_to_decrement_, nullptr))
    , tracker_to_release_(std::exchange(other.tracker_to_release_, nullptr))
{
}

ChargeHandle & ChargeHandle::operator=(ChargeHandle && other) noexcept
{
    if (this != &other)
    {
        release();
        bytes_ = std::exchange(other.bytes_, 0);
        logical_bytes_ = std::exchange(other.logical_bytes_, 0);
        counter_to_decrement_ = std::exchange(other.counter_to_decrement_, nullptr);
        tracker_to_release_ = std::exchange(other.tracker_to_release_, nullptr);
    }
    return *this;
}

ChargeHandle::~ChargeHandle()
{
    release();
}

void ChargeHandle::release() noexcept
{
    if (bytes_ == 0)
        return;

    /// Reverse the tracker charge first. Two paths:
    ///   1. Captured-tracker path (production + tests with MainThreadStatus): release goes
    ///      directly to the MemoryTracker chain we snapshotted at charge time, bypassing
    ///      the thread-local lookup that CurrentMemoryTracker::free would do. This pins
    ///      cross-thread release to the charging thread's chain (Finding 5). We use the
    ///      public `adjustWithUntrackedMemory(-bytes)` surface because MemoryTracker::free
    ///      itself is private (friend-only to CurrentMemoryTracker); the negative-argument
    ///      branch of adjustWithUntrackedMemory forwards straight to free, and free is
    ///      effectively noexcept ("free should never throw" per the comment in
    ///      MemoryTracker.cpp), which is what we need inside a destructor.
    ///   2. Null-tracker fallback (gtest binaries where CurrentThread::getMemoryTracker()
    ///      returned nullptr at charge time): CurrentMemoryTracker::alloc was itself a
    ///      no-op on the charge side (no thread-local + no total_memory_tracker), so the
    ///      symmetric free is also a no-op. Calling CurrentMemoryTracker::free here keeps
    ///      the alloc/free pair structurally mirrored. The AllocationTrace return is the
    ///      sampling hook for alloc accounting; nothing to attribute on free.
    if (tracker_to_release_)
    {
        tracker_to_release_->adjustWithUntrackedMemory(-static_cast<Int64>(bytes_));
    }
    else
    {
        [[maybe_unused]] auto trace = CurrentMemoryTracker::free(static_cast<Int64>(bytes_));
    }

    if (counter_to_decrement_)
        counter_to_decrement_->fetch_sub(static_cast<int64_t>(bytes_), std::memory_order_acq_rel);

    CurrentMetrics::sub(CurrentMetrics::ShmAdoptedBytesCurrent, static_cast<CurrentMetrics::Value>(bytes_));
    ProfileEvents::increment(ProfileEvents::ShmRetainsReleased);

    bytes_ = 0;
    logical_bytes_ = 0;
    counter_to_decrement_ = nullptr;
    tracker_to_release_ = nullptr;
}

}
