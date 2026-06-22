#include <Storages/SharedMemorySource/Tracker/ChargeHandle.h>

#include <Common/CurrentMemoryTracker.h>
#include <Common/CurrentMetrics.h>
#include <Common/CurrentThread.h>
#include <Common/MemoryTracker.h>
#include <Common/ProfileEvents.h>
#include <Common/ThreadStatus.h>

#include <base/types.h>

#include <utility>


namespace CurrentMetrics
{
    extern const Metric ShmAdoptedBytesCurrent;
    extern const Metric ShmAdoptedBytesLogicalCurrent;
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
    std::shared_ptr<AdoptedByteState> state,
    std::shared_ptr<ThreadGroup> query_group_at_charge) noexcept
    : bytes_(charged_bytes)
    , logical_bytes_(logical_bytes)
    , state_(std::move(state))
    , query_group_to_release_(std::move(query_group_at_charge))
{
}

ChargeHandle::ChargeHandle(ChargeHandle && other) noexcept
    : bytes_(std::exchange(other.bytes_, 0))
    , logical_bytes_(std::exchange(other.logical_bytes_, 0))
    , state_(std::move(other.state_))
    , query_group_to_release_(std::move(other.query_group_to_release_))
{
}

ChargeHandle & ChargeHandle::operator=(ChargeHandle && other) noexcept
{
    if (this != &other)
    {
        release();
        bytes_ = std::exchange(other.bytes_, 0);
        logical_bytes_ = std::exchange(other.logical_bytes_, 0);
        state_ = std::move(other.state_);
        query_group_to_release_ = std::move(other.query_group_to_release_);
    }
    return *this;
}

ChargeHandle::~ChargeHandle()
{
    release();
}

void ChargeHandle::release() noexcept
{
    if (bytes_ == 0 && logical_bytes_ == 0)
        return;

    /// Reverse the tracker charge first. If the destructor still runs under the same query
    /// group, use the public CurrentMemoryTracker path and immediately flush the cushion so
    /// the query-level charge is removed now rather than at the next threshold crossing. If
    /// the handle is destroyed after detaching from the query, release against the owning
    /// ThreadGroup's query tracker; this avoids the raw producer-thread MemoryTracker pointer
    /// that can dangle after ThreadStatus teardown.
    if (bytes_ != 0 && query_group_to_release_)
    {
        if (CurrentThread::getGroup() == query_group_to_release_)
        {
            [[maybe_unused]] auto trace = CurrentMemoryTracker::free(static_cast<Int64>(bytes_));
            CurrentThread::flushUntrackedMemory();
        }
        else
        {
            query_group_to_release_->memory_tracker.adjustWithUntrackedMemory(-static_cast<Int64>(bytes_));
        }
    }
    else if (bytes_ != 0)
    {
        [[maybe_unused]] auto trace = CurrentMemoryTracker::free(static_cast<Int64>(bytes_));
    }

    if (state_)
    {
        state_->charged_current.fetch_sub(static_cast<int64_t>(bytes_), std::memory_order_acq_rel);
        state_->logical_current.fetch_sub(static_cast<int64_t>(logical_bytes_), std::memory_order_acq_rel);
    }

    CurrentMetrics::sub(CurrentMetrics::ShmAdoptedBytesCurrent, static_cast<CurrentMetrics::Value>(bytes_));
    CurrentMetrics::sub(CurrentMetrics::ShmAdoptedBytesLogicalCurrent, static_cast<CurrentMetrics::Value>(logical_bytes_));
    ProfileEvents::increment(ProfileEvents::ShmRetainsReleased);

    bytes_ = 0;
    logical_bytes_ = 0;
    state_.reset();
    query_group_to_release_.reset();
}

}
