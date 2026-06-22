#include <gtest/gtest.h>

#include <Storages/SharedMemorySource/Adoption/RetainToken.h>
#include <Storages/SharedMemorySource/Tracker/ChargeHandle.h>

#include <Common/CurrentMemoryTracker.h>
#include <Common/CurrentMetrics.h>
#include <Common/CurrentThread.h>
#include <Common/Exception.h>
#include <Common/MemoryTracker.h>
#include <Common/ThreadStatus.h>
#include <Common/scope_guard_safe.h>

#include <base/types.h>

#include <atomic>
#include <functional>
#include <memory>
#include <thread>
#include <utility>

using namespace DB;

namespace CurrentMetrics
{
extern const Metric ShmAdoptedBytesCurrent;
extern const Metric ShmAdoptedBytesLogicalCurrent;
}

namespace
{

/// Mirror the production charge sequence (AdoptedByteCharger H2 does the same
/// ordering: capture query group -> alloc -> counter bump). In the gtest binary without
/// MainThreadStatus there is no thread-local MemoryTracker, so alloc() is a benign no-op
/// returning AllocationTrace(0) and `CurrentThread::getMemoryTracker()` returns nullptr;
/// the ChargeHandle's destructor will take the null-query-group fallback path
/// (CurrentMemoryTracker::free, also a no-op). We still exercise alloc's [[nodiscard]]
/// surface so any future contract change surfaces here.
void chargeForTest(const std::shared_ptr<AdoptedByteState> & state, Int64 charged_bytes, Int64 logical_bytes)
{
    [[maybe_unused]] auto trace = CurrentMemoryTracker::alloc(charged_bytes);
    state->charged_current.fetch_add(charged_bytes, std::memory_order_acq_rel);
    state->logical_current.fetch_add(logical_bytes, std::memory_order_acq_rel);
    CurrentMetrics::add(CurrentMetrics::ShmAdoptedBytesCurrent, charged_bytes);
    CurrentMetrics::add(CurrentMetrics::ShmAdoptedBytesLogicalCurrent, logical_bytes);
}

void chargeForTest(const std::shared_ptr<AdoptedByteState> & state, Int64 bytes)
{
    chargeForTest(state, bytes, bytes);
}

}


TEST(ChargeHandle, BasicAcquireRelease)
{
    /// Done-when (a): acquire+release-via-destructor pairs are balanced.
    auto state = std::make_shared<AdoptedByteState>();
    constexpr Int64 charged = 4096;
    constexpr Int64 logical = 4032;
    chargeForTest(state, charged, logical);

    {
        ChargeHandle handle(charged, logical, state);
        EXPECT_TRUE(handle.isActive());
        EXPECT_EQ(handle.bytes(), static_cast<size_t>(charged));
        EXPECT_EQ(handle.logicalBytes(), static_cast<size_t>(logical));
        EXPECT_EQ(state->charged_current.load(), charged);
        EXPECT_EQ(state->logical_current.load(), logical);
    }

    /// Counter is decremented by `charged` (not `logical`) - the I7 exact-at-boundary contract.
    EXPECT_EQ(state->charged_current.load(), 0);
    EXPECT_EQ(state->logical_current.load(), 0);
}


TEST(ChargeHandle, MoveConstructTransfersCharge)
{
    /// Done-when (c): move-construct transfers the charge cleanly.
    auto state = std::make_shared<AdoptedByteState>();
    constexpr Int64 bytes = 1024;
    chargeForTest(state, bytes);

    {
        ChargeHandle h1(bytes, bytes, state);
        {
            ChargeHandle h2(std::move(h1));
            EXPECT_FALSE(h1.isActive()); // NOLINT(bugprone-use-after-move)
            EXPECT_EQ(h1.bytes(), 0u); // NOLINT(bugprone-use-after-move)
            EXPECT_TRUE(h2.isActive());
            EXPECT_EQ(h2.bytes(), static_cast<size_t>(bytes));
            EXPECT_EQ(state->charged_current.load(), bytes);
            EXPECT_EQ(state->logical_current.load(), bytes);
        }
        /// h2's dtor releases the charge.
        EXPECT_EQ(state->charged_current.load(), 0);
        EXPECT_EQ(state->logical_current.load(), 0);
        /// h1's dtor at outer scope must be a no-op - no double-release.
    }
    EXPECT_EQ(state->charged_current.load(), 0);
    EXPECT_EQ(state->logical_current.load(), 0);
}


TEST(ChargeHandle, MoveAssignReleasesPreviousCharge)
{
    auto a = std::make_shared<AdoptedByteState>();
    auto b = std::make_shared<AdoptedByteState>();
    constexpr Int64 bytes_a = 512;
    constexpr Int64 bytes_b = 8192;
    chargeForTest(a, bytes_a);
    chargeForTest(b, bytes_b, bytes_b / 2);

    ChargeHandle h_a(bytes_a, bytes_a, a);
    ChargeHandle h_b(bytes_b, bytes_b / 2, b);

    h_a = std::move(h_b);

    /// h_a's previous charge (a) is released by move-assign; h_b's charge (b) is now in h_a.
    EXPECT_EQ(a->charged_current.load(), 0);
    EXPECT_EQ(a->logical_current.load(), 0);
    EXPECT_EQ(b->charged_current.load(), bytes_b);
    EXPECT_EQ(b->logical_current.load(), bytes_b / 2);
    EXPECT_EQ(h_a.bytes(), static_cast<size_t>(bytes_b));
    EXPECT_EQ(h_a.logicalBytes(), static_cast<size_t>(bytes_b / 2));
    EXPECT_FALSE(h_b.isActive()); // NOLINT(bugprone-use-after-move)

    h_a = ChargeHandle{};
    EXPECT_EQ(b->charged_current.load(), 0);
    EXPECT_EQ(b->logical_current.load(), 0);
}


TEST(ChargeHandle, DefaultAndMovedFromAreNoOpOnDestruction)
{
    auto state = std::make_shared<AdoptedByteState>();
    state->charged_current.store(42);
    state->logical_current.store(24);
    {
        ChargeHandle handle;
        EXPECT_FALSE(handle.isActive());
        EXPECT_EQ(handle.bytes(), 0u);
    }
    EXPECT_EQ(state->charged_current.load(), 42);
    EXPECT_EQ(state->logical_current.load(), 24);
}


TEST(ChargeHandle, ReleaseOnSameThreadMirrorsUntrackedCushion)
{
    MainThreadStatus::getInstance();
    CurrentThread::flushUntrackedMemory();
    total_memory_tracker.resetCounters();
    CurrentThread::get().memory_tracker.resetCounters();

    SCOPE_EXIT_SAFE({
        total_memory_tracker.resetCounters();
        CurrentThread::get().memory_tracker.resetCounters();
        CurrentThread::flushUntrackedMemory();
    });

    auto * captured = CurrentThread::getMemoryTracker();
    ASSERT_NE(captured, nullptr);

    constexpr Int64 bytes = 1024;
    [[maybe_unused]] auto trace = CurrentMemoryTracker::alloc(bytes);
    const Int64 before_amount = captured->get();
    EXPECT_EQ(before_amount, 0);

    auto state = std::make_shared<AdoptedByteState>();
    state->charged_current.fetch_add(bytes, std::memory_order_acq_rel);
    state->logical_current.fetch_add(bytes, std::memory_order_acq_rel);
    CurrentMetrics::add(CurrentMetrics::ShmAdoptedBytesCurrent, bytes);
    CurrentMetrics::add(CurrentMetrics::ShmAdoptedBytesLogicalCurrent, bytes);

    {
        ChargeHandle handle(bytes, bytes, state);
    }

    EXPECT_EQ(captured->get(), before_amount);
    EXPECT_EQ(state->charged_current.load(), 0);
    EXPECT_EQ(state->logical_current.load(), 0);
}


TEST(ChargeHandle, ReleaseOnDifferentThreadWithoutQueryGroupFallsBackToTotalTracker)
{
    /// H2 safety fallback: when no query group is available, ChargeHandle does not retain a
    /// raw producer-thread MemoryTracker pointer. A different, detached thread therefore falls
    /// back to CurrentMemoryTracker::free(), which can release the process-total tracker but
    /// deliberately leaves the producer thread tracker alone.
    ///
    /// Production query execution captures ThreadGroupPtr and releases the query tracker
    /// exactly. This test covers the tracker-less/unit-test path: it must not dereference a
    /// thread tracker that may already have been torn down.
    MainThreadStatus::getInstance();
    CurrentThread::flushUntrackedMemory();
    total_memory_tracker.resetCounters();
    CurrentThread::get().memory_tracker.resetCounters();

    SCOPE_EXIT_SAFE({
        total_memory_tracker.resetCounters();
        CurrentThread::get().memory_tracker.resetCounters();
        CurrentThread::flushUntrackedMemory();
    });

    auto * captured = CurrentThread::getMemoryTracker();
    ASSERT_NE(captured, nullptr);

    /// 16 MiB is well above the 4 MiB default per-thread untracked_memory cushion, so the
    /// alloc forcibly flushes through to the producer thread tracker's `amount` and the
    /// process-total tracker.
    constexpr Int64 bytes = 16 * 1024 * 1024;
    [[maybe_unused]] auto trace = CurrentMemoryTracker::alloc(bytes);

    const Int64 before_amount = captured->get();
    const Int64 before_total_amount = total_memory_tracker.get();
    EXPECT_GE(before_amount, bytes);
    EXPECT_GE(before_total_amount, bytes);

    auto state = std::make_shared<AdoptedByteState>();
    state->charged_current.fetch_add(bytes, std::memory_order_acq_rel);
    state->logical_current.fetch_add(bytes, std::memory_order_acq_rel);
    CurrentMetrics::add(CurrentMetrics::ShmAdoptedBytesCurrent, bytes);
    CurrentMetrics::add(CurrentMetrics::ShmAdoptedBytesLogicalCurrent, bytes);

    std::thread t(
        [state]
        {
            /// On this child std::thread `current_thread == nullptr` (no ThreadStatus was
            /// attached). Constructing the handle inside the thread (rather than moving one in)
            /// keeps the move-semantics out of the assertion.
            ChargeHandle handle(bytes, bytes, state);
            EXPECT_EQ(CurrentThread::getMemoryTracker(), nullptr);
        });
    t.join();

    /// Feature-local counter released.
    EXPECT_EQ(state->charged_current.load(), 0);
    EXPECT_EQ(state->logical_current.load(), 0);
    /// No raw producer-thread tracker was retained, so that tracker is untouched. The
    /// fallback path releases the process-total tracker, which is static and safe here.
    EXPECT_EQ(captured->get(), before_amount);
    const Int64 total_delta = before_total_amount - total_memory_tracker.get();
    EXPECT_GE(total_delta, bytes);
    /// Thread creation / teardown and gtest bookkeeping may also touch the process-total
    /// tracker while this cross-thread fallback path is exercised. Keep the assertion tight
    /// enough to catch a missing SHM release while allowing unrelated sub-page noise.
    EXPECT_LE(total_delta - bytes, 4096);
}


TEST(RetainToken, AliasCopiesFireOnceOnLastDrop)
{
    int fire_count = 0;
    {
        auto rt = makeRetainToken([&] { ++fire_count; });
        EXPECT_EQ(fire_count, 0);
        EXPECT_EQ(rt.use_count(), 1L);
        {
            auto rt2 = rt; // NOLINT(performance-unnecessary-copy-initialization) - testing shared_ptr alias copy
            EXPECT_EQ(fire_count, 0);
            EXPECT_EQ(rt.use_count(), 2L);
            {
                auto rt3 = rt2; // NOLINT(performance-unnecessary-copy-initialization) - testing shared_ptr alias copy
                EXPECT_EQ(fire_count, 0);
                EXPECT_EQ(rt3.use_count(), 3L);
            }
            EXPECT_EQ(rt.use_count(), 2L);
            EXPECT_EQ(fire_count, 0);
        }
        EXPECT_EQ(rt.use_count(), 1L);
        EXPECT_EQ(fire_count, 0);
    }
    EXPECT_EQ(fire_count, 1);
}


TEST(RetainToken, EmptyCallbackIsBenign)
{
    /// An empty std::function still produces a token; the Holder destructor skips the call.
    auto rt = makeRetainToken(std::function<void()>{});
    EXPECT_TRUE(static_cast<bool>(rt));
}
