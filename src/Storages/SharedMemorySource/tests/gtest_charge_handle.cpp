#include <gtest/gtest.h>

#include <Storages/SharedMemorySource/Adoption/RetainToken.h>
#include <Storages/SharedMemorySource/Tracker/ChargeHandle.h>

#include <Common/CurrentMemoryTracker.h>
#include <Common/CurrentThread.h>
#include <Common/Exception.h>
#include <Common/MemoryTracker.h>
#include <Common/ThreadStatus.h>
#include <Common/scope_guard_safe.h>

#include <base/types.h>

#include <atomic>
#include <functional>
#include <thread>
#include <utility>

using namespace DB;

namespace
{

/// Mirror the production charge sequence (AdoptedByteCharger post-Finding-5 does the same
/// ordering: capture-tracker -> alloc -> counter bump). In the gtest binary without
/// MainThreadStatus there is no thread-local MemoryTracker, so alloc() is a benign no-op
/// returning AllocationTrace(0) and `CurrentThread::getMemoryTracker()` returns nullptr;
/// the ChargeHandle's destructor will take the null-tracker fallback path
/// (CurrentMemoryTracker::free, also a no-op). We still exercise alloc's [[nodiscard]]
/// surface so any future contract change surfaces here.
void chargeForTest(std::atomic<int64_t> & counter, Int64 bytes)
{
    [[maybe_unused]] auto trace = CurrentMemoryTracker::alloc(bytes);
    counter.fetch_add(bytes, std::memory_order_acq_rel);
}

}


TEST(ChargeHandle, BasicAcquireRelease)
{
    /// Done-when (a): acquire+release-via-destructor pairs are balanced.
    std::atomic<int64_t> counter{0};
    constexpr Int64 charged = 4096;
    constexpr Int64 logical = 4032;
    chargeForTest(counter, charged);

    {
        ChargeHandle handle(charged, logical, &counter, /*tracker_at_charge=*/nullptr);
        EXPECT_TRUE(handle.isActive());
        EXPECT_EQ(handle.bytes(), static_cast<size_t>(charged));
        EXPECT_EQ(handle.logicalBytes(), static_cast<size_t>(logical));
        EXPECT_EQ(counter.load(), charged);
    }

    /// Counter is decremented by `charged` (not `logical`) - the I7 exact-at-boundary contract.
    EXPECT_EQ(counter.load(), 0);
}


TEST(ChargeHandle, MoveConstructTransfersCharge)
{
    /// Done-when (c): move-construct transfers the charge cleanly.
    std::atomic<int64_t> counter{0};
    constexpr Int64 bytes = 1024;
    chargeForTest(counter, bytes);

    {
        ChargeHandle h1(bytes, bytes, &counter, nullptr);
        {
            ChargeHandle h2(std::move(h1));
            EXPECT_FALSE(h1.isActive()); // NOLINT(bugprone-use-after-move)
            EXPECT_EQ(h1.bytes(), 0u);   // NOLINT(bugprone-use-after-move)
            EXPECT_TRUE(h2.isActive());
            EXPECT_EQ(h2.bytes(), static_cast<size_t>(bytes));
            EXPECT_EQ(counter.load(), bytes);
        }
        /// h2's dtor releases the charge.
        EXPECT_EQ(counter.load(), 0);
        /// h1's dtor at outer scope must be a no-op - no double-release.
    }
    EXPECT_EQ(counter.load(), 0);
}


TEST(ChargeHandle, MoveAssignReleasesPreviousCharge)
{
    std::atomic<int64_t> a{0};
    std::atomic<int64_t> b{0};
    constexpr Int64 bytes_a = 512;
    constexpr Int64 bytes_b = 8192;
    chargeForTest(a, bytes_a);
    chargeForTest(b, bytes_b);

    ChargeHandle h_a(bytes_a, bytes_a, &a, nullptr);
    ChargeHandle h_b(bytes_b, bytes_b / 2, &b, nullptr);

    h_a = std::move(h_b);

    /// h_a's previous charge (a) is released by move-assign; h_b's charge (b) is now in h_a.
    EXPECT_EQ(a.load(), 0);
    EXPECT_EQ(b.load(), bytes_b);
    EXPECT_EQ(h_a.bytes(), static_cast<size_t>(bytes_b));
    EXPECT_EQ(h_a.logicalBytes(), static_cast<size_t>(bytes_b / 2));
    EXPECT_FALSE(h_b.isActive()); // NOLINT(bugprone-use-after-move)

    h_a = ChargeHandle{};
    EXPECT_EQ(b.load(), 0);
}


TEST(ChargeHandle, DefaultAndMovedFromAreNoOpOnDestruction)
{
    std::atomic<int64_t> counter{42};
    {
        ChargeHandle handle;
        EXPECT_FALSE(handle.isActive());
        EXPECT_EQ(handle.bytes(), 0u);
    }
    EXPECT_EQ(counter.load(), 42);
}


TEST(ChargeHandle, ReleaseOnDifferentThreadDecrementsCapturedTracker)
{
    /// Finding 5 fix: release goes against the MemoryTracker chain that was charged
    /// (snapshotted at charge time), regardless of which thread runs the destructor.
    ///
    /// Set up MainThreadStatus so `CurrentThread::getMemoryTracker()` on the test thread
    /// returns a non-null tracker (the thread-level memory_tracker inside ThreadStatus,
    /// whose parent is `total_memory_tracker`). Charge against that captured chain on
    /// the test thread, hand the ChargeHandle to a fresh std::thread (which has no
    /// ThreadStatus and therefore no `current_thread`), and let the destructor run there.
    /// With Finding 5's pinning, the captured tracker's `amount` decrements; without it
    /// (the pre-fix world) the destructor would invoke CurrentMemoryTracker::free which
    /// finds no tracker on the child thread and silently no-ops, leaving the captured
    /// tracker's amount inflated.
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
    /// alloc forcibly flushes through to the captured tracker's `amount` (rather than
    /// being absorbed in the cushion and not visible at the tracker level).
    constexpr Int64 bytes = 16 * 1024 * 1024;
    [[maybe_unused]] auto trace = CurrentMemoryTracker::alloc(bytes);

    const Int64 before_amount = captured->get();
    EXPECT_GE(before_amount, bytes);

    std::atomic<int64_t> counter{0};
    counter.fetch_add(bytes, std::memory_order_acq_rel);

    std::thread t([&counter, captured]
    {
        /// On this child std::thread `current_thread == nullptr` (no ThreadStatus was
        /// attached). The destructor must NOT consult current_thread; it must release
        /// directly against the captured pointer. Constructing the handle inside the
        /// thread (rather than moving one in) keeps the move-semantics out of the assertion.
        ChargeHandle handle(bytes, bytes, &counter, captured);
        EXPECT_EQ(CurrentThread::getMemoryTracker(), nullptr);
    });
    t.join();

    /// Feature-local counter released.
    EXPECT_EQ(counter.load(), 0);
    /// Captured tracker decremented by exactly `bytes` (the release went to the
    /// snapshotted chain, not via current_thread on the destruction thread).
    EXPECT_EQ(before_amount - captured->get(), bytes);
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
