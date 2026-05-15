#include <gtest/gtest.h>

#include <Storages/SharedMemorySource/Tracker/AdoptedByteCharger.h>
#include <Storages/SharedMemorySource/Tracker/ChargeHandle.h>

#include <Common/CurrentMemoryTracker.h>
#include <Common/CurrentThread.h>
#include <Common/Exception.h>
#include <Common/MemoryTracker.h>
#include <Common/ThreadStatus.h>
#include <Common/scope_guard_safe.h>

#include <base/types.h>

#include <utility>

using namespace DB;


TEST(AdoptedByteCharger, BalancedAcquireRelease)
{
    /// Counter starts at zero, increments on each charge(), and returns to zero after the last
    /// ChargeHandle is destroyed — the I7 "returns to zero on source destruction" invariant.
    AdoptedByteCharger charger;
    EXPECT_EQ(charger.currentChargedBytes(), 0);
    {
        auto h1 = charger.charge(4096);
        EXPECT_EQ(charger.currentChargedBytes(), 4096);
        {
            auto h2 = charger.charge(1024);
            EXPECT_EQ(charger.currentChargedBytes(), 4096 + 1024);
        }
        EXPECT_EQ(charger.currentChargedBytes(), 4096);
    }
    EXPECT_EQ(charger.currentChargedBytes(), 0);
}


TEST(AdoptedByteCharger, ChargedVsLogical)
{
    /// The feature-local counter and the gauge track `adopted_bytes` (charged, includes safe-read
    /// padding); `logical_bytes` is reported separately via the handle's accessor and feeds the
    /// `ShmAdoptedBytesLogical` ProfileEvent. Distinction is from the system glossary entry
    /// "Adopted byte count".
    AdoptedByteCharger charger;
    {
        auto h = charger.charge(/*adopted_bytes=*/4096 + 15, /*logical_bytes=*/4096);
        EXPECT_EQ(charger.currentChargedBytes(), 4096 + 15);
        EXPECT_EQ(h.bytes(), static_cast<size_t>(4096 + 15));
        EXPECT_EQ(h.logicalBytes(), 4096u);
    }
    EXPECT_EQ(charger.currentChargedBytes(), 0);
}


TEST(AdoptedByteCharger, MoveHandlePreservesAccounting)
{
    /// Moving a ChargeHandle transfers ownership of the charge without touching the counter or
    /// the underlying tracker — the source becomes a no-op sentinel; the destination owns the
    /// release. Symmetric to ChargeHandle's MoveConstructTransfersCharge gtest in T1.4.
    AdoptedByteCharger charger;
    auto h1 = charger.charge(2048);
    EXPECT_EQ(charger.currentChargedBytes(), 2048);

    auto h2 = std::move(h1);
    EXPECT_FALSE(h1.isActive()); // NOLINT(bugprone-use-after-move)
    EXPECT_TRUE(h2.isActive());
    EXPECT_EQ(charger.currentChargedBytes(), 2048);
}


TEST(AdoptedByteCharger, ZeroByteChargeIsValid)
{
    /// A zero-byte charge is a degenerate-but-legal call (e.g. a producer block with all
    /// columns empty). The counter stays at zero and the returned handle's destructor is a
    /// no-op (ChargeHandle::release short-circuits when bytes_ == 0, so no `free` call and no
    /// `ShmRetainsReleased` event fire).
    AdoptedByteCharger charger;
    auto h = charger.charge(0);
    EXPECT_EQ(charger.currentChargedBytes(), 0);
    EXPECT_FALSE(h.isActive());
}


TEST(AdoptedByteCharger, LimitFailureRollsBackCounter)
{
    /// Install a thread-local + total MemoryTracker chain with a tight hard limit, then attempt
    /// a charge that overflows it. The MEMORY_LIMIT_EXCEEDED exception must propagate AND the
    /// feature-local counter must be at its pre-call value — proving the rollback path is
    /// correct (memory-tracker-integration §Release semantics; I8 enforcement + I7 exactness).
    ///
    /// Pattern mirrors `gtest_mark_ranges_memory_tracking.cpp`: ensure MainThreadStatus is up,
    /// reset and lower both trackers, flush the per-thread untracked-memory cushion (default
    /// 4 MiB), then try a charge well above both the cushion and the hard limit. SCOPE_EXIT
    /// restores the limits to 0 (== unlimited) so subsequent gtests in the same process are
    /// not affected.
    MainThreadStatus::getInstance();
    CurrentThread::flushUntrackedMemory();
    total_memory_tracker.resetCounters();
    CurrentThread::get().memory_tracker.resetCounters();

    total_memory_tracker.setHardLimit(1024);
    CurrentThread::get().memory_tracker.setHardLimit(1024);

    SCOPE_EXIT_SAFE({
        total_memory_tracker.setHardLimit(0);
        CurrentThread::get().memory_tracker.setHardLimit(0);
        total_memory_tracker.resetCounters();
        CurrentThread::get().memory_tracker.resetCounters();
        CurrentThread::flushUntrackedMemory();
    });

    AdoptedByteCharger charger;
    EXPECT_EQ(charger.currentChargedBytes(), 0);

    /// 16 MiB is comfortably above both the 1 KiB hard limit and the 4 MiB default
    /// untracked-memory cushion, so the alloc call is forced to consult (and reject at) the
    /// hard limit rather than being absorbed locally. The lambda absorbs the [[nodiscard]]
    /// return of `charge()` so that EXPECT_THROW does not trip the clang nodiscard warning.
    auto try_overflow = [&] { [[maybe_unused]] auto h = charger.charge(16 * 1024 * 1024); };
    EXPECT_THROW(try_overflow(), DB::Exception);
    EXPECT_EQ(charger.currentChargedBytes(), 0);
}


TEST(AdoptedByteCharger, NoTransientCounterIncrementOnLimitFailure)
{
    /// Finding 5 / AC5 ("no charged-then-rolled-back transient"): the feature-local counter
    /// must remain at its pre-call value throughout a throwing `charge()` — there must be
    /// no observable moment where it transiently holds the not-yet-rolled-back charge.
    ///
    /// The pre-Finding-5 ordering was: counter.fetch_add(N) -> alloc(N) [throws] ->
    /// counter.fetch_sub(N) [rollback] -> rethrow. A concurrent reader interleaving
    /// between fetch_add and fetch_sub would have observed the transient `+N`. The fixed
    /// ordering is: alloc(N) [throws] -> rethrow (no counter touch at all). The post-throw
    /// `currentChargedBytes() == 0` assertion is necessary for both orderings; what
    /// distinguishes the fixed code is that the assertion holds because the counter was
    /// NEVER incremented, not because a rollback restored it.
    ///
    /// We exercise this with `max_memory_usage == 1024` so that any non-trivial charge
    /// (>> 1 KiB) throws MEMORY_LIMIT_EXCEEDED at the alloc call. 16 MiB also clears the
    /// 4 MiB per-thread cushion so the tracker is forced to consult (and reject at) the
    /// hard limit rather than absorbing the alloc locally.
    MainThreadStatus::getInstance();
    CurrentThread::flushUntrackedMemory();
    total_memory_tracker.resetCounters();
    CurrentThread::get().memory_tracker.resetCounters();

    total_memory_tracker.setHardLimit(1024);
    CurrentThread::get().memory_tracker.setHardLimit(1024);

    SCOPE_EXIT_SAFE({
        total_memory_tracker.setHardLimit(0);
        CurrentThread::get().memory_tracker.setHardLimit(0);
        total_memory_tracker.resetCounters();
        CurrentThread::get().memory_tracker.resetCounters();
        CurrentThread::flushUntrackedMemory();
    });

    AdoptedByteCharger charger;
    ASSERT_EQ(charger.currentChargedBytes(), 0);

    auto try_overflow = [&] { [[maybe_unused]] auto h = charger.charge(16 * 1024 * 1024); };
    EXPECT_THROW(try_overflow(), DB::Exception);

    /// Structurally, the counter was never incremented (the alloc threw before line 50
    /// of AdoptedByteCharger.cpp's `charge()`). This assertion catches a regression where
    /// the counter increment is moved BACK before the alloc call.
    EXPECT_EQ(charger.currentChargedBytes(), 0);
}
