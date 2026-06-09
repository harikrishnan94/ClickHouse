#pragma once

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>

namespace DB::RadixJoin
{

/** A leader + helpers cooperative work pool, with NO dedicated threads and NO sleeping on a race.
  *
  * The radix post-build (scatter + leaf-HT build) must run once, in parallel, but it is triggered
  * lazily by the first probe thread that calls `joinBlock` after the build barrier. We do not want to
  * spin up our own thread pool there — the pipeline already has N probe threads arriving. So every
  * probe thread calls `run(body)` with the same `body`:
  *
  *   - the FIRST caller becomes the leader and executes `body()`. Inside it the leader issues
  *     `parallelFor(total, fn)` calls (the parallel scatter / HT-build steps);
  *   - every OTHER caller becomes a helper that waits for work and drains `parallelFor` units until
  *     the leader closes the session.
  *
  * Correct for any number of participants, including one (then `run` is fully sequential). The leader
  * never blocks inside `body()` except in `parallelFor`, so it always makes progress even if no helper
  * ever shows up. Exceptions from the leader or from any unit are propagated to all participants.
  */
class CoopPool
{
public:
    CoopPool() = default;
    CoopPool(const CoopPool &) = delete;
    CoopPool & operator=(const CoopPool &) = delete;

    /// Every participating thread calls this with the same `body`. First is the leader (runs body),
    /// the rest are helpers. Callers arriving after the session finished return immediately (after
    /// rethrowing any leader exception).
    void run(std::function<void()> body);

    /// Distribute `total` work units across the leader and any present helpers; blocks until all units
    /// finish. Called only by the leader, only from inside `body`. No-op for total == 0.
    void parallelFor(size_t total, std::function<void(size_t)> fn);

private:
    struct Job
    {
        std::function<void(size_t)> fn;
        size_t total = 0;
        std::atomic<size_t> next{0};
        std::atomic<size_t> done{0};
        std::exception_ptr exc; /// first unit exception; guarded by CoopPool::mutex
    };

    void drainJob(const std::shared_ptr<Job> & job);

    std::mutex mutex;
    std::condition_variable cv;
    std::shared_ptr<Job> current_job;       /// non-null while a parallelFor is active; guarded by mutex
    bool session_done = false;              /// leader finished body(); guarded by mutex
    std::exception_ptr leader_exception;    /// leader/unit exception to propagate; guarded by mutex
    std::atomic<bool> leader_taken{false};
};

}
