#include <Interpreters/RadixHashJoin/CoopPool.h>

namespace DB::RadixJoin
{

void CoopPool::drainJob(const std::shared_ptr<Job> & job)
{
    while (true)
    {
        const size_t idx = job->next.fetch_add(1, std::memory_order_relaxed);
        if (idx >= job->total)
            break;

        try
        {
            job->fn(idx);
        }
        catch (...)
        {
            std::lock_guard lock(mutex);
            if (!job->exc)
                job->exc = std::current_exception();
        }

        /// The thread that completes the last unit wakes the leader blocked in parallelFor. The notify
        /// must be serialized with `mutex`: the leader evaluates the `done >= total` predicate while
        /// holding `mutex` and then sleeps in cv.wait; notifying outside the lock races with that
        /// window and could be lost (leader sees done<total, this thread finishes+notifies, leader
        /// then sleeps forever). Holding `mutex` guarantees the notify lands after the leader is asleep.
        if (job->done.fetch_add(1, std::memory_order_acq_rel) + 1 == job->total)
        {
            std::lock_guard lock(mutex);
            cv.notify_all();
        }
    }
}

void CoopPool::parallelFor(size_t total, std::function<void(size_t)> fn)
{
    if (total == 0)
        return;

    auto job = std::make_shared<Job>();
    job->fn = std::move(fn);
    job->total = total;

    {
        std::lock_guard lock(mutex);
        current_job = job;
    }
    cv.notify_all(); /// wake helpers waiting for work

    drainJob(job); /// the leader pulls units too

    {
        std::unique_lock lock(mutex);
        cv.wait(lock, [&] { return job->done.load(std::memory_order_acquire) >= total; });
        current_job = nullptr;
    }

    if (job->exc)
        std::rethrow_exception(job->exc);
}

void CoopPool::run(std::function<void()> body)
{
    bool expected = false;
    if (leader_taken.compare_exchange_strong(expected, true, std::memory_order_acq_rel))
    {
        /// Leader: run body() (which issues the parallelFor steps), then close the session.
        std::exception_ptr exc;
        try
        {
            body();
        }
        catch (...)
        {
            exc = std::current_exception();
        }

        {
            std::lock_guard lock(mutex);
            leader_exception = exc;
            session_done = true;
        }
        cv.notify_all();

        if (exc)
            std::rethrow_exception(exc);
    }
    else
    {
        /// Helper (also covers callers that arrive after the session already finished).
        while (true)
        {
            std::shared_ptr<Job> job;
            {
                std::unique_lock lock(mutex);
                cv.wait(lock, [this] { return current_job != nullptr || session_done; });
                if (session_done)
                {
                    if (leader_exception)
                        std::rethrow_exception(leader_exception);
                    return;
                }
                job = current_job;
            }
            drainJob(job);
        }
    }
}

}
