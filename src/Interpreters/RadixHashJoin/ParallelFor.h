#pragma once

#include <cstddef>
#include <functional>

namespace DB::RadixJoin
{

/** The parallel-for abstraction the post-build (scatter + leaf-table build) is distributed over.
  *
  * It decouples `BuildSide` / `LeafTable` from the concrete thread mechanism: the production path
  * (`RadixHashJoin::onBuildPhaseFinish`) backs it with a `ThreadPool` (`scheduleOrThrow` + `wait`),
  * the unit tests back it with plain `std::thread`s; both honour the same contract.
  *
  * Contract for any implementation of `parallel_for(total, fn)`:
  *   - invoke `fn(unit, worker)` once for every `unit` in [0, total);
  *   - `worker` is a DENSE id in [0, num_workers) that is STABLE for the lifetime of one unit and
  *     owned by exactly one thread at a time, so a unit may use it to index a per-worker resource
  *     under a single-writer invariant (e.g. `LeafTables::build_arenas[worker]`);
  *   - units are handed out with dynamic load balancing (leaf sizes are highly skewed, so a static
  *     equal-count split would serialize on the big leaves);
  *   - an exception thrown by any unit is propagated to the caller after all workers have stopped.
  * `total == 0` is a no-op.
  */
using UnitFn = std::function<void(size_t unit, size_t worker)>;
using ParallelFor = std::function<void(size_t total, const UnitFn & fn)>;

}
