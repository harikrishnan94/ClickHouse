#pragma once

/// hashprobe-bench/sweep/sweep_manager.h
///
/// SweepManager: top-level harness entry point (Phase 2, Track G).
///
/// Spec requirements covered:
///   G.1 — build-once probe-many over (max_threads x block_size x counter_mode)
///   G.2 — build_invocations=1 assert + stderr log (C7)
///   G.3 — cfg.reps repetitions per cell, per_rep arrays, median + CV (H6)
///   G.4 — cpu_affinity, git_commit, compiler, cxx_flags in artifact (G6, I2, I3)

#include <hashprobe_bench/artifact.h>
#include <hashprobe_bench/config.h>

namespace DB::HashProbeBench
{

/// Top-level sweep orchestrator.
///
/// Construction stores a const reference to the ConfigType; the config must
/// outlive the SweepManager object.  Call run() to execute the full sweep.
class SweepManager
{
public:
    explicit SweepManager(const ConfigType & cfg);

    /// Run the full sweep: build the join engine ONCE, then probe across the
    /// configured grid (max_threads x block_size x counter_mode in {none,hw})
    /// with cfg.reps repetitions per cell.
    ///
    /// Writes <output_dir>/artifact.json and returns the populated Artifact.
    Artifact run();

private:
    const ConfigType & cfg_;
};

} // namespace DB::HashProbeBench
