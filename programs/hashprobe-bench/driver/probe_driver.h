#pragma once

/// hashprobe-bench/driver/probe_driver.h
///
/// ProbeDriver: drives the probe phase of a HashJoin with tight-loop drain semantics.
///
/// Spec requirements covered: C2, C5, C8, G2, H1, H2, probe phase definition.

#include <hashprobe_bench/artifact.h>
#include "../instrumentation/hw_counters.h"

#include <Core/Block.h>
#include <Interpreters/IJoin.h>

#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

namespace DB::HashProbeBench
{

/// Per-call TID log entry (G2, C.3).
///
/// Used to verify:
///   (a) |distinct TIDs in log| == max_threads (every thread participated)
///   (b) each probe block has exactly one TID entry (no double-dispatch)
///   (c) per-TID intervals [joinblock_start_ns, last_next_end_ns] do not overlap
///
/// One entry per joinBlock call.  Used to verify thread assignment invariants:
///   (a) |distinct TIDs| == max_threads
///   (b) each block has exactly one TID
///   (c) per-TID intervals [joinblock_start_ns, last_next_end_ns] do not overlap
struct CallerTidEntry
{
    uint64_t probe_block_idx = 0;
    uint64_t tid = 0; ///< gettid() from <sys/syscall.h>
    uint64_t joinblock_start_ns = 0; ///< CLOCK_MONOTONIC_RAW before joinBlock call
    uint64_t last_next_end_ns = 0; ///< CLOCK_MONOTONIC_RAW after the final next() call
};

/// ProbeDriver drives the probe phase of a HashJoin.
///
/// For each probe block it executes the following tight-loop drain (C8 behavioural parity
/// with JoiningTransform::readExecute, lines 240-260 of JoiningTransform.cpp):
///
///   1. Increment ProfileEvents::JoinProbeTableRowCount  (C.2, matches production line 245)
///   2. Call join->joinBlock(std::move(block))           (C.1, exactly once per probe block)
///   3. Loop: call next() until is_last                  (C.1, tight loop — no batching)
///   4. Increment ProfileEvents::JoinResultRowCount      (C.2, matches production line 235)
///   5. Emit every output block to the downstream sink   (C.1)
///   6. Call next_block->filterBySelector() when set     (C.1, matches production lines 251-253)
///   7. Reset join_result only at is_last                (C.1, matches production lines 257-258)
///
/// Timing and TID are recorded at both granularities (C.3, C.4, H2).
class ProbeDriver
{
public:
    /// Consumer that receives each output Block from next().
    using OutputSink = std::function<void(Block)>;

    /// Drain-mode tag written into the result artifact (C.5, C8 structural deviation doc).
    /// Populate BuildHeader::harness_drain_mode with this value before serialising the artifact.
    /// The harness uses a tight loop instead of JoiningTransform's one-block-per-step model;
    /// this constant documents that structural deviation for post-hoc analysis.
    static constexpr const char * DRAIN_MODE = "tight_loop";

    ProbeDriver(std::shared_ptr<IJoin> join, OutputSink sink);

    /// Process one probe block through the full drain loop (HashJoin / ConcurrentHashJoin).
    ///
    /// @param block           Probe (left-side) block.  Moved on entry (production line 246).
    /// @param probe_block_idx 0-based index of this block in the probe stream.
    /// @returns               Per-probe-block log entry (H2, C.4); also appended to getProbeBlockLog().
    ProbeBlockEntry drainBlock(Block block, uint64_t probe_block_idx, HwCounters * hw_counters = nullptr);

    // ── PartitionedHashJoin path ──────────────────────────────────────────────

    /// Phase 1 of the PHJ probe: scatter one left-side block into partitions via joinBlock().
    ///
    /// PHJ::joinBlock does NOT return actual result rows — it only scatters the probe
    /// block into per-partition slices.  This method records the wall+CPU cost of that
    /// scatter call and appends a ProbeBlockEntry to probe_block_log_ (with
    /// result_emit_wall_ns = 0, phase_probe/phase_generate all-zero, output_rows = 0).
    ///
    /// Call drainDelayedBlocks() after all scatter calls to get actual results.
    ProbeBlockEntry scatterPhjBlock(Block block, uint64_t probe_block_idx);

    /// Phase 2 of the PHJ probe: drain the IBlocksStream from getDelayedBlocks().
    ///
    /// Each call to stream->next() processes one partition (build-HT + probe + gen).
    /// The per-partition ProbePoint callbacks (phj_build_ht_start/end, phj_probe_start/end,
    /// phj_gen_start/end) are fired by DelayedBlocks.cpp and captured here into
    /// PhjPartitionEntry objects, which are appended to phj_partition_log_.
    ///
    /// Total output rows are accumulated into probe_block_log_ entries (amortised).
    void drainDelayedBlocks(HwCounters * hw_counters = nullptr);

    // ── Log accessors ──────────────────────────────────────────────────────────

    /// Per-call TID log (G2, C.3).  One entry per drainBlock() call.
    const std::vector<CallerTidEntry> & getCallerTidLog() const { return caller_tid_log_; }

    /// Per-probe-block timing log (H2, C.4).  One entry per drainBlock() call.
    const std::vector<ProbeBlockEntry> & getProbeBlockLog() const { return probe_block_log_; }

    /// Per-output-block timing log (H2, C.4).  One entry per next() call across all blocks.
    const std::vector<OutputBlockEntry> & getOutputBlockLog() const { return output_block_log_; }

    /// PHJ per-partition log.  Populated by drainDelayedBlocks(); empty for hash algorithms.
    const std::vector<PhjPartitionEntry> & getPhjPartitionLog() const { return phj_partition_log_; }

private:
    std::shared_ptr<IJoin> join_;
    OutputSink sink_;

    std::vector<CallerTidEntry> caller_tid_log_;
    std::vector<ProbeBlockEntry> probe_block_log_;
    std::vector<OutputBlockEntry> output_block_log_;
    std::vector<PhjPartitionEntry> phj_partition_log_;
};

} // namespace DB::HashProbeBench
