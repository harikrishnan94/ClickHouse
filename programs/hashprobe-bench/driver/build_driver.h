#pragma once

/// hashprobe-bench/driver/build_driver.h
///
/// Build-phase driver: constructs TableJoin + HashJoin / ConcurrentHashJoin,
/// drives addBlockToJoin calls, runs lifecycle (onBuildPhaseFinish /
/// runPostBuildPhase), and enforces the A2 / A2b fail-loudly gates.
///
/// Spec sections covered: A2, A2b, B1, B3, C1, C3, C6, G1

#include <hashprobe_bench/config.h>
#include <hashprobe_bench/types.h>

#include <Core/Block.h>
#include <Interpreters/HashJoin/HashJoin.h>
#include <Interpreters/IJoin.h>
#include <Interpreters/PartitionedHashJoin.h>

#include <memory>
#include <string>
#include <vector>

namespace DB::HashProbeBench
{

/// Full output from runBuildDriver(): the live join object + metrics + log.
struct BuildDriverOutput
{
    std::shared_ptr<IJoin> join; ///< Built join engine, ready for probe
    BuildResult result; ///< Build metrics and gate results
    std::vector<std::string> lifecycle_log; ///< Ordered lifecycle events (C6)
    std::string join_engine; ///< "HashJoin" | "ConcurrentHashJoin" | "PartitionedHashJoin" (G1)
    uint32_t slots = 1; ///< ConcurrentHashJoin slot count; 1 for others
};

// ── Block construction helpers ────────────────────────────────────────────────

/// Create a right-side sample block (schema only, zero rows) matching config.
Block makeRightSampleBlock(const ConfigType & config);

/// Create a filled Block with `num_rows` rows for the build side.
///   duplicate_keys = false: key = start_key + j
///   duplicate_keys = true : key = j % (num_rows / 2)
Block makeBuildBlock(const ConfigType & config, size_t num_rows, uint64_t start_key = 0, bool duplicate_keys = false);

// ── Type / gate helpers ───────────────────────────────────────────────────────

/// Convert HashJoin::Type to its lowercase canonical string.
std::string hashJoinTypeToString(HashJoin::Type type);

/// Returns true iff type_str is in the A2 allowed set.
bool isAllowedMapType(const std::string & type_str);

/// Returns "[HARNESS_ERROR] unsupported_config: resolved_map_type=<t>" or "".
std::string checkMapTypeGate(const std::string & type_str);

/// Returns "[HARNESS_ERROR] all_unique_keys_with_all_strictness_would_silently_promote_to_rightany" or "".
std::string checkStrictnessGate(const std::string & at_construction, const std::string & after_build);

/// Convert StrictnessConfig -> "ALL" | "ANY" | "RIGHTANY"
std::string strictnessConfigToString(StrictnessConfig s);

// ── Main driver ───────────────────────────────────────────────────────────────

/// Run the full build phase and return the live join + metrics.
///
/// On A2 violation:  emits [HARNESS_ERROR] to stderr and calls exit(1).
/// On A2b violation: emits [HARNESS_ERROR] to stderr and calls exit(1).
BuildDriverOutput runBuildDriver(const ConfigType & config, const std::vector<Block> & build_blocks, uint64_t build_distinct_keys = 0);

} // namespace DB::HashProbeBench
