#pragma once

#include <cstddef>
#include <limits>

namespace DB::HashProbeBench
{

enum class ProbePoint
{
    probe_loop_start,
    phase1_start,
    phase1_end,
    phase2_start,
    phase2_end,
    probe_loop_end,
    next_start,
    next_end,
    generate_block_start,
    generate_block_end,

    /// PartitionedHashJoin per-partition phase boundaries.
    /// Fired inside DelayedBlocks::nextImpl() once per partition.
    phj_build_ht_start, ///< Entry of per-partition HashJoin build phase
    phj_build_ht_end, ///< Exit  of per-partition HashJoin build phase
    phj_probe_start, ///< Entry of per-partition probe loop (joinBlock calls)
    phj_probe_end, ///< Exit  of per-partition probe loop
    phj_gen_start, ///< Entry of per-partition generate/drain (next() calls)
    phj_gen_end, ///< Exit  of per-partition generate/drain
};

using ProbePointCallback = void (*)(ProbePoint, void *);

inline constexpr size_t INVALID_PROBE_POINT_PARTITION = std::numeric_limits<size_t>::max();

inline thread_local ProbePointCallback g_probe_point_callback = nullptr;
inline thread_local void * g_probe_point_context = nullptr;
inline thread_local size_t g_probe_point_partition = INVALID_PROBE_POINT_PARTITION;

inline void setProbePointCallback(ProbePointCallback callback, void * context)
{
    g_probe_point_callback = callback;
    g_probe_point_context = context;
}

inline void clearProbePointCallback()
{
    g_probe_point_callback = nullptr;
    g_probe_point_context = nullptr;
}

inline void setProbePointPartition(size_t partition)
{
    g_probe_point_partition = partition;
}

inline void clearProbePointPartition()
{
    g_probe_point_partition = INVALID_PROBE_POINT_PARTITION;
}

inline size_t getProbePointPartition()
{
    return g_probe_point_partition;
}

inline void fireProbePoint(ProbePoint point)
{
    if (g_probe_point_callback)
        g_probe_point_callback(point, g_probe_point_context);
}

inline void fireProbePoint(ProbePoint point, size_t partition)
{
    setProbePointPartition(partition);
    fireProbePoint(point);
}

} // namespace DB::HashProbeBench
