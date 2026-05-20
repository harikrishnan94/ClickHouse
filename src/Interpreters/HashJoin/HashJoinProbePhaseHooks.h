#pragma once

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

inline thread_local ProbePointCallback g_probe_point_callback = nullptr;
inline thread_local void * g_probe_point_context = nullptr;

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

inline void fireProbePoint(ProbePoint point)
{
    if (g_probe_point_callback)
        g_probe_point_callback(point, g_probe_point_context);
}

} // namespace DB::HashProbeBench
