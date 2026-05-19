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
