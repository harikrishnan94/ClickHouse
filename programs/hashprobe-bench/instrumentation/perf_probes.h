#pragma once

/// hashprobe-bench/instrumentation/perf_probes.h
///
/// Harness-side per-block probe points.  All macros are no-ops; SDT/USDT
/// support has been removed.  The macros are retained so call-sites in
/// probe_driver.cpp compile without change.

#define HARNESS_PROBE_BLOCK_START()            ((void)0)
#define HARNESS_PROBE_BLOCK_END()              ((void)0)
#define HARNESS_PROBE_PHASE1_START()           ((void)0)
#define HARNESS_PROBE_PHASE1_END()             ((void)0)
#define HARNESS_PROBE_PHASE2_START()           ((void)0)
#define HARNESS_PROBE_PHASE2_END()             ((void)0)
#define HARNESS_PROBE_NEXT_START()             ((void)0)
#define HARNESS_PROBE_NEXT_END()               ((void)0)
#define HARNESS_PROBE_GENERATE_BLOCK_START()   ((void)0)
#define HARNESS_PROBE_GENERATE_BLOCK_END()     ((void)0)
