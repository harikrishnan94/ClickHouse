#pragma once

/// Thread-local knob for HashJoin generate-phase prefetch look-ahead.
///
/// Controls how many iterations ahead the gather loops in IColumn.cpp issue
/// __builtin_prefetch hints.  Set to 0 to disable; default is 8.
///
/// The hashprobe-bench harness sets this per probe run to sweep values without
/// rebuilding dbms (see programs/hashprobe-bench/sweep/sweep_manager.cpp).

namespace DB
{
inline thread_local unsigned generate_phase_prefetch_lookahead = 8;
}
