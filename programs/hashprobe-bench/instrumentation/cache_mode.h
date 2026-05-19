#pragma once

/// hashprobe-bench/instrumentation/cache_mode.h
///
/// LLC cache warm/cold controller (H5).
///
/// Warm mode: iterate the hash-table pool and maps to pull the built join
/// data into the last-level cache before probing.
///
/// Cold mode: determine LLC size from the kernel sysfs, then allocate and
/// write 2× LLC size of scratch memory to evict the join data from the LLC
/// before probing.
///
/// LLC size detection reads:
///   /sys/devices/system/cpu/cpu0/cache/index*/size
/// The highest-level cache (largest index) is used.  Falls back to a
/// conservative 32 MiB if sysfs is unavailable.

#include <cstddef>
#include <cstdint>

namespace DB { class HashJoin; }

namespace DB::HashProbeBench
{

/// Read the LLC size (in bytes) from the kernel sysfs.
/// Returns a positive value; falls back to 32 MiB if unavailable.
size_t detectLlcSizeBytes();

/// Warm the LLC by reading the full join data (pool + maps).
/// Must be called after build completes and before probe starts.
void warmLlc(const DB::HashJoin & join);

/// Evict the LLC by writing a 2× LLC-size scratch buffer twice.
/// Should be called immediately before the cold probe.
void evictLlc();

} // namespace DB::HashProbeBench
