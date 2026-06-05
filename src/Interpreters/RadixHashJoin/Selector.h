#pragma once

#include <Interpreters/RadixHashJoin/PartitionConfig.h>

#include <base/types.h>

#include <cstddef>
#include <vector>

namespace DB
{
class IColumn;
}

namespace DB::RadixHash
{

/** Per-thread selector (spec sections 4.2 steps 2-3, 4.5).
  *
  * For a key column it computes one 32-bit `IColumn::computeHashInto` per row, derives the
  * `uint16` leaf id `pid = hash >> (32 - total_bits)` (the top `total_bits`), and accumulates a
  * replicated histogram. The replication (round-robin over a small power-of-two number of copies)
  * avoids store-to-load-forwarding stalls on `++hist[pid]` while the copies fit L1/L2; they are
  * summed on read. One selector is owned per build/probe worker, so it is lock-free.
  *
  * Build and probe use the identical `computeHashInto` + `shift`, so the same key always yields
  * the same `pid` on both sides (spec invariant 15.5).
  */
class Selector
{
public:
    explicit Selector(const PartitionConfig & cfg_);

    /// For a key column over rows [0, n): write the 32-bit hash into `hash_out` (n entries) and
    /// the uint16 leaf id into `pid_out` (n entries), and update the internal histogram.
    void process(const IColumn & key_col, size_t n, UInt32 * hash_out, UInt16 * pid_out);

    /// Derive pids from a precomputed hash buffer (no re-hash). Updates the histogram.
    void pidsFromHashes(const UInt32 * hash_in, size_t n, UInt16 * pid_out);

    /// Sum the replicated histograms into `out` (resized to num_leaves). Returns the total count.
    UInt64 mergedHistogram(std::vector<UInt32> & out) const;

    size_t numReplicas() const { return replicas; }
    const PartitionConfig & config() const { return cfg; }

private:
    PartitionConfig cfg;
    size_t replicas;
    /// replicas * num_leaves counters, replica r at [r * num_leaves, (r + 1) * num_leaves).
    std::vector<UInt32> hist;

    void addToHistogram(const UInt16 * pid, size_t n);
};

/** Merge per-thread histograms (spec section 4.6 step 2): `global_hist[p] = sum_t local[t][p]`,
  * and `offset` is its exclusive prefix sum (`offset[0] = 0`, `offset[p] = offset[p-1] +
  * global_hist[p-1]`), i.e. the start index of every leaf. Returns the total row count
  * (`offset[num_leaves-1] + global_hist[num_leaves-1]`). Both outputs are resized to num_leaves.
  */
UInt64 mergeHistograms(
    const std::vector<std::vector<UInt32>> & per_thread_hist,
    size_t num_leaves,
    std::vector<UInt64> & global_hist,
    std::vector<UInt64> & offset);

}
