#pragma once

#include <Core/Block_fwd.h>
#include <Interpreters/IJoin.h>

#include <memory>
#include <mutex>
#include <optional>

namespace DB
{

class TableJoin;
class HashJoin;

/** RadixHashJoin — a radix-partitioned hash join exposed as `join_algorithm = 'radix_hash'`.
  *
  * It targets the case where the build-side hash table working set exceeds last-level cache,
  * for a single fixed-width join key of <= 8 bytes, by never copying build payload (only
  * key + ref are partitioned) and doing all partitioning as one deferred, exactly-sized
  * SWWC/NT scatter. See spec `radix_hash_join_spec.md` for the full design.
  *
  * Phase P0 (current): this class is a *passthrough*. It owns a plain `HashJoin` and forwards
  * build and probe to it, so the algorithm is selectable end-to-end and returns results that
  * are identical to `hash`. The real radix data path (THP arena, selector/histogram, SWWC/NT
  * scatter, per-leaf 16-byte-cell hash tables, custom probe transform) is added phase by phase
  * in later commits. The applicability gate (single fixed-width key of <= 8 bytes, inner join)
  * is enforced by the factory in `PlannerJoins`; mismatching joins fall back to `parallel_hash`.
  */
class RadixHashJoin : public IJoin
{
public:
    RadixHashJoin(
        std::shared_ptr<TableJoin> table_join_,
        SharedHeader right_sample_block_,
        size_t max_threads_,
        std::optional<UInt64> rhs_size_estimation_,
        UInt64 max_partitions_per_pass_,
        bool any_take_last_row_ = false);

    ~RadixHashJoin() override;

    std::string getName() const override { return "RadixHashJoin"; }

    const TableJoin & getTableJoin() const override;

    /// The build (right) side is filled by `FillingRightJoinSideTransform` in parallel; the
    /// passthrough guards the forwarded `addBlockToJoin` with a mutex (P0). Later phases make
    /// ingestion lock-free per the spec.
    bool supportParallelJoin() const override { return true; }

    bool addBlockToJoin(const Block & block, bool check_limits) override;
    bool addBlockToJoin(const Block & block, size_t num_rows, bool check_limits) override;

    void checkTypesOfKeys(const Block & block) const override;

    JoinResultPtr joinBlock(Block block) override;

    size_t getTotalRowCount() const override;
    size_t getTotalByteCount() const override;
    bool alwaysReturnsEmptySet() const override;

    IBlocksStreamPtr getNonJoinedBlocks(
        const Block & left_sample_block, const Block & result_sample_block, UInt64 max_block_size) const override;

    void onBuildPhaseFinish() override;
    bool hasPostBuildPhase() const override;
    void runPostBuildPhase() override;

private:
    std::shared_ptr<TableJoin> table_join;
    SharedHeader right_sample_block;

    /// Carried for later phases (leaf sizing / fanout / parallelism). Unused by the P0 passthrough.
    [[maybe_unused]] size_t max_threads;
    [[maybe_unused]] std::optional<UInt64> rhs_size_estimation;
    [[maybe_unused]] UInt64 max_partitions_per_pass;

    /// P0 passthrough target. Replaced by the radix-partitioned structures in later phases.
    std::unique_ptr<HashJoin> hash_join;

    /// Guards parallel `addBlockToJoin` while the passthrough delegates to a single HashJoin.
    std::mutex add_block_mutex;
};

}
