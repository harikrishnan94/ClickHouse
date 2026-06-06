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
  * It targets the case where the build-side hash table working set exceeds last-level cache, for a
  * single (or composite) fixed-width join key, by never copying build payload (only key + ref are
  * partitioned) and doing all partitioning as one deferred, exactly-sized SWWC/NT scatter, then
  * probing small per-leaf 16-byte-ish-cell hash tables that stay L2-resident. See the spec
  * `radix_hash_join_spec.md` for the full design.
  *
  * Phase P4 (current): the real build + probe data path is live for a fixed-width key whose packed
  * width is a multiple of 4 in [4, 64]:
  *   - addBlockToJoin    -> BuildStore::add        (move + select; no scatter, no payload copy)
  *   - onBuildPhaseFinish-> BuildStore::finishBuild (merge histograms, prefix sums)
  *   - runPostBuildPhase -> BuildStore::scatterToLeaves + build per-leaf hash tables + next_chain
  *                          + colptr tables
  *   - joinBlock         -> probe the leaf hash tables (chain traversal for JOIN ALL) and emit matches
  *
  * The applicability gate (inner `ALL` equi-join, fixed-width non-nullable non-LC key, packed width a
  * multiple of 4) is enforced by the factory in `PlannerJoins`. If the packed key width is outside the
  * leaf-cell range [4, 64] the join falls back to an internal passthrough `HashJoin` (still correct).
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

    /// The build (right) side is filled by `FillingRightJoinSideTransform` in parallel; the radix
    /// build path is lock-free (one BuildStore slot per build thread). The passthrough fallback
    /// serialises the forwarded `addBlockToJoin` with a mutex.
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

    size_t max_threads;
    std::optional<UInt64> rhs_size_estimation;
    UInt64 max_partitions_per_pass;

    /// Whether the live radix data path is used (true) or the passthrough HashJoin fallback (false).
    bool use_radix = false;

    /// Passthrough fallback target (only when `use_radix == false`): a plain HashJoin.
    std::unique_ptr<HashJoin> hash_join;
    /// Guards parallel `addBlockToJoin` while the passthrough delegates to a single HashJoin.
    std::mutex add_block_mutex;

    /// All radix-path state (build store, leaf hash tables, colptr tables, output plan). Defined in the
    /// .cpp so this header stays free of the RadixHash internals; null when `use_radix == false`.
    struct RadixState;
    std::unique_ptr<RadixState> state;

    JoinResultPtr joinBlockRadix(Block block) const;
};

}
