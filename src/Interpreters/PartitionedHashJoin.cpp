#include <Interpreters/PartitionedHashJoin.h>

#include <Interpreters/HashJoin/HashJoin.h>
#include <Interpreters/PartitionedHashShuffle.h>
#include <Interpreters/TableJoin.h>
#include <Common/ProfileEvents.h>
#include <Common/Stopwatch.h>
#include <Common/logger_useful.h>

namespace ProfileEvents
{
extern const Event PartitionedHashBuildShuffleMicroseconds;
extern const Event PartitionedHashBuildShufflePass0Microseconds;
extern const Event PartitionedHashBuildShuffleTrailingMicroseconds;
extern const Event PartitionedHashBuildScatterRows;
extern const Event PartitionedHashBuildBlocksMoved;
}

namespace DB
{

namespace
{

/// Per-row fixed width of a (sample, empty) column; falls back to a coarse estimate for variable types.
size_t fixedRowBytes(const ColumnPtr & col)
{
    if (col->valuesHaveFixedSize())
        return col->sizeOfValueIfFixed();
    return 16;
}

size_t groupRows(const Columns & group)
{
    return group.empty() ? 0 : group[0]->size();
}

size_t groupBytes(const Columns & group)
{
    size_t bytes = 0;
    for (const auto & col : group)
        bytes += col->byteSize();
    return bytes;
}

}

PartitionedHashJoin::PartitionedHashJoin(
    std::shared_ptr<TableJoin> table_join_,
    SharedHeader right_sample_block_,
    size_t max_threads_,
    std::optional<UInt64> rhs_size_estimation_,
    size_t max_partitions_per_pass_,
    size_t shard_by_hash_input_batch_bytes_,
    bool debug_skip_passthrough_,
    bool any_take_last_row_)
    : table_join(std::move(table_join_))
    , right_sample_block(std::move(right_sample_block_))
    , max_threads(max_threads_)
    , any_take_last_row(any_take_last_row_)
    , shard_by_hash_input_batch_bytes(shard_by_hash_input_batch_bytes_)
    , debug_skip_passthrough(debug_skip_passthrough_)
    , hash_join(std::make_unique<HashJoin>(table_join, right_sample_block, any_take_last_row))
{
    /// Map the right-side join key names to positions in the right sample block.
    const auto & key_names_right = table_join->getOnlyClause().key_names_right;
    std::vector<bool> is_key(right_sample_block->columns(), false);
    for (const auto & name : key_names_right)
    {
        const size_t pos = right_sample_block->getPositionByName(name);
        key_indices.push_back(pos);
        is_key[pos] = true;
    }

    /// Per-row byte widths of the selected right columns (key vs payload), for leaf-count derivation.
    /// rhs_size_estimation from the planner is the right-side ROW count (right_rows_estimation).
    PartitionConfigInputs cfg_inputs;
    cfg_inputs.rhs_rows_estimation = rhs_size_estimation_;
    cfg_inputs.max_partitions_per_pass = max_partitions_per_pass_;
    cfg_inputs.key_bytes = 0;
    cfg_inputs.payload_bytes = 0;
    const Columns sample_cols = right_sample_block->getColumns();
    for (size_t i = 0; i < sample_cols.size(); ++i)
    {
        const size_t w = fixedRowBytes(sample_cols[i]);
        if (is_key[i])
            cfg_inputs.key_bytes += w;
        else
            cfg_inputs.payload_bytes += w;
    }
    if (cfg_inputs.key_bytes == 0)
        cfg_inputs.key_bytes = 8;

    partition_config = derivePartitionConfig(cfg_inputs);
    leaf_chains.resize(partition_config.total_leaves);

    /// Allocate intermediate stage buffers for stages 1..numPasses-1 (stage numPasses == leaves).
    const size_t num_passes = partition_config.numPasses();
    stage_buffers.resize(num_passes);
    stage_buffer_bytes.resize(num_passes);
    UInt8 bits_so_far = 0;
    for (size_t s = 1; s < num_passes; ++s)
    {
        bits_so_far = static_cast<UInt8>(bits_so_far + partition_config.pass_bits[s - 1]);
        const size_t count = size_t{1} << bits_so_far;
        stage_buffers[s].resize(count);
        stage_buffer_bytes[s].assign(count, 0);
    }

    LOG_DEBUG(
        getLogger("PartitionedHashJoin"),
        "Partition config: {} leaves, {} passes, key_bytes={}, payload_bytes={}, batch_bytes={}",
        partition_config.total_leaves,
        num_passes,
        cfg_inputs.key_bytes,
        cfg_inputs.payload_bytes,
        shard_by_hash_input_batch_bytes);
}

bool PartitionedHashJoin::addBlockToJoin(const Block & block, bool check_limits)
{
    /// P2: the internal HashJoin keeps the 9 queries correct (joinBlock delegates to it). The radix
    /// shuffle below runs as a measured side-effect; its output (leaf_chains) is consumed by the eager
    /// HT build in P3. Both see the same right rows.
    /// Diagnostic: `debug_skip_passthrough` skips this build so the shuffle is timed without the passthrough
    /// HashJoin competing for cache/memory bandwidth (results become incorrect; measurement only).
    const bool ok = debug_skip_passthrough ? true : hash_join->addBlockToJoin(block, check_limits);

    const size_t rows = block.rows();
    if (rows == 0)
        return ok;

    ingested_rows.fetch_add(rows, std::memory_order_relaxed);

    /// Degenerate single-leaf config: no partitioning, collect raw blocks into leaf 0.
    if (partition_config.numPasses() == 0)
    {
        leaf_chains[0].push_back(block.getColumns());
        return ok;
    }

    /// Pass 0 is eager and batched like BufferedShardByHashTransform: accumulate input, flush per block
    /// when the setting is 0, else once the accumulated byte threshold is crossed.
    pending_input.push_back(block.getColumns());
    pending_input_bytes += block.bytes();
    if (shard_by_hash_input_batch_bytes == 0 || pending_input_bytes >= shard_by_hash_input_batch_bytes)
        flushPass0();

    return ok;
}

void PartitionedHashJoin::flushPass0()
{
    if (pending_input.empty())
        return;

    /// One multi-source pass-0 scatter over the accumulated input batch into the fanout0 stage-1
    /// partitions (re-deriving the hash from the key columns; nothing carried).
    const UInt32 fanout0 = UInt32{1} << partition_config.pass_bits[0];
    const UInt32 shift0 = partition_config.shiftForPass(0);

    size_t scattered_rows = 0;
    Stopwatch watch;
    std::vector<Columns> children = scatterGroupsByKeyHash(pending_input, key_indices, shift0, fanout0, scattered_rows);
    ProfileEvents::increment(ProfileEvents::PartitionedHashBuildShufflePass0Microseconds, watch.elapsedMicroseconds());
    ProfileEvents::increment(ProfileEvents::PartitionedHashBuildShuffleMicroseconds, watch.elapsedMicroseconds());
    ProfileEvents::increment(ProfileEvents::PartitionedHashBuildScatterRows, scattered_rows);
    ProfileEvents::increment(ProfileEvents::PartitionedHashBuildBlocksMoved, pending_input.size());

    pending_input.clear();
    pending_input_bytes = 0;

    for (UInt32 pid = 0; pid < fanout0; ++pid)
        pushToStage(1, pid, std::move(children[pid]));
}

void PartitionedHashJoin::pushToStage(size_t stage, size_t prefix, Columns group)
{
    if (groupRows(group) == 0)
        return;

    /// Leaf: append to the leaf chain (no carried hash to drop).
    if (stage == partition_config.numPasses())
    {
        leaf_chains[prefix].push_back(std::move(group));
        return;
    }

    /// Intermediate stage: accumulate; refine when the partition reaches the batch threshold (or always
    /// when the setting is 0).
    stage_buffer_bytes[stage][prefix] += groupBytes(group);
    stage_buffers[stage][prefix].push_back(std::move(group));
    if (shard_by_hash_input_batch_bytes == 0 || stage_buffer_bytes[stage][prefix] >= shard_by_hash_input_batch_bytes)
        refineBuffer(stage, prefix);
}

void PartitionedHashJoin::refineBuffer(size_t stage, size_t prefix)
{
    auto & chain = stage_buffers[stage][prefix];
    if (chain.empty())
        return;

    const UInt32 fanout = UInt32{1} << partition_config.pass_bits[stage];
    const UInt32 shift = partition_config.shiftForPass(stage);

    size_t scattered_rows = 0;
    Stopwatch watch;
    std::vector<Columns> children = scatterGroupsByKeyHash(chain, key_indices, shift, fanout, scattered_rows);
    const auto us = watch.elapsedMicroseconds();
    ProfileEvents::increment(ProfileEvents::PartitionedHashBuildShuffleTrailingMicroseconds, us);
    ProfileEvents::increment(ProfileEvents::PartitionedHashBuildShuffleMicroseconds, us);
    ProfileEvents::increment(ProfileEvents::PartitionedHashBuildScatterRows, scattered_rows);

    chain.clear();
    stage_buffer_bytes[stage][prefix] = 0;

    for (UInt32 pid = 0; pid < fanout; ++pid)
        pushToStage(stage + 1, prefix * fanout + pid, std::move(children[pid]));
}

void PartitionedHashJoin::onBuildPhaseFinish()
{
    /// Flush the pending input batch through pass 0, then drain every remaining buffer down the cascade.
    flushPass0();
    for (size_t stage = 1; stage < partition_config.numPasses(); ++stage)
        for (size_t prefix = 0; prefix < stage_buffers[stage].size(); ++prefix)
            refineBuffer(stage, prefix);

    /// Runtime conservation check on the REAL workload: rows in leaf_chains == rows ingested.
    size_t leaf_rows = 0;
    for (const auto & chain : leaf_chains)
        for (const auto & group : chain)
            leaf_rows += groupRows(group);

    const size_t ingested = ingested_rows.load(std::memory_order_relaxed);
    if (leaf_rows != ingested)
        LOG_ERROR(
            getLogger("PartitionedHashJoin"),
            "Build shuffle row conservation VIOLATED: leaf_chains has {} rows but {} were ingested",
            leaf_rows,
            ingested);
    else
        LOG_DEBUG(
            getLogger("PartitionedHashJoin"),
            "Build shuffle row conservation OK: {} rows across {} leaves",
            leaf_rows,
            partition_config.total_leaves);
}

void PartitionedHashJoin::checkTypesOfKeys(const Block & block) const
{
    hash_join->checkTypesOfKeys(block);
}

void PartitionedHashJoin::initialize(const Block & left_sample_block)
{
    hash_join->initialize(left_sample_block);
}

JoinResultPtr PartitionedHashJoin::joinBlock(Block block)
{
    return hash_join->joinBlock(std::move(block));
}

void PartitionedHashJoin::setTotals(const Block & block)
{
    hash_join->setTotals(block);
}

const Block & PartitionedHashJoin::getTotals() const
{
    return hash_join->getTotals();
}

size_t PartitionedHashJoin::getTotalRowCount() const
{
    return hash_join->getTotalRowCount();
}

size_t PartitionedHashJoin::getTotalByteCount() const
{
    return hash_join->getTotalByteCount();
}

bool PartitionedHashJoin::alwaysReturnsEmptySet() const
{
    return hash_join->alwaysReturnsEmptySet();
}

IBlocksStreamPtr
PartitionedHashJoin::getNonJoinedBlocks(const Block & left_sample_block, const Block & result_sample_block, UInt64 max_block_size) const
{
    return hash_join->getNonJoinedBlocks(left_sample_block, result_sample_block, max_block_size);
}

}
