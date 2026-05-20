#include <Interpreters/PartitionedHashJoin.h>
#include <Interpreters/PartitionedHashJoin/DelayedBlocks.h>
#include <Interpreters/PartitionedHashJoin/OutBlock.h>
#include <Interpreters/PartitionedHashJoin/ThreadSlot.h>

#include <Interpreters/HashJoin/HashJoin.h>

#include <Columns/ColumnNullable.h>
#include <Columns/ColumnsNumber.h>
#include <Columns/IColumn.h>

#include <Core/Block.h>
#include <Common/assert_cast.h>

#include <cstring>

namespace DB
{

PartitionedHashJoinDelayedBlocks::PartitionedHashJoinDelayedBlocks(PartitionedHashJoin & join)
    : join_(join)
{
}

/// Reconstruct a CH Block from one OutBlock (build side).
/// We use IColumn::insertRangeFrom by wrapping the arena data in a temporary column.
Block PartitionedHashJoinDelayedBlocks::buildOutBlockToBlock(const OutBlock & ob, const ShuffleSpec & spec) const
{
    if (ob.filled == 0)
        return {};

    const size_t n = ob.filled;
    const auto & schema = *join_.rightSampleBlock();
    const auto & scatter_cols = spec.scatter_cols;

    // Build the output as a set of MutableColumns matching the schema.
    MutableColumns out_cols;
    out_cols.reserve(schema.columns());
    for (size_t ci = 0; ci < schema.columns(); ++ci)
        out_cols.push_back(schema.getByPosition(ci).column->cloneEmpty());

    // Each scatter_cols entry maps to one ob.cols[si] buffer.
    // Nullable columns contribute TWO entries (is_nullable=true then is_nullmap=true).
    for (size_t si = 0; si < scatter_cols.size() && si < static_cast<size_t>(ob.num_cols); ++si)
    {
        const auto & sc = scatter_cols[si];
        const size_t col_idx = sc.block_pos;
        const size_t eb = sc.elem_bytes;

        if (col_idx >= out_cols.size() || ob.cols[si] == nullptr)
            continue;

        auto & dst = out_cols[col_idx];

        if (sc.is_nullmap)
        {
            // Null-map slot: write 1-byte entries into the ColumnNullable's null map.
            auto & dst_null = assert_cast<ColumnNullable &>(*dst);
            auto & nm_col = dst_null.getNullMapColumn();
            auto tmp = nm_col.cloneEmpty();
            tmp->insertManyDefaults(n);
            std::memcpy(const_cast<char *>(tmp->getRawData().data()), ob.cols[si], n);
            nm_col.insertRangeFrom(*tmp, 0, n);
        }
        else if (sc.is_nullable)
        {
            // Inner-data slot of a nullable column.
            auto & dst_null = assert_cast<ColumnNullable &>(*dst);
            auto & inner = dst_null.getNestedColumn();
            auto tmp = inner.cloneEmpty();
            tmp->insertManyDefaults(n);
            std::memcpy(const_cast<char *>(tmp->getRawData().data()), ob.cols[si], n * eb);
            inner.insertRangeFrom(*tmp, 0, n);
        }
        else
        {
            // Plain non-nullable column.
            auto tmp = dst->cloneEmpty();
            tmp->insertManyDefaults(n);
            std::memcpy(const_cast<char *>(tmp->getRawData().data()), ob.cols[si], n * eb);
            dst->insertRangeFrom(*tmp, 0, n);
        }
    }

    return schema.cloneWithColumns(std::move(out_cols));
}

Block PartitionedHashJoinDelayedBlocks::nextImpl()
{
    // Serialise all concurrent worker calls.
    const std::scoped_lock lg(mutex_);

    if (!ready_.empty())
    {
        Block b = std::move(ready_.front());
        ready_.pop_front();
        return b;
    }

    const size_t P = join_.numPartitions();
    const ShuffleSpec & bspec = join_.buildSpec();

    /// Pipeline contract: getDelayedBlocks() is only called after all build- and
    /// probe-side ingest is done, so slots[] is stable here. num_slots_created is
    /// updated under slot_mu by getOrAssign*Slot() and released atomically; this
    /// acquire-load is the matching pair.
    const size_t n_slots = join_.numSlots();

    while (true)
    {
        const size_t p = join_.partitionCursor().fetch_add(1, std::memory_order_relaxed);
        if (p >= P)
            return {};

        // Count total build and probe rows for this partition.
        size_t total_build = 0;
        size_t total_probe = 0;
        for (size_t s = 0; s < n_slots; ++s)
        {
            ThreadSlot & slot = join_.getSlot(s);
            if (slot.build_initialised)
                total_build += slot.build_parts[p].total_rows;
            if (slot.probe_initialised)
                total_probe += slot.probe_parts[p].total_rows;
        }

        // Skip only if both sides are completely empty — nothing to process.
        if (total_build == 0 && total_probe == 0)
            continue;

        auto part_hj = std::make_shared<HashJoin>(join_.tableJoin(), join_.rightSampleBlock(), join_.anyTakeLastRow(), total_build);

        // Feed build chunks.
        bool any_build_added = false;
        for (size_t s = 0; s < n_slots; ++s)
        {
            ThreadSlot & slot = join_.getSlot(s);
            if (!slot.build_initialised)
                continue;
            for (const OutBlock * b = slot.build_parts[p].head; b; b = b->next)
            {
                const Block blk = buildOutBlockToBlock(*b, bspec);
                if (blk.rows() > 0)
                {
                    part_hj->addBlockToJoin(blk, false);
                    any_build_added = true;
                }
            }
        }

        part_hj->onBuildPhaseFinish();
        if (part_hj->hasPostBuildPhase())
            part_hj->runPostBuildPhase();

        // Don't use alwaysReturnsEmptySet() as an early-out: for LEFT/FULL JOINs,
        // even an empty build side must be probed (left rows emit with NULLs).

        // Probe: iterate full Block slices from IColumn::scatter.
        //
        // Special case: when the build side has 0 rows AND join kind is LEFT/FULL,
        // HashJoin::onBuildPhaseFinish() has promoted ALL→RightAny (all_values_unique=true
        // for empty HT), which sets add_missing=false and drops unmatched left rows.
        // Fix: create a fresh probe-only HashJoin (no onBuildPhaseFinish) for this case.
        std::shared_ptr<HashJoin> probe_hj = part_hj;
        if (!any_build_added)
        {
            const auto k = join_.tableJoin()->kind();
            if (k == JoinKind::Left || k == JoinKind::Full)
            {
                probe_hj = std::make_shared<HashJoin>(join_.tableJoin(), join_.rightSampleBlock(), join_.anyTakeLastRow(), 0);
                // Intentionally do NOT call onBuildPhaseFinish() here:
                // the HT is empty and ALL strictness is preserved, so add_missing=true
                // and unmatched left rows are correctly emitted with NULLs.
            }
        }

        for (size_t s = 0; s < n_slots; ++s)
        {
            ThreadSlot & slot = join_.getSlot(s);
            if (!slot.probe_initialised)
                continue;
            for (const Block & probe_blk : slot.probe_parts[p].slices)
            {
                if (probe_blk.rows() == 0)
                    continue;
                auto result = probe_hj->joinBlock(Block(probe_blk));
                while (result)
                {
                    auto data = result->next();
                    if (!data.block.empty())
                        ready_.push_back(std::move(data.block));
                    if (data.is_last)
                        break;
                }
            }
        }

        // Non-joined rows for RIGHT/FULL.
        if (part_hj->hasNonJoinedRows() && join_.hasNonJoinedHeaders())
        {
            auto non_joined = part_hj->getNonJoinedBlocks(join_.nonJoinedLeftHeader(), join_.nonJoinedResultHeader(), 65536);
            if (non_joined)
            {
                while (!non_joined->isFinished())
                {
                    Block b = non_joined->next();
                    if (!b.empty())
                        ready_.push_back(std::move(b));
                }
            }
        }

        if (!ready_.empty())
        {
            Block b = std::move(ready_.front());
            ready_.pop_front();
            return b;
        }
    }
}

}
