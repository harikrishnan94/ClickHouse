#include <Interpreters/PartitionedHashShuffle.h>

#include <algorithm>

#include <Columns/ColumnsScatter.h>
#include <Common/PODArray.h>

namespace DB
{

void scatterGroupsByKeyHash(
    const std::vector<Columns> & sources,
    const std::vector<size_t> & key_indices,
    UInt32 shift,
    UInt32 fanout,
    size_t & scattered_rows,
    ScatterScratch & scratch,
    std::vector<Columns> & children)
{
    const UInt32 mask = fanout - 1;
    const size_t num_cols = sources.empty() ? 0 : sources[0].size();

    /// Reset the reusable output container to `fanout` column-groups of `num_cols` slots. The inner
    /// `Columns` were moved out by the previous caller; only the outer vector's capacity is retained.
    children.resize(fanout);
    for (auto & c : children)
    {
        c.clear();
        c.resize(num_cols);
    }

    /// Per-source pids: re-derive the hash from the key columns, slice this pass's window. `pids` grows
    /// but never shrinks, so each `PaddedPODArray` keeps its capacity across flushes.
    if (scratch.pids.size() < sources.size())
        scratch.pids.resize(sources.size());
    scratch.pids_spans.resize(sources.size());

    size_t total_rows = 0;
    for (size_t b = 0; b < sources.size(); ++b)
    {
        const size_t rows = sources[b].empty() ? 0 : sources[b][0]->size();
        total_rows += rows;
        scratch.hashes.resize(rows);
        bool initial = true;
        for (size_t ki : key_indices)
        {
            sources[b][ki]->computeHashInto(0, rows, scratch.hashes.data(), initial);
            initial = false;
        }
        auto & pids = scratch.pids[b];
        pids.resize(rows);
        for (size_t i = 0; i < rows; ++i)
            pids[i] = (scratch.hashes[i] >> shift) & mask;
        scratch.pids_spans[b] = std::span<const UInt32>(pids.data(), rows);
    }
    scattered_rows += total_rows;

    /// Empty: leave `fanout` empty children cloned from the source column types.
    if (total_rows == 0)
    {
        if (!sources.empty())
            for (auto & c : children)
                for (size_t col = 0; col < num_cols; ++col)
                    c[col] = sources[0][col]->cloneEmpty();
        return;
    }

    scratch.rows_per_shard.resize(fanout);
    std::fill(scratch.rows_per_shard.begin(), scratch.rows_per_shard.end(), UInt32{0});
    ColumnsScatter::countRowsPerShard(
        std::span<const std::span<const UInt32>>(scratch.pids_spans.data(), scratch.pids_spans.size()),
        std::span<UInt32>(scratch.rows_per_shard.data(), fanout));

    scratch.src_ptrs.resize(sources.size());
    for (size_t c = 0; c < num_cols; ++c)
    {
        for (size_t b = 0; b < sources.size(); ++b)
            scratch.src_ptrs[b] = sources[b][c].get();
        MutableColumns scattered = ColumnsScatter::scatter(
            std::span<const IColumn * const>(scratch.src_ptrs.data(), scratch.src_ptrs.size()),
            std::span<const std::span<const UInt32>>(scratch.pids_spans.data(), scratch.pids_spans.size()),
            fanout,
            std::span<const UInt32>(scratch.rows_per_shard.data(), fanout));
        for (UInt32 s = 0; s < fanout; ++s)
            children[s][c] = std::move(scattered[s]);
    }
}

std::vector<Columns> radixShuffleBlockToLeaves(
    const Columns & input_columns, const std::vector<size_t> & key_indices, const PartitionConfig & config, size_t & scattered_rows)
{
    std::vector<Columns> result;

    if (config.numPasses() == 0)
    {
        result.push_back(input_columns);
        return result;
    }

    std::vector<Columns> current;
    current.push_back(input_columns);

    /// One reusable scratch + one children buffer: this oracle scatters one stage fully before the next,
    /// so there is no nested re-entrancy and a single children buffer is enough.
    ScatterScratch scratch;
    std::vector<Columns> children;
    for (size_t pass = 0; pass < config.numPasses(); ++pass)
    {
        const UInt32 fanout = UInt32{1} << config.pass_bits[pass];
        const UInt32 shift = config.shiftForPass(pass);

        std::vector<Columns> next;
        next.reserve(current.size() * fanout);
        for (const auto & group : current)
        {
            scatterGroupsByKeyHash({group}, key_indices, shift, fanout, scattered_rows, scratch, children);
            for (auto & child : children)
                next.push_back(std::move(child));
        }
        current = std::move(next);
    }

    return current;
}

}
