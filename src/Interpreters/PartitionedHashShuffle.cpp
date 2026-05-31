#include <Interpreters/PartitionedHashShuffle.h>

#include <algorithm>

#include <Columns/ColumnsScatter.h>
#include <Common/PODArray.h>

namespace DB
{

std::vector<Columns> scatterGroupsByKeyHash(
    const std::vector<Columns> & sources, const std::vector<size_t> & key_indices, UInt32 shift, UInt32 fanout, size_t & scattered_rows)
{
    const UInt32 mask = fanout - 1;
    const size_t num_cols = sources.empty() ? 0 : sources[0].size();

    std::vector<Columns> children(fanout);
    for (auto & c : children)
        c.resize(num_cols);

    /// Per-source pids: re-derive the hash from the key columns, slice this pass's window.
    std::vector<PaddedPODArray<UInt32>> pids(sources.size());
    std::vector<std::span<const UInt32>> pids_spans(sources.size());
    PaddedPODArray<UInt32> hashes;
    size_t total_rows = 0;
    for (size_t b = 0; b < sources.size(); ++b)
    {
        const size_t rows = sources[b].empty() ? 0 : sources[b][0]->size();
        total_rows += rows;
        hashes.resize(rows);
        bool initial = true;
        for (size_t ki : key_indices)
        {
            sources[b][ki]->computeHashInto(0, rows, hashes.data(), initial);
            initial = false;
        }
        pids[b].resize(rows);
        for (size_t i = 0; i < rows; ++i)
            pids[b][i] = (hashes[i] >> shift) & mask;
        pids_spans[b] = std::span<const UInt32>(pids[b].data(), rows);
    }
    scattered_rows += total_rows;

    /// Empty: return fanout empty children cloned from the source column types.
    if (total_rows == 0)
    {
        if (!sources.empty())
            for (auto & c : children)
                for (size_t col = 0; col < num_cols; ++col)
                    c[col] = sources[0][col]->cloneEmpty();
        return children;
    }

    PaddedPODArray<UInt32> rows_per_shard;
    rows_per_shard.resize(fanout);
    std::fill(rows_per_shard.begin(), rows_per_shard.end(), UInt32{0});
    ColumnsScatter::countRowsPerShard(
        std::span<const std::span<const UInt32>>(pids_spans.data(), pids_spans.size()),
        std::span<UInt32>(rows_per_shard.data(), fanout));

    std::vector<const IColumn *> src_ptrs(sources.size());
    for (size_t c = 0; c < num_cols; ++c)
    {
        for (size_t b = 0; b < sources.size(); ++b)
            src_ptrs[b] = sources[b][c].get();
        MutableColumns scattered = ColumnsScatter::scatter(
            std::span<const IColumn * const>(src_ptrs.data(), src_ptrs.size()),
            std::span<const std::span<const UInt32>>(pids_spans.data(), pids_spans.size()),
            fanout,
            std::span<const UInt32>(rows_per_shard.data(), fanout));
        for (UInt32 s = 0; s < fanout; ++s)
            children[s][c] = std::move(scattered[s]);
    }

    return children;
}

std::vector<Columns> radixShuffleBlockToLeaves(
    const Columns & input_columns,
    const std::vector<size_t> & key_indices,
    const PartitionConfig & config,
    size_t & scattered_rows)
{
    std::vector<Columns> result;

    if (config.numPasses() == 0)
    {
        result.push_back(input_columns);
        return result;
    }

    std::vector<Columns> current;
    current.push_back(input_columns);

    for (size_t pass = 0; pass < config.numPasses(); ++pass)
    {
        const UInt32 fanout = UInt32{1} << config.pass_bits[pass];
        const UInt32 shift = config.shiftForPass(pass);

        std::vector<Columns> next;
        next.reserve(current.size() * fanout);
        for (const auto & group : current)
        {
            std::vector<Columns> children = scatterGroupsByKeyHash({group}, key_indices, shift, fanout, scattered_rows);
            for (auto & child : children)
                next.push_back(std::move(child));
        }
        current = std::move(next);
    }

    return current;
}

}
