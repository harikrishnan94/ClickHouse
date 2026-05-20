#include <Interpreters/PartitionedHashJoin/ProbePartitions.h>

#include <Columns/IColumn.h>
#include <Core/Block.h>

namespace DB
{

void probeScatterBlock(const Block & block, const uint16_t * pids, size_t n_rows, size_t P, std::vector<ProbePartition> & probe_parts)
{
    if (n_rows == 0)
        return;

    IColumn::Selector selector(n_rows);
    for (size_t i = 0; i < n_rows; ++i)
        selector[i] = pids[i];

    const size_t ncols = block.columns();

    /// Scatter each column into P sub-columns.
    std::vector<std::vector<ColumnPtr>> per_col_parts(ncols);
    for (size_t ci = 0; ci < ncols; ++ci)
    {
        auto scattered = block.getByPosition(ci).column->scatter(P, selector);
        per_col_parts[ci].resize(P);
        for (size_t p = 0; p < P; ++p)
            per_col_parts[ci][p] = std::move(scattered[p]);
    }

    /// Assemble per-partition blocks.
    for (size_t p = 0; p < P; ++p)
    {
        if (per_col_parts.empty())
            continue;
        const size_t rows_p = per_col_parts[0][p]->size();
        if (rows_p == 0)
            continue;

        MutableColumns cols;
        cols.reserve(ncols);
        for (size_t ci = 0; ci < ncols; ++ci)
            cols.push_back(IColumn::mutate(std::move(per_col_parts[ci][p])));

        probe_parts[p].slices.push_back(block.cloneWithColumns(std::move(cols)));
        probe_parts[p].total_rows += rows_p;
    }
}

}
