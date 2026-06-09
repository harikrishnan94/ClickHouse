#pragma once

#include <Core/Block.h>
#include <Columns/IColumn.h>
#include <DataTypes/IDataType.h>

#include <string>
#include <vector>

namespace DB::RadixJoin
{

/** Per-(payload column, build block) base pointers for the probe-side payload gather.
  *
  * Build payload is never co-located with the leaf cell — at emit time a matched `BuildRef{block_no,
  * row_no}` resolves the payload directly out of the accumulated build blocks. For each right payload
  * column we resolve, once per accumulated block, the `const IColumn *` of that column in that block.
  * Gathering a matched row is then a dependent lookup + one `insertFrom`: no per-row name lookup, and
  * no `IColumn` type dispatch beyond the unavoidable virtual `insertFrom`. The pointers alias the
  * accumulated blocks, which are read-only for the whole probe phase, so they stay valid.
  */
struct PayloadColumns
{
    struct Column
    {
        std::string output_name;            /// renamed output name in the joined block
        DataTypePtr type;                   /// output column type
        std::vector<const IColumn *> by_block; /// indexed by build block_no
    };

    std::vector<Column> columns;

    /// `columns_to_add` is the right payload (right sample block minus the join keys); `output_names[i]`
    /// is the renamed output name of `columns_to_add[i]`. Payload values are read from each block by the
    /// column's original name.
    void build(const std::vector<Block> & blocks, const Block & columns_to_add, const std::vector<std::string> & output_names)
    {
        const size_t num_cols = columns_to_add.columns();
        columns.clear();
        columns.reserve(num_cols);
        for (size_t col = 0; col < num_cols; ++col)
        {
            const auto & src = columns_to_add.getByPosition(col);
            Column pc;
            pc.output_name = output_names[col];
            pc.type = src.type;
            pc.by_block.resize(blocks.size(), nullptr);
            for (size_t block_idx = 0; block_idx < blocks.size(); ++block_idx)
                pc.by_block[block_idx] = blocks[block_idx].getByName(src.name).column.get();
            columns.push_back(std::move(pc));
        }
    }
};

}
