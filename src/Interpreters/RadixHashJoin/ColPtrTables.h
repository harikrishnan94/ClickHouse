#pragma once

#include <Core/Block.h>
#include <Columns/IColumn.h>
#include <DataTypes/IDataType.h>

#include <string>
#include <vector>

namespace DB::RadixHash
{

/** Per-column / per-block typed base pointers for the probe-side build-payload gather (spec section
  * 5.4; phase P4). Build payload is never co-located with the leaf cell — at emit time a matched
  * `BuildRef{block_no, row_no}` resolves the payload directly out of the accumulated build blocks.
  *
  * For every build output (payload) column we resolve, once per accumulated block, the `const IColumn *`
  * base of that column in that block (`by_block[block_no]`). Gathering a matched row is then a single
  * dependent lookup + one `insertFrom` — no per-row column name lookup and no `IColumn` type dispatch
  * beyond the unavoidable virtual `insertFrom`. (For a fixed-width column this is the analogue of the
  * spec's `const T*` table; the `IColumn *` form keeps the gather correct for any payload type.)
  *
  * The pointers alias the accumulated build blocks, which are read-only for the whole probe phase, so
  * the bases stay valid for the join's lifetime.
  */
struct ColPtrTables
{
    struct PayloadColumn
    {
        std::string output_name; /// renamedRightColumnName(original) — the name in the joined output
        DataTypePtr type; /// output column type (from sample_block_with_columns_to_add)
        std::vector<const IColumn *> by_block; /// indexed by build block_no; base of this column in that block
    };

    std::vector<PayloadColumn> payload;

    /// Build the tables from the accumulated build `blocks` and the right payload columns to add
    /// (`columns_to_add` = right_sample_block minus the join keys). `output_names[i]` is the renamed
    /// output name of `columns_to_add[i]` (e.g. `renamedRightColumnName`). The payload column values
    /// are read from each block by the column's original name.
    void build(const std::vector<Block> & blocks, const Block & columns_to_add, const std::vector<std::string> & output_names)
    {
        const size_t num_cols = columns_to_add.columns();
        payload.clear();
        payload.reserve(num_cols);
        for (size_t col = 0; col < num_cols; ++col)
        {
            const auto & src = columns_to_add.getByPosition(col);
            PayloadColumn pc;
            pc.output_name = output_names[col];
            pc.type = src.type;
            pc.by_block.resize(blocks.size(), nullptr);
            for (size_t block_idx = 0; block_idx < blocks.size(); ++block_idx)
                pc.by_block[block_idx] = blocks[block_idx].getByName(src.name).column.get();
            payload.push_back(std::move(pc));
        }
    }
};

}
