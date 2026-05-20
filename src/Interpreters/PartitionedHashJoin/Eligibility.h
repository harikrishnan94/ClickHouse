#pragma once

#include <Core/Block.h>
#include <Core/Names.h>

namespace DB
{

/// Returns true iff `right_sample` satisfies the fixed-width constraints in spec §2.1–§2.2:
///   - sum of key column fixed widths ≤ 16 bytes (128 bits)
///   - every key column AND every kept payload column is fixed-width after wrapper materialisation
///     (Nullable, LowCardinality, ColumnConst, ColumnSparse over a fixed-width inner type → OK;
///      same wrappers over variable-width inner types → not OK)
///
/// `key_names`            : right-side key column names (must exist in right_sample)
/// `kept_payload_names`   : kept right-side payload column names (must exist in right_sample)
bool isSupportedByColumns(const Block & right_sample, const Names & key_names, const Names & kept_payload_names);

/// Returns the fixed element size (in bytes) for a column after standard wrapper materialisation,
/// or 0 if the underlying type is variable-width (= ineligible).
size_t fixedElemBytes(const DataTypePtr & dt);

}
