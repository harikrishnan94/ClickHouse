#pragma once

#include <Common/RadixShuffle/ColumnPrimitives.h>


namespace DB
{

/// Wrap column primitives for the nested type X into column primitives for
/// `ColumnNullable(X)`. The wrapped scatter/reconstruct/hash treat the
/// nullable as two parallel sub-columns each subject to the round-trip
/// invariant (§3.5): the nested column's bytes at null positions are
/// preserved, not normalized.
ColumnPrimitives makeNullable(ColumnPrimitives nested);

}
