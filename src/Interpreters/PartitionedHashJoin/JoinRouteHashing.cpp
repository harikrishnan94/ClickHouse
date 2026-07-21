#include <Interpreters/PartitionedHashJoin/JoinRouteHashing.h>

#include <Columns/ColumnLowCardinality.h>
#include <Columns/ColumnString.h>
#include <Columns/ColumnsScatter.h>
#include <Interpreters/PartitionedHashJoin/DenseHyperLogLog.h>
#include <Common/PODArray.h>

namespace DB
{

namespace
{

/// How one key column feeds the per-row route accumulator. The classification must give the
/// same fold for the same VALUES on both join sides; the only cross-representation pairs the
/// planner allows without a cast are plain-vs-LowCardinality strings and nullable-vs-plain
/// (already stripped by the caller), both of which fold identical value bytes.
enum class RouteColumnKind : UInt8
{
    Fixed, /// fixed-and-contiguous values: fold the raw value bytes
    String, /// ColumnString: fold the value bytes
    LowCardinality, /// live dictionary column: fold the value bytes via getDataAt
    Generic, /// anything else: fold a vectorized value-based hash (computeHashInto)
};

struct RouteColumn
{
    RouteColumnKind kind = RouteColumnKind::Generic;
    const char * data = nullptr;
    size_t width = 0;
    const UInt64 * offsets = nullptr;
    const char * chars = nullptr;
    const IColumn * column = nullptr;
    PaddedPODArray<UInt32> generic_hash;

    explicit RouteColumn(const IColumn & col, size_t rows)
    {
        if (const auto * low_cardinality = typeid_cast<const ColumnLowCardinality *>(&col))
        {
            kind = RouteColumnKind::LowCardinality;
            column = low_cardinality;
        }
        else if (const auto * string = typeid_cast<const ColumnString *>(&col))
        {
            kind = RouteColumnKind::String;
            chars = reinterpret_cast<const char *>(string->getChars().data());
            offsets = string->getOffsets().data();
        }
        else if (col.isFixedAndContiguous())
        {
            kind = RouteColumnKind::Fixed;
            data = col.getRawData().data();
            width = col.sizeOfValueIfFixed();
        }
        else
        {
            /// Value-based and representation-independent by the `computeHashInto` contract.
            /// The hash is CRC32C-flavored like the map hash, but the `mixStep` multiply in
            /// `fold` decorrelates the route bits from the in-table bucket placement.
            kind = RouteColumnKind::Generic;
            generic_hash.resize(rows);
            col.computeHashInto(0, rows, generic_hash.data(), /*initial=*/true);
        }
    }

    ALWAYS_INLINE UInt64 fold(UInt64 h, size_t row) const
    {
        switch (kind)
        {
            case RouteColumnKind::Fixed: return ColumnsScatter::foldBytes(h, data + row * width, width);
            case RouteColumnKind::String: {
                const size_t begin = offsets[static_cast<ssize_t>(row) - 1];
                return ColumnsScatter::foldBytes(h, chars + begin, offsets[row] - begin);
            }
            case RouteColumnKind::LowCardinality: {
                const std::string_view value = column->getDataAt(row);
                return ColumnsScatter::foldBytes(h, value.data(), value.size());
            }
            case RouteColumnKind::Generic: return ColumnsScatter::mixStep(h, generic_hash[row]);
        }
    }
};

template <typename T, typename Sink>
void routeSingleNumericColumn(const char * data, size_t rows, Sink & sink)
{
    const T * values = reinterpret_cast<const T *>(data);
    for (size_t i = 0; i < rows; ++i)
        sink(i, ColumnsScatter::routeWord(static_cast<UInt64>(values[i])));
}

/// `sink(row, word)` is a compile-time-known callable inlined into the loops. The words are a
/// build/probe contract: both public entry points instantiate this one implementation, and the
/// fold/finalize chain must not change without changing both sides in lockstep.
template <typename Sink>
void computeJoinRouteWordsImpl(const ColumnRawPtrs & key_columns, size_t rows, Sink && sink)
{
    if (rows == 0)
        return;
    chassert(!key_columns.empty());

    /// The hot single-numeric-key path: the exact benchmark-proven `routeWord` on the value.
    /// Restricted to numeric columns so that a column that can pair with a different physical
    /// representation on the other side (FixedString vs LowCardinality) takes the byte fold.
    if (key_columns.size() == 1 && key_columns[0]->isNumeric() && key_columns[0]->isFixedAndContiguous())
    {
        const IColumn & column = *key_columns[0];
        const char * data = column.getRawData().data();
        switch (column.sizeOfValueIfFixed())
        {
            case 1: routeSingleNumericColumn<UInt8>(data, rows, sink); return;
            case 2: routeSingleNumericColumn<UInt16>(data, rows, sink); return;
            case 4: routeSingleNumericColumn<UInt32>(data, rows, sink); return;
            case 8: routeSingleNumericColumn<UInt64>(data, rows, sink); return;
            default: break; /// wide numerics (UInt128/UInt256/...) take the byte fold below
        }
    }

    std::vector<RouteColumn> columns;
    columns.reserve(key_columns.size());
    for (const auto * column : key_columns)
        columns.emplace_back(*column, rows);

    /// Column-outer accumulation keeps the per-column dispatch out of the row loop; the per-row
    /// fold chain (all columns in clause order) is the same as a row-outer loop would produce.
    PaddedPODArray<UInt64> accumulator(rows, 0);
    for (const auto & column : columns)
        for (size_t i = 0; i < rows; ++i)
            accumulator[i] = column.fold(accumulator[i], i);
    for (size_t i = 0; i < rows; ++i)
        sink(i, ColumnsScatter::finalizeRoute(accumulator[i]));
}

}

void computeJoinRouteWords(const ColumnRawPtrs & key_columns, size_t rows, UInt32 * words)
{
    computeJoinRouteWordsImpl(key_columns, rows, [&](size_t row, UInt32 word) { words[row] = word; });
}

void computeJoinRoutesForFill(const ColumnRawPtrs & key_columns, size_t rows, const UInt8 * skip, UInt16 * routes, DenseHyperLogLog & hll)
{
    computeJoinRouteWordsImpl(
        key_columns,
        rows,
        [&](size_t row, UInt32 word)
        {
            routes[row] = static_cast<UInt16>(word >> 16);
            if (!skip || !skip[row])
                hll.add(word);
        });
}

void computeJoinLeafIds(const ColumnRawPtrs & key_columns, size_t rows, size_t bits, UInt16 * leaf_ids)
{
    chassert(bits > 0 && bits <= 16);
    const auto shift = static_cast<UInt32>(32 - bits);
    computeJoinRouteWordsImpl(
        key_columns, rows, [&](size_t row, UInt32 word) { leaf_ids[row] = static_cast<UInt16>(word >> shift); });
}

}
