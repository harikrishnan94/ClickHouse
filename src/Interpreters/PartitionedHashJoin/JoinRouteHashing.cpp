#include <Interpreters/PartitionedHashJoin/JoinRouteHashing.h>

#include <Columns/ColumnLowCardinality.h>
#include <Columns/ColumnString.h>
#include <Columns/ColumnsScatter.h>
#include <Interpreters/PartitionedHashJoin/DenseHyperLogLog.h>
#include <Common/PODArray.h>

#include <algorithm>
#include <array>
#include <cstring>

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

    /// All-fixed fast path (multi-column numeric keys, e.g. `keys128`/`keys256`): fold each
    /// row with the accumulator in a register, in one pass over the rows. The column-outer
    /// loop below would instead stream the accumulator array through memory once per column
    /// (~4x the memops for `keys256`) - measured 6.6% of the probe query's CPU on the
    /// dimension-join loss cell. The per-row fold chain (all columns in clause order) is
    /// bit-identical to the column-outer accumulation, preserving the build/probe contract.
    if (std::ranges::all_of(columns, [](const RouteColumn & c) { return c.kind == RouteColumnKind::Fixed; }))
    {
        struct FixedSource
        {
            const char * data;
            size_t width;
        };
        std::vector<FixedSource> sources; /// STYLE_CHECK_ALLOW_STD_CONTAINERS
        sources.reserve(columns.size());
        for (const auto & column : columns)
            sources.push_back({column.data, column.width});

        /// The multi-column numeric keys are almost always all-8-byte columns (`keys128` =
        /// 2xUInt64, `keys256` = 4xUInt64, the `hashed` shapes 8x). With the width only known
        /// at runtime the row loop above re-evaluates `foldBytes`' width loop AND its tail
        /// switch per column per row, and reloads the source pointer/width pair from the
        /// vector each time - measured 11-19% of the probe query's cycles as the row-loop
        /// hot spot. A compile-time width-8 fold (bit-identical: `foldBytes(h, p, 8)` is
        /// exactly one full `mixStep` step with no tail) with the column count unrolled for
        /// the common counts keeps the column pointers in registers and the fold branch-free.
        if (std::ranges::all_of(sources, [](const FixedSource & s) { return s.width == 8; }))
        {
            auto fold_all_w8 = [&]<size_t n_columns>()
            {
                std::array<const char *, n_columns> data;
                for (size_t c = 0; c < n_columns; ++c)
                    data[c] = sources[c].data;
                for (size_t i = 0; i < rows; ++i)
                {
                    UInt64 h = 0;
                    for (size_t c = 0; c < n_columns; ++c)
                    {
                        UInt64 x;
                        memcpy(&x, data[c] + i * 8, sizeof(x));
                        h = ColumnsScatter::mixStep(h, x);
                    }
                    sink(i, ColumnsScatter::finalizeRoute(h));
                }
            };
            switch (sources.size())
            {
                case 2: fold_all_w8.template operator()<2>(); return;
                case 3: fold_all_w8.template operator()<3>(); return;
                case 4: fold_all_w8.template operator()<4>(); return;
                default: {
                    for (size_t i = 0; i < rows; ++i)
                    {
                        UInt64 h = 0;
                        for (const auto & source : sources)
                        {
                            UInt64 x;
                            memcpy(&x, source.data + i * 8, sizeof(x));
                            h = ColumnsScatter::mixStep(h, x);
                        }
                        sink(i, ColumnsScatter::finalizeRoute(h));
                    }
                    return;
                }
            }
        }

        for (size_t i = 0; i < rows; ++i)
        {
            UInt64 h = 0;
            for (const auto & source : sources)
                h = ColumnsScatter::foldBytes(h, source.data + i * source.width, source.width);
            sink(i, ColumnsScatter::finalizeRoute(h));
        }
        return;
    }

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

void computeJoinRoutesForFill(const ColumnRawPtrs & key_columns, size_t rows, UInt16 * routes)
{
    computeJoinRouteWordsImpl(key_columns, rows, [&](size_t row, UInt32 word) { routes[row] = static_cast<UInt16>(word >> 16); });
}

void computeJoinLeafIds(const ColumnRawPtrs & key_columns, size_t rows, size_t bits, UInt16 * leaf_ids)
{
    chassert(bits > 0 && bits <= 16);
    const auto shift = static_cast<UInt32>(32 - bits);
    computeJoinRouteWordsImpl(
        key_columns, rows, [&](size_t row, UInt32 word) { leaf_ids[row] = static_cast<UInt16>(word >> shift); });
}

}
