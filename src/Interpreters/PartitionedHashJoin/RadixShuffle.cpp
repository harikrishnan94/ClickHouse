#include <Interpreters/PartitionedHashJoin/PartitionOutput.h>
#include <Interpreters/PartitionedHashJoin/RadixShuffle.h>
#include <Interpreters/PartitionedHashJoin/RadixShuffleHash.h>

#include <Columns/ColumnNullable.h>
#include <Columns/IColumn.h>
#include <Common/assert_cast.h>

#include <cstring>

namespace DB
{

/// Return the raw data pointer for a scatter column, peeling nullable/const wrappers.
///   is_nullmap = true  → return the null-map byte array
///   is_nullable = true → return the nested column's raw data
///   otherwise          → return the column's own raw data
static const uint8_t * getColData(const IColumn & col, const ShuffleColDesc & sc, size_t row_start)
{
    if (sc.is_nullmap)
    {
        const auto & nc = assert_cast<const ColumnNullable &>(col);
        return reinterpret_cast<const uint8_t *>(nc.getNullMapColumn().getData().data()) + row_start;
    }
    if (sc.is_nullable)
    {
        const auto & nc = assert_cast<const ColumnNullable &>(col);
        return reinterpret_cast<const uint8_t *>(nc.getNestedColumn().getRawData().data()) + (row_start * sc.elem_bytes);
    }
    return reinterpret_cast<const uint8_t *>(col.getRawData().data()) + (row_start * sc.elem_bytes);
}

/// Template implementation: `UseSWWC` is a compile-time switch that lets the
/// compiler dead-code-eliminate the unused scatter mode and inline the inner loops.
template <bool UseSWWC>
static void shuffleBlockIntoPartitionsImpl(
    const Block & block,
    const ShuffleSpec & spec,
    std::vector<PartitionOutput> & parts,
    std::vector<RadixShuffleColumnPtr> & cols,
    ShuffleScratch & scratch,
    BumpArena & arena)
{
    const size_t rows = block.rows();
    if (rows == 0)
        return;

    const size_t P = spec.P;
    const size_t K = spec.totalCols();
    const uint64_t mask = static_cast<uint64_t>(P - 1);
    const size_t batch_sz = spec.batch_size;

    uint16_t * pids = scratch.pids.data();
    uint16_t * hist = scratch.hist.data();
    uint8_t * const pos = scratch.positions.data();
    uint8_t * const scnt = scratch.swwc_cnt.data();

    size_t row_start = 0;
    while (row_start < rows)
    {
        const size_t n = std::min(batch_sz, rows - row_start);

        // ── Phase 1: SIMD hash → pids[] ──────────────────────────────────
        // For nullable key columns, hash the INNER data (treat null rows as value 0).
        // Materialise ColumnConst first — constant-folded columns are not contiguous.
        {
            bool first = true;
            for (const auto & kc : spec.key_cols)
            {
                if (kc.is_nullmap)
                    continue; // null-map slots don't contribute to the partition hash

                const ColumnPtr col_full = block.getByPosition(kc.block_pos).column->convertToFullColumnIfConst();
                const IColumn & col = *col_full;
                const uint8_t * key_data = nullptr;
                if (kc.is_nullable)
                {
                    const auto & nc = assert_cast<const ColumnNullable &>(col);
                    key_data = reinterpret_cast<const uint8_t *>(nc.getNestedColumn().getRawData().data()) + (row_start * kc.elem_bytes);
                }
                else
                {
                    key_data = reinterpret_cast<const uint8_t *>(col.getRawData().data()) + (row_start * kc.elem_bytes);
                }

                hashOneKeyIntoIds(key_data, kc.elem_bytes, n, mask, pids, first);
                first = false;
            }
        }

        // ── Phase 2: Histogram ────────────────────────────────────────────
        scratch.clearHist(P);
        for (size_t j = 0; j < n; ++j)
            hist[pids[j]]++;

        // ── Phase 3: Pre-grow + on_grow + pre-commit ─────────────────────
        for (size_t p = 0; p < P; ++p)
        {
            if (!hist[p])
                continue;

            auto & po = parts[p];

            const bool need_grow = !po.cur || po.cur->filled + hist[p] > po.cur->capacity;
            if (need_grow)
            {
                if constexpr (UseSWWC)
                {
                    if (scnt[p] > 0)
                    {
                        for (size_t ci = 0; ci < K; ++ci)
                            cols[ci]->drain_one(p, scnt[p]);
                        scnt[p] = 0;
                    }
                }
                growPartitionOutput(po, arena, po.next_cap, spec.col_elem_bytes.data(), K);
                for (size_t ci = 0; ci < K; ++ci)
                    cols[ci]->on_grow(p, po.cur->cols[ci]);
            }

            po.cur->filled += hist[p];
            po.total_rows += hist[p];
        }

        // ── Phase 4a: positions[] (SWWC only) ────────────────────────────
        // positions[j] = staging slot for row j. swwc_cnt[p] tracks the next slot
        // to fill for partition p, wrapping mod 8 so subsequent batches reuse the
        // staging line after each scatter_staged() inline-flushes at slot==7.
        if constexpr (UseSWWC)
        {
            for (size_t j = 0; j < n; ++j)
            {
                const uint16_t p = pids[j];
                const uint8_t slot_idx = scnt[p];
                pos[j] = slot_idx;
                scnt[p] = (slot_idx + 1) & 7;
            }
        }

        // ── Phase 4b: scatter ─────────────────────────────────────────────
        // Materialise ColumnConst before calling getColData / getRawData.
        for (size_t ci = 0; ci < K; ++ci)
        {
            const auto & sc = spec.scatter_cols[ci];
            const ColumnPtr col_full = block.getByPosition(sc.block_pos).column->convertToFullColumnIfConst();
            const IColumn & col = *col_full;
            const void * src = getColData(col, sc, row_start);

            if constexpr (UseSWWC)
                cols[ci]->scatter_staged(pids, pos, src, n);
            else
                cols[ci]->scatter_direct(pids, src, n);
        }

        row_start += n;
    }
}

/// Public dispatcher: single runtime branch into one of two fully-specialised
/// implementations. Inside each specialisation, `UseSWWC` is a compile-time
/// constant so the dead branch is dropped.
void shuffleBlockIntoPartitions(
    const Block & block,
    const ShuffleSpec & spec,
    std::vector<PartitionOutput> & parts,
    std::vector<RadixShuffleColumnPtr> & cols,
    ShuffleScratch & scratch,
    BumpArena & arena)
{
    if (spec.use_swwc)
        shuffleBlockIntoPartitionsImpl<true>(block, spec, parts, cols, scratch, arena);
    else
        shuffleBlockIntoPartitionsImpl<false>(block, spec, parts, cols, scratch, arena);
}

}
