#include <Common/RadixShuffle/RadixPartitionOperator.h>

#include <Columns/ColumnNullable.h>
#include <Columns/IColumn.h>
#include <Common/RadixShuffle/ColumnPrimitives.h>
#include <Common/RadixShuffle/ColumnPrimitives/FixedWidth.h>
#include <base/types.h>

#if defined(__x86_64__)
#    include <immintrin.h>
#endif

#include <algorithm>
#include <cstdlib>
#include <cstring>


namespace DB::RadixShuffle
{

// ── RadixPartitionOperator implementation ────────────────────────────────────

template <typename TKey>
RadixPartitionOperator<TKey>::RadixPartitionOperator(
    int P, int K, std::vector<ColumnPrimitives> prims, BumpArena & arena, bool use_swwc, size_t init_cap, size_t max_cap)
    : P_(P)
    , K_(K)
    , use_swwc_(use_swwc)
    , batch_(std::max(1024, std::min(kSmartMaxBatch, P * kBatchFactor)))
    , mask_(static_cast<uint32_t>(P) - 1)
    , max_cap_(max_cap)
    , col_prims_(std::move(prims))
    , arena_(arena)
    , pids_(static_cast<size_t>(batch_))
    , hist_(static_cast<size_t>(P), 0)
    , pos_(static_cast<size_t>(batch_))
    , cnt_(static_cast<size_t>(P), 0)
{
    // Build physical column table.
    // Nullable primitives (nested != null && nested has scatter_raw) are expanded
    // into two physical leaf primitives:
    //   - makeFixedWidth<uint8_t>()  for the null map    (1 B/row, always direct scatter)
    //   - *prim.nested               for the values      (full SWWC path via leaf primitive)
    //
    // This eliminates scatterRawNullable entirely — each physical primitive calls
    // the standard scatterRawSwwcFixed<T> directly on a plain ColumnVector sub-column,
    // matching the baseline UInt64Column::scatter_staged pattern exactly.
    for (int k = 0; k < K_; ++k)
    {
        const ColumnPrimitives & prim = col_prims_[static_cast<size_t>(k)];
        const bool expandable = prim.nested != nullptr && prim.nested->scatter_raw != nullptr;
        if (expandable)
        {
            phys_prims_.push_back(makeFixedWidth<UInt8>()); // UInt8=char8_t, has explicit instantiation
            phys_col_info_.push_back({static_cast<size_t>(k), true, false});

            phys_prims_.push_back(*prim.nested);
            phys_col_info_.push_back({static_cast<size_t>(k), false, true});
        }
        else
        {
            phys_prims_.push_back(prim);
            phys_col_info_.push_back({static_cast<size_t>(k), false, false});
        }
    }
    K_phys_ = static_cast<int>(phys_prims_.size());

    // ScatterState and elem_sizes are per physical column.
    scatter_states_.reserve(static_cast<size_t>(K_phys_));
    for (int k = 0; k < K_phys_; ++k)
        scatter_states_.emplace_back(static_cast<size_t>(P_));

    elem_sizes_.resize(static_cast<size_t>(K_phys_));
    for (int k = 0; k < K_phys_; ++k)
        elem_sizes_[static_cast<size_t>(k)] = phys_prims_[static_cast<size_t>(k)].raw_elem_size;

    parts_.assign(static_cast<size_t>(P), {});
    for (auto & ps : parts_)
        ps.next_cap = init_cap;
}


/// Extract the sub-column for a physical primitive from the logical block.
static const IColumn & extractPhysCol(const DB::Columns & columns, const PhysColInfo & info)
{
    const IColumn & col = *columns[info.logical_k];
    if (info.use_null_map)
        return assert_cast<const ColumnNullable &>(col).getNullMapColumn();
    if (info.use_nested)
        return assert_cast<const ColumnNullable &>(col).getNestedColumn();
    return col;
}


template <typename TKey>
void RadixPartitionOperator<TKey>::process(const DB::Columns & columns)
{
    if (columns.empty() || columns[0]->size() == 0)
        return;
    const size_t n_total = columns[0]->size();
    for (size_t i = 0; i < n_total;)
    {
        const int n = static_cast<int>(std::min(static_cast<size_t>(batch_), n_total - i));
        runBatch(columns, i, n);
        i += static_cast<size_t>(n);
    }
}


template <typename TKey>
void RadixPartitionOperator<TKey>::runBatch(const DB::Columns & columns, size_t start, int n)
{
    uint32_t * pids = pids_.data();
    uint32_t * hist = hist_.data();

    // ── Phase 1: compute partition IDs from the first LOGICAL column ──────────
    col_prims_[0].compute_pids(col_prims_[0], *columns[0], start, n, mask_, pids);

    // ── Phase 2: histogram ────────────────────────────────────────────────
    std::memset(hist, 0, static_cast<size_t>(P_) * sizeof(uint32_t));
    for (int j = 0; j < n; ++j)
        hist[pids[j]]++;

    // ── Phase 3: pre-grow + notify PHYSICAL columns + pre-commit ─────────────
    for (int p = 0; p < P_; ++p)
    {
        if (!hist[p])
            continue;
        auto & ps = parts_[static_cast<size_t>(p)];
        if (!ps.cur || ps.cur->filled + hist[p] > ps.cur->capacity)
        {
            if (use_swwc_ && ps.cur && cnt_[static_cast<size_t>(p)])
            {
                for (int k = 0; k < K_phys_; ++k)
                    if (phys_prims_[static_cast<size_t>(k)].drain_raw)
                        phys_prims_[static_cast<size_t>(k)].drain_raw(
                            phys_prims_[static_cast<size_t>(k)],
                            static_cast<size_t>(p),
                            cnt_[static_cast<size_t>(p)],
                            scatter_states_[static_cast<size_t>(k)]);
                cnt_[static_cast<size_t>(p)] = 0;
            }
            growPart(ps, arena_, K_phys_, elem_sizes_.data(), max_cap_);
            for (int k = 0; k < K_phys_; ++k)
                phys_prims_[static_cast<size_t>(k)].on_grow_raw(
                    phys_prims_[static_cast<size_t>(k)],
                    static_cast<size_t>(p),
                    ps.cur->cols[k],
                    ps.cur->capacity,
                    scatter_states_[static_cast<size_t>(k)]);
        }
        ps.cur->filled += hist[p];
    }

    if (use_swwc_)
    {
        // ── Phase 4a: staging slots (shared across all physical columns) ──────
        constexpr uint8_t kSlotMask = static_cast<uint8_t>(64 / sizeof(TKey)) - 1;
        uint32_t * pos = pos_.data();
        uint8_t * cnt = cnt_.data();
        for (int j = 0; j < n; ++j)
        {
            const uint32_t p = pids[j];
            const uint32_t slot = cnt[p];
            pos[j] = slot;
            cnt[p] = static_cast<uint8_t>((slot + 1) & kSlotMask);
        }
        // ── Phase 4b: SWWC scatter per PHYSICAL column ────────────────────────
        for (int k = 0; k < K_phys_; ++k)
        {
            const IColumn & sub_col = extractPhysCol(columns, phys_col_info_[static_cast<size_t>(k)]);
            auto & prim = phys_prims_[static_cast<size_t>(k)];
            if (prim.scatter_raw_swwc)
                prim.scatter_raw_swwc(sub_col, start, pids, pos, n, scatter_states_[static_cast<size_t>(k)]);
            else
                prim.scatter_raw(sub_col, start, pids, n, scatter_states_[static_cast<size_t>(k)]);
        }
    }
    else
    {
        // ── Phase 4b: direct scatter per PHYSICAL column ──────────────────────
        for (int k = 0; k < K_phys_; ++k)
        {
            const IColumn & sub_col = extractPhysCol(columns, phys_col_info_[static_cast<size_t>(k)]);
            phys_prims_[static_cast<size_t>(k)].scatter_raw(
                sub_col, start, pids, n, scatter_states_[static_cast<size_t>(k)]);
        }
    }
}


template <typename TKey>
void RadixPartitionOperator<TKey>::finish()
{
    if (!use_swwc_)
        return;

#if defined(__x86_64__)
    _mm_sfence();
#endif

    for (int p = 0; p < P_; ++p)
    {
        if (!cnt_[static_cast<size_t>(p)])
            continue;
        for (int k = 0; k < K_phys_; ++k)
            if (phys_prims_[static_cast<size_t>(k)].drain_raw)
                phys_prims_[static_cast<size_t>(k)].drain_raw(
                    phys_prims_[static_cast<size_t>(k)],
                    static_cast<size_t>(p),
                    cnt_[static_cast<size_t>(p)],
                    scatter_states_[static_cast<size_t>(k)]);
        cnt_[static_cast<size_t>(p)] = 0;
    }
}


// ── explicit instantiations ───────────────────────────────────────────────────

template class RadixPartitionOperator<uint8_t>;
template class RadixPartitionOperator<uint16_t>;
template class RadixPartitionOperator<uint32_t>;
template class RadixPartitionOperator<uint64_t>;
template class RadixPartitionOperator<int8_t>;
template class RadixPartitionOperator<int16_t>;
template class RadixPartitionOperator<int32_t>;
template class RadixPartitionOperator<int64_t>;
template class RadixPartitionOperator<float>;
template class RadixPartitionOperator<double>;

} // namespace DB::RadixShuffle
