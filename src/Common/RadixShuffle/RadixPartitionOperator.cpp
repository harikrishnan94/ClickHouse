#include <Common/RadixShuffle/RadixPartitionOperator.h>

#include <Columns/IColumn.h>
#include <Common/RadixShuffle/ColumnPrimitives.h>

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
    // Construct one ScatterState per column, each sized to P partitions.
    scatter_states_.reserve(static_cast<size_t>(K_));
    for (int k = 0; k < K_; ++k)
        scatter_states_.emplace_back(static_cast<size_t>(P_));

    parts_.assign(static_cast<size_t>(P), {});
    for (auto & ps : parts_)
        ps.next_cap = init_cap;
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

    // ── Phase 1: compute partition IDs in one SIMD pass ──────────────────────
    // compute_pids does pids[j] = hash(keys[j]) & mask_ — same as the baseline
    // hashBatch32 call but dispatched through a function pointer.
    col_prims_[0].compute_pids(col_prims_[0], *columns[0], start, n, mask_, pids);

    // ── Phase 2: histogram ────────────────────────────────────────────────
    std::memset(hist, 0, static_cast<size_t>(P_) * sizeof(uint32_t));
    for (int j = 0; j < n; ++j)
        hist[pids[j]]++;

    // ── Phase 3: pre-grow + notify columns + pre-commit ───────────────────
    for (int p = 0; p < P_; ++p)
    {
        if (!hist[p])
            continue;
        auto & ps = parts_[static_cast<size_t>(p)];
        if (!ps.cur || ps.cur->filled + hist[p] > ps.cur->capacity)
        {
            // Drain any staged rows into the current block before growing.
            if (use_swwc_ && ps.cur && cnt_[static_cast<size_t>(p)])
            {
                for (int k = 0; k < K_; ++k)
                    col_prims_[static_cast<size_t>(k)].drain_raw(
                        static_cast<size_t>(p), cnt_[static_cast<size_t>(p)], scatter_states_[static_cast<size_t>(k)]);
                cnt_[static_cast<size_t>(p)] = 0;
            }
            growPart(ps, arena_, K_, sizeof(TKey), max_cap_);
            for (int k = 0; k < K_; ++k)
                col_prims_[static_cast<size_t>(k)].on_grow_raw(
                    static_cast<size_t>(p), ps.cur->cols[k], scatter_states_[static_cast<size_t>(k)]);
        }
        ps.cur->filled += hist[p]; // pre-commit (thread-private, safe)
    }

    if (use_swwc_)
    {
        // ── Phase 4a: compute staging slots (shared across columns) ───────
        uint32_t * pos = pos_.data();
        uint8_t * cnt = cnt_.data();
        for (int j = 0; j < n; ++j)
        {
            const uint32_t p = pids[j];
            const uint32_t slot = cnt[p];
            pos[j] = slot;
            cnt[p] = static_cast<uint8_t>((slot + 1) & 7);
        }
        // ── Phase 4b: SWWC scatter per column ─────────────────────────────
        for (int k = 0; k < K_; ++k)
        {
            col_prims_[static_cast<size_t>(k)].scatter_raw_swwc(
                *columns[static_cast<size_t>(k)], start, pids, pos, n, scatter_states_[static_cast<size_t>(k)]);
        }
    }
    else
    {
        // ── Phase 4b: direct scatter per column ───────────────────────────
        for (int k = 0; k < K_; ++k)
        {
            col_prims_[static_cast<size_t>(k)].scatter_raw(
                *columns[static_cast<size_t>(k)], start, pids, n, scatter_states_[static_cast<size_t>(k)]);
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
        for (int k = 0; k < K_; ++k)
            col_prims_[static_cast<size_t>(k)].drain_raw(
                static_cast<size_t>(p), cnt_[static_cast<size_t>(p)], scatter_states_[static_cast<size_t>(k)]);
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
