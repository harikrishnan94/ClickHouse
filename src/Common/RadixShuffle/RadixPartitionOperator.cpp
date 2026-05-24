#include <Common/RadixShuffle/RadixPartitionOperator.h>

#include <Columns/ColumnVector.h>
#include <Common/RadixShuffle/HashKernels.h>
#include <Common/assert_cast.h>

#if defined(__x86_64__)
#    include <immintrin.h>
#endif

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <utility>


namespace DB::RadixShuffle
{

namespace
{

template <typename TKey>
PartSchema buildSchemaForType(int K)
{
    PartSchema schema;
    size_t off = 0;
    for (int k = 0; k < K; ++k)
    {
        schema.fixed_slots.push_back({static_cast<size_t>(k), SlotRole::Values, sizeof(TKey), sizeof(TKey)});
        schema.slot_byte_offset.push_back(off);
        off += sizeof(TKey);
    }
    schema.fixed_bytes_per_row = off;
    schema.has_varlen_portion = false;
    return schema;
}

} // namespace

// ── RadixPartitionOperator implementation ────────────────────────────────────

template <typename TKey>
RadixPartitionOperator<TKey>::RadixPartitionOperator(int P, int K, std::vector<IScatterColumn *> cols, bool use_swwc)
    : P_(P)
    , K_(K)
    , use_swwc_(use_swwc)
    , batch_(std::max(1024, std::min(kSmartMaxBatch, P * kBatchFactor)))
    , mask_(static_cast<uint32_t>(P) - 1)
    , allocator_(buildSchemaForType<TKey>(K), static_cast<size_t>(P), 0)
    , cols_(std::move(cols))
    , pids_(static_cast<size_t>(batch_))
    , hist_(static_cast<size_t>(P), 0)
    , size_hist_(static_cast<size_t>(P), 0)
    , varlen_zeros_(static_cast<size_t>(P), 0)
    , grants_(static_cast<size_t>(P))
    , stale_bitset_((static_cast<size_t>(P) + 63) / 64, 0)
    , pos_(static_cast<size_t>(batch_))
    , cnt_(static_cast<size_t>(P), 0)
{
    handle_ = allocator_.acquire();
}


template <typename TKey>
RadixPartitionOperator<TKey>::~RadixPartitionOperator()
{
    finish();
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

    // ── Phase 1: hash key column → partition IDs ──────────────────────────
    const TKey * key_data = assert_cast<const ColumnVector<TKey> &>(*columns[0]).getData().data();
    for (int j = 0; j < n; ++j)
        pids[j] = hashOne32(key_data[start + j]) & mask_;

    // ── Phase 2: histogram ────────────────────────────────────────────────
    std::memset(hist, 0, static_cast<size_t>(P_) * sizeof(uint32_t));
    for (int j = 0; j < n; ++j)
        hist[pids[j]]++;

    // ── Phase 3: reserve (pre-grow + pre-commit) + notify stale columns ────
    for (int p = 0; p < P_; ++p)
        size_hist_[static_cast<size_t>(p)] = static_cast<size_t>(hist[p]);

    std::fill(stale_bitset_.begin(), stale_bitset_.end(), uint64_t{0});
    handle_->reserve(size_hist_.data(), varlen_zeros_.data(), grants_.data(), stale_bitset_.data());

    for (size_t word = 0; word < stale_bitset_.size(); ++word)
    {
        uint64_t bits = stale_bitset_[word];
        while (bits)
        {
            const size_t bit = static_cast<size_t>(__builtin_ctzll(bits));
            const size_t p = word * 64 + bit;
            if (p < static_cast<size_t>(P_))
            {
                // Drain staged rows into the old chunk before redirecting
                // the column write pointers to the newly allocated chunk.
                if (use_swwc_ && cnt_[p])
                {
                    for (auto * c : cols_)
                        c->drain_one(p, cnt_[p]);
                    cnt_[p] = 0;
                }

                const PartReservation & slice = grants_[p].slice;
                for (int k = 0; k < K_; ++k)
                {
                    void * col_base = static_cast<char *>(slice.fixed->data) + slice.fixed->slot_byte_offsets[static_cast<size_t>(k)];
                    cols_[static_cast<size_t>(k)]->on_grow(p, col_base);
                }
            }
            bits &= bits - 1;
        }
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
            const TKey * col_data = assert_cast<const ColumnVector<TKey> &>(*columns[static_cast<size_t>(k)]).getData().data();
            cols_[static_cast<size_t>(k)]->scatter_staged(pids, pos, col_data + start, n);
        }
    }
    else
    {
        // ── Phase 4b: direct scatter per column ───────────────────────────
        for (int k = 0; k < K_; ++k)
        {
            const TKey * col_data = assert_cast<const ColumnVector<TKey> &>(*columns[static_cast<size_t>(k)]).getData().data();
            cols_[static_cast<size_t>(k)]->scatter_direct(pids, col_data + start, n);
        }
    }
}


template <typename TKey>
void RadixPartitionOperator<TKey>::finish()
{
    if (!handle_)
        return;

    if (use_swwc_)
    {
#if defined(__x86_64__)
        _mm_sfence();
#endif

        for (int p = 0; p < P_; ++p)
        {
            if (!cnt_[static_cast<size_t>(p)])
                continue;
            for (auto * c : cols_)
                c->drain_one(static_cast<size_t>(p), cnt_[static_cast<size_t>(p)]);
            cnt_[static_cast<size_t>(p)] = 0;
        }
    }

    allocator_.release(handle_);
    handle_ = nullptr;
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
