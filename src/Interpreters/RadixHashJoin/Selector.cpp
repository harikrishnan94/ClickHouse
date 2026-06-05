#include <Interpreters/RadixHashJoin/Selector.h>

#include <Columns/IColumn.h>

#include <algorithm>
#include <bit>

namespace DB::RadixHash
{

namespace
{
    /// Number of replicated histograms (round-robined per row to dodge store-to-load-forwarding
    /// stalls), kept only while `replicas * num_leaves * 4 B` fits ~L1 (32 KiB). Power of two.
    size_t chooseReplicas(size_t num_leaves)
    {
        const size_t fit = num_leaves != 0 ? (32 * 1024 / (num_leaves * sizeof(UInt32))) : 1;
        const size_t r = std::min<size_t>(4, std::max<size_t>(1, fit));
        return std::bit_floor(r);
    }
}

Selector::Selector(const PartitionConfig & cfg_)
    : cfg(cfg_)
    , replicas(chooseReplicas(cfg_.num_leaves))
    , hist(replicas * cfg_.num_leaves, 0)
{
}

void Selector::addToHistogram(const UInt16 * pid, size_t n)
{
    const size_t nl = cfg.num_leaves;
    const size_t r_mask = replicas - 1; /// replicas is a power of two
    UInt32 * h = hist.data();
    for (size_t j = 0; j < n; ++j)
        ++h[(j & r_mask) * nl + pid[j]];
}

void Selector::pidsFromHashes(const UInt32 * hash_in, size_t n, UInt16 * pid_out)
{
    const UInt32 shift = cfg.shift;
    /// 64-bit shift keeps `shift == 32` (total_bits == 0) well-defined, yielding pid == 0.
    for (size_t j = 0; j < n; ++j)
        pid_out[j] = static_cast<UInt16>(static_cast<UInt64>(hash_in[j]) >> shift);
    addToHistogram(pid_out, n);
}

void Selector::process(const IColumn & key_col, size_t n, UInt32 * hash_out, UInt16 * pid_out)
{
    key_col.computeHashInto(0, n, hash_out, /*initial=*/true);
    pidsFromHashes(hash_out, n, pid_out);
}

UInt64 Selector::mergedHistogram(std::vector<UInt32> & out) const
{
    const size_t nl = cfg.num_leaves;
    out.assign(nl, 0);
    for (size_t r = 0; r < replicas; ++r)
    {
        const UInt32 * h = hist.data() + r * nl;
        for (size_t p = 0; p < nl; ++p)
            out[p] += h[p];
    }

    UInt64 total = 0;
    for (size_t p = 0; p < nl; ++p)
        total += out[p];
    return total;
}

UInt64 mergeHistograms(
    const std::vector<std::vector<UInt32>> & per_thread_hist,
    size_t num_leaves,
    std::vector<UInt64> & global_hist,
    std::vector<UInt64> & offset)
{
    global_hist.assign(num_leaves, 0);
    for (const auto & h : per_thread_hist)
        for (size_t p = 0; p < num_leaves; ++p)
            global_hist[p] += h[p];

    offset.assign(num_leaves, 0);
    UInt64 acc = 0;
    for (size_t p = 0; p < num_leaves; ++p)
    {
        offset[p] = acc;
        acc += global_hist[p];
    }
    return acc;
}

}
