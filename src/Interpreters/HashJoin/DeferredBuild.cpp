#include <Interpreters/HashJoin/DeferredBuild.h>

#include <algorithm>
#include <cmath>
#include <memory>

namespace DB
{

DeferredBuildStreamCounts::DeferredBuildStreamCounts(size_t num_clauses, size_t num_buckets)
{
    for (size_t clause = 0; clause < num_clauses; ++clause)
        clauses.emplace_back(num_buckets);
}

DeferredBuildState::DeferredBuildState(size_t num_streams, size_t num_clauses_, size_t num_buckets_)
    : num_clauses(num_clauses_)
    , num_buckets(num_buckets_)
    , streams(num_streams)
{
}

DeferredBuildState::~DeferredBuildState()
{
    for (auto & stream : streams)
        delete stream.load(std::memory_order_acquire);
}

DeferredBuildStreamCounts & DeferredBuildState::streamCounts(size_t stream)
{
    if (auto * counts = streams[stream].load(std::memory_order_relaxed))
        return *counts;
    auto counts = std::make_unique<DeferredBuildStreamCounts>(num_clauses, num_buckets);
    streams[stream].store(counts.get(), std::memory_order_release);
    return *counts.release();
}

DeferredBuildState::ClauseTotals DeferredBuildState::totals(size_t clause) const
{
    ClauseTotals result;
    result.bucket_rows.resize(num_buckets);
    DeferredBuildSketch merged;
    for (const auto & stream : streams)
    {
        const auto * counts = stream.load(std::memory_order_acquire);
        if (!counts)
            continue;
        const auto & clause_counts = counts->clauses[clause];
        merged.merge(clause_counts.sketch);
        for (size_t bucket = 0; bucket < num_buckets; ++bucket)
            result.bucket_rows[bucket] += clause_counts.bucket_rows[bucket];
    }
    for (const UInt64 bucket_rows : result.bucket_rows)
        result.rows += bucket_rows;
    /// The sketch counts distinct hashes, and never more keys than there are rows.
    result.estimate = std::min<size_t>(merged.size(), result.rows);
    return result;
}

std::vector<size_t> deferredBuildBucketReserves(size_t estimate, const std::vector<UInt64> & bucket_rows, size_t rows)
{
    std::vector<size_t> reserves(bucket_rows.size());
    if (rows == 0)
        return reserves;

    const size_t estimate_with_headroom = estimate + estimate / 10;
    const size_t target = std::min(estimate_with_headroom, rows);
    if (bucket_rows.size() == 1)
    {
        reserves[0] = target;
        return reserves;
    }

    const double even_share = static_cast<double>(target) / static_cast<double>(bucket_rows.size());
    /// Four standard deviations of the number of keys a bucket gets when the keys spread evenly.
    const auto with_spread = static_cast<size_t>(std::ceil(even_share + 4 * std::sqrt(even_share)));
    const auto max_row_share = static_cast<size_t>(4 * even_share);
    for (size_t bucket = 0; bucket < bucket_rows.size(); ++bucket)
    {
        const auto row_share = static_cast<size_t>(std::ceil(
            static_cast<double>(estimate_with_headroom) * static_cast<double>(bucket_rows[bucket]) / static_cast<double>(rows)));
        reserves[bucket] = std::min<size_t>(bucket_rows[bucket], std::max(with_spread, std::min(row_share, max_row_share)));
    }
    return reserves;
}

}
