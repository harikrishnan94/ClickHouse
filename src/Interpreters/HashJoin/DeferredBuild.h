#pragma once

#include <Common/HyperLogLogCounter.h>

#include <atomic>
#include <deque>
#include <vector>

namespace DB
{

/// The finalizer of MurmurHash3, a bijection of 32-bit words. It mixes the low 32 bits of the hash of the map before
/// the sketch splits them into an index and a rank.
struct JoinKeyHashMix
{
    UInt32 operator()(UInt32 x) const
    {
        x ^= x >> 16;
        x *= 0x85ebca6b;
        x ^= x >> 13;
        x *= 0xc2b2ae35;
        x ^= x >> 16;
        return x;
    }
};

/// 8 KiB, standard error about 1.15%.
using DeferredBuildSketch = HyperLogLogCounter<
    13,
    UInt32,
    JoinKeyHashMix,
    UInt32,
    double,
    TrivialBiasEstimator,
    HyperLogLogMode::FullFeatured,
    DenominatorMode::StableIfBig,
    HyperLogLogRegisters::Bytes,
    /*track_changes=*/true>;

/// What one build stream of a deferred build counted while it stored its blocks, for each ON clause: the keys of
/// the rows that the insert will take (non-NULL and passing the ON condition), and their number per bucket of the map.
struct DeferredBuildStreamCounts
{
    struct Clause
    {
        explicit Clause(size_t num_buckets) : bucket_rows(num_buckets) {}

        DeferredBuildSketch sketch;
        std::vector<UInt64> bucket_rows;
    };

    DeferredBuildStreamCounts(size_t num_clauses, size_t num_buckets);

    /// A deque, because the sketch cannot move.
    std::deque<Clause> clauses;
};

/// The counts of every build stream of a deferred build. Each stream creates its counts at its first block and is
/// their only writer; the counts live until the join is destroyed.
class DeferredBuildState
{
public:
    DeferredBuildState(size_t num_streams, size_t num_clauses, size_t num_buckets);
    ~DeferredBuildState();

    /// Called by stream `stream` only.
    DeferredBuildStreamCounts & streamCounts(size_t stream);

    /// Once every stream has finished: the estimated number of distinct keys of clause `clause`, and the rows per bucket.
    struct ClauseTotals
    {
        size_t estimate = 0;
        std::vector<UInt64> bucket_rows;
        size_t rows = 0;
    };
    ClauseTotals totals(size_t clause) const;

private:
    const size_t num_clauses;
    const size_t num_buckets;
    std::vector<std::atomic<DeferredBuildStreamCounts *>> streams;
};

/// The number of keys to reserve in each bucket of a map, from the estimate of distinct keys and the rows per bucket.
/// The estimate gets 10% of head-room, and a bucket gets the even share of the reserve with room for its spread, or
/// its share of the rows when that is larger, up to four even shares. No bucket gets more than its rows.
std::vector<size_t> deferredBuildBucketReserves(size_t estimate, const std::vector<UInt64> & bucket_rows, size_t rows);

}
