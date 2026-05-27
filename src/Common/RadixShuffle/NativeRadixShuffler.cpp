#include <Common/RadixShuffle/NativeRadixShuffler.h>

#include <Columns/IColumn.h>
#include <Interpreters/JoinUtils.h>
#include <Common/WeakHash.h>

#include <chrono>
#include <cstddef>


namespace
{
inline uint64_t elapsedNs(std::chrono::steady_clock::time_point t0) noexcept
{
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count());
}
} // namespace


namespace DB
{

NativeRadixShuffler::NativeRadixShuffler(int num_partitions, int num_columns)
    : num_partitions_(num_partitions)
    , num_columns_(num_columns)
    , output_(static_cast<size_t>(num_partitions))
{
}

void NativeRadixShuffler::process(const DB::Columns & columns)
{
    if (columns.empty() || columns[0]->size() == 0)
        return;

    const auto proc_t0 = std::chrono::steady_clock::now();
    const size_t n = columns[0]->size();

    // Phase 1: compute WeakHash32 over all K key columns.
    // Mirrors BufferedShardByHashTransform::generateOutputChunks exactly:
    //   hash starts from the first column's weak hash, then updates with each
    //   subsequent column — matching the multi-key hash accumulation in the PR.
    {
        const auto t = std::chrono::steady_clock::now();

        WeakHash32 hash = columns[0]->getWeakHash32();
        for (int k = 1; k < num_columns_; ++k)
            hash.update(columns[static_cast<size_t>(k)]->getWeakHash32());

        // Phase 2: build IColumn::Selector via Lemire fastrange.
        // Matches JoinCommon::hashToSelector with the Lemire lambda from the PR:
        //   selector[i] = ((intHashCRC32(hash[i]) & 0xFFFFFFFF) * num_shards) >> 32
        const size_t P = static_cast<size_t>(num_partitions_);
        scratch_selector_ = JoinCommon::hashToSelector(hash, [P](size_t h) -> size_t { return ((h & 0xFFFFFFFF) * P) >> 32; });

        timings_.hash_ns += elapsedNs(t);
    }

    // Phase 3: IColumn::scatter — scatter all K columns in one pass per column.
    {
        const auto t = std::chrono::steady_clock::now();

        std::vector<DB::Columns> per_partition(static_cast<size_t>(num_partitions_));
        for (int k = 0; k < num_columns_; ++k)
        {
            auto split = columns[static_cast<size_t>(k)]->scatter(static_cast<size_t>(num_partitions_), scratch_selector_);
            for (int p = 0; p < num_partitions_; ++p)
                per_partition[static_cast<size_t>(p)].push_back(std::move(split[static_cast<size_t>(p)]));
        }

        for (int p = 0; p < num_partitions_; ++p)
        {
            const auto & part_cols = per_partition[static_cast<size_t>(p)];
            if (!part_cols.empty() && part_cols[0]->size() != 0)
                output_[static_cast<size_t>(p)].push_back(std::move(per_partition[static_cast<size_t>(p)]));
        }

        timings_.scatter_ns += elapsedNs(t);
    }

    timings_.total_process_ns += elapsedNs(proc_t0);
    timings_.rows_processed += n;
    ++timings_.blocks_processed;
}

} // namespace DB
