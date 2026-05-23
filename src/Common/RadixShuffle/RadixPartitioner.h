#pragma once

#include <Columns/IColumn_fwd.h>
#include <Common/RadixShuffle/Allocator.h>
#include <Common/RadixShuffle/ColumnPrimitives.h>
#include <Common/RadixShuffle/PartSchema.h>
#include <Common/RadixShuffle/PartitionTypes.h>

#include <cstddef>
#include <cstdint>
#include <vector>


namespace DB::RadixShuffle
{

struct RadixPartitionerOptions
{
    /// 0 → use the algorithm's formula max(1024, min(32768, P × 16)).
    size_t batch_size_override = 0;
    AllocatorOptions allocator_options{};
};


/// Per-thread radix-partition driver.  No cross-thread sync within a phase.
/// Wraps Allocator + Handle + ScatterState + ColumnPrimitives into the
/// 4-phase pipeline from radix_partition_algo.md (direct mode; no SWWC).
///
/// Lifecycle:
///   1. Construct with the schema, primitives, P, key column indices.
///   2. Call process() for each input block (arbitrarily large; internally
///      sliced to batch_size).
///   3. Call finish() (or let the destructor do it).
///   4. Read per-partition PartReservationView lists via bucket(p).
///      Downstream callers use prim.reconstruct() to materialise columns.
class RadixPartitioner
{
public:
    /// Per-partition output: an ordered list of PartReservationView slices
    /// accumulated across all batches.  Each view refers to memory owned by
    /// the Allocator and is valid until the RadixPartitioner is destroyed.
    struct Bucket
    {
        std::vector<PartReservationView> views;
        size_t total_rows = 0;
        size_t total_varlen_bytes = 0;
    };

    /// schema and primitives must be produced by buildSchemaAndPrimitives() so
    /// that fixed_slot_indices are consistent.
    ///
    /// key_col_idxs: indices into primitives[] whose columns are hashed to
    /// derive the partition assignment.  Multiple indices are composed by
    /// chaining prim.hash calls (hashCombine accumulation).
    RadixPartitioner(
        PartSchema schema,
        std::vector<ColumnPrimitives> primitives,
        size_t partitions,
        std::vector<size_t> key_col_idxs,
        RadixPartitionerOptions options = {});

    /// Process one input block.  columns[k] maps to primitives[k].
    /// If columns[0]->size() > batchSize(), the block is sliced internally
    /// via IColumn::cut; each slice pays a copy cost.
    void process(const DB::Columns & columns);

    /// Release the Handle so buckets can be read.  Idempotent.
    /// Bucket memory remains valid until the destructor runs.
    void finish();

    [[nodiscard]] const Bucket & bucket(size_t p) const noexcept { return buckets_[p]; }
    [[nodiscard]] size_t partitions() const noexcept { return num_parts_; }
    [[nodiscard]] size_t batchSize() const noexcept { return batch_size_; }
    [[nodiscard]] const PartSchema & schema() const noexcept { return part_schema_; }
    [[nodiscard]] const std::vector<ColumnPrimitives> & primitives() const noexcept { return prims_; }

    RadixPartitioner(const RadixPartitioner &) = delete;
    RadixPartitioner & operator=(const RadixPartitioner &) = delete;
    RadixPartitioner(RadixPartitioner &&) = delete;
    RadixPartitioner & operator=(RadixPartitioner &&) = delete;
    ~RadixPartitioner();

private:
    void processBatch(const DB::Columns & columns, size_t n);

    /// Accumulate per-partition varlen byte totals from all varlen columns.
    /// Reads pids_[] which must already be computed for this batch.
    void accumulateVarlenBytes(const DB::Columns & columns, size_t n);

    PartSchema part_schema_;
    std::vector<ColumnPrimitives> prims_;
    size_t num_parts_;
    std::vector<size_t> key_col_idxs_;
    size_t batch_size_;

    Allocator allocator_;
    Handle * handle_ = nullptr;

    /// Per-column persistent write-pointer caches (survive across batches).
    std::vector<ScatterState> scatter_states_;

    /// Per-batch scratch arrays (sized to batch_size_ at construction).
    std::vector<uint32_t> hashes_;       ///< raw hash per row before reduction
    std::vector<uint16_t> pids_;         ///< partition id per row
    std::vector<size_t> hist_;           ///< row count per partition this batch
    std::vector<size_t> varlen_per_part_; ///< varlen byte total per partition
    std::vector<PartReserveGrant> grants_; ///< allocation results from reserve
    std::vector<PartReservation> dst_;     ///< flattened slice per partition
    std::vector<uint64_t> stale_bitset_;   ///< ceil(P/64) stale-fixed-chunk words

    std::vector<Bucket> buckets_; ///< accumulated per-partition output views
};

}
