#include <Common/RadixShuffle/RadixPartitioner.h>

#include <Columns/ColumnNullable.h>
#include <Columns/ColumnString.h>
#include <Columns/IColumn.h>
#include <Common/Exception.h>

#include <algorithm>
#include <stdexcept>


namespace DB
{

namespace ErrorCodes
{
extern const int BAD_ARGUMENTS;
extern const int LOGICAL_ERROR;
}

}


namespace DB::RadixShuffle
{

namespace
{

size_t computeBatchSize(size_t P, size_t override_val) noexcept
{
    if (override_val > 0)
        return override_val;
    return std::max<size_t>(1024, std::min<size_t>(32768, P * 16));
}

} // namespace


RadixPartitioner::RadixPartitioner(
    PartSchema schema,
    std::vector<ColumnPrimitives> primitives,
    size_t partitions,
    std::vector<size_t> key_col_idxs,
    RadixPartitionerOptions options)
    : part_schema_(std::move(schema))
    , prims_(std::move(primitives))
    , num_parts_(partitions)
    , key_col_idxs_(std::move(key_col_idxs))
    , batch_size_(computeBatchSize(partitions, options.batch_size_override))
    , allocator_(part_schema_, partitions, 0, options.allocator_options)
    , handle_(nullptr)
{
    if (partitions == 0)
        throw Exception(DB::ErrorCodes::BAD_ARGUMENTS, "RadixPartitioner: partitions must be > 0");
    if (key_col_idxs_.empty())
        throw Exception(DB::ErrorCodes::BAD_ARGUMENTS, "RadixPartitioner: key_col_idxs must be non-empty");
    for (size_t idx : key_col_idxs_)
        if (idx >= prims_.size())
            throw Exception(
                DB::ErrorCodes::BAD_ARGUMENTS, "RadixPartitioner: key_col_idx {} out of range (prims size {})", idx, prims_.size());

    handle_ = allocator_.acquire();

    // Scratch arrays — batch_size_ rows / P partitions.
    hashes_.resize(batch_size_);
    pids_.resize(batch_size_);
    hist_.resize(num_parts_);
    varlen_per_part_.resize(num_parts_);
    grants_.resize(num_parts_);
    dst_.resize(num_parts_);
    stale_bitset_.resize((num_parts_ + 63) / 64);
    buckets_.resize(num_parts_);

    // One ScatterState per column.
    scatter_states_.reserve(prims_.size());
    for (size_t k = 0; k < prims_.size(); ++k)
        scatter_states_.emplace_back(num_parts_);
}


RadixPartitioner::~RadixPartitioner()
{
    finish();
}


void RadixPartitioner::finish()
{
    if (handle_ != nullptr)
    {
        allocator_.release(handle_);
        handle_ = nullptr;
    }
}


void RadixPartitioner::process(const DB::Columns & columns)
{
    if (columns.empty())
        return;
    if (columns.size() != prims_.size())
        throw Exception(
            DB::ErrorCodes::BAD_ARGUMENTS,
            "RadixPartitioner::process: columns size ({}) != primitives size ({})",
            columns.size(),
            prims_.size());
    if (handle_ == nullptr)
        throw Exception(DB::ErrorCodes::LOGICAL_ERROR, "RadixPartitioner::process called after finish()");

    const size_t n_total = columns[0]->size();
    if (n_total == 0)
        return;

    if (n_total <= batch_size_)
    {
        processBatch(columns, n_total);
        return;
    }

    // Slice into batch_size_ chunks using IColumn::cut.
    // Each cut allocates a copy; this is the cost of processing large blocks.
    for (size_t off = 0; off < n_total; off += batch_size_)
    {
        const size_t n = std::min(batch_size_, n_total - off);
        DB::Columns slice(columns.size());
        for (size_t k = 0; k < columns.size(); ++k)
            slice[k] = columns[k]->cut(off, n);
        processBatch(slice, n);
    }
}


void RadixPartitioner::processBatch(const DB::Columns & columns, size_t n)
{
    // ───── Phase 1: hash chain → Lemire's fast modulo → pids ─────

    std::fill_n(hashes_.data(), n, uint32_t{0});
    for (const size_t k_idx : key_col_idxs_)
        prims_[k_idx].hash(prims_[k_idx], part_schema_, *columns[k_idx], n, hashes_.data());

    // Lemire's fast divisor: (hash × P) >> 32 maps [0, 2^32) → [0, P)
    // uniformly without a % operation.  Valid for any P ≤ 2^16 ≤ 2^32.
    for (size_t i = 0; i < n; ++i)
        pids_[i] = static_cast<uint16_t>((static_cast<uint64_t>(hashes_[i]) * static_cast<uint64_t>(num_parts_)) >> 32);

    // ───── Phase 2: per-partition histogram ─────

    std::fill(hist_.begin(), hist_.end(), size_t{0});
    for (size_t j = 0; j < n; ++j)
        ++hist_[pids_[j]];

    // ───── Phase 2.5: per-partition varlen byte totals ─────

    if (part_schema_.has_varlen_portion)
        accumulateVarlenBytes(columns, n);
    else
        std::fill(varlen_per_part_.begin(), varlen_per_part_.end(), size_t{0});

    // ───── Phase 3: reserve (pre-grow + commit) ─────

    std::fill(stale_bitset_.begin(), stale_bitset_.end(), uint64_t{0});
    handle_->reserve(hist_.data(), varlen_per_part_.data(), grants_.data(), stale_bitset_.data());

    for (size_t p = 0; p < num_parts_; ++p)
        dst_[p] = grants_[p].slice;

    // ───── Phase 4: scatter (one call per column, direct mode) ─────

    for (size_t k = 0; k < prims_.size(); ++k)
        prims_[k].scatter(
            prims_[k], part_schema_, *columns[k], pids_.data(), n, num_parts_, dst_.data(), scatter_states_[k], stale_bitset_.data());

    // ───── Bookkeeping: record PartReservationView per active partition ─────

    for (size_t p = 0; p < num_parts_; ++p)
    {
        if (dst_[p].reserved_rows == 0)
            continue;
        buckets_[p].views.push_back(
            PartReservationView{
                dst_[p].fixed,
                dst_[p].begin_row,
                dst_[p].begin_row + dst_[p].reserved_rows,
                dst_[p].data,
                dst_[p].begin_byte,
                dst_[p].begin_byte + dst_[p].reserved_bytes,
            });
        buckets_[p].total_rows += dst_[p].reserved_rows;
        buckets_[p].total_varlen_bytes += dst_[p].reserved_bytes;
    }
}


void RadixPartitioner::accumulateVarlenBytes(const DB::Columns & columns, size_t n)
{
    std::fill(varlen_per_part_.begin(), varlen_per_part_.end(), size_t{0});

    for (size_t k = 0; k < prims_.size(); ++k)
    {
        if (!prims_[k].writes_varlen)
            continue;

        const IColumn * col = columns[k].get();

        // Unwrap ColumnNullable — the varlen data lives in the nested column.
        if (const auto * nullable = typeid_cast<const ColumnNullable *>(col))
            col = &nullable->getNestedColumn();

        if (const auto * str_col = typeid_cast<const ColumnString *>(col))
        {
            const auto & offsets = str_col->getOffsets();
            UInt64 prev = 0;
            for (size_t j = 0; j < n; ++j)
            {
                const UInt64 end = offsets[j];
                varlen_per_part_[pids_[j]] += static_cast<size_t>(end - prev);
                prev = end;
            }
        }
    }
}

}
