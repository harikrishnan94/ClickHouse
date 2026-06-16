#include <Interpreters/RadixHashJoin/BuildSide.h>

#include <Interpreters/RadixHashJoin/Hll.h>
#include <Interpreters/RadixHashJoin/PackedKeyHash.h>

#include <Columns/IColumn.h>

#include <Common/Exception.h>

#include <algorithm>
#include <bit>
#include <numeric>
#include <optional>
#include <span>

namespace DB
{
namespace ErrorCodes
{
extern const int LOGICAL_ERROR;
}
}

namespace DB::RadixJoin
{

namespace
{

/// Replicate the histogram enough to break store-to-load-forwarding stalls on consecutive increments,
/// but only while all replicas still fit a small L1 budget (else a single histogram is used and the
/// count pass is naturally L2-bound, as in any radix partitioner).
size_t chooseReplicas(size_t num_leaves)
{
    const size_t fit = num_leaves != 0 ? (32 * 1024 / (num_leaves * sizeof(UInt32))) : 1;
    return std::bit_floor(std::min<size_t>(4, std::max<size_t>(1, fit)));
}

/// Route words (high 32 bits of the packed-key hash) for `n` packed keys at stride `width`. The width
/// switch is hoisted out of the per-row loop, so the inner loop runs the unrolled compile-time hash.
template <size_t width>
void computeRoutesFixed(const char * keys, size_t n, UInt32 * out)
{
    for (size_t i = 0; i < n; ++i)
        out[i] = routeBits(hashPackedKey<width>(keys + i * width));
}

void computeRoutes(const char * keys, size_t n, size_t width, UInt32 * out)
{
    switch (width)
    {
        case 4:  computeRoutesFixed<4>(keys, n, out); return;
        case 8:  computeRoutesFixed<8>(keys, n, out); return;
        case 12: computeRoutesFixed<12>(keys, n, out); return;
        case 16: computeRoutesFixed<16>(keys, n, out); return;
        case 20: computeRoutesFixed<20>(keys, n, out); return;
        case 24: computeRoutesFixed<24>(keys, n, out); return;
        case 28: computeRoutesFixed<28>(keys, n, out); return;
        case 32: computeRoutesFixed<32>(keys, n, out); return;
        case 36: computeRoutesFixed<36>(keys, n, out); return;
        case 40: computeRoutesFixed<40>(keys, n, out); return;
        case 44: computeRoutesFixed<44>(keys, n, out); return;
        case 48: computeRoutesFixed<48>(keys, n, out); return;
        case 52: computeRoutesFixed<52>(keys, n, out); return;
        case 56: computeRoutesFixed<56>(keys, n, out); return;
        case 60: computeRoutesFixed<60>(keys, n, out); return;
        case 64: computeRoutesFixed<64>(keys, n, out); return;
        default: chassert(false && "unsupported packed key width"); return;
    }
}

/// Route word (high 32 bits) AND bucket word (low 32 bits) of the same packed-key hash, for `n` keys. Used
/// on the final scatter pass when distinct-estimate sizing is on: the route drives the leaf scatter and the
/// bucket (independent of leaf selection) feeds the per-leaf HLL — both from a single hash, no recompute.
template <size_t width>
void computeRoutesAndBucketsFixed(const char * keys, size_t n, UInt32 * route_out, UInt32 * bucket_out)
{
    for (size_t i = 0; i < n; ++i)
    {
        const HashT h = hashPackedKey<width>(keys + i * width);
        route_out[i] = routeBits(h);
        bucket_out[i] = bucketBits(h);
    }
}

void computeRoutesAndBuckets(const char * keys, size_t n, size_t width, UInt32 * route_out, UInt32 * bucket_out)
{
    switch (width)
    {
        case 4:  computeRoutesAndBucketsFixed<4>(keys, n, route_out, bucket_out);  return;
        case 8:  computeRoutesAndBucketsFixed<8>(keys, n, route_out, bucket_out);  return;
        case 12: computeRoutesAndBucketsFixed<12>(keys, n, route_out, bucket_out); return;
        case 16: computeRoutesAndBucketsFixed<16>(keys, n, route_out, bucket_out); return;
        case 20: computeRoutesAndBucketsFixed<20>(keys, n, route_out, bucket_out); return;
        case 24: computeRoutesAndBucketsFixed<24>(keys, n, route_out, bucket_out); return;
        case 28: computeRoutesAndBucketsFixed<28>(keys, n, route_out, bucket_out); return;
        case 32: computeRoutesAndBucketsFixed<32>(keys, n, route_out, bucket_out); return;
        case 36: computeRoutesAndBucketsFixed<36>(keys, n, route_out, bucket_out); return;
        case 40: computeRoutesAndBucketsFixed<40>(keys, n, route_out, bucket_out); return;
        case 44: computeRoutesAndBucketsFixed<44>(keys, n, route_out, bucket_out); return;
        case 48: computeRoutesAndBucketsFixed<48>(keys, n, route_out, bucket_out); return;
        case 52: computeRoutesAndBucketsFixed<52>(keys, n, route_out, bucket_out); return;
        case 56: computeRoutesAndBucketsFixed<56>(keys, n, route_out, bucket_out); return;
        case 60: computeRoutesAndBucketsFixed<60>(keys, n, route_out, bucket_out); return;
        case 64: computeRoutesAndBucketsFixed<64>(keys, n, route_out, bucket_out); return;
        default: chassert(false && "unsupported packed key width"); return;
    }
}

/// One contiguous, 64-byte-aligned [key | ref] allocation per non-empty partition (each section
/// line-padded so the SWWC drain can write a final partial line in-bounds). This is the ONLY place
/// the build allocates per-partition output, so the post-build allocation count is O(num_parts) — the
/// no-churn property the gates check.
struct PartitionArrays
{
    std::vector<void *> key;
    std::vector<BuildRef *> ref;
    UInt64 alloc_count = 0;
};

PartitionArrays allocExactPartitions(Arena & arena, std::span<const UInt64> counts, size_t key_width, const ParallelFor * parallel_for)
{
    const size_t num_parts = counts.size();
    PartitionArrays out;
    out.key.assign(num_parts, nullptr);
    out.ref.assign(num_parts, nullptr);

    auto carve = [&](size_t part)
    {
        if (counts[part] == 0)
            return;
        const size_t key_bytes = roundUpToLine(counts[part] * key_width);
        const size_t ref_bytes = roundUpToLine(counts[part] * sizeof(BuildRef));
        char * base = static_cast<char *>(arena.allocate(key_bytes + ref_bytes, LINE_BYTES));
        out.key[part] = base;
        out.ref[part] = reinterpret_cast<BuildRef *>(base + key_bytes); /// NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)
    };

    if (parallel_for != nullptr)
        (*parallel_for)(num_parts, [&](size_t part, size_t /*worker*/) { carve(part); });
    else
        for (size_t part = 0; part < num_parts; ++part)
            carve(part);

    for (size_t part = 0; part < num_parts; ++part)
        out.alloc_count += (counts[part] != 0);
    return out;
}

}

/// Per-worker scratch for the depth-first multi-pass refinement (reused across the whole subtree).
struct BuildSide::RefineScratch
{
    explicit RefineScratch(size_t max_fanout)
        : key_scratch(max_fanout), ref_scratch(max_fanout), key_cursors(max_fanout), ref_cursors(max_fanout)
    {
    }
    ScatterScratch key_scratch;
    ScatterScratch ref_scratch;
    std::vector<void *> key_cursors;
    std::vector<BuildRef *> ref_cursors;
    std::vector<UInt32> route;
    std::vector<UInt32> bucket; /// low-32 hash per row, only filled on the final pass when HLL is on
};

/// Per-worker × per-leaf HLL partial sketches for the scatter, in one flat zero-initialised byte array
/// (worker `w`'s sketch for leaf `L` is the `m`-byte span at `(w * num_leaves + L) * m`). Each worker is a
/// single writer of its own slice during the scatter (the dense worker-id invariant), so accumulation is
/// lock-free; the slices are register-wise-max merged per leaf after the scatter.
struct BuildSide::HllScatterState
{
    HllScatterState(size_t num_workers_, size_t num_leaves_)
        : num_workers(num_workers_)
        , num_leaves(num_leaves_)
        , precision(Hll::choosePrecision(num_workers_, num_leaves_))
        , m(Hll::numRegisters(precision))
        , registers(num_workers_ * num_leaves_ * m, 0)
    {
    }

    UInt8 * sketch(size_t worker, size_t leaf) noexcept { return registers.data() + (worker * num_leaves + leaf) * m; }
    const UInt8 * sketch(size_t worker, size_t leaf) const noexcept { return registers.data() + (worker * num_leaves + leaf) * m; }

    size_t num_workers;
    size_t num_leaves;
    UInt8 precision;
    UInt32 m;
    std::vector<UInt8> registers;
};

BuildSide::LocalState::LocalState(size_t num_leaves)
    : replicas(chooseReplicas(num_leaves))
    , rep_hist(replicas * num_leaves, 0)
{
}

BuildSide::BuildSide(PartitionPlan plan_, std::vector<size_t> key_positions_, std::vector<size_t> key_widths_, size_t max_threads_)
    : part_plan(std::move(plan_))
    , key_positions(std::move(key_positions_))
    , key_widths(std::move(key_widths_))
    , max_threads(std::max<size_t>(max_threads_, 1))
{
    chassert(!key_positions.empty() && key_positions.size() == key_widths.size());

    key_offsets.resize(key_widths.size());
    key_packers.resize(key_widths.size());
    size_t acc = 0;
    for (size_t col = 0; col < key_widths.size(); ++col)
    {
        key_offsets[col] = acc;
        key_packers[col] = chooseColumnPacker(key_widths[col]);
        acc += key_widths[col];
    }
    key_width = acc;

    local.reserve(max_threads);
    for (size_t slot = 0; slot < max_threads; ++slot)
        local.push_back(std::make_unique<LocalState>(part_plan.num_leaves));
}

BuildSide::~BuildSide() = default;

void BuildSide::packKeyChunk(const Block & block, size_t row_begin, size_t rows, char * dst) const
{
    for (size_t col = 0; col < key_positions.size(); ++col)
    {
        const char * column_data = block.getByPosition(key_positions[col]).column->getRawData().data();
        key_packers[col](column_data, row_begin, rows, dst, key_width, key_offsets[col], key_widths[col]);
    }
}

void BuildSide::add(const Block & block, size_t lane)
{
    if (lane >= local.size())
        throw Exception(
            ErrorCodes::LOGICAL_ERROR,
            "RadixHashJoin BuildSide: build lane {} exceeds max_threads {}",
            lane,
            local.size());
    LocalState & state = *local[lane];

    /// (1) Zero copy: a COW shared_ptr move, no column data copied.
    Block kept = block;
    const size_t n = kept.rows();
    chassert(n <= std::numeric_limits<UInt32>::max()); /// row_no is a 0-based UInt32 (DB::BuildRef::row_no)

    if (n > 0)
    {
        /// (2) Route word per row, into a reused scratch buffer (NOT stored per row). Single-column
        /// keys hash the column's raw data directly; multi-column keys are packed a chunk at a time.
        state.route_scratch.resize(n);
        UInt32 * route = state.route_scratch.data();
        if (key_positions.size() == 1)
        {
            const char * raw = kept.getByPosition(key_positions[0]).column->getRawData().data();
            computeRoutes(raw, n, key_width, route);
        }
        else
        {
            state.pack_scratch.resize(SCATTER_CHUNK_ROWS * key_width);
            char * packed = state.pack_scratch.data();
            for (size_t begin = 0; begin < n; begin += SCATTER_CHUNK_ROWS)
            {
                const size_t chunk = std::min(SCATTER_CHUNK_ROWS, n - begin);
                packKeyChunk(kept, begin, chunk, packed);
                computeRoutes(packed, chunk, key_width, route + begin);
            }
        }

        /// (3) Accumulate into the per-worker replicated histogram (kept across all the worker's blocks).
        const size_t num_leaves = part_plan.num_leaves;
        const UInt32 safe_shift = part_plan.total_bits > 0 ? part_plan.leaf_shift : 0u; /// avoid a 32-bit shift when num_leaves==1
        const UInt32 leaf_mask = static_cast<UInt32>(num_leaves - 1);
        const size_t replica_mask = state.replicas - 1;
        UInt32 * hist = state.rep_hist.data();
        for (size_t row = 0; row < n; ++row)
            ++hist[(row & replica_mask) * num_leaves + ((route[row] >> safe_shift) & leaf_mask)];
    }

    state.blocks.push_back(std::move(kept));
    state.rows_of_block.push_back(static_cast<UInt32>(n));
}

void BuildSide::finishBuild()
{
    const size_t num_leaves = part_plan.num_leaves;

    size_t num_blocks = 0;
    for (const auto & up : local)
        if (up)
            num_blocks += up->blocks.size();
    chassert(num_blocks <= DB::BuildRef::BLOCK_NO_MASK); /// block_no uses the low 31 bits (MSB is the singleton marker)

    all_blocks.reserve(num_blocks);
    rows_per_block.reserve(num_blocks);

    /// Concatenate the per-worker stores in slot order, assigning final block_no and recording each
    /// worker's contiguous block range (the scatter's unit of static, lock-free ownership).
    for (size_t slot = 0; slot < local.size(); ++slot)
    {
        if (!local[slot] || local[slot]->blocks.empty())
            continue;
        LocalState & state = *local[slot];
        slot_block_begin.push_back(all_blocks.size());
        for (size_t bi = 0; bi < state.blocks.size(); ++bi)
        {
            all_blocks.push_back(std::move(state.blocks[bi]));
            rows_per_block.push_back(state.rows_of_block[bi]);
        }
        slot_block_end.push_back(all_blocks.size());
        used_slots.push_back(slot);
    }

    /// Fold every used slot's replicated histogram into the global per-leaf histogram.
    global_hist.assign(num_leaves, 0);
    for (size_t slot : used_slots)
    {
        const auto & state = *local[slot];
        for (size_t rep = 0; rep < state.replicas; ++rep)
        {
            const UInt32 * rep_hist = state.rep_hist.data() + rep * num_leaves;
            for (size_t leaf = 0; leaf < num_leaves; ++leaf)
                global_hist[leaf] += rep_hist[leaf];
        }
    }

    /// Exclusive per-block row offset: flat(ref) = block_base[block_no] + row_no.
    block_base.assign(rows_per_block.size() + 1, 0);
    for (size_t b = 0; b < rows_per_block.size(); ++b)
        block_base[b + 1] = block_base[b] + rows_per_block[b];
    total_rows = block_base.empty() ? 0 : block_base.back();

    finished = true;
}

LeafArrays BuildSide::makeLeafArrays() const
{
    LeafArrays out;
    out.num_leaves = part_plan.num_leaves;
    out.key_width = key_width;
    out.key_base.assign(part_plan.num_leaves, nullptr);
    out.ref_base.assign(part_plan.num_leaves, nullptr);
    out.leaf_rows.assign(part_plan.num_leaves, 0);
    return out;
}

void BuildSide::scatterBlockRanges(
    const ParallelFor & parallel_for,
    size_t num_parts,
    UInt32 shift,
    UInt32 mask,
    const std::vector<UInt64> & slot_part_offset,
    void * const * key_bases,
    BuildRef * const * ref_bases,
    std::atomic<UInt64> & total_bytes,
    bool accumulate_hll)
{
    const size_t kw = key_width;
    const bool multi_col = key_positions.size() > 1;
    const size_t num_used = used_slots.size();
    if (num_used == 0)
        return;

    /// At high fanout the scatter routes through SWWC + NT; below the threshold (or without NT) it uses
    /// the direct incremental cursors. Key and ref are scattered as two separate columns.
    const bool use_swwc = shouldUseSwwc(static_cast<int>(num_parts));

    parallel_for(num_used, [&](size_t slot, size_t worker)
    {
        const UInt64 * offsets = slot_part_offset.data() + slot * num_parts;

        std::vector<void *> kcur;
        std::vector<BuildRef *> rcur;
        std::optional<ScatterScratch> key_ss;
        std::optional<ScatterScratch> ref_ss;

        if (use_swwc)
        {
            key_ss.emplace(num_parts);
            ref_ss.emplace(num_parts);
            for (size_t part = 0; part < num_parts; ++part)
            {
                if (key_bases[part] != nullptr)
                {
                    key_ss->cursors()[part] = static_cast<char *>(key_bases[part]) + offsets[part] * kw;
                    ref_ss->cursors()[part] = reinterpret_cast<char *>(ref_bases[part]) + offsets[part] * sizeof(BuildRef); /// NOLINT
                }
            }
        }
        else
        {
            kcur.assign(num_parts, nullptr);
            rcur.assign(num_parts, nullptr);
            for (size_t part = 0; part < num_parts; ++part)
            {
                if (key_bases[part] != nullptr)
                {
                    kcur[part] = static_cast<char *>(key_bases[part]) + offsets[part] * kw;
                    rcur[part] = ref_bases[part] + offsets[part];
                }
            }
        }

        std::vector<BuildRef> refs;
        std::vector<char> packed;
        std::vector<UInt32> route(SCATTER_CHUNK_ROWS);
        std::vector<UInt32> bucket;
        if (accumulate_hll)
            bucket.resize(SCATTER_CHUNK_ROWS);
        if (multi_col)
            packed.resize(SCATTER_CHUNK_ROWS * kw);

        UInt64 local_bytes = 0;

        for (size_t block_idx = slot_block_begin[slot]; block_idx < slot_block_end[slot]; ++block_idx)
        {
            const size_t n = rows_per_block[block_idx];
            if (n == 0)
                continue;

            refs.resize(n);
            for (size_t row = 0; row < n; ++row)
                refs[row] = BuildRef(static_cast<UInt32>(block_idx), static_cast<UInt32>(row));

            const char * raw_keys = multi_col
                ? nullptr
                : all_blocks[block_idx].getByPosition(key_positions[0]).column->getRawData().data();

            for (size_t begin = 0; begin < n; begin += SCATTER_CHUNK_ROWS)
            {
                const size_t chunk = std::min(SCATTER_CHUNK_ROWS, n - begin);
                const char * keys_ptr = nullptr;
                if (multi_col)
                {
                    packKeyChunk(all_blocks[block_idx], begin, chunk, packed.data());
                    keys_ptr = packed.data();
                }
                else
                {
                    keys_ptr = raw_keys + begin * kw;
                }

                /// Recompute the route words for this chunk (the same hash `add` used for the histogram).
                /// On the final leaf-writing pass with distinct-estimate sizing on, also recover the bucket
                /// (low-32) word from the same hash and fold each key into its leaf's per-worker HLL sketch.
                if (accumulate_hll)
                {
                    computeRoutesAndBuckets(keys_ptr, chunk, kw, route.data(), bucket.data());
                    const UInt8 precision = hll_scatter->precision;
                    for (size_t row = 0; row < chunk; ++row)
                    {
                        const size_t leaf = (route[row] >> shift) & mask; /// final leaf id in a single-pass scatter
                        Hll::add(hll_scatter->sketch(worker, leaf), precision, bucket[row]);
                    }
                }
                else
                {
                    computeRoutes(keys_ptr, chunk, kw, route.data());
                }

                if (use_swwc)
                {
                    local_bytes += appendColumnSwwc(route.data(), shift, mask, chunk, keys_ptr, kw, *key_ss);
                    local_bytes += appendColumnSwwc(route.data(), shift, mask, chunk, refs.data() + begin, sizeof(BuildRef), *ref_ss);
                }
                else
                {
                    local_bytes += appendColumnDirect(route.data(), shift, mask, chunk, keys_ptr, kw, kcur.data());
                    local_bytes += appendColumnDirect(
                        route.data(), shift, mask, chunk, refs.data() + begin, sizeof(BuildRef),
                        reinterpret_cast<void **>(rcur.data())); /// NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)
                }
            }
        }

        if (use_swwc)
        {
            drainColumnSwwc(num_parts, *key_ss);
            drainColumnSwwc(num_parts, *ref_ss);
        }

        total_bytes.fetch_add(local_bytes, std::memory_order_relaxed);
    });
}

LeafArrays BuildSide::scatterSinglePass(const ParallelFor & parallel_for)
{
    LeafArrays out = makeLeafArrays();
    const size_t num_leaves = part_plan.num_leaves;
    const size_t num_used = used_slots.size();

    auto arrs = allocExactPartitions(out.arena, global_hist, key_width, &parallel_for);
    out.key_base = std::move(arrs.key);
    out.ref_base = std::move(arrs.ref);
    out.alloc_count = arrs.alloc_count;
    for (size_t leaf = 0; leaf < num_leaves; ++leaf)
        out.leaf_rows[leaf] = global_hist[leaf];

    const UInt32 shift = part_plan.total_bits > 0 ? part_plan.leaf_shift : 0u;
    const UInt32 mask = static_cast<UInt32>(num_leaves - 1);

    /// Per-(slot, leaf) start offsets within each leaf array: slot w begins where slots 0..w-1 ended.
    std::vector<UInt64> slot_off(num_used * num_leaves, 0);
    {
        std::vector<UInt64> running(num_leaves, 0);
        for (size_t worker = 0; worker < num_used; ++worker)
        {
            const auto & state = *local[used_slots[worker]];
            UInt64 * start = slot_off.data() + worker * num_leaves;
            for (size_t leaf = 0; leaf < num_leaves; ++leaf)
                start[leaf] = running[leaf];
            for (size_t rep = 0; rep < state.replicas; ++rep)
            {
                const UInt32 * rep_hist = state.rep_hist.data() + rep * num_leaves;
                for (size_t leaf = 0; leaf < num_leaves; ++leaf)
                    running[leaf] += rep_hist[leaf];
            }
        }
    }

    std::atomic<UInt64> total_bytes{0};
    /// Single pass = the final leaf-writing pass; accumulate the per-leaf HLL here when it is enabled.
    scatterBlockRanges(
        parallel_for, num_leaves, shift, mask, slot_off, out.key_base.data(), out.ref_base.data(), total_bytes,
        /*accumulate_hll=*/hll_scatter != nullptr);

    out.bytes_scattered = total_bytes.load();
    return out;
}

void BuildSide::refine(
    size_t first_leaf,
    const void * in_keys,
    const BuildRef * in_refs,
    UInt64 rows,
    size_t pass_index,
    UInt32 bits_consumed,
    LeafArrays & out,
    const std::vector<UInt64> & hist_prefix,
    RefineScratch & scratch,
    UInt64 & local_bytes,
    size_t worker)
{
    const UInt32 pass_bits = part_plan.pass_bits[pass_index];
    const size_t fanout = size_t{1} << pass_bits;
    const UInt32 new_bits = bits_consumed + pass_bits;
    const UInt32 leaf_fanout_shift = part_plan.total_bits - new_bits;
    const size_t leaves_per_child = size_t{1} << leaf_fanout_shift;
    const UInt32 routing_shift = PartitionPlan::ROUTE_BITS - new_bits;
    const UInt32 mask = static_cast<UInt32>(fanout - 1);
    const bool is_last = (pass_index + 1 == part_plan.pass_bits.size());
    const size_t kw = key_width;
    const bool use_swwc = shouldUseSwwc(static_cast<int>(fanout));

    /// Recompute the route words from the scattered packed keys (nothing is carried between passes). On the
    /// final pass with distinct-estimate sizing on, also recover the bucket (low-32) word from the same hash
    /// so each key can be folded into its leaf's per-worker HLL sketch.
    const bool accumulate_hll = is_last && hll_scatter != nullptr;
    scratch.route.resize(rows);
    if (accumulate_hll)
    {
        scratch.bucket.resize(rows);
        computeRoutesAndBuckets(static_cast<const char *>(in_keys), rows, kw, scratch.route.data(), scratch.bucket.data());
    }
    else
    {
        computeRoutes(static_cast<const char *>(in_keys), rows, kw, scratch.route.data());
    }

    auto scatter_into = [&](void * const * key_bases, BuildRef * const * ref_bases)
    {
        if (use_swwc)
        {
            scratch.key_scratch.resetFills(fanout);
            scratch.ref_scratch.resetFills(fanout);
            for (size_t child = 0; child < fanout; ++child)
            {
                scratch.key_scratch.cursors()[child] = key_bases[child];
                scratch.ref_scratch.cursors()[child] = ref_bases[child];
            }
            appendColumnSwwc(scratch.route.data(), routing_shift, mask, rows, in_keys, kw, scratch.key_scratch);
            appendColumnSwwc(scratch.route.data(), routing_shift, mask, rows, in_refs, sizeof(BuildRef), scratch.ref_scratch);
            drainColumnSwwc(fanout, scratch.key_scratch);
            drainColumnSwwc(fanout, scratch.ref_scratch);
        }
        else
        {
            for (size_t child = 0; child < fanout; ++child)
            {
                scratch.key_cursors[child] = key_bases[child];
                scratch.ref_cursors[child] = ref_bases[child];
            }
            appendColumnDirect(scratch.route.data(), routing_shift, mask, rows, in_keys, kw, scratch.key_cursors.data());
            appendColumnDirect(
                scratch.route.data(), routing_shift, mask, rows, in_refs, sizeof(BuildRef),
                reinterpret_cast<void **>(scratch.ref_cursors.data())); /// NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)
        }
        local_bytes += rows * (kw + sizeof(BuildRef));
    };

    if (is_last)
    {
        /// Final pass: scatter straight into the pre-allocated leaf arrays.
        std::vector<void *> kbase(fanout);
        std::vector<BuildRef *> rbase(fanout);
        for (size_t child = 0; child < fanout; ++child)
        {
            const size_t leaf = first_leaf + child * leaves_per_child;
            kbase[child] = out.key_base[leaf];
            rbase[child] = out.ref_base[leaf];
        }
        scatter_into(kbase.data(), rbase.data());
        if (accumulate_hll)
        {
            /// leaves_per_child == 1 on the final pass, so the routed child IS the absolute leaf offset.
            const UInt8 precision = hll_scatter->precision;
            for (UInt64 row = 0; row < rows; ++row)
            {
                const size_t leaf = first_leaf + ((scratch.route[row] >> routing_shift) & mask);
                Hll::add(hll_scatter->sketch(worker, leaf), precision, scratch.bucket[row]);
            }
        }
        return;
    }

    /// Intermediate pass: child counts from the leaf-prefix array, a RAII arena freed on return.
    std::vector<UInt64> child_counts(fanout);
    for (size_t child = 0; child < fanout; ++child)
    {
        const size_t lo = first_leaf + child * leaves_per_child;
        const size_t hi = lo + leaves_per_child;
        child_counts[child] = hist_prefix[hi] - hist_prefix[lo];
    }

    Arena child_arena;
    auto arrs = allocExactPartitions(child_arena, child_counts, kw, nullptr);
    scatter_into(arrs.key.data(), arrs.ref.data());

    /// Depth-first: finish each child's subtree before the next, freeing its block immediately so peak
    /// intermediate memory tracks the live path, not the whole tree.
    for (size_t child = 0; child < fanout; ++child)
    {
        if (child_counts[child] == 0)
            continue;
        refine(
            first_leaf + child * leaves_per_child,
            arrs.key[child],
            arrs.ref[child],
            child_counts[child],
            pass_index + 1,
            new_bits,
            out,
            hist_prefix,
            scratch,
            local_bytes,
            worker);
        child_arena.release(arrs.key[child]);
    }
}

LeafArrays BuildSide::scatterMultiPass(const ParallelFor & parallel_for)
{
    const size_t num_leaves = part_plan.num_leaves;
    const size_t kw = key_width;
    const size_t num_passes = part_plan.pass_bits.size();
    const size_t num_used = used_slots.size();

    /// hist_prefix[hi] - hist_prefix[lo] = rows in leaves [lo, hi).
    std::vector<UInt64> hist_prefix(num_leaves + 1, 0);
    std::inclusive_scan(global_hist.begin(), global_hist.end(), hist_prefix.begin() + 1);

    LeafArrays out = makeLeafArrays();
    {
        auto arrs = allocExactPartitions(out.arena, global_hist, kw, &parallel_for);
        out.key_base = std::move(arrs.key);
        out.ref_base = std::move(arrs.ref);
        out.alloc_count = arrs.alloc_count;
        for (size_t leaf = 0; leaf < num_leaves; ++leaf)
            out.leaf_rows[leaf] = global_hist[leaf];
    }

    std::atomic<UInt64> total_bytes{0};

    /// Pass 0: blocks -> 2^pass_bits[0] partitions (an intermediate arena).
    const UInt32 pass0_bits = part_plan.pass_bits[0];
    const size_t p0 = size_t{1} << pass0_bits;
    const UInt32 shift0 = PartitionPlan::ROUTE_BITS - pass0_bits;
    const UInt32 mask0 = static_cast<UInt32>(p0 - 1);
    chassert(p0 > 0 && num_leaves % p0 == 0);
    const size_t leaves_per_p0 = num_leaves / p0; // NOLINT(clang-analyzer-core.DivideZero)

    /// Per-(slot, pass-0-partition) counts and the resulting per-partition sizes + per-slot offsets.
    std::vector<UInt64> slot_hist0(num_used * p0, 0);
    for (size_t worker = 0; worker < num_used; ++worker)
    {
        const auto & state = *local[used_slots[worker]];
        for (size_t rep = 0; rep < state.replicas; ++rep)
        {
            const UInt32 * rep_hist = state.rep_hist.data() + rep * num_leaves;
            for (size_t part = 0; part < p0; ++part)
            {
                const size_t lo = part * leaves_per_p0;
                UInt64 sum = 0;
                for (size_t leaf = lo; leaf < lo + leaves_per_p0; ++leaf)
                    sum += rep_hist[leaf];
                slot_hist0[worker * p0 + part] += sum;
            }
        }
    }

    std::vector<UInt64> level0_counts(p0, 0);
    for (size_t worker = 0; worker < num_used; ++worker)
        for (size_t part = 0; part < p0; ++part)
            level0_counts[part] += slot_hist0[worker * p0 + part];

    std::vector<UInt64> slot_off0(num_used * p0, 0);
    {
        std::vector<UInt64> running(p0, 0);
        for (size_t worker = 0; worker < num_used; ++worker)
        {
            UInt64 * start = slot_off0.data() + worker * p0;
            for (size_t part = 0; part < p0; ++part)
            {
                start[part] = running[part];
                running[part] += slot_hist0[worker * p0 + part];
            }
        }
    }

    Arena level0_arena;
    auto level0 = allocExactPartitions(level0_arena, level0_counts, kw, &parallel_for);
    /// Pass 0 writes intermediate partitions, not final leaves -> never accumulate HLL here.
    scatterBlockRanges(
        parallel_for, p0, shift0, mask0, slot_off0, level0.key.data(), level0.ref.data(), total_bytes,
        /*accumulate_hll=*/false);

    /// Refine each pass-0 partition to its leaves (one parallel unit per partition).
    size_t max_refine_fanout = 1;
    for (size_t pass_idx = 1; pass_idx < num_passes; ++pass_idx)
        max_refine_fanout = std::max(max_refine_fanout, size_t{1} << part_plan.pass_bits[pass_idx]);

    parallel_for(p0, [&](size_t partition, size_t worker)
    {
        if (level0_counts[partition] == 0)
            return;
        RefineScratch scratch(max_refine_fanout);
        UInt64 local_bytes = 0;
        refine(
            partition * leaves_per_p0,
            level0.key[partition],
            level0.ref[partition],
            level0_counts[partition],
            /*pass_index=*/1,
            pass0_bits,
            out,
            hist_prefix,
            scratch,
            local_bytes,
            worker);
        level0_arena.release(level0.key[partition]);
        total_bytes.fetch_add(local_bytes, std::memory_order_relaxed);
    });

    out.bytes_scattered = total_bytes.load();
    return out;
}

LeafArrays BuildSide::scatterToLeaves(const ParallelFor & parallel_for, size_t num_workers, bool estimate_distinct_keys)
{
    chassert(finished);

    const size_t num_leaves = part_plan.num_leaves;

    /// Set up the per-worker HLL partial sketches for the duration of the scatter. The scatter passes read
    /// `hll_scatter` to accumulate on the final pass; merging happens once here afterwards.
    std::optional<HllScatterState> hll;
    if (estimate_distinct_keys && num_leaves != 0)
    {
        hll.emplace(std::max<size_t>(num_workers, 1), num_leaves);
        hll_scatter = &*hll;
    }

    LeafArrays out = part_plan.pass_bits.size() <= 1 ? scatterSinglePass(parallel_for) : scatterMultiPass(parallel_for);

    if (hll_scatter != nullptr)
    {
        /// Merge each leaf's per-worker partial sketches (register-wise max) into one estimate, clamped to
        /// the leaf's row count so the table can only ever shrink relative to row-count sizing. Parallel by
        /// leaf because the leaf count can be large.
        const UInt8 precision = hll_scatter->precision;
        const UInt32 m = hll_scatter->m;
        const size_t workers = hll_scatter->num_workers;
        out.distinct_key_estimates.assign(num_leaves, 0);
        parallel_for(num_leaves, [&](size_t leaf, size_t /*worker*/)
        {
            const UInt64 rows = out.leaf_rows[leaf];
            if (rows == 0)
                return;
            UInt8 merged[Hll::numRegisters(Hll::MAX_PRECISION)];
            std::memcpy(merged, hll_scatter->sketch(0, leaf), m);
            for (size_t w = 1; w < workers; ++w)
                Hll::merge(merged, hll_scatter->sketch(w, leaf), precision);
            const UInt64 est = Hll::estimate(merged, precision);
            out.distinct_key_estimates[leaf] = std::clamp<UInt64>(est, 1, rows);
        });
        hll_scatter = nullptr;
    }

    return out;
}

}
