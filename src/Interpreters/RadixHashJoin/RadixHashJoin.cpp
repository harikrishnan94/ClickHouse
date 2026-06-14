#include <Interpreters/RadixHashJoin/RadixHashJoin.h>

#include <Interpreters/RadixHashJoin/BuildSide.h>
#include <Interpreters/RadixHashJoin/KeyLayout.h>
#include <Interpreters/RadixHashJoin/LeafTable.h>
#include <Interpreters/RadixHashJoin/PackedKeyHash.h>
#include <Interpreters/RadixHashJoin/ParallelFor.h>
#include <Interpreters/RadixHashJoin/PartitionPlan.h>

#include <Interpreters/RowRefs.h>
#include <Interpreters/TableJoin.h>
#include <Interpreters/JoinUtils.h>
#include <Interpreters/HashJoin/HashJoin.h> /// canRemoveColumnsFromLeftBlock
#include <Interpreters/castColumn.h>

#include <Core/Block.h>
#include <Columns/ColumnsNumber.h>
#include <Columns/ColumnVector.h>
#include <Columns/IColumn.h>
#include <DataTypes/IDataType.h>

#include <Common/Exception.h>
#include <Common/Stopwatch.h>
#include <Common/ThreadPool.h>
#include <Common/ThreadGroupSwitcher.h>
#include <Common/assert_cast.h>
#include <Common/setThreadName.h>

#include <atomic>
#include <exception>
#include <mutex>
#include <numeric>

#include <unistd.h>

namespace CurrentMetrics
{
extern const Metric RadixHashJoinPoolThreads;
extern const Metric RadixHashJoinPoolThreadsActive;
extern const Metric RadixHashJoinPoolThreadsScheduled;
}

namespace DB
{

namespace ErrorCodes
{
extern const int LOGICAL_ERROR;
}

namespace
{

/// Private per-core L2 size for leaf sizing; 0 -> PartitionPlan fallback.
size_t detectL2Bytes()
{
#if defined(OS_LINUX) && defined(_SC_LEVEL2_CACHE_SIZE)
    if (const auto ret = ::sysconf(_SC_LEVEL2_CACHE_SIZE); ret > 0)
        return static_cast<size_t>(ret);
#endif
    return 0;
}

/// `RadixJoin::ParallelFor` backed by a `ThreadPool` (used the same way `ConcurrentHashJoin` drives its
/// pool: `scheduleOrThrow` + `wait`, with each task inheriting the query's thread group). Distributes
/// `total` units across up to `num_workers` tasks with dynamic work-stealing on an atomic cursor —
/// leaf sizes are highly skewed, so a static equal-count split would serialize on the big leaves. Each
/// task carries a fixed dense `worker` id in [0, num_workers) (one task per id), so a unit may index a
/// per-worker resource (e.g. `LeafTables::build_arenas[worker]`) under a single-writer invariant. The
/// first unit exception is captured and rethrown after every task has stopped.
void runParallelFor(ThreadPool & pool, size_t num_workers, const ThreadGroupPtr & thread_group, size_t total, const RadixJoin::UnitFn & fn)
{
    if (total == 0)
        return;

    const size_t workers = std::min(num_workers, total);
    std::atomic<size_t> next{0};
    std::mutex exc_mutex;
    std::exception_ptr first_exc;

    auto capture = [&](std::exception_ptr e)
    {
        std::lock_guard lock(exc_mutex);
        if (!first_exc)
            first_exc = std::move(e);
        next.store(total, std::memory_order_relaxed); /// stop the other workers from pulling more units
    };

    try
    {
        for (size_t w = 0; w < workers; ++w)
        {
            pool.scheduleOrThrow([&, w, thread_group]
            {
                ThreadGroupSwitcher switcher(thread_group, ThreadName::RADIX_JOIN);
                while (true)
                {
                    const size_t unit = next.fetch_add(1, std::memory_order_relaxed);
                    if (unit >= total)
                        break;
                    try
                    {
                        fn(unit, w);
                    }
                    catch (...)
                    {
                        capture(std::current_exception());
                        break;
                    }
                }
            });
        }
    }
    catch (...)
    {
        /// A failed schedule (or any setup error): stop the running tasks, then drain and propagate.
        capture(std::current_exception());
    }

    pool.wait();

    if (first_exc)
        std::rethrow_exception(first_exc);
}

/// Precomputed (once) output schema of the probe. One entry per output column, grouped by gather kind.
struct LeftOut
{
    String name;        /// the left column, gathered straight from the probe block by name
    DataTypePtr type;
};
struct PayOut
{
    size_t payload_idx; /// position in `columns_to_add` (also indexes `payload_right_indexes`)
    DataTypePtr type;
    String name;
};
struct ReqOut
{
    String left_source; /// the left key column the value is copied from (equi-join)
    DataTypePtr left_type;
    DataTypePtr right_type;
    String name;
};
struct OutputPlan
{
    std::vector<LeftOut> left;
    std::vector<PayOut> payload;
    std::vector<ReqOut> required;
};

/// A contiguous range of matched rows that all resolve to the SAME build block, so the block's
/// payload can be gathered in one bulk call. `begin`/`end` index the sorted match arrays.
struct BlockRun
{
    UInt32 block_no;
    UInt64 begin;
    UInt64 end;
};

/** Per-probe-lane reusable scratch. `joinBlock` runs concurrently across streams, so each probe lane
  * owns its own slot (selected by the plumbed lane index — no locking, no thread_local). Every buffer
  * is reused across blocks — capacity only ever grows — so the steady-state probe does no per-block
  * heap allocation.
  */
struct ProbeScratch
{
    /// Batch-wide multi-column key packing. Capacity is reused across batches and only ever grows, so the
    /// steady state does no per-batch heap allocation. The per-row hash is no longer materialised here —
    /// it is computed on the fly inside `collectMatches`.
    std::vector<char> packed;                   /// multi-column packed keys for the whole batch
    std::vector<const char *> kcol_src;         /// raw data of each left key column

    /// Matches (one (left_row, ref) per match), in probe order.
    std::vector<UInt32> left_rows;
    std::vector<RadixJoin::BuildRef> refs;

    /// Counting sort of the matches by build block (only when `grouped`).
    bool grouped = false;                       /// true -> use the sorted arrays + runs; false -> probe order
    std::vector<UInt64> block_start;            /// size num_blocks + 1: per-block start offset
    std::vector<UInt64> cursor;                 /// running write position per block
    std::vector<UInt32> sorted_left_rows;       /// matches reordered by build block
    std::vector<UInt32> sorted_row_no;          /// the matched build row within its block
    std::vector<BlockRun> runs;                 /// non-empty per-block runs over the sorted arrays

    /// Reused index columns for the bulk IColumn::index gathers (data reassigned, capacity reused).
    ColumnUInt32::MutablePtr left_index;
    ColumnUInt32::MutablePtr payload_index;
};

/// Read-only probe inputs, bundled so the phase functions need no access to the private State type.
struct ProbeContext
{
    const Names & key_names_left;
    const std::vector<size_t> & key_widths;
    const std::vector<size_t> & key_offsets;
    const std::vector<RadixJoin::ColumnPackFn> & key_packers;
    size_t key_width;
    const RadixJoin::LeafTables & leaf_tables;
    UInt32 leaf_shift;
    UInt32 total_bits;
    const std::vector<UInt64> & block_base;
    size_t num_leaves;
};

/// Block-scope leaf partitioning is done in batches of this many rows so the pack/hash/route scratch
/// stays bounded even when the user raises `max_block_size` (the common case is one batch per block).
constexpr size_t PROBE_BATCH_ROWS = 65536;
static_assert(PROBE_BATCH_ROWS <= 65536, "the AMAC probe slot stores the per-batch row index in a UInt16");

/// Precompute the output schema once: left columns (filtered by the analyzer rules), then payload
/// columns whose output name is not already provided by a left column, then required right-key columns
/// not already present. Mirrors the dedup order the result block must have.
void buildOutputPlan(
    bool remove_left_columns,
    const NameSet & left_output_names,
    const Block & block,
    const Block & columns_to_add,
    const std::vector<std::string> & payload_output_names,
    const Block & required_right_keys,
    const std::vector<std::string> & required_right_keys_output_names,
    const std::vector<std::string> & required_right_keys_sources,
    OutputPlan & plan)
{
    NameSet emitted;
    for (const auto & col : block)
    {
        if (remove_left_columns && !left_output_names.contains(col.name))
            continue;
        plan.left.push_back({col.name, col.type});
        emitted.insert(col.name);
    }
    for (size_t col = 0; col < columns_to_add.columns(); ++col)
    {
        const std::string & out_name = payload_output_names[col];
        if (emitted.contains(out_name))
            continue;
        plan.payload.push_back({col, columns_to_add.getByPosition(col).type, out_name});
        emitted.insert(out_name);
    }
    for (size_t key_col = 0; key_col < required_right_keys.columns(); ++key_col)
    {
        const std::string & out_name = required_right_keys_output_names[key_col];
        if (emitted.contains(out_name))
            continue;
        const auto & right_key = required_right_keys.getByPosition(key_col);
        const auto & left_src = block.getByName(required_right_keys_sources[key_col]);
        plan.required.push_back({required_right_keys_sources[key_col], left_src.type, right_key.type, out_name});
        emitted.insert(out_name);
    }
}

/// Counting-sort the probe-order matches by build block (stable within a block) into the sorted match
/// arrays + per-block runs, so each block's payload is gathered with one bulk call.
void sortMatchesByBlock(const ProbeContext & ctx, ProbeScratch & s)
{
    const size_t out_rows = s.refs.size();
    const size_t num_blocks = ctx.block_base.size() - 1;

    s.block_start.assign(num_blocks + 1, 0);
    for (const RadixJoin::BuildRef ref : s.refs)
        ++s.block_start[ref.blockNo() + 1];
    for (size_t b = 1; b <= num_blocks; ++b)
        s.block_start[b] += s.block_start[b - 1];

    s.cursor.assign(s.block_start.begin(), s.block_start.end());
    s.sorted_left_rows.resize(out_rows);
    s.sorted_row_no.resize(out_rows);
    for (size_t m = 0; m < out_rows; ++m)
    {
        const RadixJoin::BuildRef ref = s.refs[m];
        const UInt64 pos = s.cursor[ref.blockNo()]++;
        s.sorted_left_rows[pos] = s.left_rows[m];
        s.sorted_row_no[pos] = ref.rowNo();
    }

    s.runs.clear();
    for (size_t b = 0; b < num_blocks; ++b)
    {
        const UInt64 begin = s.block_start[b];
        const UInt64 end = s.block_start[b + 1];
        if (end > begin)
            s.runs.push_back({static_cast<UInt32>(b), begin, end});
    }
}

/// Pack the batch's multi-column keys into `packed` in `SCATTER_CHUNK_ROWS` tiles (chunk-aware) so each
/// tile's per-column scatter writes stay L1-resident. Single-column keys need no packing (the column's
/// raw data is used directly) and never reach here. The 64-bit hash is NOT computed here anymore — it is
/// derived from the packed key inside the probe pipeline (`collectMatches`), where its multiply-fold
/// latency overlaps with the in-flight cell misses.
template <size_t key_width>
void packBatch(
    const ProbeContext & ctx,
    const std::vector<const char *> & kcol_src,
    size_t batch_start,
    size_t bn,
    char * packed)
{
    constexpr size_t tile = RadixJoin::BuildSide::SCATTER_CHUNK_ROWS; /// 1024
    const size_t ncols = ctx.key_widths.size();
    for (size_t off = 0; off < bn; off += tile)
    {
        const size_t tn = std::min(tile, bn - off);
        char * dst = packed + off * key_width;
        for (size_t c = 0; c < ncols; ++c)
            ctx.key_packers[c](kcol_src[c], batch_start + off, tn, dst, key_width, ctx.key_offsets[c], ctx.key_widths[c]);
    }
}

/// Phase 1 — Probe. For each batch: pack (multi-column) + 64-bit hash, then the flat per-row routed
/// probe. Matches land directly in `s.left_rows`/`s.refs`, then are grouped by build block for the bulk
/// gathers when the build had duplicates.
void probeBlock(const ProbeContext & ctx, const Block & block, size_t n, ProbeScratch & s)
{
    s.left_rows.clear();
    s.refs.clear();
    s.left_rows.reserve(n);
    s.refs.reserve(n);

    const bool single_col = ctx.key_widths.size() == 1;
    const char * single_raw = nullptr;
    if (single_col)
    {
        single_raw = block.getByName(ctx.key_names_left[0]).column->getRawData().data();
    }
    else
    {
        s.kcol_src.clear();
        s.kcol_src.reserve(ctx.key_widths.size());
        for (const auto & name : ctx.key_names_left)
            s.kcol_src.push_back(block.getByName(name).column->getRawData().data());
    }

    const bool has_dups = ctx.leaf_tables.any_duplicates.load(std::memory_order_relaxed);

    for (size_t batch_start = 0; batch_start < n; batch_start += PROBE_BATCH_ROWS)
    {
        const size_t bn = std::min(PROBE_BATCH_ROWS, n - batch_start);

        const char * keys = nullptr;
        if (single_col)
        {
            keys = single_raw + batch_start * ctx.key_width;
        }
        else
        {
            s.packed.resize(bn * ctx.key_width);
            keys = s.packed.data();

            /// Phase 1a — pack the multi-column keys (chunk-aware). Single-column keys skip this entirely;
            /// the per-row hash is computed later inside `collectMatches`.
            switch (ctx.key_width)
            {
#define RHJ_PACK(W) \
    case W: \
        packBatch<W>(ctx, s.kcol_src, batch_start, bn, s.packed.data()); \
        break;
                RHJ_PACK(4)  RHJ_PACK(8)  RHJ_PACK(12) RHJ_PACK(16)
                RHJ_PACK(20) RHJ_PACK(24) RHJ_PACK(28) RHJ_PACK(32)
                RHJ_PACK(36) RHJ_PACK(40) RHJ_PACK(44) RHJ_PACK(48)
                RHJ_PACK(52) RHJ_PACK(56) RHJ_PACK(60) RHJ_PACK(64)
#undef RHJ_PACK
                default:
                    throw Exception(ErrorCodes::LOGICAL_ERROR, "RadixHashJoin: unsupported key width {}", ctx.key_width);
            }
        }

        /// Phase 1b — lookup. Matches (absolute batch-local row id) are appended to the global match
        /// buffers; `batch_start` is folded back in afterwards (a no-op for the single-batch common case).
        const size_t match_begin = s.left_rows.size();
        RadixJoin::collectMatches(
            ctx.key_width, ctx.leaf_tables.leaves.data(), ctx.leaf_shift, ctx.total_bits,
            keys, bn, /*pos_fits_u32=*/ctx.leaf_tables.max_bucket_bits <= 31, s.left_rows, s.refs);

        if (batch_start != 0)
        {
            const auto base = static_cast<UInt32>(batch_start);
            for (size_t m = match_begin; m < s.left_rows.size(); ++m)
                s.left_rows[m] += base;
        }
    }

    /// Hybrid gather decision. With a duplicate-free build the matches are ~1:1 and scattered, so a
    /// direct typed gather in match order (no sort, no temp) wins. With duplicates the probe rows fan
    /// out and grouping by build block (counting sort) gives the gather base-pointer locality.
    s.grouped = has_dups;
    if (s.grouped)
        sortMatchesByBlock(ctx, s);
}

/// Phase 2 — Gather Left. Left columns and required right-key columns are both gathered from the probe
/// block in bulk via the type-specialised IColumn::index (one call per column, no per-row dispatch).
void gatherLeft(const OutputPlan & plan, const Block & block, ProbeScratch & s, ColumnsWithTypeAndName & out)
{
    if (!s.left_index)
        s.left_index = ColumnUInt32::create();
    /// Probe order when gathering directly, build-block-grouped order when the matches were sorted.
    const std::vector<UInt32> & left_src = s.grouped ? s.sorted_left_rows : s.left_rows;
    s.left_index->getData().assign(left_src.begin(), left_src.end());
    const IColumn & index = *s.left_index;

    for (const auto & lo : plan.left)
        out.emplace_back(block.getByName(lo.name).column->index(index, 0), lo.type, lo.name);

    for (const auto & ro : plan.required)
    {
        /// Equi-join: right_key == left_key. Gather the left source in bulk, then cast to the right key
        /// type — skipping the cast when the types already match (the common case).
        ColumnPtr gathered = block.getByName(ro.left_source).column->index(index, 0);
        if (ro.left_type->equals(*ro.right_type))
        {
            out.emplace_back(gathered, ro.right_type, ro.name);
        }
        else
        {
            ColumnWithTypeAndName tmp_col(gathered, ro.left_type, ro.name);
            out.emplace_back(castColumn(tmp_col, ro.right_type), ro.right_type, ro.name);
        }
    }
}

/// Direct typed gather of one fixed-width numeric payload column in probe order: the build value of
/// each match is copied straight into the output data array. The column type is dispatched ONCE (the
/// loop body is a plain typed load/store, no per-row virtual call) and there is a single copy with no
/// temp allocation. `stored_columns[block_no]->columns[col_idx]` is the build column holding the
/// matched row; `col_idx` is the payload column's position in `right_sample_block`.
template <typename T>
void gatherNumericDirect(
    IColumn & out_col,
    const ColumnsInfo * const * stored_columns,
    size_t col_idx,
    const RadixJoin::BuildRef * refs,
    size_t out_rows)
{
    auto & dst = assert_cast<ColumnVector<T> &>(out_col).getData();
    dst.resize(out_rows);
    for (size_t i = 0; i < out_rows; ++i)
    {
        const RadixJoin::BuildRef ref = refs[i];
        const IColumn * src = stored_columns[ref.blockNo()]->columns[col_idx].get();
        dst[i] = assert_cast<const ColumnVector<T> *>(src)->getData()[ref.rowNo()];
    }
}

/// Gather one payload column directly in probe order (the duplicate-free / sparse path). Fixed-width
/// numeric columns take the typed no-dispatch path above; anything else (String, Nullable, Decimal,
/// wide ints, ...) falls back to per-row insertFrom. `col_idx` is the payload column's position in
/// `right_sample_block`; `stored_columns[block_no]->columns[col_idx]` is the build column.
void gatherPayloadDirect(
    const PayOut & po,
    const ColumnsInfo * const * stored_columns,
    size_t col_idx,
    const RadixJoin::BuildRef * refs,
    size_t out_rows,
    ColumnsWithTypeAndName & out)
{
    auto col = po.type->createColumn();
    if (out_rows == 0)
    {
        out.emplace_back(std::move(col), po.type, po.name);
        return;
    }
    switch (col->getDataType())
    {
#define RHJ_GATHER_NUMERIC(TYPE_INDEX, T) \
    case TypeIndex::TYPE_INDEX: gatherNumericDirect<T>(*col, stored_columns, col_idx, refs, out_rows); break;
        RHJ_GATHER_NUMERIC(UInt8, UInt8)   RHJ_GATHER_NUMERIC(UInt16, UInt16)
        RHJ_GATHER_NUMERIC(UInt32, UInt32) RHJ_GATHER_NUMERIC(UInt64, UInt64)
        RHJ_GATHER_NUMERIC(Int8, Int8)     RHJ_GATHER_NUMERIC(Int16, Int16)
        RHJ_GATHER_NUMERIC(Int32, Int32)   RHJ_GATHER_NUMERIC(Int64, Int64)
        RHJ_GATHER_NUMERIC(Float32, Float32) RHJ_GATHER_NUMERIC(Float64, Float64)
#undef RHJ_GATHER_NUMERIC
        default:
            col->reserve(out_rows);
            for (size_t i = 0; i < out_rows; ++i)
            {
                const RadixJoin::BuildRef ref = refs[i];
                const IColumn * src = stored_columns[ref.blockNo()]->columns[col_idx].get();
                col->insertFrom(*src, ref.rowNo());
            }
    }
    out.emplace_back(std::move(col), po.type, po.name);
}

/// Phase 3 — Gather Right. Build payload lives per build block (not co-located with the leaf cell).
/// Two paths, chosen in probeBlock: the duplicate-free direct typed gather (probe order), or, when the
/// build had duplicates, gathering one build block at a time over the sorted runs (one IColumn::index +
/// one insertRangeFrom per block). Neither path does per-row virtual dispatch on a numeric payload.
/// `stored_columns` resolves a `BuildRef::blockNo()` to the stored block's `ColumnsInfo`;
/// `payload_right_indexes[po.payload_idx]` is the payload column's position in `right_sample_block`.
void gatherRight(
    const OutputPlan & plan,
    const ColumnsInfo * const * stored_columns,
    const std::vector<size_t> & payload_right_indexes,
    ProbeScratch & s,
    ColumnsWithTypeAndName & out)
{
    if (!s.grouped)
    {
        const size_t out_rows = s.refs.size();
        for (const auto & po : plan.payload)
            gatherPayloadDirect(po, stored_columns, payload_right_indexes[po.payload_idx], s.refs.data(), out_rows, out);
        return;
    }

    const size_t out_rows = s.sorted_row_no.size();
    if (!s.payload_index)
        s.payload_index = ColumnUInt32::create();

    for (const auto & po : plan.payload)
    {
        const size_t col_idx = payload_right_indexes[po.payload_idx];
        auto col = po.type->createColumn();
        col->reserve(out_rows);
        /// `stored_columns` is only valid after the build; an empty `runs` (header path / no matches)
        /// must not touch it. Fetch the per-block source inside the loop so the empty case is a no-op.
        for (const auto & run : s.runs)
        {
            const IColumn * src = stored_columns[run.block_no]->columns[col_idx].get();
            s.payload_index->getData().assign(s.sorted_row_no.begin() + run.begin, s.sorted_row_no.begin() + run.end);
            ColumnPtr gathered = src->index(*s.payload_index, 0);
            col->insertRangeFrom(*gathered, 0, run.end - run.begin);
        }
        out.emplace_back(std::move(col), po.type, po.name);
    }
}

}

struct RadixHashJoin::State
{
    RadixJoin::PartitionPlan plan;

    Names key_names_left;
    Names key_names_right;
    std::vector<size_t> key_positions; /// key column positions in the right block (right_sample order)
    std::vector<size_t> key_widths;
    std::vector<size_t> key_offsets;   /// byte offset of each key column within a packed key
    std::vector<RadixJoin::ColumnPackFn> key_packers; /// shared with the build side
    size_t key_width = 0;

    std::unique_ptr<RadixJoin::BuildSide> build_side;

    /// The dedicated pool the eager post-build is parallelised over (sized to max_threads, created in
    /// the constructor). Mirrors `ConcurrentHashJoin::pool`.
    std::unique_ptr<ThreadPool> pool;

    std::atomic<bool> built{false};
    RadixJoin::LeafTables leaf_tables;

    /// Build-block payload resolution, shared with the other join algorithms (see RowRefs.h). Each stored
    /// right block is registered (in accumulation order) so `BuildRef::blockNo()` indexes `blocksData()`
    /// directly. `columns_infos` owns the `ColumnsInfo` objects (stable addresses; the index holds raw
    /// `const ColumnsInfo *`). `payload_right_indexes[payload_idx]` is the payload column's position in
    /// `right_sample_block` (computed once in the constructor), mirroring CHJ's `right_indexes`.
    StoredColumnsIndexPtr stored_columns_index;
    std::vector<std::unique_ptr<ColumnsInfo>> columns_infos;
    std::vector<size_t> payload_right_indexes;

    std::vector<UInt64> block_base;
    UInt64 total_rows = 0;
    size_t total_bytes = 0;

    /// Output plan (precomputed in the constructor).
    Block right_table_keys;                                /// the right join-key columns
    Block columns_to_add;                                  /// right payload columns (right sample minus keys)
    std::vector<std::string> payload_output_names;         /// renamed output name of each payload column
    Block required_right_keys;                             /// right key columns that must appear in the output
    std::vector<std::string> required_right_keys_sources;       /// the left key each is copied from
    std::vector<std::string> required_right_keys_output_names;  /// renamed output names
    bool remove_left_columns = false;
    NameSet left_output_names;

    /// Probe-side schema, precomputed once on the first joinBlock call (the header path).
    std::once_flag plan_once;
    OutputPlan out_plan;

    /// Per-lane probe scratch (joinBlock runs concurrently across streams). Sized to max_threads in the
    /// constructor; each lane owns its slot exclusively, so the probe path needs no locking. Slots are
    /// created lazily on first use by their owning lane.
    std::vector<std::unique_ptr<ProbeScratch>> lane_scratch;
};

RadixHashJoin::RadixHashJoin(
    std::shared_ptr<TableJoin> table_join_,
    SharedHeader right_sample_block_,
    size_t max_threads_,
    std::optional<UInt64> rhs_size_estimation_,
    UInt64 max_partitions_per_pass_,
    bool size_tables_by_distinct_estimate_)
    : table_join(std::move(table_join_))
    , right_sample_block(right_sample_block_)
    , max_threads(std::max<size_t>(max_threads_, 1))
    , rhs_size_estimation(rhs_size_estimation_)
    , max_partitions_per_pass(max_partitions_per_pass_)
    , size_tables_by_distinct_estimate(size_tables_by_distinct_estimate_)
    , state(std::make_unique<State>())
{
    /// The planner gate guarantees the invariants below; re-check and fail loudly if violated.
    if (!table_join->oneDisjunct())
        throw Exception(ErrorCodes::LOGICAL_ERROR, "RadixHashJoin requires a single ON clause disjunct");

    const auto & clause = table_join->getOnlyClause();
    state->key_names_right = clause.key_names_right;
    state->key_names_left = clause.key_names_left;

    if (state->key_names_right.empty())
        throw Exception(ErrorCodes::LOGICAL_ERROR, "RadixHashJoin requires at least one join key column");

    for (const auto & name : state->key_names_right)
    {
        const auto * col = right_sample_block->findByName(name);
        if (!col || !col->type->haveMaximumSizeOfValue())
            throw Exception(ErrorCodes::LOGICAL_ERROR, "RadixHashJoin: key column '{}' is not fixed-width", name);
        const size_t w = col->type->getMaximumSizeOfValueInMemory();
        if (w == 0 || w % 4 != 0)
            throw Exception(
                ErrorCodes::LOGICAL_ERROR, "RadixHashJoin: key column '{}' width {} is not a multiple of 4", name, w);
        state->key_positions.push_back(right_sample_block->getPositionByName(name));
        state->key_widths.push_back(w);
        state->key_width += w;
    }
    if (state->key_width < 4 || state->key_width > 64 || state->key_width % 4 != 0)
        throw Exception(
            ErrorCodes::LOGICAL_ERROR, "RadixHashJoin: packed key width {} must be a multiple of 4 in [4, 64]", state->key_width);

    state->key_offsets.resize(state->key_widths.size());
    std::exclusive_scan(state->key_widths.begin(), state->key_widths.end(), state->key_offsets.begin(), size_t{0});

    state->key_packers.reserve(state->key_widths.size());
    for (size_t w : state->key_widths)
        state->key_packers.push_back(RadixJoin::chooseColumnPacker(w));

    state->plan = RadixJoin::PartitionPlan::choose(rhs_size_estimation, detectL2Bytes(), max_partitions_per_pass);
    state->build_side = std::make_unique<RadixJoin::BuildSide>(state->plan, state->key_positions, state->key_widths, max_threads);

    /// Dedicated pool for the eager post-build (scatter + leaf-table build), sized to max_threads and
    /// driven exactly like ConcurrentHashJoin's: `scheduleOrThrow` + `wait`. queue_size == max_threads
    /// because `runParallelFor` schedules at most one task per worker id.
    state->pool = std::make_unique<ThreadPool>(
        CurrentMetrics::RadixHashJoinPoolThreads,
        CurrentMetrics::RadixHashJoinPoolThreadsActive,
        CurrentMetrics::RadixHashJoinPoolThreadsScheduled,
        /*max_threads_*/ max_threads,
        /*max_free_threads_*/ 0,
        /*queue_size_*/ max_threads);

    /// One probe-scratch slot per probe lane (the pipeline replicates the probe across at most max_threads
    /// lanes); each lane owns its slot, so the probe path is lock-free. Slots are filled lazily on use.
    state->lane_scratch.resize(max_threads);

    /// Output plan: split right keys vs payload, the required right keys (copied from the left), and the
    /// left columns that survive into the result (the analyzer column rules).
    JoinCommon::splitAdditionalColumns(state->key_names_right, *right_sample_block, state->right_table_keys, state->columns_to_add);

    state->payload_output_names.reserve(state->columns_to_add.columns());
    for (const auto & col : state->columns_to_add)
        state->payload_output_names.push_back(table_join->renamedRightColumnName(col.name));

    /// Resolve each payload column's position in `right_sample_block` once. Stored blocks are normalised
    /// to right-sample column order, so this position indexes the block's `ColumnsInfo::columns` directly.
    /// `payload_idx` (the position in `columns_to_add`) indexes this vector.
    state->payload_right_indexes.reserve(state->columns_to_add.columns());
    for (const auto & col : state->columns_to_add)
        state->payload_right_indexes.push_back(right_sample_block->getPositionByName(col.name));

    state->required_right_keys = table_join->getRequiredRightKeys(state->right_table_keys, state->required_right_keys_sources);
    state->required_right_keys_output_names.reserve(state->required_right_keys.columns());
    for (const auto & col : state->required_right_keys)
        state->required_right_keys_output_names.push_back(table_join->renamedRightColumnName(col.name));

    state->remove_left_columns = HashJoin::canRemoveColumnsFromLeftBlock(*table_join);
    for (const auto & col : table_join->getOutputColumns(JoinTableSide::Left))
        state->left_output_names.insert(col.name);
}

RadixHashJoin::~RadixHashJoin() = default;

const TableJoin & RadixHashJoin::getTableJoin() const
{
    return *table_join;
}

bool RadixHashJoin::addBlockToJoin(const Block & block, bool check_limits)
{
    return addBlockToJoin(block, block.rows(), check_limits, 0);
}

bool RadixHashJoin::addBlockToJoin(const Block & block, size_t num_rows, bool check_limits)
{
    return addBlockToJoin(block, num_rows, check_limits, 0);
}

bool RadixHashJoin::addBlockToJoin(const Block & block, size_t /*num_rows*/, bool /*check_limits*/, size_t build_lane)
{
    /// Normalise to the right sample structure (by name, in order) so key columns sit at the build-side
    /// key positions and every payload column is gatherable by name later. Materialise so the key
    /// columns expose contiguous raw data and the stored payload types match `columns_to_add`.
    ColumnsWithTypeAndName cols;
    cols.reserve(right_sample_block->columns());
    for (const auto & sample_col : *right_sample_block)
        cols.push_back(block.getByName(sample_col.name));
    Block normalized = materializeBlock(Block(std::move(cols)));

    state->build_side->add(normalized, build_lane);

    return true;
}

void RadixHashJoin::setTotals(const Block & block)
{
    std::lock_guard lock(totals_mutex);
    IJoin::setTotals(block);
}

void RadixHashJoin::checkTypesOfKeys(const Block & block) const
{
    JoinCommon::checkTypesOfKeys(block, state->key_names_left, *right_sample_block, state->key_names_right);
}

void RadixHashJoin::onBuildPhaseFinish()
{
    /// The eager post-build is build-phase work. JoiningTransform calls this from `work()` (heavy work
    /// allowed) exactly once, before any real probe, so the join is fully built when it returns.
    state->build_side->finishBuild(); /// merge + prefix-sum the histograms (the build barrier)
    runPostBuild();
}

void RadixHashJoin::runPostBuild()
{
    /// Capture the calling thread's group once so every pool task attaches to the same query group; the
    /// `ParallelFor` then schedules the post-build steps on `state->pool`.
    const ThreadGroupPtr thread_group = getCurrentThreadGroup();
    const size_t num_workers = max_threads;
    ThreadPool & pool = *state->pool;
    const RadixJoin::ParallelFor parallel_for = [&pool, num_workers, thread_group](size_t total, const RadixJoin::UnitFn & fn)
    {
        runParallelFor(pool, num_workers, thread_group, total, fn);
    };

    /// Deferred exact key+ref scatter into the per-leaf arrays. The route hash is recomputed from the
    /// key inside the scatter; the leaf bucket is recomputed in the leaf-table build — nothing per-row
    /// is carried, and no payload is moved.
    RadixJoin::LeafArrays leaves = state->build_side->scatterToLeaves(parallel_for, num_workers, size_tables_by_distinct_estimate);

    state->block_base = state->build_side->blockBase();
    state->total_rows = state->build_side->totalRows();

    state->leaf_tables = RadixJoin::buildLeafTables(leaves, state->total_rows, state->key_width, num_workers, parallel_for);

    /// Register each stored right block (in accumulation order) so a `BuildRef::blockNo()` indexes
    /// `blocksData()` directly. `add` returns size()-1, so block_no == build index (the chassert below).
    /// Serial, on this thread.
    {
        state->stored_columns_index = std::make_shared<StoredColumnsIndex>();
        state->columns_infos.reserve(state->build_side->blocks().size());
        for (const auto & block : state->build_side->blocks())
        {
            state->columns_infos.emplace_back(std::make_unique<ColumnsInfo>(block.getColumns()));
            const UInt32 bn = state->stored_columns_index->add(state->columns_infos.back().get()); /// NOLINT(clang-analyzer-deadcode.DeadStores)
            chassert(bn + 1 == state->columns_infos.size()); /// block_no == build index (bn read only in the chassert)
        }

        state->total_bytes = state->leaf_tables.arena.bytesReserved();
    }

    leaves = RadixJoin::LeafArrays(); /// scatter arrays no longer needed once the tables are built

    /// Publish all post-build state to the probe threads. The pipeline already orders
    /// `onBuildPhaseFinish` before any real `joinBlock`, but this release/acquire on `built` is the
    /// documented barrier: `leaf_tables`, `stored_columns_index`, `block_base`, `total_rows` are all
    /// written above and become visible to a probe thread that observes `built == true`.
    state->built.store(true, std::memory_order_release);
}

JoinResultPtr RadixHashJoin::joinBlock(Block block)
{
    return joinBlock(std::move(block), 0);
}

JoinResultPtr RadixHashJoin::joinBlock(Block block, size_t lane)
{
    /// The join is fully built by `onBuildPhaseFinish` before any real probe; `joinBlock` never builds.
    /// Before the build barrier (the header/planning path: `transformHeader` calls `joinBlock` first)
    /// `built` is still false and the block below emits the output schema only.
    State & st = *state;
    const size_t n = block.rows();

    /// Materialise the probe block so the key columns expose contiguous raw data and the emitted left
    /// columns are full (matching the materialised header JoiningTransform produces).
    block = materializeBlock(block);

    /// Precompute the output schema once (the first call is the header path, before any probe).
    std::call_once(st.plan_once, [&]
    {
        buildOutputPlan(
            st.remove_left_columns, st.left_output_names, block, st.columns_to_add, st.payload_output_names,
            st.required_right_keys, st.required_right_keys_output_names, st.required_right_keys_sources, st.out_plan);
    });

    const bool can_probe = st.built.load(std::memory_order_acquire) && n > 0;

    /// Each probe lane owns its scratch slot (filled lazily on first use), so no locking is needed.
    if (lane >= st.lane_scratch.size())
        throw Exception(
            ErrorCodes::LOGICAL_ERROR, "RadixHashJoin: probe lane {} exceeds max_threads {}", lane, st.lane_scratch.size());
    auto & slot = st.lane_scratch[lane];
    if (!slot)
        slot = std::make_unique<ProbeScratch>();
    ProbeScratch & s = *slot;

    if (can_probe)
    {
        const ProbeContext ctx{
            st.key_names_left, st.key_widths, st.key_offsets, st.key_packers, st.key_width,
            st.leaf_tables, st.plan.leaf_shift, st.plan.total_bits, st.block_base, st.leaf_tables.leaves.size()};
        probeBlock(ctx, block, n, s);
    }
    else
    {
        /// Header / empty-probe path: emit the schema only (no matches). The direct (non-grouped) path
        /// reads `left_rows` / `refs`, so clear those too.
        s.grouped = false;
        s.left_rows.clear();
        s.refs.clear();
        s.sorted_left_rows.clear();
        s.sorted_row_no.clear();
        s.runs.clear();
    }

    ColumnsWithTypeAndName out;
    out.reserve(st.out_plan.left.size() + st.out_plan.required.size() + st.out_plan.payload.size());

    gatherLeft(st.out_plan, block, s, out);
    /// `stored_columns_index` is populated only by runPostBuild; on the header / empty-probe path there
    /// are no runs/refs, so gatherRight never dereferences the (null) base — pass it through unguarded.
    const ColumnsInfo * const * stored_columns
        = st.stored_columns_index ? st.stored_columns_index->blocksData() : nullptr;
    gatherRight(st.out_plan, stored_columns, st.payload_right_indexes, s, out);

    return IJoinResult::createFromBlock(Block(std::move(out)));
}

size_t RadixHashJoin::getTotalRowCount() const
{
    return state->total_rows;
}

size_t RadixHashJoin::getTotalByteCount() const
{
    return state->total_bytes;
}

bool RadixHashJoin::alwaysReturnsEmptySet() const
{
    return state->built.load(std::memory_order_acquire) && state->total_rows == 0;
}

IBlocksStreamPtr RadixHashJoin::getNonJoinedBlocks(
    const Block & /*left_sample_block*/, const Block & /*result_sample_block*/, UInt64 /*max_block_size*/) const
{
    /// Inner join only: no non-joined right rows.
    return {};
}

}
