#include <Interpreters/RadixHashJoin/RadixHashJoin.h>

#include <Interpreters/RadixHashJoin/BuildStore.h>
#include <Interpreters/RadixHashJoin/ColPtrTables.h>
#include <Interpreters/RadixHashJoin/LeafHashTable.h>
#include <Interpreters/RadixHashJoin/PartitionConfig.h>

#include <Interpreters/TableJoin.h>
#include <Interpreters/JoinUtils.h>
#include <Interpreters/HashJoin/HashJoin.h>  /// for canRemoveColumnsFromLeftBlock only
#include <Interpreters/castColumn.h>

#include <Core/Block.h>
#include <Core/Joins.h>
#include <Columns/IColumn.h>
#include <DataTypes/IDataType.h>

#include <Common/CurrentMetrics.h>
#include <Common/Exception.h>
#include <Common/ProfileEvents.h>
#include <Common/ThreadPool.h>

#include <atomic>
#include <cstring>

#include <unistd.h>

namespace CurrentMetrics
{
extern const Metric RadixHashJoinPoolThreads;
extern const Metric RadixHashJoinPoolThreadsActive;
extern const Metric RadixHashJoinPoolThreadsScheduled;
}

namespace ProfileEvents
{
extern const Event RadixHashProbeSelectMicroseconds;
extern const Event RadixHashProbeLookupMicroseconds;
extern const Event RadixHashProbeGatherMicroseconds;
extern const Event RadixHashProbeRows;
extern const Event RadixHashOutputRows;
}

namespace DB
{

namespace ErrorCodes
{
extern const int LOGICAL_ERROR;
}

namespace
{

/// Private per-core L2 size, used to size the leaves (spec section 5.2). 0 -> PartitionConfig fallback.
size_t detectL2Bytes()
{
#if defined(OS_LINUX) && defined(_SC_LEVEL2_CACHE_SIZE)
    if (const auto ret = ::sysconf(_SC_LEVEL2_CACHE_SIZE); ret > 0)
        return static_cast<size_t>(ret);
#endif
    return 0;
}

}

/// All radix-path state. The build store, the eagerly-built leaf hash tables + next_chain, the colptr
/// gather tables, and the precomputed output plan (which left/right columns end up in the joined block,
/// and how).
struct RadixHashJoin::RadixState
{
    RadixHash::PartitionConfig cfg;

    /// Join key columns inside the right (build) block and the left (probe) block.
    Names key_names_left;
    Names key_names_right;
    std::vector<size_t> key_positions; /// positions of the key columns in the right block (== right_sample order)
    std::vector<size_t> key_widths;    /// byte width of each key column
    std::vector<size_t> key_offsets;   /// prefix sums of key_widths (byte offset of each col in a packed key)
    size_t key_width = 0;              /// packed key width (Σ key_widths), a multiple of 4 in [4, 64]

    std::unique_ptr<RadixHash::BuildStore> build_store;

    /// Filled by runPostBuildPhase.
    std::atomic<bool> built{false};
    RadixHash::LeafHashTables leaf_hts;
    RadixHash::ColPtrTables colptr;
    std::vector<UInt64> block_base;
    UInt64 total_rows = 0;
    size_t total_bytes = 0;

    /// --- output plan (precomputed in the constructor) ---
    Block right_table_keys;                              /// the right join-key columns
    Block columns_to_add;                                /// right payload columns (right_sample_block minus keys)
    std::vector<std::string> payload_output_names;       /// renamed output name of each columns_to_add column
    Block required_right_keys;                           /// right key columns that must also appear in the output
    std::vector<std::string> required_right_keys_sources;      /// the left key column each is copied from
    std::vector<std::string> required_right_keys_output_names; /// renamed output names
    bool remove_left_columns = false;  /// analyzer: drop left columns not in the join result
    NameSet left_output_names;         /// left columns kept in the result (when remove_left_columns)
};


RadixHashJoin::RadixHashJoin(
    std::shared_ptr<TableJoin> table_join_,
    SharedHeader right_sample_block_,
    size_t max_threads_,
    std::optional<UInt64> rhs_size_estimation_,
    UInt64 max_partitions_per_pass_)
    : table_join(std::move(table_join_))
    , right_sample_block(right_sample_block_)
    , max_threads(std::max<size_t>(max_threads_, 1))
    , rhs_size_estimation(rhs_size_estimation_)
    , max_partitions_per_pass(max_partitions_per_pass_)
    , state(std::make_unique<RadixState>())
{
    /// The planner gate (radixHashJoinApplicable in PlannerJoins.cpp) guarantees all invariants below.
    /// Fail loudly if they are somehow violated rather than silently falling back.
    if (!table_join->oneDisjunct())
        throw Exception(ErrorCodes::LOGICAL_ERROR, "RadixHashJoin requires a single ON clause disjunct");

    const auto & clause = table_join->getOnlyClause();
    state->key_names_right = clause.key_names_right;
    state->key_names_left  = clause.key_names_left;

    if (state->key_names_right.empty())
        throw Exception(ErrorCodes::LOGICAL_ERROR, "RadixHashJoin requires at least one join key column");

    for (const auto & name : state->key_names_right)
    {
        const auto * col = right_sample_block->findByName(name);
        if (!col || !col->type->haveMaximumSizeOfValue())
            throw Exception(
                ErrorCodes::LOGICAL_ERROR, "RadixHashJoin: key column '{}' is not fixed-width", name);
        const size_t w = col->type->getMaximumSizeOfValueInMemory();
        if (w == 0 || w % 4 != 0)
            throw Exception(
                ErrorCodes::LOGICAL_ERROR,
                "RadixHashJoin: key column '{}' has width {} which is not a multiple of 4", name, w);
        state->key_positions.push_back(right_sample_block->getPositionByName(name));
        state->key_widths.push_back(w);
        state->key_width += w;
    }
    if (state->key_width < 4 || state->key_width > 64 || state->key_width % 4 != 0)
        throw Exception(
            ErrorCodes::LOGICAL_ERROR,
            "RadixHashJoin: packed key width {} must be a multiple of 4 in [4, 64]", state->key_width);

    state->key_offsets.assign(state->key_widths.size(), 0);
    for (size_t c = 1; c < state->key_widths.size(); ++c)
        state->key_offsets[c] = state->key_offsets[c - 1] + state->key_widths[c - 1];

    state->cfg = RadixHash::PartitionConfig::make(rhs_size_estimation, detectL2Bytes(), max_partitions_per_pass);
    state->build_store = std::make_unique<RadixHash::BuildStore>(
        state->cfg, state->key_positions, state->key_widths, max_threads);

    /// Output plan: split right keys vs payload, the required right keys (copied from the left side),
    /// and which left columns survive into the result (spec section 5.4, analyzer column rules).
    JoinCommon::splitAdditionalColumns(
        state->key_names_right, *right_sample_block, state->right_table_keys, state->columns_to_add);

    state->payload_output_names.reserve(state->columns_to_add.columns());
    for (const auto & col : state->columns_to_add)
        state->payload_output_names.push_back(table_join->renamedRightColumnName(col.name));

    state->required_right_keys = table_join->getRequiredRightKeys(
        state->right_table_keys, state->required_right_keys_sources);
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

bool RadixHashJoin::addBlockToJoin(const Block & block, bool /*check_limits*/)
{
    /// Normalise to the right_sample_block structure (by name, in order) so the key columns sit at the
    /// BuildStore key positions and every payload column is gatherable by name later. Materialise so the
    /// key columns expose contiguous raw data and the stored payload column types match `columns_to_add`.
    ColumnsWithTypeAndName cols;
    cols.reserve(right_sample_block->columns());
    for (const auto & sample_col : *right_sample_block)
        cols.push_back(block.getByName(sample_col.name));
    Block normalized = materializeBlock(Block(std::move(cols)));

    state->build_store->add(normalized);
    return true;
}

bool RadixHashJoin::addBlockToJoin(const Block & block, size_t /*num_rows*/, bool check_limits)
{
    return addBlockToJoin(block, check_limits);
}

void RadixHashJoin::checkTypesOfKeys(const Block & block) const
{
    JoinCommon::checkTypesOfKeys(block, state->key_names_left, *right_sample_block, state->key_names_right);
}

void RadixHashJoin::onBuildPhaseFinish()
{
    state->build_store->finishBuild();
}

bool RadixHashJoin::hasPostBuildPhase() const
{
    return true;
}

void RadixHashJoin::runPostBuildPhase()
{
    ThreadPool pool(
        CurrentMetrics::RadixHashJoinPoolThreads,
        CurrentMetrics::RadixHashJoinPoolThreadsActive,
        CurrentMetrics::RadixHashJoinPoolThreadsScheduled,
        max_threads);

    /// Deferred exact key+ref scatter, additionally carrying the per-row routing hash into the leaves so
    /// the leaf-HT bucket Fibonacci-mixes the exact hash the probe side derives (spec section 5.6).
    RadixHash::LeafArrays leaves = state->build_store->scatterToLeaves(
        pool, max_threads, /*with_leaf_hash=*/true);

    state->block_base = state->build_store->blockBase();
    state->total_rows = state->build_store->totalRows();

    /// Build the per-leaf hash tables + the shared next_chain (THP-backed: the random inserts/lookups
    /// are the most TLB-sensitive structure, spec section 4.4 / open question Q1).
    state->leaf_hts = RadixHash::buildLeafHashTables(
        leaves, state->block_base, state->total_rows, state->key_width, pool, max_threads, /*use_thp=*/true);

    /// Per-column/per-block gather pointers for the build payload (spec section 5.4).
    state->colptr.build(
        state->build_store->blocks(), state->columns_to_add, state->payload_output_names);

    state->total_bytes = state->leaf_hts.arena.bytesReserved();

    /// `leaves` (the scattered key/ref/hash arrays) is no longer needed once the HT is built.
    leaves = RadixHash::LeafArrays();

    state->built.store(true, std::memory_order_release);
}

JoinResultPtr RadixHashJoin::joinBlock(Block block)
{
    const RadixState & st = *state;
    const size_t n = block.rows();

    /// Materialise the probe block so the key columns expose contiguous raw data (getRawData) and the
    /// emitted left columns are full — matching the materialised header (JoiningTransform::transformHeader
    /// runs materializeBlockInplace on the result of joinBlock).
    block = materializeBlock(block);

    std::vector<UInt32> left_rows; /// output row -> probe-block row index
    std::vector<RadixShuffle::BuildRef> refs; /// output row -> matched build ref

    const bool can_probe = st.built.load(std::memory_order_acquire) && n > 0;
    if (can_probe)
    {
        /// Phase 1 — selector: one chained 32-bit computeHashInto over the left key columns.
        Stopwatch sw_sel;
        std::vector<UInt32> hashes(n, 0);
        for (size_t c = 0; c < st.key_names_left.size(); ++c)
            block.getByName(st.key_names_left[c]).column->computeHashInto(
                0, n, hashes.data(), /*initial=*/c == 0);

        /// Pack the left key row-major to the same layout the build side scattered. Single-column keys
        /// use the column's raw data directly (zero-copy fast path).
        const void * packed_ptr = nullptr;
        std::vector<char> packed;
        if (st.key_widths.size() == 1)
        {
            packed_ptr = block.getByName(st.key_names_left[0]).column->getRawData().data();
        }
        else
        {
            packed.resize(n * st.key_width);
            for (size_t c = 0; c < st.key_widths.size(); ++c)
            {
                const char * src = block.getByName(st.key_names_left[c]).column->getRawData().data();
                const size_t w   = st.key_widths[c];
                const size_t off = st.key_offsets[c];
                for (size_t r = 0; r < n; ++r)
                    std::memcpy(packed.data() + r * st.key_width + off, src + r * w, w);
            }
            packed_ptr = packed.data();
        }
        ProfileEvents::increment(ProfileEvents::RadixHashProbeSelectMicroseconds, sw_sel.elapsedMicroseconds());

        /// Phase A — direct leaf-HT lookup + next_chain chain traversal (JOIN ALL).
        Stopwatch sw_lookup;
        RadixHash::collectMatches(
            st.key_width, st.leaf_hts.leaves.data(), st.cfg.shift, st.cfg.total_bits,
            st.block_base.data(), hashes.data(), packed_ptr, n, left_rows, refs);
        ProfileEvents::increment(ProfileEvents::RadixHashProbeLookupMicroseconds, sw_lookup.elapsedMicroseconds());

        ProfileEvents::increment(ProfileEvents::RadixHashProbeRows, n);
        ProfileEvents::increment(ProfileEvents::RadixHashOutputRows, left_rows.size());
    }

    /// Phase 4 — emit: gather the matched left + build columns into one output block. The schema is
    /// produced even when there are no rows (header computation), so downstream sees a stable header.
    Stopwatch sw_gather;
    const size_t out_rows = left_rows.size();
    ColumnsWithTypeAndName out_cols;

    /// (a) Left output columns — each replicated via the gathered left-row indices.
    for (const auto & col : block)
    {
        if (st.remove_left_columns && !st.left_output_names.contains(col.name))
            continue;
        auto new_col = col.type->createColumn();
        new_col->reserve(out_rows);
        for (size_t m = 0; m < out_rows; ++m)
            new_col->insertFrom(*col.column, left_rows[m]);
        out_cols.emplace_back(std::move(new_col), col.type, col.name);
    }

    Block result(std::move(out_cols));

    /// (b) Build payload columns — gathered by BuildRef through the colptr tables.
    for (size_t c = 0; c < st.columns_to_add.columns(); ++c)
    {
        const std::string & out_name = st.payload_output_names[c];
        if (result.has(out_name))
            continue; /// already provided by a left column (AddedColumns rule)
        const auto & type = st.columns_to_add.getByPosition(c).type;
        auto col = type->createColumn();
        col->reserve(out_rows);
        if (out_rows > 0)
        {
            const auto & by_block = st.colptr.payload[c].by_block;
            for (size_t m = 0; m < out_rows; ++m)
            {
                const RadixShuffle::BuildRef ref = refs[m];
                col->insertFrom(*by_block[ref.block_no], ref.row_no - 1); /// row_no is 1-based
            }
        }
        result.insert(ColumnWithTypeAndName(std::move(col), type, out_name));
    }

    /// (c) Required right key columns: an equi-join match has right_key == left_key, so copy from
    /// the corresponding left key column and cast to the right key type (spec / HashJoin parity).
    for (size_t i = 0; i < st.required_right_keys.columns(); ++i)
    {
        const std::string & out_name = st.required_right_keys_output_names[i];
        if (result.has(out_name))
            continue;
        const auto & right_key = st.required_right_keys.getByPosition(i);
        const auto & left_src  = block.getByName(st.required_right_keys_sources[i]);

        auto tmp = left_src.type->createColumn();
        tmp->reserve(out_rows);
        for (size_t m = 0; m < out_rows; ++m)
            tmp->insertFrom(*left_src.column, left_rows[m]);

        ColumnWithTypeAndName tmp_col(std::move(tmp), left_src.type, out_name);
        ColumnPtr casted = castColumn(tmp_col, right_key.type);
        result.insert(ColumnWithTypeAndName(casted, right_key.type, out_name));
    }

    ProfileEvents::increment(ProfileEvents::RadixHashProbeGatherMicroseconds, sw_gather.elapsedMicroseconds());

    return IJoinResult::createFromBlock(std::move(result));
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
    /// Inner join: an empty build side yields an empty result. Only meaningful once build is finished.
    return state->built.load(std::memory_order_acquire) && state->total_rows == 0;
}

IBlocksStreamPtr RadixHashJoin::getNonJoinedBlocks(
    const Block & /*left_sample_block*/, const Block & /*result_sample_block*/, UInt64 /*max_block_size*/) const
{
    /// Inner join only in v1: there are no non-joined right rows.
    return {};
}

}
