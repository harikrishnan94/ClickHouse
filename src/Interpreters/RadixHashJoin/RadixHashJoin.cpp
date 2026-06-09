#include <Interpreters/RadixHashJoin/RadixHashJoin.h>

#include <Interpreters/RadixHashJoin/BuildStore.h>
#include <Interpreters/RadixHashJoin/ColPtrTables.h>
#include <Interpreters/RadixHashJoin/KeyPacking.h>
#include <Interpreters/RadixHashJoin/LeafHashTable.h>
#include <Interpreters/RadixHashJoin/PartitionConfig.h>
#include <Interpreters/RadixHashJoin/RapidHash.h>

#include <Interpreters/TableJoin.h>
#include <Interpreters/JoinUtils.h>
#include <Interpreters/HashJoin/HashJoin.h>  /// for canRemoveColumnsFromLeftBlock only
#include <Interpreters/castColumn.h>

#include <Core/Block.h>
#include <Core/Joins.h>
#include <Columns/IColumn.h>
#include <Columns/ColumnsNumber.h>
#include <DataTypes/IDataType.h>

#include <Common/Exception.h>
#include <Common/ProfileEvents.h>

#include <atomic>
#include <cstring>
#include <numeric>

#include <unistd.h>

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
    std::vector<RadixHash::PackKeyColumnFn> key_packers; /// per-column width-specialized packer (shared with BuildStore)
    size_t key_width = 0;              /// packed key width (Σ key_widths), a multiple of 4 in [4, 64]

    std::unique_ptr<RadixHash::BuildStore> build_store;

    /// Set in onBuildPhaseFinish (after finishBuild); guards the header/planning path in ensureBuilt.
    std::atomic<bool> build_phase_finished{false};
    /// Cooperative pool used to run the post-build (scatter + HT build) on the probe threads.
    RadixHash::CoopPool coord;

    /// Set to true once the leaf HTs are fully built and ready for probing.
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

    state->key_offsets.resize(state->key_widths.size());
    std::exclusive_scan(state->key_widths.begin(), state->key_widths.end(), state->key_offsets.begin(), size_t{0});

    /// One width-specialized packer per key column (same table the build side uses, KeyPacking.h), so the
    /// probe packs composite keys row-major to the identical layout with no runtime-width memcpy.
    state->key_packers.reserve(state->key_widths.size());
    for (size_t w : state->key_widths)
        state->key_packers.push_back(RadixHash::chooseKeyPacker(w));

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
    state->build_store->finishBuild();
    state->build_phase_finished.store(true, std::memory_order_release);
}

void RadixHashJoin::runPostBuild()
{
    /// Deferred exact key+ref scatter. The routing hash (top bits of the RapidHash) is recomputed from the
    /// key columns inside the scatter (not stored per row); the leaf-HT bucket is recomputed from the key
    /// (RapidHash low bits), so no per-row hash is scattered to the leaves.
    RadixHash::LeafArrays leaves = state->build_store->scatterToLeaves(state->coord);

    state->block_base = state->build_store->blockBase();
    state->total_rows = state->build_store->totalRows();

    /// Build the per-leaf hash tables + the shared next_chain. The arena is jemalloc-backed (no
    /// mmap/THP); allocation + zeroing of the cells and next_chain run in parallel across the workers.
    state->leaf_hts = RadixHash::buildLeafHashTables(
        leaves, state->block_base, state->total_rows, state->key_width, state->coord);

    /// Per-column/per-block gather pointers for the build payload (spec section 5.4).
    state->colptr.build(
        state->build_store->blocks(), state->columns_to_add, state->payload_output_names);

    state->total_bytes = state->leaf_hts.arena.bytesReserved();

    /// `leaves` (the scattered key/ref/hash arrays) is no longer needed once the HT is built.
    leaves = RadixHash::LeafArrays();

    state->built.store(true, std::memory_order_release);
}

void RadixHashJoin::ensureBuilt()
{
    /// Fast path: already built.
    if (state->built.load(std::memory_order_acquire))
        return;

    /// Header/planning path: transformHeader calls joinBlock before onBuildPhaseFinish.
    /// The build barrier has not run yet — return safely so joinBlock emits only the output schema.
    if (!state->build_phase_finished.load(std::memory_order_acquire))
        return;

    /// Cooperative post-build: first probe thread is the leader and runs runPostBuild();
    /// subsequent probe threads act as helpers draining parallelFor work units.
    state->coord.run([this] { runPostBuild(); });
}

JoinResultPtr RadixHashJoin::joinBlock(Block block)
{
    /// Cooperative post-build: on the first real call after onBuildPhaseFinish, the probe threads
    /// collectively run the scatter + leaf-HT build before probing.
    ensureBuilt();

    const RadixState & st = *state;
    const size_t n = block.rows();

    /// Materialise the probe block so the key columns expose contiguous raw data (getRawData) and the
    /// emitted left columns are full — matching the materialised header (JoiningTransform::transformHeader
    /// runs materializeBlockInplace on the result of joinBlock).
    block = materializeBlock(block);

    const bool can_probe = st.built.load(std::memory_order_acquire) && n > 0;

    /// --- Output plan (computed once; the schema is produced even for an empty/header block). Preserves
    /// the previous column order and dedup rules: left columns; then payload columns whose output name is
    /// not already provided by a left column; then required right-key columns not already present. No
    /// per-row materialisation here — left/required columns are gathered in bulk by index() at the end.
    struct LeftOut  { const IColumn * src; DataTypePtr type; String name; };
    struct PayOut   { size_t payload_idx; DataTypePtr type; String name; };
    struct ReqOut   { const IColumn * left_src; DataTypePtr left_type; DataTypePtr right_type; String name; };
    std::vector<LeftOut> left_out;
    std::vector<PayOut> pay_out;
    std::vector<ReqOut> req_out;
    NameSet emitted_names;

    for (const auto & col : block)
    {
        if (st.remove_left_columns && !st.left_output_names.contains(col.name))
            continue;
        left_out.push_back({col.column.get(), col.type, col.name});
        emitted_names.insert(col.name);
    }
    for (size_t col = 0; col < st.columns_to_add.columns(); ++col)
    {
        const std::string & out_name = st.payload_output_names[col];
        if (emitted_names.contains(out_name))
            continue; /// already provided by a left column (AddedColumns rule)
        pay_out.push_back({col, st.columns_to_add.getByPosition(col).type, out_name});
        emitted_names.insert(out_name);
    }
    for (size_t key_col = 0; key_col < st.required_right_keys.columns(); ++key_col)
    {
        const std::string & out_name = st.required_right_keys_output_names[key_col];
        if (emitted_names.contains(out_name))
            continue;
        const auto & right_key = st.required_right_keys.getByPosition(key_col);
        const auto & left_src  = block.getByName(st.required_right_keys_sources[key_col]);
        req_out.push_back({left_src.column.get(), left_src.type, right_key.type, out_name});
        emitted_names.insert(out_name);
    }

    /// Match buffers (global probe-row index + matched BuildRef, in chain order). Reserved to the inner-join
    /// lower bound (exactly `n` at 1:1) so the lookup does not re-grow them.
    std::vector<UInt32> left_rows;
    std::vector<RadixShuffle::BuildRef> refs;
    left_rows.reserve(n);
    refs.reserve(n);
    UInt64 sel_us = 0;
    UInt64 lookup_us = 0;

    if (can_probe)
    {
        /// Tile the selector + lookup in TILE-row tiles so the packed-key and hash scratch stay L1-resident
        /// and the hashes never round-trip to DRAM between selector and lookup. TILE matches the build
        /// scatter's chunk size (BuildStore::SCATTER_CHUNK_ROWS). Scratch is thread-local, reused across
        /// tiles AND joinBlock calls (the probe is per-stream, single-threaded per instance). The emit
        /// (gather) is done ONCE after the loop in bulk, so it is not tiled.
        constexpr size_t tile_rows = RadixHash::BuildStore::SCATTER_CHUNK_ROWS; /// 1024
        thread_local std::vector<char> packed_scratch;
        thread_local std::vector<UInt64> hashes_scratch;
        thread_local std::vector<UInt32> tile_left;       /// tile-local probe row index (0..tn)
        thread_local std::vector<RadixShuffle::BuildRef> tile_refs;

        const bool single_col = st.key_widths.size() == 1;
        const char * single_raw = nullptr;
        std::vector<const char *> kcol_src; /// raw bases of each key column (multi-column only)
        if (single_col)
        {
            single_raw = block.getByName(st.key_names_left[0]).column->getRawData().data();
        }
        else
        {
            packed_scratch.resize(tile_rows * st.key_width);
            kcol_src.reserve(st.key_widths.size());
            for (const auto & name : st.key_names_left)
                kcol_src.push_back(block.getByName(name).column->getRawData().data());
        }
        hashes_scratch.resize(tile_rows);

        const bool has_chain = st.leaf_hts.next_chain != nullptr;

        for (size_t tile_start = 0; tile_start < n; tile_start += tile_rows)
        {
            const size_t tn = std::min(tile_rows, n - tile_start);

            /// Phase 1 — pack (multi-column) + 64-bit RapidHash of each packed key ONCE into the reused
            /// scratch (the identical function/bytes the build used, so a key routes to the same leaf and
            /// bucket on both sides). Single-column keys point straight at the column's raw data (zero-copy).
            Stopwatch sw_sel;
            const char * keys = nullptr;
            if (single_col)
            {
                keys = single_raw + tile_start * st.key_width;
            }
            else
            {
                char * dst = packed_scratch.data();
                for (size_t c = 0; c < st.key_widths.size(); ++c)
                    st.key_packers[c](kcol_src[c], tile_start, tn, dst, st.key_width, st.key_offsets[c], st.key_widths[c]);
                keys = dst;
            }
            UInt64 * hashes = hashes_scratch.data();
            switch (st.key_width)
            {
#define RHJ_PROBE_HASH(W) \
                case W: for (size_t i = 0; i < tn; ++i) hashes[i] = rapidhash::hash<W>(keys + i * (W)); break;
                RHJ_PROBE_HASH(4)  RHJ_PROBE_HASH(8)  RHJ_PROBE_HASH(12) RHJ_PROBE_HASH(16)
                RHJ_PROBE_HASH(20) RHJ_PROBE_HASH(24) RHJ_PROBE_HASH(28) RHJ_PROBE_HASH(32)
                RHJ_PROBE_HASH(36) RHJ_PROBE_HASH(40) RHJ_PROBE_HASH(44) RHJ_PROBE_HASH(48)
                RHJ_PROBE_HASH(52) RHJ_PROBE_HASH(56) RHJ_PROBE_HASH(60) RHJ_PROBE_HASH(64)
#undef RHJ_PROBE_HASH
                default:
                    throw Exception(
                        ErrorCodes::LOGICAL_ERROR,
                        "RadixHashJoin: unsupported key width {} (multiple of 4 in [4, 64])", st.key_width);
            }
            sel_us += sw_sel.elapsedMicroseconds();

            /// Phase A — direct leaf-HT lookup for this tile into the reused tile buffers, then append to
            /// the global match buffers with the tile offset applied to the probe-row index.
            Stopwatch sw_lookup;
            tile_left.clear();
            tile_refs.clear();
            RadixHash::collectMatches(
                st.key_width, has_chain,
                st.leaf_hts.leaves.data(), st.cfg.shift, st.cfg.total_bits,
                st.block_base.data(), hashes, keys, tn, tile_left, tile_refs);
            const auto tile_base = static_cast<UInt32>(tile_start);
            for (UInt32 r : tile_left)
                left_rows.push_back(tile_base + r);
            refs.insert(refs.end(), tile_refs.begin(), tile_refs.end());
            lookup_us += sw_lookup.elapsedMicroseconds();
        }

        ProfileEvents::increment(ProfileEvents::RadixHashProbeSelectMicroseconds, sel_us);
        ProfileEvents::increment(ProfileEvents::RadixHashProbeLookupMicroseconds, lookup_us);
        ProfileEvents::increment(ProfileEvents::RadixHashProbeRows, n);
        ProfileEvents::increment(ProfileEvents::RadixHashOutputRows, left_rows.size());
    }

    /// Phase 4 — emit. Left and required-key columns are gathered in BULK via the type-specialised
    /// IColumn::index (precedent: HashJoin's ScatteredBlock / replicate paths) instead of per-row virtual
    /// insertFrom — the gathered probe rows are random across the block but the gather is one specialised
    /// call per column. The build payload is gathered per matched BuildRef (random across build blocks; the
    /// only per-row path, as in HashJoin's AddedColumns). The schema is produced even with 0 matches.
    Stopwatch sw_gather;
    const size_t out_rows = left_rows.size();

    auto index_col = ColumnUInt32::create();
    index_col->getData().insert(left_rows.begin(), left_rows.end());

    ColumnsWithTypeAndName out_cols;
    out_cols.reserve(left_out.size());
    for (const auto & lo : left_out)
        out_cols.emplace_back(lo.src->index(*index_col, 0), lo.type, lo.name);
    Block result(std::move(out_cols));

    for (const auto & po : pay_out)
    {
        auto col_data = po.type->createColumn();
        col_data->reserve(out_rows);
        if (out_rows > 0)
        {
            const auto & by_block = st.colptr.payload[po.payload_idx].by_block;
            for (size_t m = 0; m < out_rows; ++m)
            {
                const RadixShuffle::BuildRef ref = refs[m];
                col_data->insertFrom(*by_block[ref.block_no], ref.row_no); /// row_no is 0-based
            }
        }
        result.insert(ColumnWithTypeAndName(std::move(col_data), po.type, po.name));
    }

    for (const auto & ro : req_out)
    {
        /// Equi-join match has right_key == left_key: gather the left source in bulk, then cast to the
        /// right key type — skipping the cast entirely when the types already match (the common case after
        /// join-key type unification).
        ColumnPtr gathered = ro.left_src->index(*index_col, 0);
        if (ro.left_type->equals(*ro.right_type))
        {
            result.insert(ColumnWithTypeAndName(gathered, ro.right_type, ro.name));
        }
        else
        {
            ColumnWithTypeAndName tmp_col(gathered, ro.left_type, ro.name);
            ColumnPtr casted = castColumn(tmp_col, ro.right_type);
            result.insert(ColumnWithTypeAndName(casted, ro.right_type, ro.name));
        }
    }

    if (can_probe)
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
