#!/usr/bin/env python3
"""The loop enumeration, as data.

This is the single source of truth for Gate G0.1. Every entry was written from
reading the code (the file:line ranges are the evidence), not from looking at
profiler output -- which is what gives the gate the power to fail: if the reading
missed a loop, a sampled symbol will fail to map and the gate goes red.

`symbols` are regexes matched against the *demangled leaf in-join frame*. A loop may
legitimately own several symbols (a template instantiated per key getter, a helper
that got outlined). A symbol may legitimately map to several loops when the compiler
inlined them into one function -- that is recorded as `shared_symbol` on the loop and
is a fact about the codegen, not a defect in the enumeration.
"""

from __future__ import annotations

# --------------------------------------------------------------------------
# What counts as "inside the join". Fixed in PREREG P0.1 before samples were
# looked at, so it cannot be tuned to make the gate pass.
# --------------------------------------------------------------------------

IN_JOIN_MARKERS = [
    r"DB::HashJoin",
    r"DB::Unified::",
    r"DB::ConcurrentHashJoin",
    r"DB::JoinStuff",
    r"NotJoinedHash",
    r"AddedColumns",
    r"HashJoinResult",
    r"RowRefList",
    r"StoredColumnsIndex",
    r"TwoLevelHashTable",
    r"HashTable<",
    r"HashMapTable",
    r"ColumnsHashing",
    r"JoiningTransform",
    r"FillingRightJoinSide",
    r"NonJoinedBlocksTransform",
    r"DB::JoinCommon",
    r"ScatteredBlock",
    r"KnownRowsHolder",
    r"joinDispatch",
    r"SpillingHashJoin",
]

# --------------------------------------------------------------------------
# The enumeration
# --------------------------------------------------------------------------
# fields: id, name, subphase, impls, multiplicity, formula, refs, symbols, note

HASH, PAR, UNI = "hash", "parallel_hash", "unified_hash"
ALL3 = [HASH, PAR, UNI]

LOOPS = [
    # ---------------- build: pre-insert per-block/per-row preparation --------
    dict(id="B0", name="right-block materialisation (const/sparse/LC -> full)",
         subphase="build prep", impls=ALL3, mult="per column per block",
         formula="D_cols * B",
         refs="JoinUtils.cpp:124-156 (all); ConcurrentHashJoin.cpp:304",
         symbols=[r"::materializeColumnsFromRightBlock", r"DB::JoinCommon::materializeColumn"],
         note="shared helper, identical for all three"),
    dict(id="B1", name="nullable key unfold + null-map OR",
         subphase="build prep", impls=ALL3, mult="per row when >=2 nullable keys, else per column",
         formula="R*B if nullable_keys>=2 else D_cols*B",
         refs="NullableUtils.cpp:84-109",
         symbols=[r"DB::extractNestedColumnsAndNullMap"],
         note="shared helper"),
    dict(id="B2", name="null-in-selector scan (RIGHT/FULL null-key save)",
         subphase="build prep", impls=[HASH, UNI], mult="per row (early exit)",
         formula="<=R*B",
         refs="HashJoin/HashJoin.cpp:758-765; UnifiedHashJoin/HashJoin.cpp:915-922",
         symbols=[r"DB::HashJoin::addBlockToJoin", r"DB::Unified::HashJoin::addBlockToJoin"],
         shared_symbol=True,
         note="inlined into addBlockToJoin; shares its symbol with B3, B12"),
    dict(id="B3", name="not_joined_map mark for ON-filtered rows",
         subphase="build prep", impls=[HASH, UNI], mult="per row",
         formula="R*B",
         refs="HashJoin/HashJoin.cpp:789-790; UnifiedHashJoin/HashJoin.cpp:946-947",
         symbols=[r"DB::HashJoin::addBlockToJoin", r"DB::Unified::HashJoin::addBlockToJoin"],
         shared_symbol=True, note="inlined into addBlockToJoin"),
    dict(id="B4", name="per-row used-flags reinit at build",
         subphase="build prep", impls=[HASH, UNI], mult="per row (x2: set then clear)",
         formula="2*R*B",
         refs="HashJoin/JoinUsedFlags.h:81-84; UnifiedHashJoin/JoinUsedFlags.h:83-86",
         symbols=[r"JoinUsedFlags::reinit", r"DB::JoinStuff::JoinUsedFlags"],
         note="flagged kinds only"),

    # ---------------- build: routing / scatter -------------------------------
    dict(id="B5", name="per-row routing hash (calculateHashes)",
         subphase="build scatter", impls=[PAR], mult="per row",
         formula="R*B",
         refs="ConcurrentHashJoin.cpp:605-606",
         symbols=[r"calculateHashes", r"DB::ConcurrentHashJoin::addBlockToJoin",
                  r"DB::ConcurrentHashJoin::dispatchBlock"],
         note="no counterpart in hash; unified's counterpart is B7 (fused into scatter)"),
    dict(id="B6", name="per-row hash -> slot selector",
         subphase="build scatter", impls=[PAR], mult="per row",
         formula="R*B",
         refs="ConcurrentHashJoin.cpp:588-594 hashToSelector",
         symbols=[r"hashToSelector", r"DB::ConcurrentHashJoin::addBlockToJoin",
                  r"DB::ConcurrentHashJoin::dispatchBlock"],
         shared_symbol=True, note="no counterpart in hash"),
    dict(id="B7", name="per-row bucket scatter pass 1: hash + per-bucket count",
         subphase="build scatter", impls=[UNI], mult="per row per clause",
         formula="R*B*C",
         refs="UnifiedHashJoin/HashJoinMethodsImpl.h:180-194",
         symbols=[r"Unified::HashJoinMethods.*scatterByBucket",
                  r"Unified::HashJoinMethods.*insertFromBlockImpl"],
         note="skipped entirely when num_buckets==1 (HashJoin.cpp:967-970)"),
    dict(id="B8", name="per-row bucket scatter pass 2: place row index",
         subphase="build scatter", impls=[UNI], mult="per row per clause",
         formula="R*B*C",
         refs="UnifiedHashJoin/HashJoinMethodsImpl.h:205-206",
         symbols=[r"Unified::HashJoinMethods.*scatterByBucket"],
         shared_symbol=True, note="same symbol as B7"),
    dict(id="B9", name="per-row scatter into per-slot blocks (copy or index)",
         subphase="build scatter", impls=[PAR], mult="per row (x columns on copy path)",
         formula="R*B (indices) or R*B*cols (copy)",
         refs="ConcurrentHashJoin.cpp:694-701 (copy), :719-723 (indices)",
         symbols=[r"scatterBlocksByCopying", r"scatterBlocksWithSelector",
                  r"DB::IColumn::scatter", r"::scatter\(", r"scatterImpl",
                  r"selectDispatchBlock", r"DB::ConcurrentHashJoin::dispatchBlock"],
         note="no counterpart: unified scatters row indices only, never copies columns"),
    dict(id="B10", name="per-bucket index-array reserve",
         subphase="build scatter", impls=[UNI], mult="per bucket per block per clause",
         formula="K*B*C",
         refs="UnifiedHashJoin/HashJoinMethodsImpl.h:198-203",
         symbols=[r"Unified::HashJoinMethods.*scatterByBucket"],
         shared_symbol=True, note="same symbol as B7"),

    # ---------------- build: insert ------------------------------------------
    dict(id="B11", name="per-row emplace main loop (null/mask skip + Inserter)",
         subphase="build insert", impls=ALL3, mult="per inserted row per clause",
         formula="R'*B*C",
         refs="HashJoin/HashJoinMethodsImpl.h:284-310; UnifiedHashJoin/HashJoinMethodsImpl.h:387-417",
         symbols=[r"HashJoinMethods.*insertFromBlockImplTypeCase",
                  r"HashJoinMethods.*insertFromBlockImpl",
                  r"Unified::HashJoinMethods.*insertFromBlockImpl",
                  r"::Inserter", r"insertOne", r"insertAll", r"insertAsof"],
         note="parallel_hash reaches the hash copy of this via its slot's HashJoin"),
    dict(id="B12", name="per-row build software prefetch",
         subphase="build insert", impls=ALL3, mult="per inserted row",
         formula="R'*B",
         refs="HashJoin/HashJoinMethodsImpl.h:277-287; UnifiedHashJoin/HashJoinMethodsImpl.h:376-390",
         symbols=[r"HashJoinMethods.*insertFromBlockImpl", r"prefetch"],
         shared_symbol=True, note="same loop as B11; ablatable by setting"),
    dict(id="B13", name="hash-table linear probe chain on emplace",
         subphase="build insert", impls=ALL3, mult="per inserted row (x chain length)",
         formula="R'*B*avg_chain",
         refs="HashTable.h:446-454 findCell; emplace path",
         symbols=[r"HashTable<.*>::(findCell|emplace|resize|emplaceNonZero)",
                  r"HashMapTable", r"TwoLevelHashTable<.*>::emplace",
                  r"::grower", r"HashTableGrower"],
         note="same container code for all three; differs only in table size/count"),
    dict(id="B14", name="RowRefList append + arena allocation",
         subphase="build insert", impls=ALL3, mult="per duplicate-key row",
         formula="(R'-distinct)*B",
         refs="RowRefs.h:207-277",
         symbols=[r"RowRefList", r"DB::Arena", r"Arena::alloc", r"ArenaWithFreeLists"],
         note="unified uses a per-bucket arena, hash/parallel one arena per (slot's) join"),

    # ---------------- build: locking / accounting ----------------------------
    dict(id="B15", name="per-block per-bucket try_lock drain loop",
         subphase="build insert", impls=[UNI], mult="per non-empty bucket per block per clause",
         formula="<=K*B*C successful holds; attempts can exceed this",
         refs="UnifiedHashJoin/HashJoin.cpp:156-163 (pending scan), :177-195 (drain)",
         symbols=[r"Unified::.*insertIntoBuckets", r"Unified::HashJoin::addBlockToJoin"],
         note="lock L1; analysed in table (B)"),
    dict(id="B16", name="per-bucket byte-delta accounting (2 x 21-way switch + atomic)",
         subphase="build insert", impls=[UNI], mult="per locked bucket per block per clause",
         formula="<=K*B*C",
         refs="UnifiedHashJoin/HashJoin.cpp:139-147",
         symbols=[r"Unified::.*insertIntoBuckets", r"getBucketBufferSizeInBytes",
                  r"Unified::.*bucketBytes"],
         shared_symbol=True, note="inside the bucket critical section"),
    dict(id="B17", name="per-slot try_lock drain loop",
         subphase="build insert", impls=[PAR], mult="<= S passes per block",
         formula="<=S*B attempts, B successful inserts",
         refs="ConcurrentHashJoin.cpp:325-361",
         symbols=[r"DB::ConcurrentHashJoin::addBlockToJoin"],
         note="lock L2; analysed in table (B)"),
    dict(id="B18", name="blocks_mutex block registration",
         subphase="build insert", impls=[UNI], mult="per block",
         formula="B (+0-3 extra per block)",
         refs="UnifiedHashJoin/HashJoin.cpp:863-884",
         symbols=[r"Unified::HashJoin::addBlockToJoin", r"StoredColumnsIndex"],
         note="lock L3"),

    # ---------------- build: post-build --------------------------------------
    dict(id="B19", name="post-build bucket-capacity prefix pass",
         subphase="build finalise", impls=[UNI], mult="per bucket, once per build",
         formula="K per map, once",
         refs="TwoLevelHashTable.h:97-101 BucketPrefixSums::compute; "
              "UnifiedHashJoin/HashJoin.cpp:2087-2098 freezeMapsForProbing, :2482",
         symbols=[r"BucketPrefixSums", r"Unified::HashJoin::freezeMapsForProbing",
                  r"computeBucketPrefix"],
         note="O(buckets), not O(rows) -- enumerated because the mission asks for it"),
    dict(id="B20", name="post-build slot->slot0 bucket merge",
         subphase="build finalise", impls=[PAR], mult="per bucket per slot, once per build",
         formula="(K/S)*(S-1), once",
         refs="ConcurrentHashJoin.cpp:817-833",
         symbols=[r"DB::ConcurrentHashJoin::onBuildPhaseFinish"],
         note="no counterpart: unified shares one map from the start"),
    dict(id="B21", name="owned-bucket pre-reserve before first insert",
         subphase="build finalise", impls=[PAR], mult="per owned bucket, once per slot",
         formula="(K/S) per slot, once",
         refs="ConcurrentHashJoin.cpp:162-163 reserveSpaceInHashMaps",
         symbols=[r"reserveSpaceInHashMaps"],
         note="no counterpart: hash/unified grow lazily"),

    # ---------------- probe ---------------------------------------------------
    # P0 was added after G0.1 went red on its symbols. The reading had treated
    # per-block probe setup as scaffolding; the profiler shows it holding real
    # samples, and JoinOnKeyColumns does per-column work proportional to clauses,
    # so it is enumerated rather than excluded. See WORKLOG F2.
    dict(id="P0", name="per-block probe setup (joinBlockImpl + JoinOnKeyColumns)",
         subphase="probe lookup", impls=ALL3, mult="per probe block per clause",
         formula="B_probe*C",
         refs="HashJoin/HashJoinMethodsImpl.h:142-148; UnifiedHashJoin/...:235-241",
         symbols=[r"HashJoinMethods.*joinBlockImpl", r"JoinOnKeyColumns",
                  r"::materializeColumnsFromLeftBlock"],
         note="per block, not per row; enumerated because it received samples"),
    dict(id="P1", name="probe main per-row loop (joinRightColumns)",
         subphase="probe lookup", impls=ALL3, mult="per probe row",
         formula="R_probe",
         refs="HashJoin/HashJoinMethodsImpl.h:594-631; UnifiedHashJoin/HashJoinMethodsImpl.h:704-741",
         symbols=[r"HashJoinMethods.*joinRightColumns",
                  r"Unified::HashJoinMethods.*joinRightColumns"],
         note="the dominant probe loop; everything below is inlined into it"),
    dict(id="P2", name="per-row adaptive lookahead prefetch",
         subphase="probe lookup", impls=ALL3, mult="per probe row",
         formula="R_probe",
         refs="HashJoin/HashJoinMethodsImpl.h:587-597; UnifiedHashJoin/HashJoinMethodsImpl.h:697-707",
         symbols=[r"HashJoinMethods.*joinRightColumns", r"prefetchByHash", r"Prober.*prefetch"],
         shared_symbol=True, note="unified prefetches through Prober, baseline through the map"),
    dict(id="P3", name="per-row key extract + hash + find",
         subphase="probe lookup", impls=ALL3, mult="per probe row",
         formula="R_probe",
         refs="ColumnsHashingImpl.h findKey/findKeyImpl; TwoLevelHashTable.h:554-563 Prober::find",
         symbols=[r"ColumnsHashing", r"HashMethod", r"findKeyImpl", r"findKey",
                  r"TwoLevelHashTable<.*>::(Prober|find)", r"HashTable<.*>::find",
                  r"getKeyHolder", r"DB::ColumnString", r"DB::ColumnLowCardinality"],
         note="THE divergence point: baseline calls find on the map, unified through Prober"),
    dict(id="P4", name="per-matched-row global cell offset (offsetInternal)",
         subphase="probe match", impls=ALL3, mult="per matched row",
         formula="M",
         refs="TwoLevelHashTable.h:571-574 Prober::offsetInternal; "
              "HashJoin/KeyGetter.h:19 use_offset=true; UnifiedHashJoin/HashJoinMethods.h:90",
         symbols=[r"offsetInternal", r"offsetInternalAtBucket", r"BucketPrefixSums"],
         shared_symbol=True,
         note="baseline computes it unconditionally; unified only when need_flags"),
    dict(id="P5", name="per-matched-row setUsed / setUsedOnce",
         subphase="probe match", impls=ALL3, mult="per matched row",
         formula="M",
         refs="HashJoin/JoinUsedFlags.h:119-148, :202-227; UnifiedHashJoin/JoinUsedFlags.h:124-152",
         symbols=[r"JoinUsedFlags::setUsed", r"JoinUsedFlags::setUsedOnce",
                  r"JoinStuff::JoinUsedFlags"],
         shared_symbol=True, note="flagged kinds only"),
    dict(id="P6", name="per-matched-row RowRefList walk (addFoundRowAll)",
         subphase="probe match", impls=ALL3, mult="per matched right row",
         formula="M",
         refs="KnownRowsHolder.h:109-124, :140-144",
         symbols=[r"addFoundRowAll", r"KnownRowsHolder", r"refsOf"],
         shared_symbol=True, note="verified textually identical between trees"),
    dict(id="P7", name="per-matched-row appendFromBlock",
         subphase="probe match", impls=ALL3, mult="per matched row (x right columns on eager path)",
         formula="M or M*C_cols",
         refs="AddedColumns.h:310-317, :322-326 (eager); :338-341 (lazy)",
         symbols=[r"AddedColumns.*appendFromBlock", r"AddedColumns"],
         note="verified textually identical between trees"),
    dict(id="P8", name="per-row multi-disjunct probe loop",
         subphase="probe lookup", impls=ALL3, mult="per probe row per clause",
         formula="R_probe*C",
         refs="HashJoin/HashJoinMethodsImpl.h:711-752; UnifiedHashJoin/...:827-868",
         symbols=[r"HashJoinMethods.*joinRightColumns"],
         shared_symbol=True, note="flag_per_row=true instantiation"),
    dict(id="P9", name="additional-filter probe loops (2 phases)",
         subphase="probe match", impls=ALL3, mult="per probe row, then per candidate row",
         formula="R_probe + sum(M_i)",
         refs="HashJoin/HashJoinMethodsImpl.h:945-1130; UnifiedHashJoin/...:1067-1252",
         symbols=[r"joinRightColumnsWithAdditionalFilter"],
         note="not exercised by the benchmark matrix (no residual filter); enumerated for completeness"),
    dict(id="P10", name="per-block Prober construction",
         subphase="probe lookup", impls=[UNI], mult="per probe block per clause",
         formula="B_probe*C",
         refs="TwoLevelHashTable.h:531-550; UnifiedHashJoin/HashJoinMethodsImpl.h:694, :813-814",
         symbols=[r"TwoLevelHashTable<.*>::prober", r"Unified::HashJoinMethods.*joinRightColumns"],
         shared_symbol=True, note="no counterpart: baseline uses the map directly"),

    # ---------------- result gather ------------------------------------------
    dict(id="G1", name="per-chunk partial replicate offsets",
         subphase="result gather", impls=ALL3, mult="per output chunk row",
         formula="R_out",
         refs="HashJoinResult.cpp:486-498",
         symbols=[r"HashJoinResult"],
         note="verified textually identical between trees"),
    dict(id="G2", name="per-output-row row-ref walk (buildOutputFromBlocks)",
         subphase="result gather", impls=ALL3, mult="per output row",
         formula="O",
         refs="AddedColumns.cpp:248-273, :254-258, :173-227",
         symbols=[r"buildOutputFromBlocks", r"LazyOutput", r"buildOutputFromRowRefLists",
                  r"buildJoinGetOutput"],
         note="verified textually identical between trees"),
    dict(id="G3", name="per-column per-output-row gather (insertFrom / gather)",
         subphase="result gather", impls=ALL3, mult="per right column per output row",
         formula="C_cols*O",
         refs="AddedColumns.cpp:116-121, :137-151; HashJoinResult.cpp:134-177",
         symbols=[r"fillFromRowRefs", r"appendRightColumns", r"insertFrom", r"gather",
                  r"IColumn::insertFrom", r"ColumnVector.*insert", r"ColumnString.*insert"],
         note="verified textually identical between trees"),
    dict(id="G4", name="lazy-default fill",
         subphase="result gather", impls=ALL3, mult="per right column per gap",
         formula="C_cols*gaps",
         refs="AddedColumns.h:285-286",
         symbols=[r"applyLazyDefaults"],
         note="verified textually identical between trees; ALSO ICF-folded "
              "(icf_census.json) -- codegen delta provably zero"),
    # G5 was added after G0.1 went red: ScatteredBlock::filterBySelector held 0.53%
    # of in-join samples and the reading had missed it entirely. It is the left-side
    # materialisation of the probe block and is genuinely per-row.
    dict(id="G5", name="left-block materialisation by selector (filterBySelector)",
         subphase="result gather", impls=ALL3, mult="per output row per left column",
         formula="O*C_left",
         refs="HashJoin/ScatteredBlock.h:287-341; called from "
              "{HashJoin,UnifiedHashJoin}/HashJoinResult.cpp:444-448,573-577 and "
              "JoiningTransform.cpp:252",
         symbols=[r"ScatteredBlock::filterBySelector", r"ScatteredBlock::filter\(",
                  r"transformColumnsWithSharedIndex", r"ColumnReplicated"],
         note="ScatteredBlock.h is a SINGLE shared header -- there is no Unified copy "
              "(find src -name ScatteredBlock.h returns one path), so all three "
              "implementations run literally the same code. Codegen delta zero by "
              "construction."),

    # ---------------- non-joined scan ----------------------------------------
    dict(id="N1", name="per-cell scan of owned buckets (offset + used test)",
         subphase="non-joined scan", impls=[HASH, PAR, UNI], mult="per map cell in owned buckets",
         formula="Cells/T",
         refs="HashJoin/HashJoin.cpp:1429-1443; UnifiedHashJoin/HashJoin.cpp:1513-1530",
         symbols=[r"NotJoinedHash.*fillColumns", r"Unified::NotJoinedHash"],
         note="the already-closed A7/N1 loop; unified now uses offsetInternalAtBucket"),
    dict(id="N2", name="per-cell flat scan (single-level map)",
         subphase="non-joined scan", impls=[HASH], mult="per map cell",
         formula="Cells",
         refs="HashJoin/HashJoin.cpp:1448-1462",
         symbols=[r"NotJoinedHash.*fillColumns"],
         shared_symbol=True,
         note="no counterpart: unified is always bucketed, parallel is always two-level after merge"),
    dict(id="N3", name="per-row block scan with per-row isUsed (flag_per_row)",
         subphase="non-joined scan", impls=ALL3, mult="per stored build row",
         formula="R_build",
         refs="HashJoin/HashJoin.cpp:1382-1389; UnifiedHashJoin/HashJoin.cpp:1468-1475",
         symbols=[r"NotJoinedHash.*fillColumns"],
         shared_symbol=True, note="alternative to N1 when flags are per-row"),
    dict(id="N4", name="bucket-range skip for parallel non-joined streams",
         subphase="non-joined scan", impls=[HASH, PAR, UNI], mult="per bucket",
         formula="K",
         refs="HashJoin/HashJoin.cpp:1411-1423; UnifiedHashJoin/HashJoin.cpp:1496-1508",
         symbols=[r"NotJoinedHash.*fillColumns", r"isBucketInRange"],
         shared_symbol=True, note="unified partitions by stream_idx/num_streams"),
    dict(id="N5", name="per-cell RowRefList collect",
         subphase="non-joined scan", impls=ALL3, mult="per row in an unmatched cell's list",
         formula="unmatched_rows",
         refs="HashJoin/HashJoin.cpp:1264-1269; UnifiedHashJoin/HashJoin.cpp:1350-1355",
         symbols=[r"CollectorNonJoined", r"NotJoinedHash"],
         shared_symbol=True, note=""),
    dict(id="N6", name="null-key row scan",
         subphase="non-joined scan", impls=ALL3, mult="per null-key build row",
         formula="null_rows",
         refs="HashJoin/HashJoin.cpp:1489-1505; UnifiedHashJoin/HashJoin.cpp:1557-1573",
         symbols=[r"fillNullsFromBlocks"],
         note=""),
    dict(id="N7", name="non-joined column fill",
         subphase="non-joined scan", impls=ALL3, mult="per column per emitted row",
         formula="C_cols*O_nj",
         refs="HashJoin/HashJoin.cpp:1466-1467; UnifiedHashJoin/HashJoin.cpp:1533-1534",
         symbols=[r"fillFromBlocksAndRowNumbers", r"NotJoinedHash.*fillColumns"],
         shared_symbol=True, note=""),
]

# --------------------------------------------------------------------------
# Explicit exclusions: symbols that can appear inside a join stack but are not
# per-row join loops. Each needs a reason; "it did not fit" is not a reason.
# --------------------------------------------------------------------------

EXCLUSIONS = [
    (r"JoiningTransform::(work|prepare|transform|readExecute|onFinish|onConsume|onGenerate)\b",
     "pipeline transform scaffolding, per block not per row"),
    (r"FillingRightJoinSide\w*::(work|prepare|transform|onConsume)\b",
     "pipeline transform scaffolding, per block not per row"),
    (r"NonJoinedBlocksTransform::(work|prepare|generate|NonJoinedBlocksTransform|~)",
     "pipeline transform scaffolding/lifetime, per block or once per query"),
    (r"DelayedJoinedBlocks",
     "delayed-blocks plumbing; reads zero elapsed on these queries (prior mission E7.1)"),
    (r"SpillingHashJoin",
     "the spill wrapper is identical for all three and disabled here "
     "(max_bytes_before_external_join=0), so it is not a source of asymmetry"),
    (r"DB::HashJoin::(HashJoin|~HashJoin|getTotals|setTotals|checkTypesOfKeys|initRightBlockStructure|isFilled|alwaysReturnsEmptySet)\b",
     "construction/teardown/metadata, once per join"),
    (r"DB::Unified::HashJoin::(HashJoin|~HashJoin|getTotals|setTotals|checkTypesOfKeys|initRightBlockStructure|isFilled|alwaysReturnsEmptySet)\b",
     "construction/teardown/metadata, once per join"),
    (r"DB::ConcurrentHashJoin::(ConcurrentHashJoin|~ConcurrentHashJoin|getTotals|setTotals|checkTypesOfKeys|alwaysReturnsEmptySet)\b",
     "construction/teardown/metadata, once per join"),
    (r"allocate_shared.*(JoiningTransform|NonJoinedBlocksTransform|FillingRightJoinSide)",
     "pipeline construction, once per query"),
    (r"MapsTemplate<.*>::operator=",
     "map handle copy, once per build (parallel_hash copies maps to every slot at "
     "onBuildPhaseFinish); not per row"),
    (r"DB::JoinCommon::hasNonJoinedBlocks",
     "planner-side predicate, once per query"),
    (r"(onBuildPhaseFinish|runPostBuildPhase|tryConvertToFixedHashMap|canConvertToFixedHashMap|rerange|reinitUsedFlags|finalizePerRowFlags|recomputeBucketBytes|shrinkStoredBlocksToFit|invalidateEmitTable)",
     "once-per-build finalisation, not a per-row loop; B19/B20 cover the parts that are per-bucket"),
    (r"(getTotalRowCount|getTotalByteCount|sizeInBytes|allocatedBytes|getUsedRows)",
     "accounting queries, per block at most"),

    # --- added after the first G0.1 run went red. Each names why the symbol is
    # --- not a per-row join loop; none of them widens far enough to swallow a
    # --- loop that IS enumerated (checked by the self-test in enumerate.py).
    (r"::(joinBlock|joinScatteredBlock|getNonJoinedBlocks|savedBlockSample|"
     r"supportParallelNonJoinedBlocksProcessing|getBucketBufferSize\b)\b",
     "per-probe-block or per-query driver; the per-row work it calls is enumerated "
     "separately (P0/P1/N1)"),
    (r"RightTableData::~RightTableData|NullMapHolder|::~ScatteredBlock|"
     r"ScatteredBlock::(ScatteredBlock|operator=|cut)|construct_at.*ScatteredBlock",
     "block/table lifetime: construction, move and destruction, per block not per row"),
    (r"JoinOnKeyColumns::~JoinOnKeyColumns",
     "per-block teardown of the probe key view; its construction is enumerated as P0"),
    (r"DB::JoinCommon::(checkTypesOfKeys|getColumnAsMask|getCurrentQueryMemoryUsage|"
     r"getRawPointers|convertColumnsToNullable|removeColumnNullability)",
     "per-block or once-per-query key/type plumbing shared identically by all three"),
    (r"(getMinBytesForPrefetchInJoin|copyEmptyColumns|filterColumnsPresentInSampleBlock)",
     "per-block configuration/plumbing, not per row"),
    (r"std::__1::(vector|deque)<.*>::(reserve|__emplace_back_slow_path|"
     r"__init_with_size|__add_back_capacity|~deque)",
     "container growth for per-block bookkeeping vectors, amortised per block"),
    # The ICF exclusion is narrow on purpose: it names the specific folded symbols
    # from codegen/icf_census.json rather than excluding a whole namespace.
    (r"::canRemoveColumnsFromLeftBlock",
     "identical-code-folded across the two trees (icf_census.json: one address holds "
     "both DB::HashJoin:: and DB::Unified::HashJoin:: names), so the attributed name "
     "is arbitrary; the function is per-block metadata either way"),
]


def build_registry():
    import re
    loops = []
    for spec in LOOPS:
        s = dict(spec)
        s["_re"] = [re.compile(p) for p in spec["symbols"]]
        loops.append(s)
    excl = [(re.compile(p), why) for p, why in EXCLUSIONS]
    in_join = re.compile("|".join(IN_JOIN_MARKERS))
    return loops, excl, in_join


def classify_symbol(sym, loops, excl):
    """Return (list_of_loop_ids, exclusion_reason_or_None)."""
    hits = [l["id"] for l in loops if any(r.search(sym) for r in l["_re"])]
    if hits:
        return hits, None
    for r, why in excl:
        if r.search(sym):
            return [], why
    return [], None
