SELECT
    '__JOIN_MERGETREE_ASSERT__',
    probe_count,
    build_count,
    joined_count
FROM (SELECT count() AS probe_count FROM (SELECT
    if(
        intDiv(toUInt128(cycle * 268435456 + card_bucket * 4194304 + rank + 1) * 536870912, 536870912)
            > intDiv(toUInt128(cycle * 268435456 + card_bucket * 4194304 + rank) * 536870912, 536870912),
        hit_k,
        miss_k
    ) AS k
FROM join_mergetree_bench_probe
PREWHERE (cycle < 2 AND card_bucket < 64)) AS probe_rows) AS probe_count_source
CROSS JOIN (SELECT count() AS build_count FROM (SELECT k
FROM join_mergetree_bench_build
PREWHERE occurrence < 1 AND card_bucket < 64) AS build_rows) AS build_count_source
CROSS JOIN (SELECT count() AS joined_count FROM (SELECT
    if(
        intDiv(toUInt128(cycle * 268435456 + card_bucket * 4194304 + rank + 1) * 536870912, 536870912)
            > intDiv(toUInt128(cycle * 268435456 + card_bucket * 4194304 + rank) * 536870912, 536870912),
        hit_k,
        miss_k
    ) AS k
FROM join_mergetree_bench_probe
PREWHERE (cycle < 2 AND card_bucket < 64)) AS p INNER JOIN (SELECT k
FROM join_mergetree_bench_build
PREWHERE occurrence < 1 AND card_bucket < 64) AS b USING (k) SETTINGS join_algorithm = 'parallel_hash', max_threads = 96, query_plan_join_swap_table = false, enable_analyzer = 1, enable_join_runtime_filters = 0, max_bytes_before_external_join = 0, max_bytes_ratio_before_external_join = 0, max_memory_usage = 100000000000) AS joined_count_source
FORMAT TabSeparatedRaw
