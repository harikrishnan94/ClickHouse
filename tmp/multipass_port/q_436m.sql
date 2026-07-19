SELECT p.p_p0, p.p_p1, p.p_p2, p.p_p3, p.p_p4, p.p_p5, p.p_p6 FROM (SELECT
    if(
        intDiv(toUInt128(cycle * 436207616 + card_bucket * 4194304 + rank + 1) * 872415232, 872415232)
            > intDiv(toUInt128(cycle * 436207616 + card_bucket * 4194304 + rank) * 872415232, 872415232),
        hit_k,
        miss_k
    ) AS k,
    p_p0,
    p_p1,
    p_p2,
    p_p3,
    p_p4,
    p_p5,
    p_p6
FROM join_mergetree_bench_probe
PREWHERE (cycle < 2 AND card_bucket < 104)) AS p INNER JOIN (SELECT k
FROM join_mergetree_bench_build
PREWHERE occurrence < 1 AND card_bucket < 104) AS b USING (k) SETTINGS join_algorithm = 'partitioned_hash', max_threads = 32, query_plan_join_swap_table = false, enable_analyzer = 1, enable_join_runtime_filters = 0, max_bytes_before_external_join = 0, max_bytes_ratio_before_external_join = 0, max_memory_usage = 100000000000 FORMAT Null;
