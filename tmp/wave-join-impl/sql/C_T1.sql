SELECT p.p_p0, p.p_p1, p.p_p2, p.p_p3, p.p_p4, p.p_p5, p.p_p6, b.b_p0, b.b_p1, b.b_p2, b.b_p3, b.b_p4, b.b_p5, b.b_p6 FROM (SELECT
    if(
        intDiv(toUInt128(cycle * 268435456 + card_bucket * 4194304 + rank + 1) * 1073741824, 1073741824)
            > intDiv(toUInt128(cycle * 268435456 + card_bucket * 4194304 + rank) * 1073741824, 1073741824),
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
PREWHERE (cycle < 4 AND card_bucket < 64)) AS p INNER JOIN (SELECT k, b_p0, b_p1, b_p2, b_p3, b_p4, b_p5, b_p6
FROM join_mergetree_bench_build
PREWHERE occurrence < 1 AND card_bucket < 64) AS b USING (k) SETTINGS join_algorithm = 'radix_join', max_threads = 1, query_plan_join_swap_table = false, enable_analyzer = 1, enable_join_runtime_filters = 0, max_bytes_before_external_join = 0, max_bytes_ratio_before_external_join = 0, max_memory_usage = 100000000000 FORMAT Null;
SELECT p.p_p0, p.p_p1, p.p_p2, p.p_p3, p.p_p4, p.p_p5, p.p_p6, b.b_p0, b.b_p1, b.b_p2, b.b_p3, b.b_p4, b.b_p5, b.b_p6 FROM (SELECT
    if(
        intDiv(toUInt128(cycle * 268435456 + card_bucket * 4194304 + rank + 1) * 1073741824, 1073741824)
            > intDiv(toUInt128(cycle * 268435456 + card_bucket * 4194304 + rank) * 1073741824, 1073741824),
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
PREWHERE (cycle < 4 AND card_bucket < 64)) AS p INNER JOIN (SELECT k, b_p0, b_p1, b_p2, b_p3, b_p4, b_p5, b_p6
FROM join_mergetree_bench_build
PREWHERE occurrence < 1 AND card_bucket < 64) AS b USING (k) SETTINGS join_algorithm = 'radix_join', max_threads = 1, query_plan_join_swap_table = false, enable_analyzer = 1, enable_join_runtime_filters = 0, max_bytes_before_external_join = 0, max_bytes_ratio_before_external_join = 0, max_memory_usage = 100000000000 FORMAT Null;
