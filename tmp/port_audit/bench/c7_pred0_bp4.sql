SELECT p.p_p0, b.b_p0, b.b_p1, b.b_p2, b.b_p3 FROM (SELECT
    if(
        intDiv(toUInt128(cycle * 524288000 + card_bucket * 1 + rank + 1) * 524288000, 524288000)
            > intDiv(toUInt128(cycle * 524288000 + card_bucket * 1 + rank) * 524288000, 524288000),
        hit_k,
        miss_k
    ) AS k,
    p_p0
FROM join_mergetree_bench_probe
PREWHERE (cycle < 1 AND card_bucket < 524288000)) AS p INNER JOIN (SELECT k, b_p0, b_p1, b_p2, b_p3
FROM join_mergetree_bench_build
PREWHERE occurrence < 1 AND card_bucket < 524288000) AS b USING (k) SETTINGS join_algorithm = 'partitioned_hash', max_threads = 32, query_plan_join_swap_table = false, enable_analyzer = 1, enable_join_runtime_filters = 0, max_bytes_before_external_join = 0, max_bytes_ratio_before_external_join = 0, max_memory_usage = 100000000000 FORMAT Null;
SELECT 'teardown_us_bp4', value FROM system.events WHERE event = 'PartitionedHashJoinTeardownMicroseconds' SETTINGS system_events_show_zero_values=1;
