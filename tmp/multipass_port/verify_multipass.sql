-- G4b: bit-exact-in-aggregate verification of a genuinely multi-pass (>13-bit) plan.
-- Build side: 524288000 distinct keys, every key twice (RowRefList path); probe: 600M rows of
-- which 75.7M miss (LEFT JOIN emits them with default values). The pair (count, sum of a
-- row hash) is order-independent, so it compares `partitioned_hash` against `parallel_hash`
-- exactly on both the matched multiset and the non-matched LEFT rows.
SELECT
    'partitioned_hash' AS algo,
    count() AS cnt,
    sum(cityHash64(k, pv, bv)) AS h
FROM
(
    SELECT number AS k, number * 7 AS pv FROM numbers_mt(600000000)
) AS p
LEFT JOIN
(
    SELECT intDiv(number, 2) AS k, number AS bv FROM numbers_mt(1048576000)
) AS b
USING (k)
SETTINGS join_algorithm = 'partitioned_hash', max_threads = 32, query_plan_join_swap_table = false, enable_analyzer = 1, enable_join_runtime_filters = 0, max_bytes_before_external_join = 0, max_bytes_ratio_before_external_join = 0, max_memory_usage = 200000000000;

SELECT
    'parallel_hash' AS algo,
    count() AS cnt,
    sum(cityHash64(k, pv, bv)) AS h
FROM
(
    SELECT number AS k, number * 7 AS pv FROM numbers_mt(600000000)
) AS p
LEFT JOIN
(
    SELECT intDiv(number, 2) AS k, number AS bv FROM numbers_mt(1048576000)
) AS b
USING (k)
SETTINGS join_algorithm = 'parallel_hash', max_threads = 32, query_plan_join_swap_table = false, enable_analyzer = 1, enable_join_runtime_filters = 0, max_bytes_before_external_join = 0, max_bytes_ratio_before_external_join = 0, max_memory_usage = 200000000000;
