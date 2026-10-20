-- An `ANY LEFT` build with a right-side ON filter drops the stored blocks that insert nothing. The deferred build checks
-- `max_bytes_in_join` after the inserts, when those blocks are gone, so it must not throw where the streaming build
-- succeeds. `keep = 1` holds only for k < 2000, so most right blocks insert nothing.

SET join_algorithm = 'parallel_hash';
SET collect_hash_table_stats_during_joins = 0;
SET parallel_hash_join_threshold = 0;
SET max_bytes_before_external_join = 0;
SET max_bytes_ratio_before_external_join = 0;
SET join_use_nulls = 0;
SET max_threads = 4;
SET query_plan_join_swap_table = 'false';
SET enable_parallel_replicas = 0;

DROP TABLE IF EXISTS t_skip_build;
DROP TABLE IF EXISTS t_skip_probe;
CREATE TABLE t_skip_build (k UInt64, keep UInt8, v UInt64) ENGINE = MergeTree ORDER BY tuple();
CREATE TABLE t_skip_probe (k UInt64, v UInt64) ENGINE = MergeTree ORDER BY tuple();

INSERT INTO t_skip_build SELECT number, (number < 2000) AS keep, number * 3 FROM numbers(1000000);
INSERT INTO t_skip_probe SELECT number, number * 7 FROM numbers(1000000);

SELECT 'any_left_skip', count(), sum(cityHash64(l.k, l.v, r.k, r.keep, r.v))
FROM t_skip_probe l ANY LEFT JOIN t_skip_build r ON l.k = r.k AND r.keep = 1
SETTINGS enable_hash_join_deferred_build = 0, join_overflow_mode = 'throw', max_bytes_in_join = 16000000;

SELECT 'any_left_skip', count(), sum(cityHash64(l.k, l.v, r.k, r.keep, r.v))
FROM t_skip_probe l ANY LEFT JOIN t_skip_build r ON l.k = r.k AND r.keep = 1
SETTINGS enable_hash_join_deferred_build = 1, join_overflow_mode = 'throw', max_bytes_in_join = 16000000,
         log_comment = '04414_skip_discarded';

SYSTEM FLUSH LOGS query_log;
SELECT 'deferred build engaged',
    countIf(ProfileEvents['HashJoinDeferredPreallocatedElementsInHashTables'] > 0) = count(),
    max(ProfileEvents['HashJoinDeferredPreallocatedElementsInHashTables']) BETWEEN 1 AND 60000
FROM system.query_log
WHERE current_database = currentDatabase() AND type = 'QueryFinish' AND query_kind = 'Select'
    AND log_comment = '04414_skip_discarded';

DROP TABLE t_skip_build;
DROP TABLE t_skip_probe;
