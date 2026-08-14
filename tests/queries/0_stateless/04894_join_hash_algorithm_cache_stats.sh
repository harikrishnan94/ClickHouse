#!/usr/bin/env bash
# Tags: long

# Regression test for D-37 on uhj-parity: hash-table cache keys (which feed the size histogram
# used to preallocate a join's right-side hash table, the `rhs_size_estimation` override, and
# runtime-filter sizing) were computed only when `PARALLEL_HASH` was literally present in
# `join_algorithm`, even though `hash` and `parallel_hash` are meant to be aliases from
# `preferParallelHashLayout`'s point of view (layout selection itself ignores the algorithm
# list entirely). With `join_algorithm='hash'` only, a hot run's right-side hash table never
# preallocated from the previous run's stats.
#
# The test forces `join_algorithm='hash'` (never lists `parallel_hash`), runs the same join
# twice, and asserts that the second run preallocates exactly N entries for the right-side hash
# table — i.e. that `hash` gets the same cache-key treatment as `parallel_hash`.

CUR_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=../shell_config.sh
. "$CUR_DIR"/../shell_config.sh

opts=(
    --enable_analyzer=1
    --join_algorithm='hash'
)

# Large enough that the size estimate (>= default parallel_hash_join_threshold) selects the
# parallel layout, whose per-slot reserve is what actually consumes the cached size hint.
# `reserveSlot` reserves `hint / num_slots` per slot and sums exactly that back into the event,
# so a hint not evenly divisible by `num_slots` (a power of two, at most 256) would truncate the
# expected total. N is a multiple of 256 so the sum matches N regardless of `num_slots`.
N=1024000
T1="join_hash_cache_stats_t1"; T2="join_hash_cache_stats_t2"

$CLICKHOUSE_CLIENT -q "
  DROP TABLE IF EXISTS $T1;
  DROP TABLE IF EXISTS $T2;

  CREATE TABLE $T1(a UInt32) ENGINE=MergeTree ORDER BY ();
  INSERT INTO $T1 SELECT number FROM numbers_mt($N);

  CREATE TABLE $T2(a UInt32) ENGINE=MergeTree ORDER BY ();
  INSERT INTO $T2 SELECT number FROM numbers_mt($N);
"

SQL="SELECT count() FROM $T1 INNER JOIN $T2 ON $T1.a = $T2.a"

cold_id="join_hash_algorithm_cache_stats_cold_$RANDOM$RANDOM"
hot_id="join_hash_algorithm_cache_stats_hot_$RANDOM$RANDOM"

$CLICKHOUSE_CLIENT "${opts[@]}" --query_id="$cold_id" -q "$SQL" --format Null
$CLICKHOUSE_CLIENT "${opts[@]}" --query_id="$hot_id"  -q "$SQL" --format Null

$CLICKHOUSE_CLIENT -q "SYSTEM FLUSH LOGS query_log"

# Cold run: nothing in the cache yet -> no preallocation expected.
$CLICKHOUSE_CLIENT --param_query_id="$cold_id" -q "
  SELECT if(any(ProfileEvents['HashJoinPreallocatedElementsInHashTables']) = 0, '1', 'Error: ' || any(query_id) || ' got prealloc=' || toString(any(ProfileEvents['HashJoinPreallocatedElementsInHashTables'])))
    FROM system.query_log
   WHERE event_date >= yesterday() AND event_time >= now() - 600 AND query_id = {query_id:String} AND current_database = currentDatabase() AND type = 'QueryFinish'
"

# Hot run: `join_algorithm='hash'` alone must still compute cache keys and reuse the stats
# collected by the cold run, exactly like listing `parallel_hash` does.
$CLICKHOUSE_CLIENT --param_query_id="$hot_id" --param_expected_prealloc=$N -q "
  SELECT if(any(ProfileEvents['HashJoinPreallocatedElementsInHashTables']) = {expected_prealloc:UInt64}, '1', 'Error: ' || any(query_id) || ' got prealloc=' || toString(any(ProfileEvents['HashJoinPreallocatedElementsInHashTables'])))
    FROM system.query_log
   WHERE event_date >= yesterday() AND event_time >= now() - 600 AND query_id = {query_id:String} AND current_database = currentDatabase() AND type = 'QueryFinish'
"

$CLICKHOUSE_CLIENT -q "
  DROP TABLE $T1;
  DROP TABLE $T2;
"
