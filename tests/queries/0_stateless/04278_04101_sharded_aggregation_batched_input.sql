-- Test that `shard_by_hash_input_batch_bytes` (batched input via ColumnsScatter::scatter)
-- produces per-key-identical GROUP BY results to the per-chunk default.
--
-- Each block compares `arraySort(groupArray((k, cnt)))` between
--   shard_by_hash_input_batch_bytes = 1048576   (multi-chunk batched scatter)
--   shard_by_hash_input_batch_bytes = 0         (per-chunk ColumnsScatter flush)
-- and outputs 1 (equal) or 0 (different). A bug that misroutes any row to the
-- wrong shard will produce a different per-key count and trigger 0.

SET enable_sharding_aggregator = 1;
SET max_threads = 4;

-- ── UInt64 key ────────────────────────────────────────────────────────────────
SELECT 'UInt64';
SELECT
(
    SELECT arraySort(groupArray((k, cnt)))
    FROM (SELECT k, count() AS cnt FROM (SELECT number % 100 AS k FROM numbers_mt(10000)) GROUP BY k
          SETTINGS shard_by_hash_input_batch_bytes = 1048576)
) =
(
    SELECT arraySort(groupArray((k, cnt)))
    FROM (SELECT k, count() AS cnt FROM (SELECT number % 100 AS k FROM numbers_mt(10000)) GROUP BY k
          SETTINGS shard_by_hash_input_batch_bytes = 0)
);

-- ── String key ────────────────────────────────────────────────────────────────
SELECT 'String';
SELECT
(
    SELECT arraySort(groupArray((k, cnt)))
    FROM (SELECT k, count() AS cnt FROM (SELECT toString(number % 50) AS k FROM numbers_mt(5000)) GROUP BY k
          SETTINGS shard_by_hash_input_batch_bytes = 1048576)
) =
(
    SELECT arraySort(groupArray((k, cnt)))
    FROM (SELECT k, count() AS cnt FROM (SELECT toString(number % 50) AS k FROM numbers_mt(5000)) GROUP BY k
          SETTINGS shard_by_hash_input_batch_bytes = 0)
);

-- ── Nullable(UInt32) key ──────────────────────────────────────────────────────
SELECT 'NullableUInt32';
SELECT
(
    SELECT arraySort(groupArray((k, cnt)))
    FROM (SELECT k, count() AS cnt FROM (SELECT if(number % 7 = 0, NULL, toUInt32(number % 30)) AS k FROM numbers_mt(5000)) GROUP BY k
          SETTINGS shard_by_hash_input_batch_bytes = 1048576)
) =
(
    SELECT arraySort(groupArray((k, cnt)))
    FROM (SELECT k, count() AS cnt FROM (SELECT if(number % 7 = 0, NULL, toUInt32(number % 30)) AS k FROM numbers_mt(5000)) GROUP BY k
          SETTINGS shard_by_hash_input_batch_bytes = 0)
);

-- ── Decimal64 key ─────────────────────────────────────────────────────────────
SELECT 'Decimal64';
SELECT
(
    SELECT arraySort(groupArray((k, cnt)))
    FROM (SELECT k, count() AS cnt FROM (SELECT toDecimal64(number % 40, 2) AS k FROM numbers_mt(4000)) GROUP BY k
          SETTINGS shard_by_hash_input_batch_bytes = 1048576)
) =
(
    SELECT arraySort(groupArray((k, cnt)))
    FROM (SELECT k, count() AS cnt FROM (SELECT toDecimal64(number % 40, 2) AS k FROM numbers_mt(4000)) GROUP BY k
          SETTINGS shard_by_hash_input_batch_bytes = 0)
);

-- ── Tuple(UInt32, String) key ─────────────────────────────────────────────────
SELECT 'TupleUInt32String';
SELECT
(
    SELECT arraySort(groupArray((k, cnt)))
    FROM (SELECT k, count() AS cnt FROM (SELECT (toUInt32(number % 10), toString(number % 5)) AS k FROM numbers_mt(2000)) GROUP BY k
          SETTINGS shard_by_hash_input_batch_bytes = 1048576)
) =
(
    SELECT arraySort(groupArray((k, cnt)))
    FROM (SELECT k, count() AS cnt FROM (SELECT (toUInt32(number % 10), toString(number % 5)) AS k FROM numbers_mt(2000)) GROUP BY k
          SETTINGS shard_by_hash_input_batch_bytes = 0)
);
