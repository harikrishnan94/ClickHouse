-- PHJ pipeline shape test: verify EXPLAIN PIPELINE shows the right processors.

SET join_algorithm = 'partitioned_hash';
SET max_threads = 2;

-- For an eligible INNER JOIN with UInt64 key, PHJ should be selected.
-- The pipeline must contain FillingRightJoinSideTransform and DelayedJoinedBlocksWorkerTransform.
EXPLAIN PIPELINE
SELECT t1.a, t2.b
FROM (SELECT number AS a FROM numbers(1000)) t1
INNER JOIN (SELECT number AS b FROM numbers(1000)) t2 ON t1.a = t2.b
FORMAT LineAsString;

-- Ineligible query (String key) with partitioned_hash,hash: should NOT show PHJ.
SET join_algorithm = 'partitioned_hash,hash';
EXPLAIN PIPELINE
SELECT toString(t1.a) AS s
FROM (SELECT number AS a FROM numbers(10)) t1
INNER JOIN (SELECT number AS a FROM numbers(10)) t2 ON toString(t1.a) = toString(t2.a)
FORMAT LineAsString;
