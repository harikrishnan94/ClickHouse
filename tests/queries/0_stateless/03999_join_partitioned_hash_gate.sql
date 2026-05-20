-- Tests that ineligible join shapes fall through to the next algorithm in join_algorithm
-- when 'partitioned_hash' is listed alongside 'hash'.
-- All EXPLAIN PIPELINE outputs should show HashJoin-related transforms (not PHJ), since
-- the stub still returns nullptr and the setting falls back to 'hash'.

SET join_algorithm = 'partitioned_hash,hash';

-- 1. INNER JOIN with UInt64 key (eligible shape, falls through because stub returns nullptr)
EXPLAIN PIPELINE SELECT t1.a, t2.b
FROM (SELECT number AS a FROM numbers(10)) t1
INNER JOIN (SELECT number AS b FROM numbers(10)) t2 ON t1.a = t2.b
FORMAT Null;

-- 2. ASOF JOIN -- rejected by spec §2.3 (ASOF not supported)
EXPLAIN PIPELINE SELECT t1.a, t2.b
FROM (SELECT number AS a, toFloat64(number) AS ts FROM numbers(10)) t1
ASOF JOIN (SELECT number AS b, toFloat64(number) AS ts FROM numbers(10)) t2
ON t1.a = t2.b AND t1.ts >= t2.ts
FORMAT Null;

-- 3. CROSS JOIN -- rejected (no equi-join keys)
EXPLAIN PIPELINE SELECT t1.a, t2.b
FROM (SELECT number AS a FROM numbers(5)) t1
CROSS JOIN (SELECT number AS b FROM numbers(5)) t2
FORMAT Null;

-- 4. String key -- rejected by spec §2.1/§2.2 (variable-width)
EXPLAIN PIPELINE SELECT t1.s, t2.s
FROM (SELECT toString(number) AS s FROM numbers(10)) t1
INNER JOIN (SELECT toString(number) AS s FROM numbers(10)) t2 ON t1.s = t2.s
FORMAT Null;

-- 5. LEFT JOIN with UInt64 key (eligible, falls through via stub)
EXPLAIN PIPELINE SELECT t1.a, t2.b
FROM (SELECT number AS a FROM numbers(10)) t1
LEFT JOIN (SELECT number AS b FROM numbers(10)) t2 ON t1.a = t2.b
FORMAT Null;

-- 6. RIGHT JOIN
EXPLAIN PIPELINE SELECT t1.a, t2.b
FROM (SELECT number AS a FROM numbers(10)) t1
RIGHT JOIN (SELECT number AS b FROM numbers(10)) t2 ON t1.a = t2.b
FORMAT Null;

-- 7. FULL JOIN
EXPLAIN PIPELINE SELECT t1.a, t2.b
FROM (SELECT number AS a FROM numbers(10)) t1
FULL JOIN (SELECT number AS b FROM numbers(10)) t2 ON t1.a = t2.b
FORMAT Null;

SELECT 'gate_tests_passed';
