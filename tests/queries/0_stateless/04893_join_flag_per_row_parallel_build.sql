-- Regression test for D-39 on uhj-parity: build workers used to append to one shared
-- `pending_per_row_flags` vector with no synchronization whenever a join needed per-row used
-- flags (multiple disjuncts, or RIGHT/FULL with mixed inequality) and ran a parallel build. A
-- lost or duplicated (block_no, flags) entry silently drops or double-counts unmatched right
-- rows. This exercises a multi-disjunct FULL JOIN with several concurrent build threads and
-- asserts exact, hand-computed row-count and checksum invariants that such a race would break.
--
-- Right table has 200000 rows. Rows [0, 100) match the left table via the first disjunct, rows
-- [100000, 100100) match via the second disjunct; every other right row is unmatched. All 100
-- left rows are matched (each by exactly two right rows), so the FULL JOIN has no unmatched-left
-- rows, which keeps the expected counts simple:
--   matched pairs = 200, unmatched right rows = 200000 - 200 = 199800, total = 200000.

SET max_threads = 8;
SET join_algorithm = 'parallel_hash';
SET parallel_hash_join_threshold = 1;
SET max_block_size = 8192;

DROP TABLE IF EXISTS t1_04893;
DROP TABLE IF EXISTS t2_04893;

CREATE TABLE t1_04893 (number UInt64) ENGINE = MergeTree ORDER BY number;
CREATE TABLE t2_04893 (number UInt64) ENGINE = MergeTree ORDER BY number;

INSERT INTO t1_04893 SELECT number FROM numbers(100);
INSERT INTO t2_04893 SELECT number FROM numbers(200000);

SELECT count(), sum(t2_04893.number), sum(t1_04893.number)
FROM t1_04893
FULL JOIN t2_04893
    ON t1_04893.number = t2_04893.number OR t1_04893.number + 100000 = t2_04893.number;

DROP TABLE t1_04893;
DROP TABLE t2_04893;
