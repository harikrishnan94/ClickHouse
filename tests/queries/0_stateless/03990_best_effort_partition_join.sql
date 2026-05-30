-- Tags: no-random-settings
-- best_effort_partition join is wired only into the new analyzer.
SET enable_analyzer = 1;

DROP TABLE IF EXISTS bep_left;
DROP TABLE IF EXISTS bep_right;

CREATE TABLE bep_left (k UInt64, k2 UInt32, v String) ENGINE = MergeTree ORDER BY tuple();
CREATE TABLE bep_right (k UInt64, k2 UInt32, rk Int64) ENGINE = MergeTree ORDER BY tuple();

INSERT INTO bep_left SELECT number % 1000, number % 50, toString(number) FROM numbers(20000);
INSERT INTO bep_right SELECT number % 1000, number % 50, number * 7 FROM numbers(4000);

-- Single-key INNER ALL join, default probe buffer: result matches the `hash` algorithm.
-- sum(cityHash64(...)) is an order-independent multiset fingerprint of the join output.
SELECT
    (SELECT sum(cityHash64(l.k, l.v, r.rk)) FROM bep_left l INNER JOIN bep_right r ON l.k = r.k
       SETTINGS join_algorithm = 'best_effort_partition')
  = (SELECT sum(cityHash64(l.k, l.v, r.rk)) FROM bep_left l INNER JOIN bep_right r ON l.k = r.k
       SETTINGS join_algorithm = 'hash');

-- Output cardinality matches.
SELECT
    (SELECT count() FROM bep_left l INNER JOIN bep_right r ON l.k = r.k
       SETTINGS join_algorithm = 'best_effort_partition')
  = (SELECT count() FROM bep_left l INNER JOIN bep_right r ON l.k = r.k
       SETTINGS join_algorithm = 'hash');

-- Tiny probe buffer forces mid-stream eviction (the streaming probe path).
SELECT
    (SELECT sum(cityHash64(l.k, l.v, r.rk)) FROM bep_left l INNER JOIN bep_right r ON l.k = r.k
       SETTINGS join_algorithm = 'best_effort_partition', max_bytes_in_join_probe_buffer = 1)
  = (SELECT sum(cityHash64(l.k, l.v, r.rk)) FROM bep_left l INNER JOIN bep_right r ON l.k = r.k
       SETTINGS join_algorithm = 'hash');

-- Small per-pass fan-out forces multi-pass radix refinement.
SELECT
    (SELECT sum(cityHash64(l.k, l.v, r.rk)) FROM bep_left l INNER JOIN bep_right r ON l.k = r.k
       SETTINGS join_algorithm = 'best_effort_partition', max_partitions_per_pass = 2, max_bytes_in_join_probe_buffer = 4096)
  = (SELECT sum(cityHash64(l.k, l.v, r.rk)) FROM bep_left l INNER JOIN bep_right r ON l.k = r.k
       SETTINGS join_algorithm = 'hash');

-- Composite key INNER ALL join with eviction.
SELECT
    (SELECT sum(cityHash64(l.k, l.k2, l.v, r.rk)) FROM bep_left l INNER JOIN bep_right r ON l.k = r.k AND l.k2 = r.k2
       SETTINGS join_algorithm = 'best_effort_partition', max_bytes_in_join_probe_buffer = 1)
  = (SELECT sum(cityHash64(l.k, l.k2, l.v, r.rk)) FROM bep_left l INNER JOIN bep_right r ON l.k = r.k AND l.k2 = r.k2
       SETTINGS join_algorithm = 'hash');

-- Empty right side: no matches.
SELECT count() FROM bep_left l INNER JOIN (SELECT * FROM bep_right WHERE 0) r ON l.k = r.k
  SETTINGS join_algorithm = 'best_effort_partition';

-- Unsupported join kinds error out (v1 supports only INNER ALL).
SELECT count() FROM bep_left l LEFT JOIN bep_right r ON l.k = r.k
  SETTINGS join_algorithm = 'best_effort_partition'; -- { serverError NOT_IMPLEMENTED }

DROP TABLE bep_left;
DROP TABLE bep_right;
