CREATE TABLE l (k UInt64, v UInt64) ENGINE = MergeTree ORDER BY k;
CREATE TABLE r (k UInt64, w UInt64) ENGINE = MergeTree ORDER BY k;
INSERT INTO l SELECT number, number * 3 FROM numbers(200000);
INSERT INTO r SELECT number * 2, number * 7 FROM numbers(100000);
