-- A `Join` table keeps the keys of every INSERT. String keys live in the arena of the build stream that inserted
-- them. A `Join` table has one build stream, and its arena must live as long as the table, not as long as one INSERT.
-- Each new key is copied into that arena, and each INSERT adds keys and overwrites others. After the last INSERT,
-- `joinGet`, a JOIN and a read of the table must find every key, and `joinGet` must find them again after DETACH and
-- ATTACH. After TRUNCATE the table is empty, and `joinGet` finds the keys of the next INSERT.

SET enable_parallel_replicas = 0;

DROP TABLE IF EXISTS sj_last;
DROP TABLE IF EXISTS sj_first;
DROP TABLE IF EXISTS sj_all;
DROP TABLE IF EXISTS sj_lc;
DROP TABLE IF EXISTS sj_pair;

CREATE TABLE sj_last (s String, v UInt64) ENGINE = Join(ANY, LEFT, s) SETTINGS join_any_take_last_row = 1;
CREATE TABLE sj_first (s String, v UInt64) ENGINE = Join(ANY, LEFT, s);
CREATE TABLE sj_all (s String, v UInt64) ENGINE = Join(ALL, LEFT, s);
CREATE TABLE sj_lc (s LowCardinality(String), v UInt64) ENGINE = Join(ANY, LEFT, s) SETTINGS join_any_take_last_row = 1;
CREATE TABLE sj_pair (a UInt32, b UInt32, v UInt64) ENGINE = Join(ANY, LEFT, a, b);

-- INSERT j (0 to 4) writes the keys 0 to 999 + 100 * j, each with the value j * 1000000 + key.
INSERT INTO sj_last SELECT concat(repeat('k', 40), toString(number)), number FROM numbers(1000);
INSERT INTO sj_last SELECT concat(repeat('k', 40), toString(number)), 1000000 + number FROM numbers(1100);
INSERT INTO sj_last SELECT concat(repeat('k', 40), toString(number)), 2000000 + number FROM numbers(1200);
INSERT INTO sj_last SELECT concat(repeat('k', 40), toString(number)), 3000000 + number FROM numbers(1300);
INSERT INTO sj_last SELECT concat(repeat('k', 40), toString(number)), 4000000 + number FROM numbers(1400);
INSERT INTO sj_first SELECT concat(repeat('k', 40), toString(number)), number FROM numbers(1000);
INSERT INTO sj_first SELECT concat(repeat('k', 40), toString(number)), 1000000 + number FROM numbers(1100);
INSERT INTO sj_first SELECT concat(repeat('k', 40), toString(number)), 2000000 + number FROM numbers(1200);
INSERT INTO sj_first SELECT concat(repeat('k', 40), toString(number)), 3000000 + number FROM numbers(1300);
INSERT INTO sj_first SELECT concat(repeat('k', 40), toString(number)), 4000000 + number FROM numbers(1400);
INSERT INTO sj_all SELECT concat(repeat('k', 40), toString(number)), number FROM numbers(1000);
INSERT INTO sj_all SELECT concat(repeat('k', 40), toString(number)), 1000000 + number FROM numbers(1100);
INSERT INTO sj_all SELECT concat(repeat('k', 40), toString(number)), 2000000 + number FROM numbers(1200);
INSERT INTO sj_lc SELECT concat(repeat('k', 40), toString(number % 300)), number FROM numbers(1000);
INSERT INTO sj_lc SELECT concat(repeat('k', 40), toString(number % 300)), 1000000 + number FROM numbers(1000);
INSERT INTO sj_pair SELECT number, number % 7, number FROM numbers(1000);
INSERT INTO sj_pair SELECT number, number % 7, 1000000 + number FROM numbers(1000, 500);

-- The last value of key i is 4000000 + i; the first is j_min * 1000000 + i, where j_min is the first INSERT with i.
SELECT 'last', sum(joinGet('sj_last', 'v', concat(repeat('k', 40), toString(number)))) = sum(4000000 + number),
    countIf(isNull(joinGetOrNull('sj_last', 'v', concat(repeat('k', 40), toString(number + 1400))))) = 1400
FROM numbers(1400);
SELECT 'first', sum(joinGet('sj_first', 'v', concat(repeat('k', 40), toString(number))))
    = sum(if(number < 1000, 0, intDiv(number - 1000, 100) + 1) * 1000000 + number)
FROM numbers(1400);
SELECT 'join', count(), sum(r.v) = (SELECT sum(4000000 + number) FROM numbers(1400))
FROM (SELECT concat(repeat('k', 40), toString(number)) AS s FROM numbers(1500)) l ANY LEFT JOIN sj_last r ON l.s = r.s
WHERE r.v != 0;
SELECT 'read', count(), sum(v) = (SELECT sum(4000000 + number) FROM numbers(1400)) FROM sj_last;
SELECT 'total_rows', total_rows FROM system.tables WHERE database = currentDatabase() AND name = 'sj_last';
-- ALL keeps every row of every INSERT: 1000 + 1100 + 1200 rows.
SELECT 'all', count(), sum(r.v) = (SELECT sum(number) FROM numbers(1000)) + (SELECT sum(1000000 + number) FROM numbers(1100))
    + (SELECT sum(2000000 + number) FROM numbers(1200))
FROM (SELECT concat(repeat('k', 40), toString(number)) AS s FROM numbers(1200)) l ALL LEFT JOIN sj_all r ON l.s = r.s;
-- The last row of key j in the second INSERT is number j + 900 for j < 100 and j + 600 otherwise.
SELECT 'lc', sum(joinGet('sj_lc', 'v', concat(repeat('k', 40), toString(number))))
    = sum(1000000 + if(number < 100, number + 900, number + 600))
FROM numbers(300);
SELECT 'pair', sum(joinGet('sj_pair', 'v', toUInt32(number), toUInt32(number % 7))) = sum(if(number < 1000, number, 1000000 + number))
FROM numbers(1500);
-- `(UInt32, UInt32)` is a `keys64` map, which a read of the table can unpack (`04760_storage_join_composite_key_read`).
SELECT 'pair read', count(), sum(v) FROM sj_pair;

DETACH TABLE sj_last;
ATTACH TABLE sj_last;
SELECT 'last after attach', sum(joinGet('sj_last', 'v', concat(repeat('k', 40), toString(number)))) = sum(4000000 + number)
FROM numbers(1400);

TRUNCATE TABLE sj_last;
SELECT 'truncated', count() FROM sj_last;
INSERT INTO sj_last SELECT concat(repeat('k', 40), toString(number)), 5000000 + number FROM numbers(10);
SELECT 'after truncate', sum(joinGet('sj_last', 'v', concat(repeat('k', 40), toString(number)))) = sum(5000000 + number)
FROM numbers(10);

DROP TABLE sj_last;
DROP TABLE sj_first;
DROP TABLE sj_all;
DROP TABLE sj_lc;
DROP TABLE sj_pair;

