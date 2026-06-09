-- Tags: no-random-settings
-- The `radix_hash` join algorithm must produce results identical to `hash` for every join shape:
-- when its key gate applies (a single or composite fixed-width key whose packed width is a multiple
-- of 4 in [4, 64]) it runs the radix path, and otherwise (String / LowCardinality / Nullable / sub-4B
-- or non-multiple-of-4 keys) it falls back to `parallel_hash`. Either way the output must match `hash`.

SET enable_analyzer = 1;

DROP TABLE IF EXISTS rhj_b;
DROP TABLE IF EXISTS rhj_p;

CREATE TABLE rhj_b (a UInt64, b UInt32, c UInt8, s String, lc LowCardinality(String), n Nullable(UInt64), pay UInt64) ENGINE = Memory;
CREATE TABLE rhj_p (a UInt64, b UInt32, c UInt8, s String, lc LowCardinality(String), n Nullable(UInt64), pay UInt64) ENGINE = Memory;

-- Overlapping but unequal key ranges on the two sides, with duplicate keys (many-to-many).
INSERT INTO rhj_b SELECT number % 100, toUInt32(number % 100), toUInt8(number % 100), toString(number % 100), toString(number % 100), if(number % 7 = 0, NULL, number % 100), number FROM numbers(300);
INSERT INTO rhj_p SELECT number % 150, toUInt32(number % 150), toUInt8(number % 150), toString(number % 150), toString(number % 150), if(number % 5 = 0, NULL, number % 150), number FROM numbers(200);

-- Each row prints the case name and 1 when radix_hash agrees with hash on (count, value fingerprint).
SELECT 'single_u64', (SELECT (count(), sum(cityHash64(p.pay, b.pay))) FROM rhj_p AS p INNER JOIN rhj_b AS b ON p.a = b.a SETTINGS join_algorithm = 'radix_hash')
                   = (SELECT (count(), sum(cityHash64(p.pay, b.pay))) FROM rhj_p AS p INNER JOIN rhj_b AS b ON p.a = b.a SETTINGS join_algorithm = 'hash');

SELECT 'single_u32', (SELECT (count(), sum(cityHash64(p.pay, b.pay))) FROM rhj_p AS p INNER JOIN rhj_b AS b ON p.b = b.b SETTINGS join_algorithm = 'radix_hash')
                   = (SELECT (count(), sum(cityHash64(p.pay, b.pay))) FROM rhj_p AS p INNER JOIN rhj_b AS b ON p.b = b.b SETTINGS join_algorithm = 'hash');

SELECT 'composite_u64_u32', (SELECT (count(), sum(cityHash64(p.pay, b.pay))) FROM rhj_p AS p INNER JOIN rhj_b AS b ON p.a = b.a AND p.b = b.b SETTINGS join_algorithm = 'radix_hash')
                          = (SELECT (count(), sum(cityHash64(p.pay, b.pay))) FROM rhj_p AS p INNER JOIN rhj_b AS b ON p.a = b.a AND p.b = b.b SETTINGS join_algorithm = 'hash');

SELECT 'fallback_u8', (SELECT (count(), sum(cityHash64(p.pay, b.pay))) FROM rhj_p AS p INNER JOIN rhj_b AS b ON p.c = b.c SETTINGS join_algorithm = 'radix_hash')
                    = (SELECT (count(), sum(cityHash64(p.pay, b.pay))) FROM rhj_p AS p INNER JOIN rhj_b AS b ON p.c = b.c SETTINGS join_algorithm = 'hash');

SELECT 'fallback_string', (SELECT (count(), sum(cityHash64(p.pay, b.pay))) FROM rhj_p AS p INNER JOIN rhj_b AS b ON p.s = b.s SETTINGS join_algorithm = 'radix_hash')
                        = (SELECT (count(), sum(cityHash64(p.pay, b.pay))) FROM rhj_p AS p INNER JOIN rhj_b AS b ON p.s = b.s SETTINGS join_algorithm = 'hash');

SELECT 'fallback_lowcard', (SELECT (count(), sum(cityHash64(p.pay, b.pay))) FROM rhj_p AS p INNER JOIN rhj_b AS b ON p.lc = b.lc SETTINGS join_algorithm = 'radix_hash')
                         = (SELECT (count(), sum(cityHash64(p.pay, b.pay))) FROM rhj_p AS p INNER JOIN rhj_b AS b ON p.lc = b.lc SETTINGS join_algorithm = 'hash');

SELECT 'fallback_nullable', (SELECT (count(), sum(cityHash64(p.pay, b.pay))) FROM rhj_p AS p INNER JOIN rhj_b AS b ON p.n = b.n SETTINGS join_algorithm = 'radix_hash')
                          = (SELECT (count(), sum(cityHash64(p.pay, b.pay))) FROM rhj_p AS p INNER JOIN rhj_b AS b ON p.n = b.n SETTINGS join_algorithm = 'hash');

SELECT 'fallback_composite_nm4', (SELECT (count(), sum(cityHash64(p.pay, b.pay))) FROM rhj_p AS p INNER JOIN rhj_b AS b ON p.a = b.a AND p.c = b.c SETTINGS join_algorithm = 'radix_hash')
                               = (SELECT (count(), sum(cityHash64(p.pay, b.pay))) FROM rhj_p AS p INNER JOIN rhj_b AS b ON p.a = b.a AND p.c = b.c SETTINGS join_algorithm = 'hash');

DROP TABLE rhj_b;
DROP TABLE rhj_p;
