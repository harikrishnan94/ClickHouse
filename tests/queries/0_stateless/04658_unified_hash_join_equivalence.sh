#!/usr/bin/env bash
# Tags: no-fasttest

CUR_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=../shell_config.sh
. "$CUR_DIR"/../shell_config.sh

set -e

LOCAL_CLIENT="${CLICKHOUSE_BINARY:-clickhouse} local --multiquery"

compare_join_algorithms()
{
    local query="$1"
    local hash_result unified_result

    hash_result=$($LOCAL_CLIENT --join_algorithm=hash --enable_analyzer=1 -q "$query")
    unified_result=$($LOCAL_CLIENT --join_algorithm=unified_hash --enable_analyzer=1 -q "$query")

    if [ "$hash_result" != "$unified_result" ]; then
        echo "Mismatch for query:"
        echo "$query"
        echo "--- hash ---"
        echo "$hash_result"
        echo "--- unified_hash ---"
        echo "$unified_result"
        exit 1
    fi
}

LEFT_DATA="SELECT * FROM VALUES('k UInt64, s String, n Nullable(UInt64)', (1, 'a', 10), (2, 'b', NULL), (3, 'c', 30), (4, 'd', 40))"
RIGHT_DATA="SELECT * FROM VALUES('k UInt64, v Int32, s String, n Nullable(UInt64)', (1, 100, 'a', 1), (2, 200, 'b', NULL), (3, 300, 'c', 3), (5, 500, 'e', 5))"

compare_join_algorithms "WITH uhj_left AS ($LEFT_DATA), uhj_right AS ($RIGHT_DATA) SELECT l.k, r.v FROM uhj_left AS l INNER JOIN uhj_right AS r ON l.k = r.k ORDER BY l.k, r.v"
compare_join_algorithms "WITH uhj_left AS ($LEFT_DATA), uhj_right AS ($RIGHT_DATA) SELECT l.k, r.v FROM uhj_left AS l LEFT JOIN uhj_right AS r ON l.k = r.k ORDER BY l.k, r.v"
compare_join_algorithms "WITH uhj_left AS ($LEFT_DATA), uhj_right AS ($RIGHT_DATA) SELECT l.k, r.v FROM uhj_left AS l RIGHT JOIN uhj_right AS r ON l.k = r.k ORDER BY l.k, r.v"
compare_join_algorithms "WITH uhj_left AS ($LEFT_DATA), uhj_right AS ($RIGHT_DATA) SELECT l.k, r.v FROM uhj_left AS l FULL JOIN uhj_right AS r ON l.k = r.k ORDER BY l.k, r.v"

compare_join_algorithms "WITH uhj_left AS ($LEFT_DATA), uhj_right AS ($RIGHT_DATA) SELECT l.k, r.v FROM uhj_left AS l INNER ANY JOIN uhj_right AS r ON l.k = r.k ORDER BY l.k, r.v"
compare_join_algorithms "WITH uhj_left AS ($LEFT_DATA), uhj_right AS ($RIGHT_DATA) SELECT l.k, r.v FROM uhj_left AS l LEFT ANY JOIN uhj_right AS r ON l.k = r.k ORDER BY l.k, r.v"
compare_join_algorithms "WITH uhj_left AS ($LEFT_DATA), uhj_right AS ($RIGHT_DATA) SELECT l.k FROM uhj_left AS l LEFT SEMI JOIN uhj_right AS r ON l.k = r.k ORDER BY l.k"
compare_join_algorithms "WITH uhj_left AS ($LEFT_DATA), uhj_right AS ($RIGHT_DATA) SELECT l.k FROM uhj_left AS l LEFT ANTI JOIN uhj_right AS r ON l.k = r.k ORDER BY l.k"

compare_join_algorithms "WITH uhj_left AS ($LEFT_DATA), uhj_right AS ($RIGHT_DATA) SELECT l.k, r.v FROM uhj_left AS l INNER JOIN uhj_right AS r USING (k) ORDER BY l.k, r.v"
compare_join_algorithms "WITH uhj_left AS ($LEFT_DATA), uhj_right AS ($RIGHT_DATA) SELECT l.k, r.v FROM uhj_left AS l INNER JOIN uhj_right AS r ON l.n = r.n ORDER BY l.k, r.v"

compare_join_algorithms "WITH uhj_left AS ($LEFT_DATA), uhj_right AS ($RIGHT_DATA) SELECT l.k, r.v FROM uhj_left AS l INNER JOIN uhj_right AS r ON l.k = r.k AND l.s = r.s ORDER BY l.k, r.v"
compare_join_algorithms "WITH uhj_left AS ($LEFT_DATA), uhj_right AS ($RIGHT_DATA) SELECT l.k, r.v FROM uhj_left AS l INNER JOIN uhj_right AS r ON l.k = r.k OR l.k = 3 ORDER BY l.k, r.v"

compare_join_algorithms "WITH uhj_left AS ($LEFT_DATA), uhj_right AS ($RIGHT_DATA) SELECT l.k, r.v FROM uhj_left AS l INNER JOIN uhj_right AS r ON l.k = r.k ORDER BY l.k, r.v SETTINGS max_bytes_before_external_join = 1"

echo "OK"
