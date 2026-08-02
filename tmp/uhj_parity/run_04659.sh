#!/usr/bin/env bash
# 04659 equivalence without duplicate --max_threads (client rejects duplicates).
# Mirrors tests/queries/0_stateless/04659_unified_hash_join_parallel_build.sh logic.
set -euo pipefail
CH=(/mnt/ch/ClickHouse/build/reldeb/programs/clickhouse client --host 127.0.0.1 --port "${UHJ_PORT:-9101}")

RIGHT_DUP="SELECT number % 200000 AS k, toInt64(number) AS v, toString(number % 200000) AS s FROM numbers_mt(300000)"
RIGHT_UNIQ="SELECT number AS k, toInt64(number) * 2 AS v, toString(number) AS s FROM numbers_mt(200000)"
LEFT="SELECT number % 250000 AS k, toString(number % 250000) AS s FROM numbers_mt(300000)"

compare_join_algorithms()
{
    local query="$1"; shift
    local hash_result unified_result
    hash_result=$("${CH[@]}" --enable_analyzer=1 --max_block_size=10000 "$@" --join_algorithm=hash -q "$query")
    unified_result=$("${CH[@]}" --enable_analyzer=1 --max_block_size=10000 "$@" --join_algorithm=unified_hash -q "$query")
    if [ "$hash_result" != "$unified_result" ]; then
        echo "Mismatch for query: $query"
        echo "extra: $*"
        echo "--- hash ---"; echo "$hash_result"
        echo "--- unified_hash ---"; echo "$unified_result"
        exit 1
    fi
}

run_all_queries()
{
    local threads="${1:-16}"; shift || true
    local extra=("$@")
    compare_join_algorithms "WITH l AS ($LEFT), r AS ($RIGHT_DUP)
        SELECT count(), sum(l.k), sum(r.v) FROM l INNER JOIN r ON l.k = r.k" --max_threads="$threads" "${extra[@]}"
    compare_join_algorithms "WITH l AS ($LEFT), r AS ($RIGHT_DUP)
        SELECT count(), sum(l.k), sum(r.v) FROM l LEFT JOIN r ON l.k = r.k" --max_threads="$threads" "${extra[@]}"
    compare_join_algorithms "WITH l AS ($LEFT), r AS ($RIGHT_DUP)
        SELECT count(), sum(l.k), sum(r.v) FROM l RIGHT JOIN r ON l.k = r.k" --max_threads="$threads" "${extra[@]}"
    compare_join_algorithms "WITH l AS ($LEFT), r AS ($RIGHT_DUP)
        SELECT count(), sum(l.k), sum(r.v) FROM l FULL JOIN r ON l.k = r.k" --max_threads="$threads" "${extra[@]}"
    compare_join_algorithms "WITH l AS ($LEFT), r AS ($RIGHT_DUP)
        SELECT count(), sum(r.v) FROM l INNER JOIN r ON l.s = r.s" --max_threads="$threads" "${extra[@]}"
    compare_join_algorithms "WITH l AS ($LEFT), r AS ($RIGHT_DUP)
        SELECT count(), sum(r.v) FROM l INNER JOIN r ON l.k = r.k AND l.s = r.s" --max_threads="$threads" "${extra[@]}"
    compare_join_algorithms "WITH l AS ($LEFT), r AS ($RIGHT_UNIQ)
        SELECT count(), sum(l.k), sum(r.v) FROM l LEFT ANY JOIN r ON l.k = r.k" --max_threads="$threads" "${extra[@]}"
    compare_join_algorithms "WITH l AS ($LEFT), r AS ($RIGHT_UNIQ)
        SELECT count(), sum(l.k) FROM l LEFT SEMI JOIN r ON l.k = r.k" --max_threads="$threads" "${extra[@]}"
    compare_join_algorithms "WITH l AS ($LEFT), r AS ($RIGHT_UNIQ)
        SELECT count(), sum(l.k) FROM l LEFT ANTI JOIN r ON l.k = r.k" --max_threads="$threads" "${extra[@]}"
}

run_all_queries 16
run_all_queries 16 --max_bytes_before_external_join=100000000
run_all_queries 16 --max_bytes_before_external_join=8000000

for threads in 1 3 64; do
    compare_join_algorithms "WITH l AS ($LEFT), r AS ($RIGHT_DUP)
        SELECT count(), sum(l.k), sum(r.v) FROM l INNER JOIN r ON l.k = r.k" --max_threads="$threads"
    compare_join_algorithms "WITH l AS ($LEFT), r AS ($RIGHT_DUP)
        SELECT count(), sum(l.k), sum(r.v) FROM l FULL JOIN r ON l.k = r.k" --max_threads="$threads"
    compare_join_algorithms "WITH l AS ($LEFT), r AS ($RIGHT_DUP)
        SELECT count(), sum(r.v) FROM l INNER JOIN r ON l.s = r.s" --max_threads="$threads"
    compare_join_algorithms "WITH l AS ($LEFT), r AS ($RIGHT_UNIQ)
        SELECT count(), sum(l.k) FROM l LEFT SEMI JOIN r ON l.k = r.k" --max_threads="$threads"
done

count_filling_transforms()
{
    "${CH[@]}" --enable_analyzer=1 --max_threads=8 --join_algorithm=unified_hash "$@" -q \
        "EXPLAIN PIPELINE WITH l AS ($LEFT), r AS ($RIGHT_DUP)
            SELECT count() FROM l INNER JOIN r ON l.k = r.k" | grep -c FillingRightJoinSide
}

for transforms in $(count_filling_transforms) $(count_filling_transforms --max_bytes_before_external_join=100000000); do
    if [ "$transforms" -lt 2 ]; then
        echo "Expected parallel FillingRightJoinSide >=2, got $transforms"
        exit 1
    fi
done

echo "OK"
echo "JOB_EXIT=0"
