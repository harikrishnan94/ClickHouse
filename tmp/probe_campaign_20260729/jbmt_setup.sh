#!/bin/bash
# Stand up two local jbmt servers, one per arm, on byte-identical data.
#
# Both arms MUST see identical data or jbmt's cross-arm (row_count, checksum)
# oracle - the only correctness oracle this campaign has for real-suite units -
# would fail on every unit. So the synthetic `keys_store` is generated ONCE on
# arm A's root and arm B's root is then a hardlink clone of it: identical bytes,
# and no second copy of the 392 GB real-suite data either.
#
# Usage: jbmt_setup.sh
set -euo pipefail

CAMP=/mnt/ch/ClickHouse/tmp/probe_campaign_20260729
SRC=/mnt/data/jbmt_server/data
ROOT=/mnt/data/probe_camp_jbmt
BIN_A=$CAMP/bins/clickhouse-baseline-a05f3ee81ff.bin
BIN_B=$CAMP/bins/clickhouse-phjph-fa5667f2da7.bin
SRV=$CAMP/jbmt/join_bench_mt_servers.sh

mkdir -p "$ROOT/a" "$ROOT/b"
cd "$CAMP/jbmt"

echo "=== clone real-suite data root for arm A $(date -u +%FT%TZ)"
[ -d "$ROOT/a/data" ] || "$SRV" clone "$SRC" "$ROOT/a/data"
du -sh --apparent-size "$ROOT/a/data" | tail -1

echo "=== start arm A on the clone $(date -u +%FT%TZ)"
"$SRV" start armA "$BIN_A" "$ROOT/a/data" 9005 8125 "$ROOT/a/scratch"

echo "=== prepare-keys (synthetic suite) $(date -u +%FT%TZ)"
python3 join_bench_mt.py prepare-keys --binary "$BIN_A" --port 9005

echo "=== verify real-suite fingerprints, both tiers $(date -u +%FT%TZ)"
python3 join_bench_mt.py verify --tier a --binary "$BIN_A" --port 9005 \
    --reference /mnt/data/jbmt_server/loads.a.json
python3 join_bench_mt.py verify --tier b --binary "$BIN_A" --port 9005 \
    --reference /mnt/data/jbmt_server/loads.b.json

echo "=== stop arm A so its root is quiescent for cloning $(date -u +%FT%TZ)"
"$SRV" stop "$ROOT/a/scratch"
sleep 5

echo "=== clone arm A -> arm B (identical bytes, incl. keys_store) $(date -u +%FT%TZ)"
rm -rf "$ROOT/b/data"
"$SRV" clone "$ROOT/a/data" "$ROOT/b/data"

echo "=== start both arms $(date -u +%FT%TZ)"
"$SRV" start armA "$BIN_A" "$ROOT/a/data" 9005 8125 "$ROOT/a/scratch"
"$SRV" start armB "$BIN_B" "$ROOT/b/data" 9006 8126 "$ROOT/b/scratch"

echo "=== both arms up: confirm identity and algorithm availability $(date -u +%FT%TZ)"
for p in 9005 9006; do
    echo "--- port $p"
    "$BIN_A" client --port "$p" -q "SELECT value FROM system.build_options WHERE name = 'GIT_HASH'"
    "$BIN_A" client --port "$p" -q "SELECT count() FROM keys_store.k0"
done
echo "=== READY $(date -u +%FT%TZ)"
