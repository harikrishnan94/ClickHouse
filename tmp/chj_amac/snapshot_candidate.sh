#!/bin/bash
# Rebuild the candidate binary from the main repo at the current phj-ph HEAD
# and snapshot it into tmp/chj_amac/bins/ with its sha256 (MANIFEST.tsv is
# append-only; the last row for a given name describes the snapshot on disk).
# Re-runs cmake first so the embedded GIT_HASH is refreshed (it is baked at
# configure time and known to go stale); sha256 remains the identity of record.
set -u

MAIN_REPO=/mnt/ch/ClickHouse
BDIR=$MAIN_REPO/build/reldeb
BINS=$MAIN_REPO/tmp/chj_amac/bins

if [ -n "$(git -C "$MAIN_REPO" status --porcelain)" ]
then
    printf "FAILED: working tree not clean; refusing to snapshot an unlabelled state\n" >&2
    git -C "$MAIN_REPO" status --porcelain >&2
    exit 1
fi

HEAD=$(git -C "$MAIN_REPO" rev-parse HEAD)
SHORT=$(git -C "$MAIN_REPO" rev-parse --short "$HEAD")
TAG=candidate-$SHORT
LOG=build_${TAG}.log

( cd "$BDIR" && cmake . > "cmake_${TAG}.log" 2>&1 ) || {
    printf "FAILED: cmake re-configure (see %s/cmake_%s.log)\n" "$BDIR" "$TAG" >&2
    exit 1
}
ninja -C "$BDIR" clickhouse > "$BDIR/$LOG" 2>&1 || {
    printf "FAILED: ninja (see %s/%s)\n" "$BDIR" "$LOG" >&2
    exit 1
}

mkdir -p "$BINS"
SNAP=$BINS/clickhouse-${TAG}.bin
cp "$BDIR/programs/clickhouse" "$SNAP" || exit 1
SHA=$(sha256sum "$SNAP" | cut -d' ' -f1)
BYTES=$(stat -c%s "$SNAP")
printf "%s\t%s\t%s\t%s\t%s\t%s\n" \
    "clickhouse-${TAG}.bin" "$SHA" "$BYTES" "$HEAD" "$BDIR/$LOG" "$(date -u +%FT%TZ)" \
    >> "$BINS/MANIFEST.tsv"
printf "BUILT %s: %s sha256=%s\n" "$TAG" "$SNAP" "$SHA"
