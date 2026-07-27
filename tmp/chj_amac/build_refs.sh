#!/bin/bash
# Build the two reference binaries for the AMAC mission, sequentially:
#   1. baseline: branch concurrent-hash-join-profile-events (a05f3ee81ff)
#   2. disasm reference: branch ahj (cf465cfbe23)
# Each: hardlinked worktree -> cmake (candidate's flags) -> ninja (log in the
# build dir) -> snapshot binary + sha256 into tmp/chj_amac/bins/MANIFEST.tsv.
set -u

MAIN_REPO=/mnt/ch/ClickHouse
BINS=$MAIN_REPO/tmp/chj_amac/bins
SETUP=$MAIN_REPO/tmp/chj_amac/worktree_setup.sh

CMAKE_FLAGS=(
    -G Ninja
    -DCMAKE_BUILD_TYPE=RelWithDebInfo
    -DCMAKE_C_COMPILER=/usr/local/bin/clang-22
    -DCMAKE_CXX_COMPILER=/usr/local/bin/clang++-22
    -DCMAKE_C_COMPILER_LAUNCHER=ccache
    -DCMAKE_CXX_COMPILER_LAUNCHER=ccache
    -DENABLE_RUST=OFF
)

build_one() {
    local branch=$1 expected_head=$2 worktree=$3 tag=$4
    local bdir=$worktree/build/reldeb
    local log=build_${tag}.log

    if [ ! -d "$worktree" ]
    then
        bash "$SETUP" "$MAIN_REPO" "$worktree" "$branch" || return 1
    fi

    local head
    head=$(git -C "$worktree" rev-parse HEAD)
    if [ "$head" != "$expected_head" ]
    then
        printf "FAILED: %s is at %s, expected %s\n" "$worktree" "$head" "$expected_head" >&2
        return 1
    fi

    mkdir -p "$bdir"
    ( cd "$bdir" && cmake "${CMAKE_FLAGS[@]}" "$worktree" > cmake_${tag}.log 2>&1 ) || {
        printf "FAILED: cmake for %s (see %s/cmake_%s.log)\n" "$tag" "$bdir" "$tag" >&2
        return 1
    }
    ninja -C "$bdir" clickhouse > "$bdir/$log" 2>&1 || {
        printf "FAILED: ninja for %s (see %s/%s)\n" "$tag" "$bdir" "$log" >&2
        return 1
    }

    local bin=$bdir/programs/clickhouse
    local snap=$BINS/clickhouse-${tag}.bin
    cp "$bin" "$snap" || return 1
    local sha bytes
    sha=$(sha256sum "$snap" | cut -d' ' -f1)
    bytes=$(stat -c%s "$snap")
    printf "%s\t%s\t%s\t%s\t%s\t%s\n" \
        "clickhouse-${tag}.bin" "$sha" "$bytes" "$head" "$bdir/$log" "$(date -u +%FT%TZ)" \
        >> "$BINS/MANIFEST.tsv"
    printf "BUILT %s: %s sha256=%s\n" "$tag" "$snap" "$sha"
}

if [ ! -f "$BINS/MANIFEST.tsv" ]
then
    printf "name\tsha256\tbytes\tsource_commit\tbuild_log\tbuilt_at\n" > "$BINS/MANIFEST.tsv"
fi

build_one concurrent-hash-join-profile-events a05f3ee81ff8411759637fa367aad62e72726e71 \
    /mnt/ch/ClickHouse-concurrent-hash-join-profile-events baseline-a05f3ee81ff || exit 1
build_one ahj cf465cfbe23a14f982d1bc36510f3e311ce6379f \
    /mnt/ch/ClickHouse-ahj ahj-cf465cfbe23 || exit 1

printf "ALL REFERENCE BUILDS DONE\n"
