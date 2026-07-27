#!/bin/bash
# Create a ClickHouse worktree with hardlinked submodules, following the
# repo-local create-worktree skill (.claude/skills/create-worktree/SKILL.md)
# build/test path. Usage: worktree_setup.sh <MAIN_REPO> <WORKTREE_PATH> <BRANCH>
set -u

MAIN_REPO=$1
WORKTREE_PATH=$2
BRANCH=$3

if [ -e "$WORKTREE_PATH" ]
then
    printf "FAILED: %s already exists\n" "$WORKTREE_PATH" >&2
    exit 1
fi

git -C "$MAIN_REPO" worktree add --no-checkout "$WORKTREE_PATH" "$BRANCH" || exit 1

GIT_COMMON_DIR=$(git -C "$MAIN_REPO" rev-parse --git-common-dir)
case "$GIT_COMMON_DIR" in
    /*) GIT_DIR=$GIT_COMMON_DIR ;;
    *) GIT_DIR=$MAIN_REPO/$GIT_COMMON_DIR ;;
esac
WORKTREE_ENTRY=$(basename "$WORKTREE_PATH")

# Hardlink-copy the modules directory from the main repo while checking out the
# parent worktree files. These write disjoint paths, and the module copy only
# needs the worktree metadata created by `git worktree add --no-checkout`.
( cp -al "$GIT_DIR/modules" \
         "$GIT_DIR/worktrees/$WORKTREE_ENTRY/modules" ) &
cp_pid=$!

parent_checkout_status=0
git -C "$WORKTREE_PATH" \
    -c checkout.workers=0 \
    -c core.fsync=none \
    -c gc.auto=0 \
    checkout -q -f HEAD -- . || parent_checkout_status=$?

modules_copy_status=0
wait "$cp_pid" || modules_copy_status=$?

if [ "$parent_checkout_status" -ne 0 ]
then
    printf "FAILED: parent checkout\n" >&2
    exit "$parent_checkout_status"
fi

if [ "$modules_copy_status" -ne 0 ]
then
    printf "FAILED: cp -al modules\n" >&2
    exit "$modules_copy_status"
fi

git -C "$WORKTREE_PATH" submodule init &
init_pid=$!

# Fix the worktree pointer inside each submodule's config and config.worktree
# files in one tree walk.
find "$GIT_DIR/worktrees/$WORKTREE_ENTRY/modules" \
    \( -name config -o -name config.worktree \) -exec \
    sed -i "s|worktree = .*/contrib/|worktree = $WORKTREE_PATH/contrib/|" {} +

if ! wait "$init_pid"
then
    printf "FAILED: submodule init\n" >&2
    exit 1
fi

CPU_COUNT=$(nproc)
DEFAULT_SUBMODULE_CHECKOUT_WORKERS=8
if [ "$DEFAULT_SUBMODULE_CHECKOUT_WORKERS" -gt "$CPU_COUNT" ]
then
    DEFAULT_SUBMODULE_CHECKOUT_WORKERS=$CPU_COUNT
fi

SUBMODULE_CHECKOUT_WORKERS=${SUBMODULE_CHECKOUT_WORKERS:-$DEFAULT_SUBMODULE_CHECKOUT_WORKERS}
if [ "$SUBMODULE_CHECKOUT_WORKERS" -lt 1 ]
then
    printf "FAILED: SUBMODULE_CHECKOUT_WORKERS must be positive\n" >&2
    exit 1
fi

SUBMODULE_JOBS=${SUBMODULE_JOBS:-$(( CPU_COUNT / SUBMODULE_CHECKOUT_WORKERS ))}
if [ "$SUBMODULE_JOBS" -lt 1 ]
then
    SUBMODULE_JOBS=1
fi

# This direct materialization path intentionally bypasses `git submodule update`.
# Refuse custom update commands because skipping those would change semantics.
( git -C "$WORKTREE_PATH" config --file .gitmodules --get-regexp "^submodule\\..*\\.update$" 2>/dev/null || true ) |
    while IFS=" " read -r config_key update_command
    do
        case "$update_command" in
            "!"*)
                printf "FAILED: custom submodule update command is unsupported on local hardlink path: %s\n" "$config_key" >&2
                exit 1
                ;;
        esac
    done || exit 1

# Materialize submodule working trees in one parallel pass. If a commit is
# missing from the hardlinked module data, `git checkout` fails locally
# without fetching.
GITLINKS=$(git -C "$WORKTREE_PATH" ls-files -s |
    sed -n "s/^160000 \([0-9a-f][0-9a-f]*\) 0[[:space:]]\(.*\)$/\1 \2/p")
GITLINK_COUNT=$(printf "%s\n" "$GITLINKS" | sed -n '$=')
SUBMODULE_COUNT=$(git -C "$WORKTREE_PATH" config --file .gitmodules --get-regexp "^submodule\\..*\\.path$" |
    sed -n '$=')
if [ "${GITLINK_COUNT:-0}" != "${SUBMODULE_COUNT:-0}" ]
then
    printf "FAILED: gitlink count %s does not match .gitmodules count %s\n" "${GITLINK_COUNT:-0}" "${SUBMODULE_COUNT:-0}" >&2
    exit 1
fi

# Start known heavy submodules first, then emit the full gitlink list and keep
# the first occurrence of each path.
{
    for submodule_path in \
        contrib/llvm-project \
        contrib/google-cloud-cpp \
        contrib/aws \
        contrib/openssl \
        contrib/icu \
        contrib/boost \
        contrib/rust_vendor \
        contrib/sysroot \
        contrib/grpc \
        contrib/arrow \
        contrib/curl \
        contrib/rocksdb \
        contrib/postgres \
        contrib/wasmtime
    do
        printf "%s\n" "$GITLINKS" |
            awk -v p="$submodule_path" '$2 == p { print; exit }'
    done

    printf "%s\n" "$GITLINKS"
} |
    awk "!seen[\$2]++ { print }" |
    while IFS=" " read -r expected_commit submodule_path
    do
        printf "%s\0%s\0" "$expected_commit" "$submodule_path"
    done |
    xargs -0 -r -n2 -P "$SUBMODULE_JOBS" sh -c '
        worktree_path=$1
        git_dir=$2
        worktree_entry=$3
        checkout_workers=$4
        expected_commit=$5
        submodule_path=$6

        if [ -z "$expected_commit" ] || [ -z "$submodule_path" ]
        then
            printf "FAILED: empty submodule checkout tuple\n" >&2
            exit 1
        fi

        module_git_dir="$git_dir/worktrees/$worktree_entry/modules/$submodule_path"
        module_worktree="$worktree_path/$submodule_path"

        mkdir -p "$module_worktree" || exit 1
        printf "gitdir: %s\n" "$module_git_dir" > "$module_worktree/.git" || exit 1

        if ! git --git-dir="$module_git_dir" --work-tree="$module_worktree" \
            -c advice.detachedHead=false \
            -c checkout.workers="$checkout_workers" \
            -c checkout.thresholdForParallelism=100 \
            -c index.threads=true \
            -c core.fsync=none \
            -c gc.auto=0 \
            checkout -q -f --detach "$expected_commit"
        then
            printf "FAILED: %s: commit %s is missing from the local mirror\n" "$submodule_path" "$expected_commit" >&2
            exit 1
        fi
    ' sh "$WORKTREE_PATH" "$GIT_DIR" "$WORKTREE_ENTRY" "$SUBMODULE_CHECKOUT_WORKERS"

submodule_checkout_status=$?
if [ "$submodule_checkout_status" -ne 0 ]
then
    printf "ERROR: a submodule could not be checked out at the commit the superproject records.\n" >&2
    printf "Fetch it in the main repo, then re-run: git -C %s submodule update --init\n" "$MAIN_REPO" >&2
    exit "$submodule_checkout_status"
fi

printf "WORKTREE READY: %s @ %s\n" "$WORKTREE_PATH" "$(git -C "$WORKTREE_PATH" rev-parse HEAD)"
