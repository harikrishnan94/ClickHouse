"""Integration tests for the shm() table function (zero-copy SHM source, Phase 5).

Covers:
  * T5.2 / AC1   - bit-identical functional correctness vs an in-process C++ reference
                   path (utils/shm-producer --print-reference-values).
  * T5.3 / AC4   - pollable wiring (max_threads = 1, max_threads = 4) + cancellation
                   under T (T = 5 / 30 / 100 x interactive_delay per build mode).
  * T5.4 / AC5   - MemoryTracker baseline + limit-failure rollback + tracker reflection.
  * T5.5 / AC6   - producer-misbehaviour failure-class matrix (one test per named class,
                   including attach EACCES, readiness-locator-unresolvable, and
                   mid-publication crash).
  * T5.6 / AC7   - >=1000-iteration leak audit (stable fds + SHM segments + tracker baseline).
  * T5.6 / AC10  - republish-after-retain + held-chunk byte stability under retain
                   (proves consumer's observed bytes == published bytes via the seed
                   reference even when the producer is contending for the same slot).

Specs:
  - ~/auto_click/specs/system.md                     - AC1, AC7.
  - ~/auto_click/specs/pollable-shm-source.md        - AC4, AC6, Failure classes table.
  - ~/auto_click/specs/memory-tracker-integration.md - AC5, I7, I8.
  - ~/auto_click/specs/shm-block-stream.md           - AC10.

These tests drive shm-producer (plan task T4.1) as a detached background
process inside the test container. The producer's CLI scenarios reproduce every AC6 class.
"""

import os
import re
import sys
import threading
import time
import uuid

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from helpers.client import QueryRuntimeException
from helpers.cluster import ClickHouseCluster

# ---------------------------------------------------------------------------
# Error codes (mirrored from src/Common/ErrorCodes.cpp; plan tasks T0.2 + T2.3).
# Tests assert on these integers, per the AC6 brief: "asserts on the failure
# class, not on string content or generic exception type".
# ---------------------------------------------------------------------------
ERR_MEMORY_LIMIT_EXCEEDED = 241
ERR_SUPPORT_IS_DISABLED = 344
ERR_SHM_ATTACH_FAILED = 772
ERR_SHM_HANDSHAKE_INVALID = 773
ERR_SHM_SCHEMA_MISMATCH = 774
ERR_SHM_BLOCK_FRAMING_INVALID = 775
ERR_SHM_BUFFER_LAYOUT_INVALID = 776
ERR_SHM_PRODUCER_STALL = 777
ERR_SHM_PRODUCER_DEATH_BEFORE_EOS = 778

# ---------------------------------------------------------------------------
# Schema + AC1 query (system.md AC1).
# ---------------------------------------------------------------------------
SCHEMA = "id UInt64, v1 UInt64, v2 UInt64, s1 String, s2 String"

AC1_QUERY_TEMPLATE = (
    "SELECT count(), sum(id), sum(v1), sum(v2),"
    " sum(cityHash64(s1)), sum(cityHash64(s2)),"
    " sum(length(s1)), sum(length(s2))"
    " FROM shm('{name}', '{schema}')"
)

PRODUCER_BIN = "shm-producer"

# ---------------------------------------------------------------------------
# Cluster / fixtures
# ---------------------------------------------------------------------------
cluster = ClickHouseCluster(__file__)
node = cluster.add_instance("node", stay_alive=True)


@pytest.fixture(scope="module", autouse=True)
def start_cluster():
    try:
        cluster.start()
        if not _producer_binary_installed():
            # Fail LOUDLY rather than skip: a missing producer is a build /
            # CI-mount regression that must surface, not a silently passed
            # AC1/AC4/AC5/AC6/AC7/AC10 suite. The binary is installed by
            # utils/shm-producer/CMakeLists.txt; if you see this in CI the
            # docker-compose mount in tests/integration/helpers/cluster.py
            # (or the runner image) likely needs the same single-file bind
            # the `clickhouse` binary gets.
            pytest.fail(
                f"'{PRODUCER_BIN}' is not on PATH in the test container; "
                "install/build it before running this suite (see "
                "utils/shm-producer/CMakeLists.txt; plan task T4.1)."
            )
        yield cluster
    finally:
        # Best-effort: clean up any producer left behind by an aborted test.
        node.exec_in_container(
            ["bash", "-c", f"pkill -9 -f '{PRODUCER_BIN}' 2>/dev/null; true"],
            nothrow=True,
        )
        cluster.shutdown()


@pytest.fixture
def fresh_shm_name(request):
    """Per-test unique SHM name with before/after cleanup of /dev/shm + socket + producer."""
    # POSIX shm names should be safe identifiers; restrict to [A-Za-z0-9_].
    base = re.sub(r"[^A-Za-z0-9_]", "_", request.node.name)
    name = f"shm_test_{base}_{int(time.time() * 1000) & 0xFFFFFFFF}_{os.getpid()}"
    _kill_producers_for(name)
    _cleanup_shm(name)
    try:
        yield name
    finally:
        _kill_producers_for(name)
        _cleanup_shm(name)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
_CODE_RE = re.compile(r"Code:\s*(\d+)")


def _producer_binary_installed():
    out = node.exec_in_container(
        ["bash", "-c", f"command -v {PRODUCER_BIN} 2>/dev/null || true"],
        nothrow=True,
    )
    return bool(out and out.strip())


def _shm_basename(name):
    """Strip leading '/' so we can address the /dev/shm filename uniformly."""
    return name.lstrip("/").replace("/", "_")


def _socket_path(name):
    return f"/tmp/clickhouse_shm_{_shm_basename(name)}.sock"


def _devshm_path(name):
    return f"/dev/shm/{_shm_basename(name)}"


def _cleanup_shm(name):
    """Remove stale SHM file + control socket. Idempotent."""
    node.exec_in_container(
        ["bash", "-c", f"rm -f {_devshm_path(name)} {_socket_path(name)}"],
        nothrow=True,
    )


def _kill_producers_for(name):
    """Kill any shm-producer process spawned for this specific SHM name."""
    # pkill -f matches the full command line; the --name flag is unique per test.
    node.exec_in_container(
        ["bash", "-c", f"pkill -9 -f '{PRODUCER_BIN}.*--name {name}' 2>/dev/null; true"],
        nothrow=True,
    )


def _start_producer(
    name,
    *,
    rows,
    seed=42,
    scenario="normal",
    scenario_arg=10,
    ring_depth=4,
    data_region_size=None,
    rows_per_block=None,
    wait_ready=True,
    ready_timeout=10.0,
):
    """Spawn shm-producer as a detached subprocess on the test node.

    Optionally blocks until the SHM file + control socket appear on disk (so a
    subsequent SELECT can attach without racing the producer's setup phase).
    """
    cmd = [
        PRODUCER_BIN,
        "--name", name,
        "--rows", str(rows),
        "--seed", str(seed),
        "--scenario", scenario,
        "--scenario-arg", str(scenario_arg),
        "--ring-depth", str(ring_depth),
    ]
    if data_region_size is not None:
        cmd += ["--data-region-size", str(data_region_size)]
    if rows_per_block is not None:
        cmd += ["--rows-per-block", str(rows_per_block)]
    node.exec_in_container(cmd, detach=True)
    if wait_ready:
        _wait_for_producer_ready(name, ready_timeout)


def _wait_for_producer_ready(name, timeout):
    """Poll until both /dev/shm/<name> and /tmp/clickhouse_shm_<name>.sock exist.

    The producer creates the SHM file first (with the handshake magic written
    LAST under release ordering - see InProcessProducer.cpp::populateHandshake),
    then binds the Unix socket. Once both exist, the consumer can safely attach.
    """
    deadline = time.time() + timeout
    check = (
        f"[ -e {_devshm_path(name)} ] && [ -S {_socket_path(name)} ] "
        "&& echo ok || echo no"
    )
    while time.time() < deadline:
        result = node.exec_in_container(["bash", "-c", check], nothrow=True)
        if result and result.strip() == "ok":
            return
        time.sleep(0.05)
    raise RuntimeError(
        f"Producer not ready for SHM '{name}' within {timeout}s; "
        f"missing {_devshm_path(name)} or {_socket_path(name)}"
    )


def _extract_error_code(err):
    """Return the integer following the first 'Code: <NNN>' in an error.

    Accepts either a raw stderr string (from node.query_and_get_error) or
    a QueryRuntimeException (from node.query failures).
    """
    if isinstance(err, QueryRuntimeException):
        err_text = f"{err}\n{err.stderr or ''}"
    else:
        err_text = str(err)
    match = _CODE_RE.search(err_text)
    if match is None:
        raise AssertionError(f"No 'Code: NNN' marker in error string:\n{err_text}")
    return int(match.group(1))


def _ac1_sql(name, *, schema=SCHEMA):
    return AC1_QUERY_TEMPLATE.format(name=name, schema=schema)


def _run_ac1_query(name, *, settings=None, query_id=None, schema=SCHEMA):
    base_settings = {"allow_experimental_shm_table_function": 1}
    if settings:
        base_settings.update(settings)
    return node.query(_ac1_sql(name, schema=schema), settings=base_settings, query_id=query_id)


def _shm_metric_value(metric):
    """Read a single Shm* gauge from system.metrics."""
    raw = node.query(
        f"SELECT value FROM system.metrics WHERE metric = '{metric}'"
    ).strip()
    return int(raw) if raw else 0


def _shm_event_value(event):
    """Read a single Shm* cumulative counter from system.events.

    Returns 0 when the event has not been emitted yet (the row is absent).
    """
    raw = node.query(
        f"SELECT value FROM system.events WHERE event = '{event}'"
    ).strip()
    return int(raw) if raw else 0


def _devshm_count():
    """Number of entries in /dev/shm (used by AC7 leak audit)."""
    return int(
        node.exec_in_container(
            ["bash", "-c", "ls -1 /dev/shm 2>/dev/null | wc -l"]
        ).strip()
    )


def _server_fd_count():
    """Number of open file descriptors held by clickhouse-server (AC7)."""
    pid = node.get_process_pid("clickhouse server")
    assert pid is not None, "clickhouse-server is not running"
    return int(
        node.exec_in_container(
            ["bash", "-c", f"ls -1 /proc/{pid}/fd 2>/dev/null | wc -l"]
        ).strip()
    )


def _flush_logs():
    node.query("SYSTEM FLUSH LOGS")


# ---------------------------------------------------------------------------
# AC1 reference oracle: invoke the standalone shm-producer with
# --print-reference-values to compute the AC1 8-tuple from (seed, rows) in C++,
# matching the producer's BlockBuilder rng-draw order byte-for-byte. This is the
# independent C++ reference path system.md AC1 calls for; without it the test
# could only check internal determinism (two runs of the same seed produce the
# same result), not bit-identity against a reference computation.
# ---------------------------------------------------------------------------

REFERENCE_VALUES_COLS = 8


def _print_reference_values(*, rows, seed):
    """Run `shm-producer --print-reference-values --rows N --seed S` and parse.

    Returns the 8 string-typed reference columns matching what
    `node.query(AC1_QUERY_TEMPLATE).strip().split('\t')` produces against a
    clean `--scenario normal --rows N --seed S` producer.
    """
    out = node.exec_in_container(
        [PRODUCER_BIN, "--rows", str(rows), "--seed", str(seed), "--print-reference-values"]
    ).strip()
    parts = out.split("\t")
    assert len(parts) == REFERENCE_VALUES_COLS, (
        f"expected {REFERENCE_VALUES_COLS} reference values, got {len(parts)}: {out!r}"
    )
    return parts


# Pin the spec's reference unit so the budget below is interpretable without
# build-side knowledge of interactive_delay's default. Per pollable-shm-source.md
# AC4 T is expressed as a multiple of interactive_delay; we set the multiplicand
# explicitly in cancellation-related queries (via _cancel_settings()) so the
# 5x/30x/100x conversion below is a literal computation, not an inference.
INTERACTIVE_DELAY_US = 100_000  # 100 ms — matches the server-side default.


def _cancellation_budget_sec():
    """AC4 build-mode-parameterised cancellation budget T.

    Per pollable-shm-source.md AC4 the spec budgets are 5 x interactive_delay
    (Release), 30 x (sanitisers), 100 x (Debug). With INTERACTIVE_DELAY_US =
    100 ms that yields 0.5 s / 3.0 s / 10.0 s respectively. We add a single
    1.0 s additive overhead to absorb the docker-exec round-trip for
    KILL QUERY ... SYNC and the post-kill wait observation; this is *additive*
    (not multiplicative) because the docker overhead is a fixed cost per call,
    not a constant factor on the source's unwind cost. Exceeding the resulting
    budget is the test-level signal for stop-condition S4.
    """
    docker_exec_overhead = 1.0
    if node.is_debug_build():
        return 100 * (INTERACTIVE_DELAY_US / 1_000_000) + docker_exec_overhead   # 11.0 s
    if (
        node.is_built_with_address_sanitizer()
        or node.is_built_with_thread_sanitizer()
        or node.is_built_with_memory_sanitizer()
        or node.is_built_with_sanitizer("undefined")
    ):
        return 30 * (INTERACTIVE_DELAY_US / 1_000_000) + docker_exec_overhead    # 4.0 s
    return 5 * (INTERACTIVE_DELAY_US / 1_000_000) + docker_exec_overhead         # 1.5 s


def _cancel_settings():
    """Query settings used by cancellation tests. interactive_delay pins the
    spec's reference unit so _cancellation_budget_sec()'s 5x/30x/100x mapping
    is a literal computation, independent of whatever default the build uses.
    """
    return {
        "allow_experimental_shm_table_function": 1,
        "interactive_delay": INTERACTIVE_DELAY_US,
    }


# ===========================================================================
# T5.2 - AC1 functional correctness
# ===========================================================================


def test_ac1_bit_identical(fresh_shm_name):
    """system.md AC1: bit-identical functional correctness against a reference path.

    Strategy. `shm-producer --print-reference-values --rows N --seed S` replays
    BlockBuilder's per-row rng draws (a single std::mt19937_64 seeded with S
    drives v1, v2, s1_len, s1_chars, s2_len, s2_chars in fixed order; id[i] = i)
    and prints the AC1 8-tuple -- count, sum(id), sum(v1), sum(v2),
    sum(cityHash64(s1)), sum(cityHash64(s2)), sum(length(s1)), sum(length(s2))
    -- TAB-separated. That is the independent C++ reference path the spec
    asks for ("compared against a reference path (e.g. Values or Native)").
    cityHash64 uses the same CityHash_v1_0_2::CityHash64 the SQL function
    dispatches to (src/Functions/FunctionsHashing.h::ImplCityHash64::apply),
    so the bytes-in/hash-out mapping is identical across both code paths.

    The earlier two-producer determinism proxy only proved internal stability
    (same seed -> same result) -- it could NOT distinguish a source that drops
    every other block from a source that drains correctly (both would be
    deterministic). This rewrite catches that class of bug.
    """
    rows = 1000
    seed = 42

    reference = _print_reference_values(rows=rows, seed=seed)

    # ring_depth=1 forces per-block producer/consumer cooperation so every
    # block reaches the consumer; the AC1 aggregates require ALL rows to
    # compare 1:1 against the reference.
    _start_producer(fresh_shm_name, rows=rows, seed=seed, rows_per_block=128, ring_depth=1)

    observed = _run_ac1_query(fresh_shm_name).strip().split("\t")

    assert observed == reference, (
        "AC1 bit-identity failed against the C++ reference path.\n"
        f"  reference (--print-reference-values --rows {rows} --seed {seed}):\n"
        f"    {reference!r}\n"
        f"  observed (live shm() source):\n"
        f"    {observed!r}"
    )


# ===========================================================================
# T5.3 - AC4 pollable wiring + cancellation
# ===========================================================================


def test_ac4_max_threads_one(fresh_shm_name):
    """pollable-shm-source.md AC4: AC1 query under max_threads=1 drains cleanly.

    ring_depth=1 makes the producer wait for the consumer between every block,
    so the full row set is guaranteed to be drained (no slot-cycle overwrites).
    """
    rows = 1000
    _start_producer(fresh_shm_name, rows=rows, rows_per_block=128, ring_depth=1)
    result = _run_ac1_query(fresh_shm_name, settings={"max_threads": 1}).strip()
    cols = result.split("\t")
    assert int(cols[0]) == rows, f"count() = {cols[0]}, expected {rows}"
    assert int(cols[1]) == rows * (rows - 1) // 2, f"sum(id) = {cols[1]}"


def test_ac4_max_threads_many(fresh_shm_name):
    """pollable-shm-source.md AC4: AC1 query under max_threads=4 drains cleanly.

    The source exposes a single stream (N11), so max_threads only widens the
    downstream pipeline; the source itself stays serial. This test asserts that
    multi-threaded execution does not race with the source's async/poll path.
    ring_depth=1 again forces full-row drain (see test_ac4_max_threads_one).
    """
    rows = 1000
    _start_producer(fresh_shm_name, rows=rows, rows_per_block=128, ring_depth=1)
    result = _run_ac1_query(fresh_shm_name, settings={"max_threads": 4}).strip()
    cols = result.split("\t")
    assert int(cols[0]) == rows, f"count() = {cols[0]}, expected {rows}"
    assert int(cols[1]) == rows * (rows - 1) // 2, f"sum(id) = {cols[1]}"


def test_ac4_cancellation(fresh_shm_name):
    """pollable-shm-source.md AC4 cancellation sub-test (and I9 bounded reclaim).

    Drives the producer into a stall after a couple of blocks; the source then
    sits in Status::Async waiting on the eventfd. We KILL QUERY ... SYNC and
    assert the executor returns control within the build-mode budget T, that
    the source releases SHM resources (ShmAdoptedBytesCurrent returns to zero), and
    that the executor does NOT require producer cooperation to unwind.
    """
    _start_producer(
        fresh_shm_name, rows=1_000_000,
        scenario="stall-after", scenario_arg=2, rows_per_block=128,
    )

    query_id = f"shm_cancel_{uuid.uuid4().hex}"
    holder = {}

    def _runner():
        try:
            node.query(
                _ac1_sql(fresh_shm_name),
                settings=_cancel_settings(),
                query_id=query_id,
            )
            holder["finished"] = True
        except QueryRuntimeException as exc:
            holder["err"] = exc

    thread = threading.Thread(target=_runner, daemon=True)
    thread.start()

    # Wait until the source has actually attached AND retained at least one
    # block. ShmAdoptedBytesCurrent > 0 is the live witness (per plan T2.3 the
    # gauge is incremented on charge() and decremented on release()); 10 s of
    # slack covers clickhouse-client startup + the 2-block-then-stall window
    # under sanitiser / debug builds.
    deadline = time.time() + 10.0
    while time.time() < deadline and _shm_metric_value("ShmAdoptedBytesCurrent") == 0:
        time.sleep(0.05)
    assert _shm_metric_value("ShmAdoptedBytesCurrent") > 0, (
        "Source never charged any adopted bytes within 10 s; "
        "test cannot validate cancellation under an actively-stalled source."
    )

    budget = _cancellation_budget_sec()
    kill_start = time.time()
    # SYNC waits for the killed query to fully terminate server-side, so the
    # SYNC duration equals the source's cancel-to-unwind latency.
    node.query(f"KILL QUERY WHERE query_id = '{query_id}' SYNC")
    thread.join(timeout=budget + 5.0)
    elapsed = time.time() - kill_start

    assert not thread.is_alive(), (
        f"Query did not terminate within {budget + 5.0}s of KILL QUERY SYNC "
        f"(elapsed={elapsed:.2f}s); stop-condition S4 would fire."
    )
    assert elapsed < budget, (
        f"Cancellation took {elapsed:.2f}s, exceeds build-mode budget T={budget}s; "
        "this is the test-level signal for S4."
    )

    # I9: SHM resources are reclaimed without producer cooperation. The live-bytes
    # gauge returns to zero once the source destructs and all charge handles are
    # released. We poll briefly because the source's RAII teardown races KILL SYNC.
    deadline = time.time() + 5.0
    while time.time() < deadline and _shm_metric_value("ShmAdoptedBytesCurrent") != 0:
        time.sleep(0.05)
    assert _shm_metric_value("ShmAdoptedBytesCurrent") == 0, (
        "ShmAdoptedBytesCurrent did not return to 0 after cancellation; "
        "the source leaked a retained block (I9 violation)."
    )


# ===========================================================================
# T5.4 - AC5 MemoryTracker correctness + limit failure
# ===========================================================================


def test_ac5_baseline(fresh_shm_name):
    """memory-tracker-integration.md AC5 baseline: counters reset after the source dies.

    Asserts I7's "feature-local adopted-byte counter returns to zero on source
    destruction" via ShmAdoptedBytesCurrent, and that adopted-byte traffic is
    visible in the cumulative ProfileEvents counter ShmAdoptedBytesCharged.
    """
    _start_producer(fresh_shm_name, rows=10000, rows_per_block=512)

    before_charged = _shm_event_value("ShmAdoptedBytesCharged")
    before_logical = _shm_event_value("ShmAdoptedBytesLogical")
    _run_ac1_query(fresh_shm_name)
    after_charged = _shm_event_value("ShmAdoptedBytesCharged")
    after_logical = _shm_event_value("ShmAdoptedBytesLogical")

    assert _shm_metric_value("ShmAdoptedBytesCurrent") == 0, (
        "ShmAdoptedBytesCurrent did not return to zero after the source destructed."
    )
    assert after_charged > before_charged, (
        f"ShmAdoptedBytesCharged did not advance: before={before_charged} after={after_charged}"
    )
    assert after_logical > before_logical, (
        f"ShmAdoptedBytesLogical did not advance: before={before_logical} after={after_logical}"
    )
    # Charged includes safe-read padding, so charged_delta >= logical_delta over the
    # same set of blocks. This sanity-checks the glossary's "logical vs charged".
    assert (after_charged - before_charged) >= (after_logical - before_logical), (
        "Charged delta should be >= logical delta (charged = logical + safe-read padding)."
    )


def test_ac5_limit_failure(fresh_shm_name):
    """memory-tracker-integration.md AC5 limit-failure sub-test (and I8 enforcement).

    Runs the AC1 query under a max_memory_usage so low that even the very first
    block adoption would exceed it. The source must:
      * raise MEMORY_LIMIT_EXCEEDED at the charge step (per the Failure-classes
        table row 'memory-limit-exceeded'),
      * release the just-acquired wire retain before propagation (so no SHM
        segment leaks; I5 + I10),
      * leave ShmAdoptedBytesCurrent at the pre-query baseline.
    """
    # rows_per_block=512 x ~190 B/row ~= 96 KB charge per block; max_memory_usage
    # = 64 KB guarantees the first charge exceeds the limit (and remains
    # comfortably above parser/planner overhead so we know it was the source
    # that tripped, not analysis).
    _start_producer(
        fresh_shm_name, rows=10000, rows_per_block=512,
        data_region_size=16 * 1024 * 1024,
    )

    shm_count_before = _devshm_count()
    current_before = _shm_metric_value("ShmAdoptedBytesCurrent")

    err = node.query_and_get_error(
        _ac1_sql(fresh_shm_name),
        settings={
            "allow_experimental_shm_table_function": 1,
            "max_memory_usage": 65536,
            # untracked slack < per-block charge so the tracker sees the charge promptly.
            "max_untracked_memory": 1,
        },
    )
    assert _extract_error_code(err) == ERR_MEMORY_LIMIT_EXCEEDED, (
        f"Expected MEMORY_LIMIT_EXCEEDED ({ERR_MEMORY_LIMIT_EXCEEDED}), got:\n{err}"
    )

    # I7 + I10: rollback leaves the feature-local counter unchanged.
    assert _shm_metric_value("ShmAdoptedBytesCurrent") == current_before, (
        f"ShmAdoptedBytesCurrent drifted across a limit-failure: "
        f"before={current_before}, after={_shm_metric_value('ShmAdoptedBytesCurrent')}"
    )

    # No leaked SHM segment from the consumer side (the producer keeps its own).
    shm_count_after = _devshm_count()
    assert abs(shm_count_after - shm_count_before) <= 1, (
        f"/dev/shm count drifted across limit-failure: "
        f"before={shm_count_before} after={shm_count_after}"
    )


def test_ac5_tracker_reflection(fresh_shm_name):
    """memory-tracker-integration.md AC5 tracker-reflection sub-test.

    Spec formula (memory-tracker-integration.md AC5):
        peak_charged - max_threads * max_untracked_memory <= tracker_peak

    Implementation note on the peak_charged proxy: AC5 talks about the PEAK
    instantaneous live charged-adopted-byte count -- the high-water mark of
    the per-source feature-local counter. That counter is not exposed via any
    system.* surface today (it lives behind the AdoptedByteCharger integration
    boundary). As a pragmatic stand-in this assertion uses the cumulative
    `ShmAdoptedBytesCharged` delta over the query lifetime as the peak_charged
    value -- a CONSERVATIVE over-estimate (cumulative >= peak), which turns
    the spec inequality into a STRICTLY STRONGER assertion than AC5 literally
    requires. The true peak read via system.metrics ShmAdoptedBytesCurrent
    would need a sampling thread; if this proxy proves too strict on real CI
    workloads (cumulative >> peak on streaming-aggregation queries that
    charge-then-release each block), switch to a sampler or wire a test-only
    per-source-peak accessor through AdoptedByteCharger.
    """
    _start_producer(fresh_shm_name, rows=10000, rows_per_block=512)

    # AC5 inputs we control: pin max_threads and max_untracked_memory so the
    # spec's slack term `max_threads * max_untracked_memory` is computable here
    # without re-reading either value from system.* surfaces.
    max_threads = 1
    max_untracked = 1

    before_charged = _shm_event_value("ShmAdoptedBytesCharged")
    query_id = f"shm_track_{uuid.uuid4().hex}"
    _run_ac1_query(
        fresh_shm_name,
        settings={"max_untracked_memory": max_untracked, "max_threads": max_threads},
        query_id=query_id,
    )
    after_charged = _shm_event_value("ShmAdoptedBytesCharged")
    # cumulative-since-baseline proxy for peak_charged; see docstring.
    peak_charged = after_charged - before_charged
    assert peak_charged > 0, "Producer published zero adopted bytes; cannot test tracker."

    _flush_logs()
    peak_text = node.query(
        f"SELECT memory_usage FROM system.query_log"
        f" WHERE query_id = '{query_id}' AND type = 'QueryFinish'"
        f" ORDER BY event_time_microseconds DESC LIMIT 1"
    ).strip()
    assert peak_text, f"system.query_log has no QueryFinish row for query_id={query_id}"
    tracker_peak = int(peak_text)
    slack = max_threads * max_untracked

    # Spec: peak_charged - max_threads * max_untracked_memory <= tracker_peak.
    assert peak_charged - slack <= tracker_peak, (
        f"AC5 tracker-reflection failed: "
        f"peak_charged={peak_charged} B (cumulative ShmAdoptedBytesCharged delta over query lifetime), "
        f"max_threads={max_threads}, max_untracked_memory={max_untracked} B, "
        f"slack={slack} B, tracker_peak={tracker_peak} B. "
        f"Spec: peak_charged - max_threads*max_untracked_memory <= tracker_peak."
    )


def test_ac5_tracker_returns_to_baseline(fresh_shm_name):
    """memory-tracker-integration.md AC5 clause 3: after source destruction,
    the (process-wide) MemoryTracker chain returns to within
    `max_threads * max_untracked_memory` of its pre-query baseline.

    The query-level MemoryTracker is a child of the server's top-level
    `MemoryTracking` tracker; when the query ends and its child is destroyed,
    its bytes propagate up to the parent and are subtracted. If the source's
    RAII charge release ran cleanly on destruction (I7 + I10), the parent
    tracker returns to baseline (modulo a small absolute fudge for
    background system-tables activity between the two reads -- see below).

    This complements `test_ac5_baseline` (which asserts the feature-local
    counter ShmAdoptedBytesCurrent returns to zero) and
    `test_ac5_tracker_reflection` (which asserts the AC5 inequality during
    the query); it is the third clause of AC5: post-query baseline restoration.
    """
    max_threads = 1
    max_untracked = 1  # bytes; keeps the spec's slack term arithmetically tight
    slack = max_threads * max_untracked

    # 1) Capture baseline BEFORE any query. `MemoryTracking` is the canonical
    #    server-wide tracker -- the parent of every query-level tracker, so a
    #    leaked child charge would show up here as residual drift.
    baseline_tracker = int(node.query(
        "SELECT value FROM system.metrics WHERE metric = 'MemoryTracking'"
    ).strip())

    _start_producer(fresh_shm_name, rows=10000, seed=42,
                    scenario="normal", ring_depth=1)
    result = _run_ac1_query(
        fresh_shm_name,
        settings={
            "max_threads": max_threads,
            "max_untracked_memory": max_untracked,
        },
    )
    assert result.strip() != "", "AC1 query returned empty result; cannot test tracker return."

    # 2) Give the source a moment to release on query end. The source dtor runs
    #    synchronously as the pipeline tears down, but ChargeHandle destructors
    #    may race the test's next observation by a tiny scheduler window.
    time.sleep(0.5)

    # 3) The spec bound is `max_threads * max_untracked_memory`, but in a
    #    non-quiescent server the gap between two `MemoryTracking` reads also
    #    sees background activity (query_log inserts, async_metric_log, etc.).
    #    The `+ 65536` fudge absorbs that; it is NOT spec-mandated, only a
    #    stability margin. The feature-local assertion below (ShmAdoptedBytesCurrent
    #    == 0) is the tight bound; this assertion proves the chain reflects it.
    post_tracker = int(node.query(
        "SELECT value FROM system.metrics WHERE metric = 'MemoryTracking'"
    ).strip())
    delta = abs(post_tracker - baseline_tracker)
    assert delta <= slack + 65536, (
        f"AC5 clause 3 violated: tracker delta {delta} B > slack {slack} B + 64 KiB fudge; "
        f"baseline={baseline_tracker} post={post_tracker}. Likely a missed release in the "
        f"source dtor or a charge that leaked past pipeline teardown."
    )

    # 4) Tighter, feature-local bound: ShmAdoptedBytesCurrent must be exactly 0.
    #    Same assertion as test_ac5_baseline, repeated here so this test stands
    #    on its own as the AC5-clause-3 witness (the spec ties the two together).
    adopted_now = _shm_metric_value("ShmAdoptedBytesCurrent")
    assert adopted_now == 0, (
        f"ShmAdoptedBytesCurrent={adopted_now}, expected 0 after source destruction."
    )


# ===========================================================================
# T5.5 - AC6 producer-misbehaviour matrix (one test per failure class)
# ===========================================================================
#
# Each test reproduces one row of the
# [Failure classes](pollable-shm-source.md#interfaces--contracts) table
# and asserts on the *integer* error code (NOT the human-readable message),
# per the AC6 brief: "test asserts on the failure class, not on string
# content or generic exception type. 'Some exception' does not satisfy AC6."
#
# Mapping (taken from src/Common/ErrorCodes.cpp; plan task T0.2):
#   * non-conforming buffer geometry   -> 776 SHM_BUFFER_LAYOUT_INVALID
#                                            (OffsetOverflow trips precondition 14/18/19)
#   * producer crash mid-publication   -> 775 SHM_BLOCK_FRAMING_INVALID
#                                            (BadSlotIdentity trips precondition 9 — the
#                                             slot identity field a partial publication
#                                             would leave behind)
#   * producer death pre-EOS           -> 778 SHM_PRODUCER_DEATH_BEFORE_EOS
#   * producer stall                   -> 777 SHM_PRODUCER_STALL
#   * attach: missing object (ENOENT)  -> 772 SHM_ATTACH_FAILED
#   * attach: inaccessible (EACCES)    -> 772 SHM_ATTACH_FAILED (chmod 000 on the file)
#   * readiness-locator unresolvable   -> 772 SHM_ATTACH_FAILED (control-socket ENOENT)
#   * handshake invalid                -> 773 SHM_HANDSHAKE_INVALID
#   * schema mismatch (count)          -> 774 SHM_SCHEMA_MISMATCH
#   * schema mismatch (type)           -> 774 SHM_SCHEMA_MISMATCH


def _expect_error_code(
    name,
    expected_code,
    *,
    schema=SCHEMA,
    settings_override=None,
    sql_override=None,
):
    """Run the AC1 query (or a caller-supplied SQL) and assert the error code."""
    sql = sql_override if sql_override is not None else _ac1_sql(name, schema=schema)
    settings = {"allow_experimental_shm_table_function": 1}
    if settings_override:
        settings.update(settings_override)
    err = node.query_and_get_error(sql, settings=settings)
    code = _extract_error_code(err)
    assert code == expected_code, (
        f"Expected Code: {expected_code} but observed Code: {code}.\nError:\n{err}"
    )


def test_ac6_non_conforming_buffer(fresh_shm_name):
    """AC6 row "non-conforming buffer geometry" -> SHM_BUFFER_LAYOUT_INVALID (776).

    The producer's --scenario abort-mid-publish uses Malformation::OffsetOverflow
    (preconditions 14 / 18 / 19: declared sizes overflow the data region). Per
    the AC6 Failure-classes table that is the buffer-layout-invalid class.
    Renamed from `test_ac6_malformed_block` to make the spec-row mapping
    unambiguous -- "malformed block" was loose terminology that could equally
    refer to the block-framing-invalid row, which is now covered separately by
    `test_ac6_block_framing_invalid_via_bad_slot_identity` (the malformation
    path) and `test_ac6_mid_publication_crash_real` (the real-crash path).

    Timing-deterministic setup: we pick rows = (scenario_arg + 1) * rows_per_block
    so the producer publishes exactly scenario_arg+1 data blocks (one of which is
    malformed) followed by EOS, for a total of scenario_arg+2 publishes. With
    ring_depth=4 and scenario_arg+2 <= 4 each publish lands in a distinct slot
    and the malformed block is GUARANTEED to remain in slot N when the consumer
    attaches; it cannot be overwritten by a subsequent cycle. This eliminates
    the publish/attach race that a long-running producer would introduce.
    """
    _start_producer(
        fresh_shm_name, rows=3 * 128,  # 3 data blocks + EOS, all in distinct slots
        scenario="abort-mid-publish", scenario_arg=2, rows_per_block=128,
        ring_depth=4,
    )
    _expect_error_code(fresh_shm_name, ERR_SHM_BUFFER_LAYOUT_INVALID)


def test_ac6_block_framing_invalid_via_bad_slot_identity(fresh_shm_name):
    """AC6 block-framing-invalid CLASS via the BadSlotIdentity malformation
    path -> SHM_BLOCK_FRAMING_INVALID (775).

    The producer's --scenario block-framing-invalid uses Malformation::BadSlotIdentity
    (precondition 9: slot identity != position) on a fully PUBLISHED block.
    The consumer's findNextReadySlot scan visits the corrupted slot and raises
    SHM_BLOCK_FRAMING_INVALID, exercising the same failure-class throw site
    the "crash mid-publication" row of the AC6 Failure-classes table maps to
    -- but reaching it via a DIFFERENT trigger (a published-but-malformed slot,
    not a slot left in WRITING by a real crash). The complementary "real
    crash" trigger is covered by `test_ac6_mid_publication_crash_real` below;
    keeping both ensures the throw site is exercised under both reachable
    producer-side conditions.

    Renamed from `test_ac6_mid_publication_crash` -- the previous name
    conflated this test (which is about the BAD-SLOT-IDENTITY malformation
    on a completed block) with the spec's "producer crash mid-publication"
    row (which strictly requires a slot left in WRITING by an ungraceful
    exit -- see the new `_real` test). The fork-based gtest
    `ProducerCrashMidPublicationYieldsBlockFramingInvalid` covers the same
    real-crash path at the unit level.

    BadSlotIdentity is chosen over BadSequence because BadSequence only takes
    effect when a slot's seq counter is already > 1 (slot has been re-published),
    which would require either consumer cooperation or a ring-wrap.
    BadSlotIdentity triggers unconditionally on the first publish to a slot, so
    the same timing-deterministic setup as test_ac6_non_conforming_buffer
    applies: rows = (scenario_arg + 1) * rows_per_block with ring_depth = 4
    keeps every publish in a distinct slot so the malformation cannot be
    overwritten by a cycle before the consumer attaches.
    """
    _start_producer(
        fresh_shm_name, rows=3 * 128,  # 3 data blocks + EOS, all in distinct slots
        scenario="block-framing-invalid", scenario_arg=2, rows_per_block=128,
        ring_depth=4,
    )
    _expect_error_code(fresh_shm_name, ERR_SHM_BLOCK_FRAMING_INVALID)


def test_ac6_mid_publication_crash_real(fresh_shm_name):
    """AC6 row "producer crash mid-publication" -> SHM_BLOCK_FRAMING_INVALID (775).

    Unlike `test_ac6_block_framing_invalid_via_bad_slot_identity`, which
    publishes a fully-completed but malformed slot, this test exercises the
    spec row LITERALLY: the producer puts a slot in WRITING via the test-only
    escape hatch `setSlotStateForTesting` and then `_exit(1)`s before W->P,
    leaving the slot table exactly the way an ungraceful crash would. The
    consumer's control-socket connection then observes POLLHUP on the
    established peer fd, `checkProducerDeath` scans the slot table, sees the
    WRITING slot, and throws SHM_BLOCK_FRAMING_INVALID (NOT
    SHM_PRODUCER_DEATH_BEFORE_EOS -- the WRITING slot is the discriminator,
    per the round-2 F6 fix in `PollableShmSource::checkProducerDeath`).

    Timing: shm-producer's --scenario mid-publication-crash deliberately
    sleeps ~2 s AFTER setting the slot to WRITING and BEFORE _exit(1). That
    window lets THIS test's consumer attach via the still-live control socket
    AND drain the 2 prior PUBLISHED blocks BEFORE the peer-end close, so the
    resulting POLLHUP lands on an ESTABLISHED connection (yielding 775 via
    checkProducerDeath) and NOT on a fresh connect() (which would return
    ECONNREFUSED -> SHM_ATTACH_FAILED). Mirrors the 2 s sleep in the analogous
    fork-based gtest `ProducerCrashMidPublicationYieldsBlockFramingInvalid`.

    Layout: rows = 3 * 128 with rows_per_block = 128 and scenario_arg = 2 so
    the producer publishes blocks 0+1 (slots 0+1 PUBLISHED), then the loop
    re-enters with blocks_published == 2, hits the scenario branch, sets
    slot 2 to WRITING, sleeps, and exits. ring_depth = 4 keeps every
    publish in a distinct slot so the WRITING slot cannot be overwritten
    by a cycle before the consumer attaches.
    """
    _start_producer(
        fresh_shm_name, rows=3 * 128,
        scenario="mid-publication-crash", scenario_arg=2, rows_per_block=128,
        ring_depth=4,
    )
    err = node.query_and_get_error(
        _ac1_sql(fresh_shm_name),
        settings={"allow_experimental_shm_table_function": 1},
    )
    code = _extract_error_code(err)
    assert code == ERR_SHM_BLOCK_FRAMING_INVALID, (
        f"Expected SHM_BLOCK_FRAMING_INVALID ({ERR_SHM_BLOCK_FRAMING_INVALID}), "
        f"got Code: {code}.\nError:\n{err}"
    )
    # The throw site in PollableShmSource::checkProducerDeath formats the
    # message as: "...producer died mid-publication (slot N in WRITING state
    # ...) -- precondition 25 + AC6 mid-publication branch". Accept any of
    # the spec-aligned tokens so a future message-wording tweak (which is
    # not a spec change) does not silently flip this assertion.
    err_text = str(err).lower()
    assert any(tok in err_text for tok in ("writing", "mid-publication", "precondition")), (
        f"Expected SHM_BLOCK_FRAMING_INVALID error to mention WRITING / "
        f"mid-publication / precondition; got:\n{err}"
    )


def test_ac6_producer_crash(fresh_shm_name):
    """AC6 producer death pre-EOS -> producer-death-before-eos (778).

    The producer-death class fires when the consumer's control-socket connection
    observes POLLHUP (or equivalent) BEFORE the consumer has observed EOS. That
    requires the consumer to have completed its socket attach while the producer
    was still alive. The producer's own --scenario crash-after races the consumer
    attach (the producer publishes its target block count in microseconds and
    then _exit(1)s, often before the consumer has even connected to the socket;
    in that case the consumer fails with SHM_ATTACH_FAILED instead). To make the
    test deterministic, we instead:
      1. spawn the producer in --scenario normal --rows <large> so it stays alive
         indefinitely (publishing blocks the consumer can drain);
      2. start the consumer query in a background thread;
      3. wait for the source to actually attach (ShmAdoptedBytesCurrent > 0);
      4. SIGKILL the producer to simulate the abrupt death;
      5. wait for the consumer thread to surface the typed exception.

    This guarantees the consumer has attached BEFORE the producer dies; the
    POLLHUP-after-attach path is what the AC6 class is specifically about.
    """
    _start_producer(
        fresh_shm_name, rows=10_000_000,
        scenario="normal", rows_per_block=128, ring_depth=4,
    )

    query_id = f"shm_crash_{uuid.uuid4().hex}"
    holder = {}

    def _runner():
        try:
            node.query(
                _ac1_sql(fresh_shm_name),
                settings={"allow_experimental_shm_table_function": 1},
                query_id=query_id,
            )
            holder["finished_normally"] = True
        except QueryRuntimeException as exc:
            holder["err"] = exc

    thread = threading.Thread(target=_runner, daemon=True)
    thread.start()

    # Wait until the source has attached AND has retained at least one slot:
    # ShmAdoptedBytesCurrent > 0 means a charge handle is live (per plan T2.3
    # the gauge is incremented under the source's RAII charge sequence).
    deadline = time.time() + 15.0
    while time.time() < deadline and _shm_metric_value("ShmAdoptedBytesCurrent") == 0:
        time.sleep(0.05)
    assert _shm_metric_value("ShmAdoptedBytesCurrent") > 0, (
        "Source never charged any adopted bytes within 15 s; cannot stage a "
        "death-after-attach scenario."
    )

    # SIGKILL the producer (the dtor's shm_unlink/socket cleanup never runs, so
    # the SHM file lingers but the control-socket fd closes with the process and
    # the consumer's connected socket sees POLLHUP).
    node.exec_in_container(
        ["bash", "-c", f"pkill -KILL -f '{PRODUCER_BIN}.*--name {fresh_shm_name}' 2>/dev/null; true"],
        nothrow=True,
    )

    thread.join(timeout=30.0)
    assert not thread.is_alive(), "Consumer thread did not surface an exception within 30 s of producer death."
    err = holder.get("err")
    assert err is not None, (
        f"Query finished without an exception under producer death; holder={holder!r}"
    )
    assert _extract_error_code(err) == ERR_SHM_PRODUCER_DEATH_BEFORE_EOS, (
        f"Expected SHM_PRODUCER_DEATH_BEFORE_EOS (778), got: {err.stderr or err}"
    )


def test_ac6_producer_stall(fresh_shm_name):
    """AC6 producer stall -> producer-stall (777) within the I12 stall budget.

    The producer's --scenario stall-after publishes scenario_arg blocks then
    calls stallProducer() (which makes any further publishBlock throw) and
    exits the publishing loop, but the process stays alive in sleepUntilShutdown.
    Critically: the producer process and its control socket remain alive after
    the stall, so the consumer can attach at any later time and the failure mode
    is the consumer's stall-timer, NOT a control-socket POLLHUP. That makes
    this test race-free regardless of producer publish-vs-consumer-attach order.

    We tighten shm_source_stall_timeout_ms to 1 s so the test does not pay the
    30 s default budget. The actual budget enforcement code path is identical;
    we only shrink the constant.
    """
    _start_producer(
        fresh_shm_name, rows=100_000,
        scenario="stall-after", scenario_arg=2, rows_per_block=128,
    )
    start = time.time()
    _expect_error_code(
        fresh_shm_name, ERR_SHM_PRODUCER_STALL,
        settings_override={"shm_source_stall_timeout_ms": 1000},
    )
    elapsed = time.time() - start
    # Budget is 1 s + a generous slack for executor scheduling. Under sanitisers
    # and debug builds I12's wall-clock budget enforcement is the same code; we
    # allow up to 30 s of slack for executor cadence + KILL/RPC overhead. The
    # test asserts I12 *bounds* the wait, not its exact value.
    assert elapsed < 30.0, (
        f"Producer stall surface took {elapsed:.1f}s with a 1 s budget; "
        "I12's bounded-stall guarantee may be broken."
    )


def test_ac6_attach_missing(fresh_shm_name):
    """AC6 attach-failed (object missing) -> SHM_ATTACH_FAILED (772).

    No producer spawned; shm_open(name) returns ENOENT.
    """
    _expect_error_code(fresh_shm_name, ERR_SHM_ATTACH_FAILED)


def test_ac6_attach_inaccessible(fresh_shm_name):
    """AC6 attach-failed (object inaccessible) -> SHM_ATTACH_FAILED (772).

    Distinct from test_ac6_attach_missing: the /dev/shm entry exists, but the
    consumer's shm_open(O_RDONLY) returns EACCES because the file mode forbids
    access. The Attach-time observable failures row collapses every shm_open
    errno (ENOENT, EACCES, EINVAL, ENAMETOOLONG, ...) into a single failure
    class, SHM_ATTACH_FAILED; we keep one test per spec row by exercising both
    the missing-object and inaccessible-object cases.

    The producer creates the SHM file with the default mode (clickhouse-owner +
    group). We chmod 000 it *after* the producer is ready so the consumer's
    subsequent shm_open hits EACCES; the producer's already-open fd is
    unaffected (mode checks only happen at open(2) time, not on existing fds).
    """
    _start_producer(fresh_shm_name, rows=1000, rows_per_block=128)
    shm_path = _devshm_path(fresh_shm_name)
    node.exec_in_container(["bash", "-c", f"chmod 000 {shm_path}"])
    try:
        _expect_error_code(fresh_shm_name, ERR_SHM_ATTACH_FAILED)
    finally:
        # Restore perms so the per-test cleanup fixture's `rm -f` does not
        # depend on /dev/shm's sticky-bit semantics for cleanup of a 000 file.
        node.exec_in_container(
            ["bash", "-c", f"chmod 600 {shm_path} 2>/dev/null; true"],
            nothrow=True,
        )


def test_ac6_readiness_locator_unresolvable(fresh_shm_name):
    """AC6 row "readiness-fd locator unresolvable" -> SHM_ATTACH_FAILED (772).

    The producer's --scenario socket-missing constructs the InProcessProducer
    normally (creates the SHM segment + binds the control socket) and then
    unlinks /tmp/clickhouse_shm_<name>.sock -- the filesystem entry the
    consumer connect()s to. unlink() removes only the path entry, so any
    *existing* connected socket would remain valid; the consumer's *new*
    connect() attempts get ENOENT. ControlSocketClient::connectAndReceiveEventFd
    then raises SHM_ATTACH_FAILED per the Attach-time observable failures table.

    We bypass `_wait_for_producer_ready` because that helper polls for BOTH
    /dev/shm/<name> AND /tmp/clickhouse_shm_<name>.sock; the socket file is
    gone by design here. Instead we poll only for the /dev/shm file before
    issuing the query.
    """
    _start_producer(
        fresh_shm_name, rows=1, seed=1, scenario="socket-missing", wait_ready=False,
    )
    shm_path = _devshm_path(fresh_shm_name)
    deadline = time.time() + 10.0
    while time.time() < deadline:
        out = node.exec_in_container(
            ["bash", "-c", f"[ -e {shm_path} ] && echo ok || echo no"],
            nothrow=True,
        )
        if out and out.strip() == "ok":
            break
        time.sleep(0.05)
    else:
        raise RuntimeError(
            f"socket-missing producer never created {shm_path} within 10 s; "
            "cannot validate readiness-locator-unresolvable."
        )
    _expect_error_code(fresh_shm_name, ERR_SHM_ATTACH_FAILED)


def test_ac6_handshake_invalid(fresh_shm_name):
    """AC6 handshake-invalid -> SHM_HANDSHAKE_INVALID (773).

    The SHM file exists with the right minimum size but the handshake magic
    is zero (writing /dev/zero into it). Precondition 1 (magic match) fires
    on the consumer's first acquire-load of the handshake region.
    """
    base = _shm_basename(fresh_shm_name)
    # 32 KiB so the file comfortably exceeds sizeof(HandshakeRegion) (~128 B).
    node.exec_in_container(
        ["bash", "-c", f"dd if=/dev/zero of=/dev/shm/{base} bs=4096 count=8 2>/dev/null"]
    )
    try:
        _expect_error_code(fresh_shm_name, ERR_SHM_HANDSHAKE_INVALID)
    finally:
        node.exec_in_container(["bash", "-c", f"rm -f /dev/shm/{base}"], nothrow=True)


def test_ac6_schema_mismatch_count(fresh_shm_name):
    """AC6 schema-mismatch (column count) -> SHM_SCHEMA_MISMATCH (774).

    Producer publishes 5 columns; SQL declares 3. Cross-validation at handshake
    fires precondition 4 (schema_count mismatch).
    """
    _start_producer(fresh_shm_name, rows=100, rows_per_block=64)
    fewer = "id UInt64, v1 UInt64, v2 UInt64"
    _expect_error_code(
        fresh_shm_name, ERR_SHM_SCHEMA_MISMATCH,
        sql_override=f"SELECT count() FROM shm('{fresh_shm_name}', '{fewer}')",
    )


def test_ac6_schema_mismatch_type(fresh_shm_name):
    """AC6 schema-mismatch (type) -> SHM_SCHEMA_MISMATCH (774).

    Producer publishes id as UInt64; SQL declares id as String. Cross-validation
    fires precondition 6 (SQL-declared type vs producer-declared type equality).
    """
    _start_producer(fresh_shm_name, rows=100, rows_per_block=64)
    wrong = "id String, v1 UInt64, v2 UInt64, s1 String, s2 String"
    _expect_error_code(
        fresh_shm_name, ERR_SHM_SCHEMA_MISMATCH,
        sql_override=f"SELECT count() FROM shm('{fresh_shm_name}', '{wrong}')",
    )


# ===========================================================================
# T5.6 - AC7 leak audit + AC10 retain integrity
# ===========================================================================


def test_ac7_leak_audit(fresh_shm_name):
    """system.md AC7: >=1000 iterations of the shm() query with stable resource counters.

    Per AC7 we drive a single server process through 1000 queries and assert
    stable fd count (clickhouse-server side), stable /dev/shm segment count,
    stable ShmAdoptedBytesCurrent gauge, and balanced retain-acquire /
    retain-release event deltas (system spec I5 + I10).

    Note on per-iteration row counts: after the producer signals EOS, subsequent
    consumer attaches see a fixed leftover slot-table state (the most-recent K
    blocks plus the EOS marker). Each query drains those, finishes, and the
    source destructs. We assert *no leaks*; per-iteration row counts are NOT
    the focus of this test (system AC1 owns functional correctness).
    """
    rows = 1000
    _start_producer(fresh_shm_name, rows=rows, rows_per_block=128, ring_depth=4)

    iterations = 1000
    sql = f"SELECT count() FROM shm('{fresh_shm_name}', '{SCHEMA}')"
    settings = {"allow_experimental_shm_table_function": 1}

    fd_before = _server_fd_count()
    shm_before = _devshm_count()
    current_before = _shm_metric_value("ShmAdoptedBytesCurrent")
    retains_before = _shm_event_value("ShmRetainsAcquired")
    releases_before = _shm_event_value("ShmRetainsReleased")

    for i in range(iterations):
        node.query(sql, settings=settings)
        # Periodic in-loop sanity check on the live gauge - catches a per-iteration
        # leak early instead of waiting for the post-loop assertion to fail.
        if (i + 1) % 200 == 0:
            observed = _shm_metric_value("ShmAdoptedBytesCurrent")
            assert observed == current_before, (
                f"ShmAdoptedBytesCurrent drifted at iter {i + 1}: "
                f"baseline={current_before}, observed={observed}"
            )

    fd_after = _server_fd_count()
    shm_after = _devshm_count()
    current_after = _shm_metric_value("ShmAdoptedBytesCurrent")
    retains_after = _shm_event_value("ShmRetainsAcquired")
    releases_after = _shm_event_value("ShmRetainsReleased")

    # AC7: fd count stability. Allow some slack for housekeeping fds (log
    # rotation, incidental connections, query-log flush threads). The bug we
    # care about is the source forgetting to close its eventfd or control-
    # socket fd per query; that would push the delta to ~iterations (here,
    # ~2000), not single digits.
    assert abs(fd_after - fd_before) <= 32, (
        f"fd count drifted by {fd_after - fd_before} across {iterations} iterations "
        f"(before={fd_before} after={fd_after}); the source likely leaks fds."
    )

    # AC7: /dev/shm segment stability. The producer holds exactly one segment
    # for its full lifetime; the source must not create or leak any. Allow +/-1
    # slack for unrelated server-internal SHM use.
    assert abs(shm_after - shm_before) <= 1, (
        f"/dev/shm segment count drifted by {shm_after - shm_before} "
        f"(before={shm_before} after={shm_after}); SHM segment leak suspected."
    )

    # I7: feature-local current adopted bytes returns to baseline.
    assert current_after == current_before, (
        f"ShmAdoptedBytesCurrent drifted across {iterations} iterations: "
        f"before={current_before} after={current_after}"
    )

    # I5 + I10: every retain acquisition is matched by exactly one release.
    delta_retains = retains_after - retains_before
    delta_releases = releases_after - releases_before
    assert delta_retains == delta_releases, (
        f"Retain/release pair leak across {iterations} iterations: "
        f"acquired={delta_retains} released={delta_releases}"
    )
    assert delta_retains > 0, (
        "No retains acquired across the leak-audit loop; the source did not "
        "actually do any adoption work, so AC7's stability check is moot."
    )


def test_ac10_retain_reuse(fresh_shm_name):
    """shm-block-stream.md AC10: republish-after-retain - cooperative slot reuse.

    The producer scenario republish-after-retain publishes block 0, blocks on
    waitForRetainToRelease(0), then publishes the rest of the stream into the
    same ring (so slot 0 IS republished, with a different sequence number,
    once the consumer has dropped its retain). The consumer side sees:
      1. block 0 in slot 0, retain++, adopt, downstream aggregate, retain--;
      2. the producer wakes from waitForRetainToRelease, publishes blocks 1..N;
      3. the consumer drains those + EOS.
    We assert the AC1 query completes with full row coverage AND that the
    retain-acquire / retain-release event counters balance - proving the
    consumer did not skip a release (which would have wedged the producer).

    Limitation: AC10's stricter "bytes visible through a held Chunk remain
    bit-identical while the retain is live" claim requires capturing a Chunk
    in C++ and inspecting its byte pointers; that is owned by the T5.1 gtest
    (plan task T5.1: gtest_ac3_adoption_proof.cpp / AC10 sub-assertion).
    """
    rows = 1000
    retains_before = _shm_event_value("ShmRetainsAcquired")
    releases_before = _shm_event_value("ShmRetainsReleased")
    current_before = _shm_metric_value("ShmAdoptedBytesCurrent")

    # ring_depth=1 forces per-block cooperation so every row reaches the consumer;
    # the republish-after-retain scenario's explicit waitForRetainToRelease(0) is
    # still exercised (and still serializes block 1's publish on slot 0 retain).
    _start_producer(
        fresh_shm_name, rows=rows,
        scenario="republish-after-retain", rows_per_block=128, ring_depth=1,
    )
    result = _run_ac1_query(fresh_shm_name).strip()
    cols = result.split("\t")
    assert int(cols[0]) == rows, f"AC10 count() = {cols[0]}, expected {rows}"
    assert int(cols[1]) == rows * (rows - 1) // 2, (
        f"AC10 sum(id) = {cols[1]}, expected {rows * (rows - 1) // 2}"
    )

    retains_after = _shm_event_value("ShmRetainsAcquired")
    releases_after = _shm_event_value("ShmRetainsReleased")
    delta_retains = retains_after - retains_before
    delta_releases = releases_after - releases_before
    assert delta_retains == delta_releases, (
        f"AC10 retain/release imbalance under republish: "
        f"acquired={delta_retains} released={delta_releases}"
    )
    assert _shm_metric_value("ShmAdoptedBytesCurrent") == current_before, (
        "AC10: ShmAdoptedBytesCurrent did not return to baseline after the "
        "republish-after-retain query."
    )


def test_ac10_held_chunk_bytes_stable(fresh_shm_name):
    """shm-block-stream.md AC10: held-chunk bytes remain bit-identical under
    a contended republish.

    Companion to test_ac10_retain_reuse, which only validates the cooperative
    retain/release wake-up (count + sum(id) is enough to prove no rows were
    skipped). This test instead validates the AC10 byte-stability invariant:
    while the consumer holds a Chunk that references slot N's SHM bytes, the
    producer's republish into slot N MUST NOT mutate those bytes. The state
    machine fix (Phase A) guarantees this end-to-end by gating slot reuse on
    consumer release.

    Strategy. With --scenario republish-after-retain + ring_depth=1, EVERY
    producer publish lands in slot 0 (only one slot in the ring); the producer
    publishes block 0, waits on waitForRetainToRelease(0), then publishes
    block 1 into the same slot, repeating until all rows are emitted. If the
    consumer ever observed mid-republish bytes (e.g. half-old, half-new s1/s2
    chars), the per-row cityHash64(s1) and cityHash64(s2) sums would diverge
    from the seed-derived reference path; ANY byte-level corruption inside
    any held chunk would surface as a mismatch in one of the 6 hash/length
    columns. The reference path is the same independent C++ replay used by
    test_ac1_bit_identical, so the comparison is bit-identity, not internal
    determinism.
    """
    rows = 1000
    seed = 42

    reference = _print_reference_values(rows=rows, seed=seed)

    _start_producer(
        fresh_shm_name, rows=rows, seed=seed,
        scenario="republish-after-retain", rows_per_block=128, ring_depth=1,
    )

    observed = _run_ac1_query(fresh_shm_name).strip().split("\t")

    assert observed == reference, (
        "AC10 held-chunk byte-stability failed: the consumer's observed bytes "
        "diverged from the seed-derived reference under contended slot reuse.\n"
        f"  reference (--print-reference-values --rows {rows} --seed {seed}):\n"
        f"    {reference!r}\n"
        f"  observed (--scenario republish-after-retain, ring_depth=1):\n"
        f"    {observed!r}"
    )
