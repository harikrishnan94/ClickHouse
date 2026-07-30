#!/usr/bin/env python3
"""Score a probe-phase A/B of two ClickHouse binaries on two independent metrics.

    probe_cost      = ConcurrentHashJoinProbeDispatchMicroseconds
                    + ConcurrentHashJoinProbeLookupMicroseconds
    projection_cost = ConcurrentHashJoinProbeMicroseconds - probe_cost

`probe_cost` is jbmt's `probe_lookup` phase split for `parallel_hash`: dispatch
belongs to it because routing and key preparation are part of getting to the
cell. `projection_cost` is the residual, and it is the only way to see column
materialization here, because the two events that would name the gather
directly - `HashJoinResultBuildOutputMicroseconds` (build side) and
`HashJoinResultFilterLeftMicroseconds` (probe side) - are not registered in
this source tree. The residual is therefore a single unsplit number covering
both sides; `--check-decomposition` fails if that ever stops being true, or if
the two arms are not decomposed identically.

Each metric is scored and verdicted independently, on its own median, its own
spread and its own band. They are never summed and never netted: a cell that
wins on one and loses on the other is reported as both.

Consumes either harness's JSONL:

  * fleet_ab  - one line per timed run: `cell`, `arm`, `arm_role`, `run`,
                `valid`, `events` (7 fixed keys), `checksum`, `duration_us`.
  * jbmt v1/v2 - one line per unit: `unit_id`, `arms{name:{algorithms{algo:
                {events_per_run[], durations_ms[], row_count, checksum}}}}`
                (v1: `algorithms` at top level, one implicit arm).

Exit code is the point: any failed check exits non-zero and prints the
offending rows, so a gate cannot be satisfied by prose.
"""
import argparse
import glob as globmod
import json
import pathlib
import statistics
import sys

DISPATCH = "ConcurrentHashJoinProbeDispatchMicroseconds"
LOOKUP = "ConcurrentHashJoinProbeLookupMicroseconds"
PROBE_TOTAL = "ConcurrentHashJoinProbeMicroseconds"
PATH_EVENT = "ConcurrentHashJoinBuildMicroseconds"

# Events that would split `projection_cost` per side if this tree registered them.
GATHER_EVENTS = ("HashJoinResultBuildOutputMicroseconds", "HashJoinResultFilterLeftMicroseconds")

# Path assertion events of every other join algorithm: must be absent or zero.
FOREIGN_PATH_EVENTS = ("PartitionedHashJoinBuildMicroseconds",
                       "PartitionedHashJoinProbeMicroseconds",
                       "PartitionedHashJoinProbeLookupMicroseconds")

METRICS = ("probe_cost", "projection_cost")
MIN_RUNS = 5
BAND_FLOOR = 0.03
ALGO = "parallel_hash"


class Row:
    """One timed run of one arm on one unit, normalized across both harnesses."""

    __slots__ = ("unit", "arm", "role", "run", "events", "duration_us", "valid",
                 "invalid_reason", "row_count", "expected_rows", "checksum", "algo",
                 "source", "protocol", "direction", "binary_sha256", "axes")

    def __init__(self, **kw):
        for k in self.__slots__:
            setattr(self, k, kw.get(k))

    def metric(self, name):
        """None when the row cannot support the metric at all."""
        total = self.events.get(PROBE_TOTAL)
        if total is None:
            return None
        # ProfileEvents omits counters that never fired, so an absent dispatch or
        # lookup key means zero microseconds, not missing data.
        probe = self.events.get(DISPATCH, 0) + self.events.get(LOOKUP, 0)
        if name == "probe_cost":
            return probe
        if name == "projection_cost":
            return total - probe
        raise KeyError(name)


def expand(specs):
    """A --results value is a glob, a comma list of globs, or a repeated flag."""
    paths = []
    for spec in specs:
        for part in spec.split(","):
            part = part.strip()
            if not part:
                continue
            hits = sorted(globmod.glob(part))
            if not hits:
                if pathlib.Path(part).exists():
                    hits = [part]
                else:
                    sys.stderr.write(f"FAILED: --results matched nothing: {part}\n")
                    sys.exit(2)
            paths.extend(hits)
    if not paths:
        sys.stderr.write("FAILED: --results expanded to no files\n")
        sys.exit(2)
    return paths


def load(paths):
    """Normalize both harnesses' JSONL into Row objects, newest attempt winning."""
    fleet, jbmt = {}, {}
    for path in paths:
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                if "cell" in r and "arm_role" in r:
                    # fleet_ab: resume/retry appends, so the last attempt wins.
                    fleet[(r["cell"], r["arm_role"], r.get("run"))] = (r, path)
                elif "unit_id" in r:
                    jbmt[r["unit_id"]] = (r, path)
                else:
                    sys.stderr.write(f"FAILED: unrecognized record in {path}: {sorted(r)[:8]}\n")
                    sys.exit(2)

    rows = []
    for (cell, role, run), (r, path) in fleet.items():
        axes = r.get("cell_axes") or {}
        rows.append(Row(unit=cell, arm=r.get("arm"), role=role, run=run,
                        events=r.get("events") or {}, duration_us=r.get("duration_us"),
                        valid=bool(r.get("valid")), invalid_reason=r.get("invalid_reason"),
                        row_count=r.get("rows"), expected_rows=r.get("expected_rows"),
                        checksum=r.get("checksum"), algo=axes.get("algo"), source=path,
                        protocol=r.get("protocol", "abab"), direction=r.get("direction"),
                        binary_sha256=r.get("binary_sha256"), axes=axes))

    for unit_id, (r, path) in jbmt.items():
        arms = r.get("arms")
        if arms:
            per_arm = {name: a.get("algorithms") or {} for name, a in arms.items()}
            shas = {name: a.get("binary_sha256") for name, a in arms.items()}
        else:  # v1: a single implicit arm
            per_arm = {r.get("arm") or "solo": r.get("algorithms") or {}}
            shas = {name: None for name in per_arm}
        unit_status = r.get("status")
        # A unit the harness abandoned (INVALID / ERROR / MISSING_DATA / OVER_BUDGET) carries no
        # per-algorithm stats at all. It still has to SHOW UP as NO-VERDICT with the harness's
        # reason - a unit that vanishes from the report is a silently dropped unit.
        if not any(algos for algos in per_arm.values()):
            for arm in per_arm:
                rows.append(Row(unit=unit_id, arm=arm, role=None, run=0, events={},
                                duration_us=None, valid=False,
                                invalid_reason=f"jbmt unit status {unit_status}: {r.get('reason')}",
                                row_count=None, expected_rows=None, checksum=None,
                                algo=None, source=path, protocol="jbmt-abab", direction=None,
                                binary_sha256=shas.get(arm), axes=r.get("meta") or {}))
            continue
        for arm, algos in per_arm.items():
            for algo, st in algos.items():
                events_per_run = st.get("events_per_run") or []
                durations = st.get("durations_ms") or []
                bad = None
                if st.get("status") not in ("OK",):
                    bad = f"jbmt arm status {st.get('status')}: {st.get('reason')}"
                elif unit_status not in ("OK",):
                    bad = f"jbmt unit status {unit_status}: {r.get('reason')}"
                for i, ev in enumerate(events_per_run):
                    dur_ms = durations[i] if i < len(durations) else None
                    rows.append(Row(unit=unit_id, arm=arm, role=None, run=i, events=ev,
                                    duration_us=(dur_ms * 1000 if dur_ms is not None else None),
                                    valid=(bad is None), invalid_reason=bad,
                                    row_count=st.get("row_count"), expected_rows=r.get("expected_rows_closed_form"),
                                    checksum=st.get("checksum"), algo=algo, source=path,
                                    protocol="jbmt-abab", direction=None,
                                    binary_sha256=shas.get(arm), axes=r.get("meta") or {}))
    return rows


def pick_arms(rows, arm_a, arm_b):
    """Return (a_rows, b_rows) keyed by arm label, or by A/B role when labels tie.

    The A/A gate is specified as `--arm-a X --arm-b X`, so identical labels are
    legal and are split on fleet_ab's arm_role instead.
    """
    if arm_a == arm_b:
        a = [r for r in rows if r.arm == arm_a and r.role == "A"]
        b = [r for r in rows if r.arm == arm_b and r.role == "B"]
        if not a or not b:
            a = [r for r in rows if r.role == "A"]
            b = [r for r in rows if r.role == "B"]
        return a, b
    return ([r for r in rows if r.arm == arm_a],
            [r for r in rows if r.arm == arm_b])


def correctness(cell_rows):
    """The harness's own oracle: row counts as expected, checksums equal across arms."""
    problems = []
    for r in cell_rows:
        if r.expected_rows is not None and r.row_count is not None and r.row_count != r.expected_rows:
            problems.append(f"row_count {r.row_count} != expected {r.expected_rows}")
            break
    per_arm = {}
    for r in cell_rows:
        if r.checksum is not None:
            per_arm.setdefault(r.arm if r.role is None else r.role, set()).add(str(r.checksum))
    unstable = {k: v for k, v in per_arm.items() if len(v) != 1}
    if unstable:
        problems.append(f"checksum unstable within arm: {unstable}")
    distinct = {next(iter(v)) for v in per_arm.values() if len(v) == 1}
    if len(distinct) > 1:
        problems.append(f"cross-arm checksum mismatch: {per_arm}")
    if not per_arm:
        problems.append("no checksum recorded")
    return problems


def score_metric(a_rows, b_rows, metric, band_override, min_runs):
    """Per-arm median, delta, band and verdict for ONE metric. No cross-metric netting."""
    vals = {}
    for label, rs in (("A", a_rows), ("B", b_rows)):
        v = [r.metric(metric) for r in rs]
        if any(x is None for x in v):
            return {"verdict": "NO-VERDICT", "reason": f"{PROBE_TOTAL} absent on arm {label}"}
        vals[label] = v
    if any(len(v) < min_runs for v in vals.values()):
        return {"verdict": "NO-VERDICT",
                "reason": f"insufficient valid runs (A={len(vals['A'])} B={len(vals['B'])}, need {min_runs})"}
    if any(x < 0 for v in vals.values() for x in v):
        return {"verdict": "NO-VERDICT", "reason": f"negative {metric} (decomposition broken)"}

    med = {k: statistics.median(v) for k, v in vals.items()}
    spread = {k: (statistics.pstdev(v) if len(v) > 1 else 0.0) for k, v in vals.items()}
    rel_spread = max((spread[k] / med[k]) if med[k] else 0.0 for k in ("A", "B"))
    scale = max(med["A"], med["B"])
    if scale == 0:
        return {"verdict": "NO-VERDICT", "reason": f"{metric} median is zero on both arms",
                "median_a": 0, "median_b": 0}
    band = max(BAND_FLOOR, rel_spread) if band_override is None else band_override
    diff = med["B"] - med["A"]
    if abs(diff) <= band * scale:
        verdict = "TIE"
    elif diff < 0:
        verdict = "WIN"
    else:
        verdict = "LOSS"
    return {"verdict": verdict, "median_a": med["A"], "median_b": med["B"],
            "delta_us": diff, "delta_pct": (diff / med["A"] * 100.0) if med["A"] else float("inf"),
            "band_pct": band * 100.0, "rel_spread_pct": rel_spread * 100.0,
            "n_a": len(vals["A"]), "n_b": len(vals["B"])}


def raw_components(a_rows, b_rows):
    """Dispatch, lookup and probe total medians, so a reviewer can re-derive both metrics."""
    out = {}
    for label, rs in (("a", a_rows), ("b", b_rows)):
        for short, key in (("dispatch", DISPATCH), ("lookup", LOOKUP), ("probe_total", PROBE_TOTAL)):
            vals = [r.events.get(key, 0) for r in rs]
            out[f"{short}_{label}"] = statistics.median(vals) if vals else None
        walls = [r.duration_us for r in rs if r.duration_us is not None]
        out[f"wall_{label}"] = statistics.median(walls) if walls else None
    return out


def analyse(rows, arm_a, arm_b, band_override=None, min_runs=MIN_RUNS):
    by_unit = {}
    for r in rows:
        by_unit.setdefault(r.unit, []).append(r)

    cells = []
    for unit, unit_rows in sorted(by_unit.items()):
        a_all, b_all = pick_arms(unit_rows, arm_a, arm_b)
        entry = {"unit": unit, "axes": unit_rows[0].axes or {},
                 "n_rows": {"A": len(a_all), "B": len(b_all)},
                 "n_invalid": sum(1 for r in a_all + b_all if not r.valid)}

        blockers = []
        if not a_all or not b_all:
            blockers.append(f"arm rows missing (A={len(a_all)} B={len(b_all)}) for arms "
                            f"{arm_a!r}/{arm_b!r}")
        # A cell voided by the harness is NO-VERDICT with the harness's own reason -
        # never quietly rescored, never counted as a tie.
        void = sorted({r.invalid_reason for r in a_all + b_all
                       if not r.valid and r.invalid_reason})
        if void:
            blockers.append("harness voided: " + "; ".join(void))
        protocols = {r.protocol for r in a_all + b_all}
        if len(protocols) > 1:
            blockers.append(f"mixed protocol {sorted(protocols)}")

        a_valid = [r for r in a_all if r.valid]
        b_valid = [r for r in b_all if r.valid]
        if a_valid and b_valid:
            blockers += correctness(a_valid + b_valid)

        entry["blockers"] = blockers
        entry["metrics"] = {}
        if blockers:
            for m in METRICS:
                entry["metrics"][m] = {"verdict": "NO-VERDICT", "reason": "; ".join(blockers)}
            entry["raw"] = raw_components(a_valid or a_all, b_valid or b_all)
        else:
            for m in METRICS:
                entry["metrics"][m] = score_metric(a_valid, b_valid, m, band_override, min_runs)
            entry["raw"] = raw_components(a_valid, b_valid)
        cells.append(entry)
    return cells


# ---------------------------------------------------------------- checks (gates)

def check_decomposition(rows, arm_a, arm_b):
    """G0-b: the residual is well formed and both arms decompose identically."""
    fails = []
    per_arm_gather = {}
    for r in rows:
        if not r.valid:
            continue
        arm_key = r.arm if r.role is None else f"{r.arm}/{r.role}"
        total = r.events.get(PROBE_TOTAL)
        if total is None:
            continue
        probe = r.events.get(DISPATCH, 0) + r.events.get(LOOKUP, 0)
        residual = total - probe
        if probe + residual != total:
            fails.append(f"{r.unit} {arm_key} run {r.run}: probe_cost+projection_cost "
                         f"{probe + residual} != {PROBE_TOTAL} {total}")
        if residual < 0:
            fails.append(f"{r.unit} {arm_key} run {r.run}: projection_cost {residual} < 0 "
                         f"(dispatch {r.events.get(DISPATCH, 0)} + lookup {r.events.get(LOOKUP, 0)} "
                         f"> probe total {total})")
        present = tuple(sorted(e for e in GATHER_EVENTS if e in r.events))
        per_arm_gather.setdefault(arm_key, set()).add(present)

    print("G0-b decomposition:")
    for arm_key in sorted(per_arm_gather):
        sets = per_arm_gather[arm_key]
        print(f"  arm {arm_key}: gather events present = "
              f"{sorted({p for s in sets for p in s}) or '<none>'}")
    distinct = {frozenset(p for s in sets for p in s) for sets in per_arm_gather.values()}
    if len(distinct) > 1:
        fails.append(f"gather-event presence differs across arms: "
                     f"{ {k: sorted({p for s in v for p in s}) for k, v in per_arm_gather.items()} }")
    if len(distinct) == 1 and next(iter(distinct)):
        print(f"  NOTE: gather events ARE present ({sorted(next(iter(distinct)))}) - "
              f"projection_cost can be split per side; this report does not.")
    else:
        print("  gather events absent on both arms => projection_cost is an unsplit residual")
    print(f"  rows checked: {sum(1 for r in rows if r.valid)}   violations: {len(fails)}")
    for f in fails[:40]:
        print(f"    FAIL {f}")
    if len(fails) > 40:
        print(f"    ... and {len(fails) - 40} more")
    return fails


def check_path_event(rows, cells):
    """G0-c: parallel_hash and nothing else, proven per timed run."""
    fails = []
    checked = 0
    for r in rows:
        if not r.valid:
            continue
        checked += 1
        arm_key = r.arm if r.role is None else f"{r.arm}/{r.role}"
        if not r.events.get(PATH_EVENT, 0) > 0:
            fails.append(f"{r.unit} {arm_key} run {r.run}: {PATH_EVENT} = "
                         f"{r.events.get(PATH_EVENT)} (not > 0)")
        for ev in FOREIGN_PATH_EVENTS:
            if r.events.get(ev, 0):
                fails.append(f"{r.unit} {arm_key} run {r.run}: foreign path event {ev} = {r.events[ev]}")
        if r.algo is not None and r.algo != ALGO:
            fails.append(f"{r.unit} {arm_key} run {r.run}: join_algorithm recorded as {r.algo!r}")
    for c in cells:
        if ".hash" in c["unit"]:
            fails.append(f"{c['unit']}: cell id carries the .hash modifier")
    print("G0-c only-parallel_hash:")
    print(f"  timed runs checked: {checked}   violations: {len(fails)}")
    for f in fails[:40]:
        print(f"    FAIL {f}")
    if len(fails) > 40:
        print(f"    ... and {len(fails) - 40} more")
    return fails


def check_expect_cells(cells, expect, metrics):
    """Exactly N cells must carry a verdict on every requested metric."""
    fails = []
    scored = [c for c in cells if all(c["metrics"][m]["verdict"] != "NO-VERDICT" for m in metrics)]
    print(f"coverage: {len(scored)} cells with a verdict on {'+'.join(metrics)}, "
          f"expected {expect}   (total cells seen: {len(cells)})")
    if len(scored) != expect:
        fails.append(f"--expect-cells {expect} but {len(scored)} cells have a verdict on all of "
                     f"{metrics}")
        for c in cells:
            reasons = {m: c["metrics"][m].get("reason") for m in metrics
                       if c["metrics"][m]["verdict"] == "NO-VERDICT"}
            if reasons:
                print(f"    NO-VERDICT {c['unit']}: {reasons}")
    return fails


def check_unit_set(cells, spec, metrics, seen_not_scored=False):
    """Set equality, not a count: a loosened filter cannot fake this."""
    path, _, field = spec.partition(":")
    field = field or "cell_id"
    data = json.loads(pathlib.Path(path).read_text())
    want = {d[field] for d in data} if isinstance(data, list) else set(data)
    got = ({c["unit"] for c in cells} if seen_not_scored else
           {c["unit"] for c in cells
            if all(c["metrics"][m]["verdict"] != "NO-VERDICT" for m in metrics)})
    missing, extra = sorted(want - got), sorted(got - want)
    print(f"unit set from {path}:{field}: expected {len(want)}, scored {len(got)}, "
          f"missing {len(missing)}, extra {len(extra)}")
    print(f"  set equality: {'YES' if not missing and not extra else 'NO'}")
    for u in missing[:20]:
        print(f"    MISSING {u}")
    for u in extra[:20]:
        print(f"    EXTRA   {u}")
    fails = []
    if missing or extra:
        fails.append(f"scored unit set != {path}:{field} "
                     f"({len(missing)} missing, {len(extra)} extra)")
    return fails


def check_aa(cells, metrics, min_cells):
    """G0-a: identical binaries must produce TIE everywhere, on BOTH metrics."""
    fails = []
    print("G0-a A/A control:")
    for m in metrics:
        scored = [c for c in cells if c["metrics"][m]["verdict"] != "NO-VERDICT"]
        offenders = [c for c in scored if c["metrics"][m]["verdict"] != "TIE"]
        floor = max((abs(c["metrics"][m]["delta_pct"]) for c in scored), default=0.0)
        floor_us = max((abs(c["metrics"][m]["delta_us"]) for c in scored), default=0.0)
        print(f"  {m}: {len(scored)} scored, {len(offenders)} non-TIE, "
              f"empirical noise floor = {floor:.2f}% ({floor_us:,.0f} us largest |delta|)")
        for c in offenders:
            r = c["metrics"][m]
            print(f"    FAIL {c['unit']}: {r['verdict']} {r['delta_pct']:+.1f}% "
                  f"(band {r['band_pct']:.1f}%)")
            fails.append(f"A/A {m} {c['unit']}: {r['verdict']}")
        if len(scored) < min_cells:
            fails.append(f"A/A {m}: only {len(scored)} cells carry a verdict, need {min_cells} "
                         f"for the control to have power")
            print(f"    FAIL only {len(scored)} scored cells, need {min_cells}")
    return fails


def compare_order(cells, other_cells, metrics, fail_on_effect=False):
    """G1-b: per metric, which cells' verdicts flip between block orders.

    Gate G1-b is specified to exit 0 and print a possibly empty list, because an
    order effect is a finding to report per cell, not a failure. That leaves the
    check unable to go red, so `--fail-on-order-effect` makes the flip list
    enforceable and gives the check demonstrable power to fail.
    """
    by_other = {c["unit"]: c for c in other_cells}
    fails = []
    print("G1-b block-order comparison (ABBA vs BAAB):")
    for m in metrics:
        flips, common = [], 0
        for c in cells:
            o = by_other.get(c["unit"])
            if not o:
                continue
            v1, v2 = c["metrics"][m]["verdict"], o["metrics"][m]["verdict"]
            if "NO-VERDICT" in (v1, v2):
                continue
            common += 1
            if v1 != v2:
                flips.append((c["unit"], v1, v2, c["metrics"][m]["delta_pct"], o["metrics"][m]["delta_pct"]))
        print(f"  {m}: {common} cells with verdicts in both orders, {len(flips)} disagree")
        for u, v1, v2, d1, d2 in flips:
            print(f"    ORDER-EFFECT {u}: ABBA {v1} ({d1:+.1f}%) vs BAAB {v2} ({d2:+.1f}%)")
            if fail_on_effect:
                fails.append(f"order effect on {m}: {u} ABBA {v1} vs BAAB {v2}")
        if not flips:
            print("    (empty list - no cell's verdict depends on block order)")
    return fails


# ---------------------------------------------------------------- reporting

def fmt_ms(us):
    return "-" if us is None else f"{us / 1000.0:,.1f}"


def report(cells, metrics, label):
    print(f"\n=== {label}: per-metric verdicts "
          f"(band = max({BAND_FLOOR:.0%}, per-arm relative spread), evaluated per metric) ===")
    for m in metrics:
        scored = [c for c in cells if c["metrics"][m]["verdict"] != "NO-VERDICT"]
        counts = {v: sum(1 for c in scored if c["metrics"][m]["verdict"] == v)
                  for v in ("WIN", "TIE", "LOSS")}
        tot_a = sum(c["metrics"][m]["median_a"] for c in scored)
        tot_b = sum(c["metrics"][m]["median_b"] for c in scored)
        agg = ((tot_b - tot_a) / tot_a * 100.0) if tot_a else 0.0
        print(f"\n{m}: verdicts {len(scored)}   win={counts['WIN']} tie={counts['TIE']} "
              f"loss={counts['LOSS']}   no-verdict={len(cells) - len(scored)}")
        print(f"  aggregate {fmt_ms(tot_a)} ms -> {fmt_ms(tot_b)} ms ({agg:+.1f}%)")
        if scored:
            print(f"  median per-cell delta {statistics.median(c['metrics'][m]['delta_pct'] for c in scored):+.1f}%")
        losses = sorted((c for c in scored if c["metrics"][m]["verdict"] == "LOSS"),
                        key=lambda c: -c["metrics"][m]["delta_pct"])
        print(f"  LOSSES on {m}: {len(losses)}")
        if losses:
            print(f"    {'cell':<46} {'A ms':>10} {'B ms':>10} {'delta':>8} {'band':>6} "
                  f"{'disp A->B (ms)':>22} {'lookup A->B (ms)':>22} {'probe tot A->B (ms)':>24}")
            for c in losses:
                r, w = c["metrics"][m], c["raw"]
                print(f"    {c['unit']:<46} {fmt_ms(r['median_a']):>10} {fmt_ms(r['median_b']):>10} "
                      f"{r['delta_pct']:>+7.1f}% {r['band_pct']:>5.1f}% "
                      f"{fmt_ms(w['dispatch_a']) + '->' + fmt_ms(w['dispatch_b']):>22} "
                      f"{fmt_ms(w['lookup_a']) + '->' + fmt_ms(w['lookup_b']):>22} "
                      f"{fmt_ms(w['probe_total_a']) + '->' + fmt_ms(w['probe_total_b']):>24}")
        wins = sorted((c for c in scored if c["metrics"][m]["verdict"] == "WIN"),
                      key=lambda c: c["metrics"][m]["delta_pct"])
        print(f"  WINS on {m}: {len(wins)}")
        for c in wins:
            r = c["metrics"][m]
            print(f"    {c['unit']:<46} {fmt_ms(r['median_a']):>10} -> {fmt_ms(r['median_b']):>10} "
                  f"{r['delta_pct']:>+7.1f}% (band {r['band_pct']:.1f}%)")

    if len(metrics) > 1:
        opposed = []
        for c in cells:
            v = {m: c["metrics"][m]["verdict"] for m in metrics}
            if {"WIN", "LOSS"} <= set(v.values()):
                opposed.append(c)
        print(f"\nCELLS WHOSE TWO METRICS MOVE IN OPPOSITE DIRECTIONS: {len(opposed)} "
              f"(listed in both lists above, never netted)")
        for c in opposed:
            parts = " ".join(f"{m}={c['metrics'][m]['verdict']}({c['metrics'][m]['delta_pct']:+.1f}%)"
                             for m in metrics)
            print(f"    {c['unit']:<46} {parts}")

    no_verdict = [c for c in cells
                  if any(c["metrics"][m]["verdict"] == "NO-VERDICT" for m in metrics)]
    print(f"\nNO-VERDICT CELLS: {len(no_verdict)}")
    for c in no_verdict:
        reasons = {m: c["metrics"][m].get("reason") for m in metrics
                   if c["metrics"][m]["verdict"] == "NO-VERDICT"}
        print(f"    {c['unit']:<46} {reasons}")


def tsv_safe(v):
    """A harness reason can quote a server exception, which brings newlines and tabs with it.

    Left raw, one such reason silently splits a row across lines and every column after it
    is misread, so the TSV has to flatten whitespace rather than trust the source string.
    """
    if v is None:
        return ""
    return " ".join(str(v).split())


def write_tsv(cells, metrics, path):
    cols = ["unit"]
    for m in metrics:
        cols += [f"{m}_verdict", f"{m}_median_a_us", f"{m}_median_b_us", f"{m}_delta_us",
                 f"{m}_delta_pct", f"{m}_band_pct", f"{m}_reason"]
    cols += ["dispatch_a_us", "dispatch_b_us", "lookup_a_us", "lookup_b_us",
             "probe_total_a_us", "probe_total_b_us", "wall_a_us", "wall_b_us"]
    with open(path, "w") as fh:
        fh.write("\t".join(cols) + "\n")
        for c in cells:
            vals = [c["unit"]]
            for m in metrics:
                r = c["metrics"][m]
                vals += [r["verdict"], r.get("median_a", ""), r.get("median_b", ""),
                         r.get("delta_us", ""), f"{r['delta_pct']:.3f}" if "delta_pct" in r else "",
                         f"{r['band_pct']:.3f}" if "band_pct" in r else "", r.get("reason", "")]
            w = c["raw"]
            vals += [w.get(k) for k in ("dispatch_a", "dispatch_b", "lookup_a", "lookup_b",
                                        "probe_total_a", "probe_total_b", "wall_a", "wall_b")]
            fh.write("\t".join(tsv_safe(v) for v in vals) + "\n")
    print(f"\nper-cell TSV written to {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", action="append", required=True,
                    help="glob or comma list of harness JSONL; repeatable")
    ap.add_argument("--arm-a", default=None, help="arm label of the baseline arm")
    ap.add_argument("--arm-b", default=None, help="arm label of the candidate arm")
    ap.add_argument("--metric", choices=["probe_cost", "projection_cost", "both"], default="both")
    ap.add_argument("--expect-cells", type=int, help="fail unless exactly N cells have a verdict")
    ap.add_argument("--expect-unit-set", help="PATH:FIELD - fail unless the scored unit set equals it")
    ap.add_argument("--expect-unit-set-seen", action="store_true",
                    help="with --expect-unit-set, compare the set of units MEASURED rather than "
                         "the set that earned a verdict; proves the sweep ran the intended units "
                         "even where some are legitimately NO-VERDICT")
    ap.add_argument("--check-decomposition", action="store_true", help="G0-b")
    ap.add_argument("--check-path-event", action="store_true", help="G0-c")
    ap.add_argument("--aa-control", action="store_true", help="G0-a: every cell must TIE on both metrics")
    ap.add_argument("--aa-min-cells", type=int, default=8,
                    help="minimum scored cells for the A/A control to have power (default 8)")
    ap.add_argument("--compare-order", help="G1-b: second results glob to compare verdicts against")
    ap.add_argument("--fail-on-order-effect", action="store_true",
                    help="with --compare-order, exit non-zero if any cell's verdict flips between "
                         "block orders (G1-b as specified only prints the list, so this is what "
                         "gives that check power to fail)")
    ap.add_argument("--min-runs", type=int, default=MIN_RUNS)
    ap.add_argument("--band-override", type=float, default=None,
                    help="force the band to exactly this fraction. FOR GATE-POWER TESTING ONLY "
                         "(e.g. --band-override 0 must turn the A/A control red); never for acceptance")
    ap.add_argument("--label", default="results")
    ap.add_argument("--out-tsv")
    ap.add_argument("--out-json")
    ap.add_argument("--quiet-report", action="store_true", help="run checks only, skip the verdict tables")
    args = ap.parse_args()

    metrics = METRICS if args.metric == "both" else (args.metric,)
    paths = expand(args.results)
    rows = load(paths)
    arms_seen = sorted({(r.arm, r.role) for r in rows})
    arm_a = args.arm_a
    arm_b = args.arm_b
    if arm_a is None or arm_b is None:
        labels = sorted({r.arm for r in rows if r.arm})
        roles = sorted({r.role for r in rows if r.role})
        if len(labels) == 2:
            arm_a, arm_b = arm_a or labels[0], arm_b or labels[1]
        elif len(labels) == 1 and roles == ["A", "B"]:
            arm_a = arm_b = labels[0]
        else:
            sys.stderr.write(f"FAILED: pass --arm-a/--arm-b; arms present: {arms_seen}\n")
            return 2

    print(f"probe_ab_report: {len(paths)} file(s), {len(rows)} timed-run rows, "
          f"arms present {arms_seen}")
    print(f"arm A = {arm_a!r}   arm B = {arm_b!r}   metrics = {list(metrics)}")
    if args.band_override is not None:
        print(f"*** BAND OVERRIDE {args.band_override} - gate-power testing only, NOT acceptance ***")

    cells = analyse(rows, arm_a, arm_b, args.band_override, args.min_runs)
    fails = []
    if args.check_decomposition:
        fails += check_decomposition(rows, arm_a, arm_b)
    if args.check_path_event:
        fails += check_path_event(rows, cells)
    if args.expect_cells is not None:
        fails += check_expect_cells(cells, args.expect_cells, metrics)
    if args.expect_unit_set:
        fails += check_unit_set(cells, args.expect_unit_set, metrics,
                                args.expect_unit_set_seen)
    if args.aa_control:
        fails += check_aa(cells, metrics, args.aa_min_cells)
    if args.compare_order:
        other = analyse(load(expand([args.compare_order])), arm_a, arm_b,
                        args.band_override, args.min_runs)
        fails += compare_order(cells, other, metrics, args.fail_on_order_effect)

    if not args.quiet_report:
        report(cells, metrics, args.label)
    if args.out_tsv:
        write_tsv(cells, metrics, args.out_tsv)
    if args.out_json:
        pathlib.Path(args.out_json).write_text(json.dumps(
            {"label": args.label, "arm_a": arm_a, "arm_b": arm_b, "metrics": list(metrics),
             "files": paths, "cells": cells, "failures": fails}, indent=1, default=str))
        print(f"machine-readable report written to {args.out_json}")

    print(f"\nCHECK SUMMARY: {'PASS' if not fails else 'FAIL'} ({len(fails)} failed check(s))")
    for f in fails[:60]:
        print(f"  FAILED: {f}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
