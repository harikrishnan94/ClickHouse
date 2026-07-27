#!/usr/bin/env python3
"""parity_driver.py — case executor / comparator / engagement checker for the
parallel_hash parity harness. Invoked by run_parity.sh; can be used manually
for debugging single phases.

Modes:
  run      --cases F --client "BIN client --host H --port P" --outdir D
           Executes every case's setup (deduped) and both verdict queries,
           writing results/<id>.tsv and results/<id>.chk via client-side
           INTO OUTFILE. Queries are batched (--chunk per client invocation);
           a failed chunk falls back to per-case execution to isolate errors.
  compare  --cases F --base-dir D1 --cand-dir D2 --logdir L
           Byte-diffs (a), cross-checks (b), verifies chk count == TSV line
           count, writes logs/<id>.divergence.txt + logs/<id>.repro.sql for
           every divergence (does NOT stop at the first), prints
           'COMPARE {json}' as its last line. Matched-errors (both arms raise
           the identical genuine DB::Exception) are parity-preserving but
           budgeted (default: fail if > 4 or > 2% of cases; override with an
           explicit --acknowledge-matched-errors N), and at least
           --min-compared-pct (default 90%) of cases must produce real
           comparisons; violations are emitted in 'gate_failures'.
  engage   --cases F --client "..." --logdir L
           AMAC force-pass: per-family subset with log_comment tagging;
           asserts the AMAC engagement counters (contract constants imported
           from parity_gen) are > 0 in system.query_log. Only called when
           run_parity.sh detected the counters in the candidate binary.
           Prints 'ENGAGE {json}' as its last line.
  audit    --cases F --client "..." --label ARM
           Informational: per-family subset, reports the seven shared
           ConcurrentHashJoin* ProfileEvents from system.query_log (did
           parallel_hash actually engage?). Prints 'AUDIT {json}'.

Python: stdlib only.
"""

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import parity_gen  # single source of the Units 2-3 contract constants

CLIENT_TIMEOUT_S = 600


def load_cases(path, limit=None):
    cases = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    if limit is not None and limit > 0:
        cases = cases[:limit]
    return cases


def run_client(client_argv, *, query=None, queries_file=None):
    argv = list(client_argv)
    if query is not None:
        argv += ["-q", query]
    if queries_file is not None:
        argv += ["--queries-file", queries_file]
    try:
        proc = subprocess.run(argv, capture_output=True, text=True, timeout=CLIENT_TIMEOUT_S)
    except subprocess.TimeoutExpired as e:
        # Fail-close: a timeout is an ordinary failed invocation (rc 124), so
        # a timed-out chunk falls back to per-case execution and a timed-out
        # case is recorded as a case error — never an aborted run.
        def as_text(v):
            return v.decode(errors="replace") if isinstance(v, bytes) else (v or "")
        return 124, as_text(e.stdout), (
            f"client timeout after {CLIENT_TIMEOUT_S}s (harness fail-close)\n" + as_text(e.stderr))
    return proc.returncode, proc.stdout, proc.stderr


def outfile_stmt(verdict_query, path):
    """Splice INTO OUTFILE before the trailing FORMAT TSV (generator contract)."""
    suffix = "FORMAT TSV"
    assert verdict_query.endswith(suffix), verdict_query[-80:]
    head = verdict_query[: -len(suffix)]
    return f"{head}INTO OUTFILE '{path}' TRUNCATE {suffix}"


def result_paths(outdir, case_id):
    d = os.path.join(outdir, "results")
    return os.path.join(d, case_id + ".tsv"), os.path.join(d, case_id + ".chk")


def chunked(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i : i + n]


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------

def cmd_run(args):
    cases = load_cases(args.cases, args.limit)
    client_argv = shlex.split(args.client)
    outdir = os.path.abspath(args.outdir)
    scripts = os.path.join(outdir, "scripts")
    results = os.path.join(outdir, "results")
    os.makedirs(scripts, exist_ok=True)
    os.makedirs(results, exist_ok=True)

    t0 = time.time()
    errors = {}

    # Phase 1: deduped setup (ddl + fill). A setup failure is fatal: the
    # matrix must be valid SQL, so we stop rather than mass-skip.
    seen = {}
    for c in cases:
        script = "\n".join(c["ddl"] + c["fill"])
        seen.setdefault(script, c["id"])
    for i, (script, first_case) in enumerate(seen.items()):
        path = os.path.join(scripts, f"setup_{i:03d}.sql")
        with open(path, "w") as f:
            f.write(script + "\n")
        rc, _, err = run_client(client_argv, queries_file=path)
        if rc != 0:
            print(f"FATAL: setup {path} (first case {first_case}) failed rc={rc}:\n{err[-4000:]}",
                  file=sys.stderr)
            return 2
    print(f"setup: {len(seen)} unique table sets filled in {time.time() - t0:.1f}s", flush=True)

    # Phase 2: verdict queries, chunked; per-case fallback on chunk error.
    def case_stmts(c):
        tsv, chk = result_paths(outdir, c["id"])
        return [outfile_stmt(c["verdict_tsv"], tsv) + ";",
                outfile_stmt(c["verdict_chk"], chk) + ";"]

    def clear_outputs(c):
        for p in result_paths(outdir, c["id"]):
            if os.path.exists(p):
                os.remove(p)

    for ci, chunk in enumerate(chunked(cases, args.chunk)):
        for c in chunk:
            clear_outputs(c)
        path = os.path.join(scripts, f"chunk_{ci:03d}.sql")
        with open(path, "w") as f:
            for c in chunk:
                f.write("\n".join(case_stmts(c)) + "\n")
        tc = time.time()
        rc, _, err = run_client(client_argv, queries_file=path)
        if rc != 0:
            print(f"chunk {ci}: rc={rc}, falling back to per-case execution", flush=True)
            for c in chunk:
                clear_outputs(c)
                single = os.path.join(scripts, f"case_{c['id']}.sql")
                with open(single, "w") as f:
                    f.write("\n".join(case_stmts(c)) + "\n")
                crc, _, cerr = run_client(client_argv, queries_file=single)
                if crc != 0:
                    errors[c["id"]] = cerr.strip()[-2000:]
                    clear_outputs(c)
        print(f"chunk {ci}: {len(chunk)} cases in {time.time() - tc:.1f}s", flush=True)

    # Phase 3: sanity — every non-errored case must have both outputs.
    for c in cases:
        if c["id"] in errors:
            continue
        tsv, chk = result_paths(outdir, c["id"])
        if not (os.path.exists(tsv) and os.path.exists(chk)):
            errors[c["id"]] = "missing output file(s) after successful client run"

    summary = {"executed": len(cases), "errors": errors, "duration_s": round(time.time() - t0, 1)}
    with open(os.path.join(outdir, "run_summary.json"), "w") as f:
        json.dump(summary, f, indent=1, sort_keys=True)
    print("RUN " + json.dumps({"executed": len(cases), "n_errors": len(errors),
                               "duration_s": summary["duration_s"]}))
    return 0


# ---------------------------------------------------------------------------
# compare
# ---------------------------------------------------------------------------

SERVER_EXCEPTION_RE = re.compile(r"Code: (\d+)\. DB::Exception: (.*)")


def normalize_error(text):
    """Reduce a client stderr blob to a comparable server-exception signature,
    or None when the blob carries no genuine `DB::Exception` line. Connection
    failures (`DB::NetException`), client/harness errors, timeouts, and
    sentinels like 'missing output file(s)' therefore can NEVER be
    matched-errors: only real server exceptions qualify. The error Code
    number is preserved verbatim (never collapsed). The `Received from
    host:port.` clause is removed entirely, so ports cannot survive into the
    signature; only genuinely varying tokens are collapsed afterwards — hex
    addresses and remaining digit runs (row counts / chunk sizes vary with
    scheduling). Used ONLY to decide whether two arms that BOTH errored
    raised the same exception (matched-error == parity-preserving);
    one-sided errors are always divergences."""
    m = None
    for m in SERVER_EXCEPTION_RE.finditer(text):
        pass  # keep the last match: the server-side exception line
    if m is None:
        return None
    code, body = m.group(1), m.group(2)
    body = re.sub(r"^Received from \S+\. DB::Exception: ", "", body)
    body = re.sub(r"\(version [^)]*\)", "", body)
    body = re.sub(r"0x[0-9a-fA-F]+", "ADDR", body)
    body = re.sub(r"\d+", "N", body)
    body = " ".join(body.split())
    return f"Code: {code}. DB::Exception: {body}"


def first_diff_lines(a: bytes, b: bytes, limit=3):
    la, lb = a.split(b"\n"), b.split(b"\n")
    out = []
    for i in range(max(len(la), len(lb))):
        va = la[i] if i < len(la) else b"<absent>"
        vb = lb[i] if i < len(lb) else b"<absent>"
        if va != vb:
            out.append((i + 1, va[:300], vb[:300]))
            if len(out) >= limit:
                break
    return out


def write_divergence(logdir, case, status, detail, base_label, cand_label, base_client, cand_client,
                     suffix=".divergence.txt"):
    os.makedirs(logdir, exist_ok=True)
    repro_sql = os.path.join(logdir, case["id"] + ".repro.sql")
    with open(repro_sql, "w") as f:
        f.write("\n".join(case["ddl"] + case["fill"]) + "\n")
        f.write(case["verdict_tsv"] + ";\n")
        f.write(case["verdict_chk"] + ";\n")
    path = os.path.join(logdir, case["id"] + suffix)
    with open(path, "w") as f:
        f.write(f"CASE {case['id']}\n")
        f.write(f"family={case['family']} shape={case['shape']} kind={case['kind']} "
                f"strictness={case['strictness']} variant={case['variant']}\n")
        f.write(f"settings={json.dumps(case['settings'], sort_keys=True)}\n")
        f.write(f"STATUS: {status}\n\n{detail}\n\n")
        f.write("REPRO (each arm; the .repro.sql recreates tables, refills, and runs both verdict queries):\n")
        f.write(f"  # {base_label}\n  {base_client} --queries-file {repro_sql}\n")
        f.write(f"  # {cand_label}\n  {cand_client} --queries-file {repro_sql}\n")
    return path


def cmd_compare(args):
    cases = load_cases(args.cases, args.limit)
    base_dir, cand_dir = os.path.abspath(args.base_dir), os.path.abspath(args.cand_dir)

    def load_errors(d):
        p = os.path.join(d, "run_summary.json")
        if not os.path.exists(p):
            return None
        with open(p) as f:
            return json.load(f)["errors"]

    base_errors, cand_errors = load_errors(base_dir), load_errors(cand_dir)
    if base_errors is None or cand_errors is None:
        print("FATAL: missing run_summary.json in one of the run dirs", file=sys.stderr)
        return 2

    divergences = []
    matched_errors = []
    compared = 0  # cases where both arms produced outputs that were byte-diffed
    failed = 0    # cases with errors / missing outputs, not matched-error
    for c in cases:
        cid = c["id"]
        status, detail = None, ""
        eb, ec = base_errors.get(cid), cand_errors.get(cid)
        sig_b = normalize_error(eb) if eb else None
        if eb and ec and sig_b is not None and sig_b == normalize_error(ec):
            # Both arms raised the SAME genuine server exception: behavior
            # matches, so this is parity-preserving (a pre-existing product
            # exception, not an arm divergence). Reported loudly, capped by
            # the matched-error budget below. If a later candidate makes the
            # error appear/disappear on one arm only, that becomes an
            # error-<arm> divergence below. Non-DB::Exception errors
            # (connection refused, harness sentinels) never reach here:
            # normalize_error returns None for them.
            detail = (f"identical exception on both arms (signature: "
                      f"{sig_b}):\n\nbaseline error:\n{eb}\n\ncandidate error:\n{ec}")
            write_divergence(args.logdir, c, "matched-error-both-arms", detail,
                             args.base_label, args.cand_label,
                             args.base_client, args.cand_client,
                             suffix=".matched-error.txt")
            matched_errors.append(cid)
            print(f"MATCHED-ERROR {cid}: {sig_b}", flush=True)
            continue
        if eb or ec:
            failed += 1
            status = "error-both-arms-different" if (eb and ec) else ("error-baseline" if eb else "error-candidate")
            detail = f"baseline error:\n{eb or '<none>'}\n\ncandidate error:\n{ec or '<none>'}"
        else:
            compared += 1
            tsv_b, chk_b = result_paths(base_dir, cid)
            tsv_c, chk_c = result_paths(cand_dir, cid)
            with open(tsv_b, "rb") as f:
                a = f.read()
            with open(tsv_c, "rb") as f:
                b = f.read()
            with open(chk_b, "rb") as f:
                ka = f.read()
            with open(chk_c, "rb") as f:
                kb = f.read()
            tsv_eq, chk_eq = a == b, ka == kb
            if not tsv_eq or not chk_eq:
                if not tsv_eq and chk_eq:
                    status = "tsv-mismatch-chk-match"  # checksum too weak or column subset issue
                elif tsv_eq and not chk_eq:
                    status = "tsv-match-chk-mismatch"  # harness anomaly: bytes equal, checksum differs
                else:
                    status = "tsv-and-chk-mismatch"
                diffs = first_diff_lines(a, b)
                lines = [f"tsv bytes: baseline={len(a)} candidate={len(b)}",
                         f"chk baseline: {ka.decode(errors='replace').strip()!r}",
                         f"chk candidate: {kb.decode(errors='replace').strip()!r}"]
                for ln, va, vb in diffs:
                    lines.append(f"first-diff line {ln}:")
                    lines.append(f"  baseline : {va!r}")
                    lines.append(f"  candidate: {vb!r}")
                detail = "\n".join(lines)
            else:
                # internal cross-check: chk row count must equal TSV line count
                # (TSV escapes embedded newlines, so lines == rows; verified).
                try:
                    cnt = int(ka.split(b"\t")[0])
                except ValueError:
                    cnt = -1
                n_lines = a.count(b"\n")
                if cnt != n_lines:
                    status = "count-vs-lines-mismatch"
                    detail = f"chk count()={cnt} but TSV has {n_lines} lines (harness anomaly)"
        if status:
            p = write_divergence(args.logdir, c, status, detail,
                                 args.base_label, args.cand_label,
                                 args.base_client, args.cand_client)
            divergences.append({"id": cid, "status": status, "log": p})
            print(f"DIVERGENCE {cid}: {status}", flush=True)

    # Gate guards (fail-close). Matched errors are parity-preserving
    # individually, but too many of them mean the matrix is broken (bad
    # setting, dead server, harness bug) and the gate would pass vacuously —
    # hence the budget and the minimum-compared floor.
    n_matched = len(matched_errors)
    gate_failures = []
    if args.acknowledge_matched_errors is not None:
        if n_matched > args.acknowledge_matched_errors:
            gate_failures.append(f"matched-errors {n_matched} exceed the explicitly acknowledged "
                                 f"{args.acknowledge_matched_errors}")
    elif n_matched > args.max_matched_errors:
        gate_failures.append(f"matched-errors {n_matched} > budget {args.max_matched_errors} "
                             f"(override requires --acknowledge-matched-errors N)")
    elif n_matched * 100.0 > len(cases) * args.max_matched_error_pct:
        gate_failures.append(f"matched-errors {n_matched} > {args.max_matched_error_pct}% of {len(cases)} cases "
                             f"(override requires --acknowledge-matched-errors N)")
    if compared * 100.0 < len(cases) * args.min_compared_pct:
        gate_failures.append(f"only {compared}/{len(cases)} cases produced real comparisons "
                             f"(< {args.min_compared_pct}% floor)")
    for g in gate_failures:
        print(f"GATE-FAILURE {g}", flush=True)

    families = sorted({c["family"] for c in cases})
    combos = sorted({(c["kind"], c["strictness"], c["variant"]) for c in cases})
    print("COMPARE " + json.dumps({
        "cases": len(cases),
        "compared": compared,
        "failed": failed,
        "families": len(families),
        "combos": len(combos),
        "divergences": len(divergences),
        "diverged": [d["id"] for d in divergences][:100],
        "matched_errors": n_matched,
        "matched_error_ids": matched_errors[:100],
        "gate_failures": gate_failures,
    }))
    return 0


# ---------------------------------------------------------------------------
# per-family subset selection (engage / audit)
# ---------------------------------------------------------------------------

def family_subset(cases):
    def rank(c):
        return (
            0 if (c["kind"], c["strictness"], c["variant"]) == ("INNER", "ALL", "std") else
            1 if (c["kind"], c["strictness"]) == ("LEFT", "ALL") else 2,
            c["id"],
        )
    by_family = {}
    for c in cases:
        f = c["family"]
        if f not in by_family or rank(c) < rank(by_family[f]):
            by_family[f] = c
    return [by_family[f] for f in sorted(by_family)]


def tagged_chk_query(case, log_comment):
    q = case["verdict_chk"]
    suffix = "FORMAT TSV"
    assert q.endswith(suffix)
    return f"{q[: -len(suffix)]}, log_comment = '{log_comment}', log_queries = 1 {suffix}"


def query_log_events(client_argv, log_comment, events):
    sel = ", ".join(f"ProfileEvents['{e}']" for e in events)
    q = (f"SELECT {sel} FROM system.query_log WHERE type = 'QueryFinish' "
         f"AND log_comment = '{log_comment}' AND event_date >= yesterday() "
         f"ORDER BY event_time_microseconds DESC LIMIT 1 FORMAT TSV")
    rc, out, err = run_client(client_argv, query=q)
    if rc != 0 or not out.strip():
        return None, err
    return [int(x) for x in out.strip().split("\t")], None


def run_subset_tagged(cases, client_argv, prefix):
    """Run setup + tagged chk query for the per-family subset; returns
    {family: (case, log_comment, error_or_None)}."""
    out = {}
    done_setup = set()
    for c in family_subset(cases):
        script = "\n".join(c["ddl"] + c["fill"])
        if script not in done_setup:
            rc, _, err = run_client(client_argv, query=script)
            if rc != 0:
                out[c["family"]] = (c, None, f"setup failed: {err[-1000:]}")
                continue
            done_setup.add(script)
        lc = f"{prefix}_{c['id']}"
        rc, _, err = run_client(client_argv, query=tagged_chk_query(c, lc))
        out[c["family"]] = (c, lc, None if rc == 0 else err[-1000:])
    run_client(client_argv, query="SYSTEM FLUSH LOGS")
    time.sleep(2)  # query_log flush_interval_milliseconds=1000 safety margin
    return out


def cmd_engage(args):
    cases = load_cases(args.cases, args.limit)
    client_argv = shlex.split(args.client)
    subset = run_subset_tagged(cases, client_argv, "parity_amac")
    assert_events = list(parity_gen.AMAC_ASSERT_POSITIVE_EVENTS)
    all_events = list(parity_gen.AMAC_ENGAGEMENT_EVENTS) + list(parity_gen.SHARED_PROFILE_EVENTS)
    engaged, failures, report = 0, [], {}
    for family, (case, lc, err) in sorted(subset.items()):
        if err is not None:
            failures.append(family)
            report[family] = {"error": err}
            continue
        vals, qerr = query_log_events(client_argv, lc, all_events)
        if vals is None:
            failures.append(family)
            report[family] = {"error": f"query_log lookup failed: {qerr}"}
            continue
        ev = dict(zip(all_events, vals))
        report[family] = ev
        if all(ev[e] > 0 for e in assert_events):
            engaged += 1
        else:
            failures.append(family)
    for family in failures:
        case = subset[family][0]
        detail = json.dumps(report[family], sort_keys=True, indent=1)
        write_divergence(args.logdir, case, "amac-force-not-engaged",
                         f"asserted events {assert_events} not all > 0 under "
                         f"{parity_gen.AMAC_ENV_VAR}=force:\n{detail}",
                         "candidate(force)", "candidate(force)", args.client, args.client)
    print("ENGAGE " + json.dumps({"engaged": engaged, "total": len(subset),
                                  "failures": sorted(failures), "report": report}, sort_keys=True))
    return 0


def cmd_audit(args):
    cases = load_cases(args.cases, args.limit)
    client_argv = shlex.split(args.client)
    subset = run_subset_tagged(cases, client_argv, f"parity_audit_{args.label}")
    events = list(parity_gen.SHARED_PROFILE_EVENTS)
    report = {}
    for family, (case, lc, err) in sorted(subset.items()):
        if err is not None:
            report[family] = {"error": err}
            continue
        vals, qerr = query_log_events(client_argv, lc, events)
        report[family] = {"error": qerr} if vals is None else dict(zip(events, vals))
    probe_ev = "ConcurrentHashJoinProbeMicroseconds"
    engaged = sorted(f for f, r in report.items() if r.get(probe_ev, 0) > 0)
    print("AUDIT " + json.dumps({"label": args.label, "parallel_hash_engaged_families": engaged,
                                 "total_families": len(report), "report": report}, sort_keys=True))
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="mode", required=True)

    pr = sub.add_parser("run")
    pr.add_argument("--cases", required=True)
    pr.add_argument("--client", required=True)
    pr.add_argument("--outdir", required=True)
    pr.add_argument("--limit", type=int, default=None)
    pr.add_argument("--chunk", type=int, default=32)
    pr.set_defaults(func=cmd_run)

    pc = sub.add_parser("compare")
    pc.add_argument("--cases", required=True)
    pc.add_argument("--base-dir", required=True)
    pc.add_argument("--cand-dir", required=True)
    pc.add_argument("--logdir", required=True)
    pc.add_argument("--max-matched-errors", type=int, default=4,
                    help="matched-error budget: more than this fails the gate (default 4)")
    pc.add_argument("--max-matched-error-pct", type=float, default=2.0,
                    help="matched-error budget as %% of cases (default 2)")
    pc.add_argument("--min-compared-pct", type=float, default=90.0,
                    help="fail unless at least this %% of cases produced real comparisons (default 90)")
    pc.add_argument("--acknowledge-matched-errors", type=int, default=None,
                    help="explicit acknowledgment: allow up to N matched-errors, overriding the budget")
    pc.add_argument("--base-label", default="baseline")
    pc.add_argument("--cand-label", default="candidate")
    pc.add_argument("--base-client", default="<baseline_bin> client --host 127.0.0.1 --port 19101")
    pc.add_argument("--cand-client", default="<candidate_bin> client --host 127.0.0.1 --port 19201")
    pc.add_argument("--limit", type=int, default=None)
    pc.set_defaults(func=cmd_compare)

    pe = sub.add_parser("engage")
    pe.add_argument("--cases", required=True)
    pe.add_argument("--client", required=True)
    pe.add_argument("--logdir", required=True)
    pe.add_argument("--limit", type=int, default=None)
    pe.set_defaults(func=cmd_engage)

    pa = sub.add_parser("audit")
    pa.add_argument("--cases", required=True)
    pa.add_argument("--client", required=True)
    pa.add_argument("--label", required=True)
    pa.add_argument("--limit", type=int, default=None)
    pa.set_defaults(func=cmd_audit)

    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
