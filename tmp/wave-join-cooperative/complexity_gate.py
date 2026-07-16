#!/usr/bin/env python3
"""Gate 8 — structural simplicity (FROZEN before any production C++ edit).

REGISTER AMENDMENT v2 (2026-07-16, USER SIGN-OFF — recorded in WORKLOG.md):
Gate A's original criterion (candidate probe-control NCNB <= 75% of baseline)
is structurally unreachable for the sealed cooperative design: the baseline
outsourced its probe data plane to build-shared machinery (excluded as
byte-identical shared code) while the contract requires that data plane to run
as claimable jobs on executor lanes (~150 candidate-only lines). With the
user's sign-off Gate A becomes: candidate probe-control NCNB <= 115% of
baseline; and Gate B is TIGHTENED from "fewer" to "at least 25% fewer"
synchronization-primitive declarations. Gates C and D are unchanged. The
measurement definitions (NCNB, unit partition, shared units, declarations)
are unchanged.

Usage:
  complexity_gate.py --baseline tmp/wave-join-cooperative/RadixHashJoin.before.cpp \
                     --candidate src/Interpreters/RadixHashJoin/RadixHashJoin.cpp

Frozen metric definitions (any later edit is a register amendment):

* NCNB(text): non-comment, non-blank line count. Block and line comments are
  stripped with a small string-literal-aware scanner; a line counts if any
  code characters remain.

* Unit partition: both files are split into top-level braced units (functions,
  classes, structs, enums) by brace matching that is aware of comments and
  string/char literals. Namespace braces are transparent (their contents are
  scanned, the namespace line itself is preamble). A unit's NAME is the last
  identifier before '(' for functions (qualified like RadixHashJoin::joinBlock
  when present) or after struct/class/enum. A unit in the candidate is SHARED
  iff a unit with the same name AND byte-identical normalized body (NCNB lines
  joined) exists in the baseline. Shared units are the untouched build-side
  machinery; everything else — changed, new, or removed units plus all
  preamble (includes, usings, constants, namespace scaffolding) — counts as
  PROBE-CONTROL for its file. This makes any counting-boundary trick
  (renaming, reshuffling, moving code between units) count AGAINST the
  candidate, never for it.

* Gate A (size):    probe_control(candidate) <= 1.15 * probe_control(baseline)
                    [amendment v2; originally 0.75 — see header]
* Gate B (sync):    synchronization-primitive DECLARATIONS in probe-control
                    code: candidate <= 0.75 * baseline (at least 25% fewer)
                    [amendment v2; originally merely fewer]. A declaration is a line
                    declaring an object/member/local of: std::atomic*,
                    std::mutex, std::shared_mutex, SharedMutex, std::
                    condition_variable*, ConcurrentBoundedQueue, std::latch,
                    std::barrier, std::counting_semaphore/binary_semaphore,
                    std::promise, std::future, OnceFlag/std::once_flag.
* Gate C (one core wave engine):
                    - at most ONE 'enum class' declaration in the candidate's
                      probe-control code (the single drain-phase machine), and
                    - the candidate's IBlocksStream subclass (delayed-blocks
                      adapter) is thin: <= 40 NCNB lines, and
                    - exactly one IJoinResult subclass in probe-control code.
* Gate D (no moved code):
                    - NCNB(src/Interpreters/RadixHashJoin/RadixHashJoin.h)
                      <= 62 + 10 (frozen baseline 62), and
                    - src/Interpreters/RadixHashJoin/ contains exactly
                      {RadixHashJoin.cpp, RadixHashJoin.h, WaveJoinProbe.tla}.

Exit 0 only if every gate passes. The independent verifier audits these
definitions and recomputes the metrics.
"""

import argparse
import os
import re
import sys

HEADER_PATH = "src/Interpreters/RadixHashJoin/RadixHashJoin.h"
HEADER_BASELINE_NCNB = 62
HEADER_SLACK = 10
DIR_PATH = "src/Interpreters/RadixHashJoin"
DIR_EXPECTED = {"RadixHashJoin.cpp", "RadixHashJoin.h", "WaveJoinProbe.tla"}

SYNC_DECL = re.compile(
    r"^\s*(?:mutable\s+|static\s+|const\s+)*"
    r"(?:std::)?(?:atomic(?:_bool|_int|_size_t)?\s*<|atomic_flag\b|mutex\b|shared_mutex\b|"
    r"condition_variable(?:_any)?\b|latch\b|barrier\b|counting_semaphore\b|"
    r"binary_semaphore\b|promise\s*<|future\s*<|once_flag\b)"
    r"|^\s*(?:mutable\s+)?(?:DB::)?(?:SharedMutex|ConcurrentBoundedQueue)\b"
)


def strip_comments(src: str) -> str:
    """Remove // and /* */ comments, string-literal aware."""
    out = []
    i, n = 0, len(src)
    while i < n:
        c = src[i]
        if c == '"' or c == "'":
            quote = c
            out.append(c)
            i += 1
            while i < n:
                out.append(src[i])
                if src[i] == "\\":
                    if i + 1 < n:
                        out.append(src[i + 1])
                        i += 2
                        continue
                elif src[i] == quote:
                    i += 1
                    break
                i += 1
            continue
        if c == "/" and i + 1 < n and src[i + 1] == "/":
            while i < n and src[i] != "\n":
                i += 1
            continue
        if c == "/" and i + 1 < n and src[i + 1] == "*":
            i += 2
            while i + 1 < n and not (src[i] == "*" and src[i + 1] == "/"):
                if src[i] == "\n":
                    out.append("\n")
                i += 1
            i += 2
            continue
        out.append(c)
        i += 1
    return "".join(out)


def ncnb_lines(text: str):
    return [line.strip() for line in strip_comments(text).split("\n") if line.strip()]


def parse_units(src: str):
    """Split into (name, normalized_body, ncnb_count) top-level units.

    Walks the comment-stripped source tracking brace depth; namespace braces
    are transparent. A unit starts at depth 0/transparent scope when a line
    opens a brace (or a signature accumulates until its opening brace) and
    ends when its brace closes. Everything outside units is preamble.
    """
    text = strip_comments(src)
    lines = text.split("\n")

    units = []
    preamble_lines = []

    depth = 0
    ns_depths = []  # depths at which a namespace opened (transparent)
    unit_lines = []
    unit_open_depth = None
    pending_sig = []

    def flush_unit():
        nonlocal unit_lines
        body = [line.strip() for line in unit_lines if line.strip()]
        name = unit_name(body)
        units.append((name, "\n".join(body), len(body)))
        unit_lines = []

    def unit_name(body):
        head = " ".join(body[:8])
        m = re.search(r"\b(?:struct|class|enum(?:\s+class)?)\s+([A-Za-z_]\w*)", head)
        if m:
            return m.group(1)
        m = re.search(r"([A-Za-z_][\w:]*)\s*\(", head)
        if m:
            return m.group(1)
        return body[0][:40] if body else "<anon>"

    for raw in lines:
        line = raw.rstrip()
        stripped = line.strip()
        opens = line.count("{")
        closes = line.count("}")

        in_unit = unit_open_depth is not None
        top_level = depth == len(ns_depths)  # only transparent namespace scopes open

        if not in_unit and top_level:
            ns_re = r"^\s*(inline\s+)?namespace\b[^;{]*$"
            is_ns_inline = bool(re.match(r"^\s*(inline\s+)?namespace\b[^;]*\{", line))
            # Allman style: 'namespace X' line followed by a lone '{' line.
            is_ns_allman = (
                stripped == "{"
                and len(pending_sig) == 1
                and bool(re.match(ns_re, pending_sig[0]))
            )
            if is_ns_inline or is_ns_allman:
                if is_ns_allman:
                    preamble_lines.extend(pending_sig)
                    pending_sig = []
                preamble_lines.append(stripped)
                depth += opens - closes
                ns_depths.append(depth)
                continue
            if stripped in ("}", "};") and closes and ns_depths and depth - closes < ns_depths[-1]:
                preamble_lines.append(stripped)
                depth += opens - closes
                ns_depths.pop()
                continue
            if opens:
                unit_lines = pending_sig + [stripped]
                pending_sig = []
                unit_open_depth = depth
                depth += opens - closes
                if depth == unit_open_depth and (closes or stripped.endswith("};") or stripped.endswith("}")):
                    unit_open_depth = None
                    flush_unit()
                continue
            # signature accumulation: a non-empty line that is not a complete
            # statement (no trailing ';') may be a multi-line signature head
            if stripped and not stripped.endswith(";"):
                pending_sig.append(stripped)
            else:
                if pending_sig:
                    preamble_lines.extend(pending_sig)
                    pending_sig = []
                if stripped:
                    preamble_lines.append(stripped)
            continue

        if in_unit:
            if stripped:
                unit_lines.append(stripped)
            depth += opens - closes
            if depth <= unit_open_depth:
                unit_open_depth = None
                flush_unit()
            continue

        # inside a namespace body handled by top_level logic above; anything
        # else (shouldn't happen) is preamble
        if stripped:
            preamble_lines.append(stripped)
        depth += opens - closes

    if pending_sig:
        preamble_lines.extend(pending_sig)
    if unit_lines:
        flush_unit()

    return units, preamble_lines


def probe_control(units, preamble, shared_keys):
    """NCNB lines not belonging to shared identical units."""
    total = len(preamble)
    texts = []
    for name, body, count in units:
        if (name, body) in shared_keys:
            continue
        total += count
        texts.append(body)
    return total, "\n".join(texts) + "\n" + "\n".join(preamble)


def count_sync_decls(text: str) -> int:
    return sum(1 for line in text.split("\n") if SYNC_DECL.search(line))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--candidate", required=True)
    args = ap.parse_args()

    base_src = open(args.baseline).read()
    cand_src = open(args.candidate).read()

    base_units, base_pre = parse_units(base_src)
    cand_units, cand_pre = parse_units(cand_src)

    base_keys = {(n, b) for n, b, _ in base_units}
    cand_keys = {(n, b) for n, b, _ in cand_units}
    shared = base_keys & cand_keys

    base_probe, base_text = probe_control(base_units, base_pre, shared)
    cand_probe, cand_text = probe_control(cand_units, cand_pre, shared)

    failures = []

    # Gate A (amendment v2: no growth beyond baseline + 15%)
    limit = 1.15 * base_probe
    print(f"Gate A (size, amended): probe-control NCNB baseline={base_probe} candidate={cand_probe} "
          f"(limit {limit:.0f}, delta {(cand_probe - base_probe) / base_probe:+.1%})")
    if not cand_probe <= limit:
        failures.append("Gate A (amended): candidate probe-control lines exceed 115% of baseline")

    # Gate B (amendment v2: at least 25% fewer)
    base_sync = count_sync_decls(base_text)
    cand_sync = count_sync_decls(cand_text)
    print(f"Gate B (sync decls, amended): baseline={base_sync} candidate={cand_sync} "
          f"(limit {0.75 * base_sync:.1f})")
    if not cand_sync <= 0.75 * base_sync:
        failures.append("Gate B (amended): candidate does not have at least 25% fewer sync-primitive declarations")

    # Gate C
    enums = len(re.findall(r"\benum\s+class\b", cand_text))
    print(f"Gate C (one engine): 'enum class' count in probe-control = {enums} (<= 1 required)")
    if enums > 1:
        failures.append("Gate C: more than one enum class (suspected second machine)")
    stream_units = [(n, b, c) for n, b, c in cand_units
                    if re.search(r":\s*(?:public\s+)?IBlocksStream\b", b)]
    for n, b, c in stream_units:
        print(f"Gate C (thin delayed adapter): {n} = {c} NCNB lines (<= 40 required)")
        if c > 40:
            failures.append(f"Gate C: delayed-blocks adapter {n} is {c} NCNB lines (> 40)")
    result_units = [(n, b, c) for n, b, c in cand_units
                    if re.search(r":\s*(?:public\s+)?IJoinResult\b", b)]
    print(f"Gate C (one result type): IJoinResult subclasses = {len(result_units)}")
    if len(result_units) != 1:
        failures.append("Gate C: expected exactly one IJoinResult subclass in probe-control code")

    # Gate D
    header_ncnb = len(ncnb_lines(open(HEADER_PATH).read()))
    print(f"Gate D (header): NCNB={header_ncnb} (baseline {HEADER_BASELINE_NCNB}, "
          f"limit {HEADER_BASELINE_NCNB + HEADER_SLACK})")
    if header_ncnb > HEADER_BASELINE_NCNB + HEADER_SLACK:
        failures.append("Gate D: header grew beyond the frozen slack (moved-code suspicion)")
    actual_files = {f for f in os.listdir(DIR_PATH) if not f.startswith(".")}
    print(f"Gate D (files): {sorted(actual_files)}")
    if actual_files != DIR_EXPECTED:
        failures.append(f"Gate D: unexpected files in {DIR_PATH}: {sorted(actual_files ^ DIR_EXPECTED)}")

    print()
    if failures:
        print("STRUCTURAL GATE: FAIL")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    print("STRUCTURAL GATE: PASS")
    sys.exit(0)


if __name__ == "__main__":
    main()
