#!/usr/bin/env python3
"""Reusable codegen analysis: disassemble, count, measure dependent-load depth, and
run `llvm-mca`.

Three commands:

    codegen.py dis   --binary B --symbol RE [--index N]      -> .asm artefact
    codegen.py count --asm F [--range LO:HI ...]             -> instruction classes
    codegen.py mca   --asm F [--range LO:HI ...]             -> cycles / IPC / bottleneck

The instruction-class regexes are copied verbatim from the prior mission's
`perf/codegen/logs/count.py`, so counts here are comparable with the numbers already in
`codegen/N1_nonjoined_scan.md` rather than counted by a different rule.

Two things this adds over the inherited counter:

**Dependent-load chain depth.** The mission's loop schema asks for it and an
instruction count cannot supply it. Computed as the longest chain of loads within the
range where each load's *address* register was defined by an earlier load's
destination -- the pointer-chase depth, which is what makes a loop latency-bound
rather than throughput-bound and is exactly what `llvm-mca` prices and a raw count
cannot.

**A real objdump -> mca extraction step.** The trap recorded in the prior mission's
WORKLOG E13 is that `llvm-mca` will not eat `llvm-objdump` output: the file header,
the `<symbol>:` label lines and `--symbolize-operands`' `<L1>` branch targets all fail
to assemble, and stripping the targets naively leaves `b.hs` with no operand. Here the
`<Lnn>` targets are rewritten to real local labels and emitted as `.Lnn:` definitions
where they fall inside the analysed range; a branch to a label outside the range is
retargeted to the end of the block, which keeps it assembling while changing nothing
about the straight-line resource usage mca prices. Every such rewrite is reported, so
the caveat travels with the number.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BIN = os.path.join(HERE, "..", "perf", "bin")
OBJDUMP = os.path.join(BIN, "llvm-objdump")
NM = os.path.join(BIN, "llvm-nm")
MCA = os.path.join(BIN, "llvm-mca")
MCPU = "neoverse-v2"
TRIPLE = "aarch64-unknown-linux-gnu"

# --- verbatim from perf/codegen/logs/count.py, so counts stay comparable ---------
LOAD = re.compile(r'^(ldr|ldrb|ldrh|ldrsw|ldrsb|ldrsh|ldp|ldur|ldurb|ldarb|ldar|ldapr|ldaxr|ldxr)\b')
STORE = re.compile(r'^(str|strb|strh|stp|stur|sturb|stlr|stlrb|stxr|stlxr)\b')
BRANCH = re.compile(r'^(b|b\.[a-z]+|cbz|cbnz|tbz|tbnz|br|ret)\b')
CALL = re.compile(r'^(bl|blr)\b')
SPILL_RE = re.compile(r'\[(sp|x29)[,\]]')

REG = re.compile(r'\b([wx](?:[0-9]|[12][0-9]|30)|sp|xzr|wzr)\b')


def sh(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stderr[-3000:], file=sys.stderr)
        raise SystemExit(f"command failed: {' '.join(cmd[:4])}...")
    return r.stdout


# ---------------------------------------------------------------- disassemble
def find_symbols(binary, pattern):
    out = sh([NM, "--defined-only", "--demangle", binary])
    rx = re.compile(pattern)
    hits = []
    for line in out.splitlines():
        parts = line.split(" ", 2)
        if len(parts) != 3 or parts[1].upper() not in ("T", "W"):
            continue
        if rx.search(parts[2]):
            hits.append((int(parts[0], 16), parts[2]))
    return hits


def disassemble(binary, addr, size, out_path):
    txt = sh([OBJDUMP, "-d", "--symbolize-operands", "--no-show-raw-insn",
              f"--start-address={hex(addr)}", f"--stop-address={hex(addr + size)}",
              binary])
    with open(out_path, "w") as fh:
        fh.write(txt)
    return txt


def parse_asm(path):
    """address -> instruction text, plus label lines."""
    insns, labels = {}, {}
    last = None
    for line in open(path):
        line = line.rstrip("\n")
        m = re.match(r'^\s*<(L\d+)>:\s*$', line)
        if m:
            labels[m.group(1)] = None       # resolved to the next instruction's address
            last = m.group(1)
            continue
        m = re.match(r'^\s*([0-9a-f]+):\s+(.*)$', line)
        if not m:
            continue
        addr = int(m.group(1), 16)
        text = re.sub(r'\s*//.*$', '', m.group(2)).strip()
        text = re.sub(r'\s+', ' ', text)
        insns[addr] = text
        if last is not None:
            labels[last] = addr
            last = None
    return insns, labels


# ---------------------------------------------------------------------- count
def in_ranges(addr, ranges):
    return any(lo <= addr <= hi for lo, hi in ranges)


def selected(insns, ranges):
    keys = sorted(insns)
    if not ranges:
        return [(a, insns[a]) for a in keys]
    return [(a, insns[a]) for a in keys if in_ranges(a, ranges)]


def count(insns, ranges, label="block"):
    total = loads = stores = branches = calls = 0
    spill_st = spill_ld = nops = 0
    for _addr, t in selected(insns, ranges):
        op = t.split(" ")[0]
        if op == "nop":
            nops += 1
            continue
        total += 1
        is_spill = bool(SPILL_RE.search(t))
        if CALL.match(op):
            calls += 1
            branches += 1
        elif BRANCH.match(op):
            branches += 1
        elif LOAD.match(op):
            loads += 1
            if is_spill:
                spill_ld += 1
        elif STORE.match(op):
            stores += 1
            if is_spill:
                spill_st += 1
    return dict(label=label, insns=total, loads=loads, stores=stores, branches=branches,
                calls=calls, spill_st=spill_st, spill_ld=spill_ld, nops_excluded=nops,
                dep_load_depth=dep_load_depth(insns, ranges))


def dep_load_depth(insns, ranges):
    """Longest chain of loads whose address operand came from an earlier load.

    Linear scan with a per-register 'depth of the load that last defined it'. A load
    whose base register has depth d produces depth d+1. Conservative in both
    directions and stated as such: it does not follow control flow, and it treats any
    register appearing inside `[...]` as an address operand.
    """
    depth = {}
    best = 0
    for _addr, t in selected(insns, ranges):
        op = t.split(" ")[0]
        mem = re.search(r'\[([^\]]*)\]', t)
        addr_regs = set(REG.findall(mem.group(1))) if mem else set()
        if LOAD.match(op):
            base = max((depth.get(r, 0) for r in addr_regs), default=0)
            d = base + 1
            best = max(best, d)
            for dst in REG.findall(t.split("[")[0])[:2 if op in ("ldp", "ldnp") else 1]:
                depth[dst] = d
        else:
            # a non-load that consumes a loaded value propagates its depth
            src = max((depth.get(r, 0) for r in REG.findall(t)), default=0)
            dsts = REG.findall(t.split(",")[0]) if "," in t else []
            for dst in dsts[:1]:
                if not STORE.match(op) and not BRANCH.match(op):
                    depth[dst] = src
    return best


def show(d):
    print(f"{d['label']:<40} insns={d['insns']:>4} loads={d['loads']:>3} stores={d['stores']:>3} "
          f"branches={d['branches']:>3} calls={d['calls']:>2} spill(st/ld)={d['spill_st']}/{d['spill_ld']} "
          f"dep_load_depth={d['dep_load_depth']}  [nops excluded {d['nops_excluded']}]")


# ------------------------------------------------------------------------ mca
def to_mca_asm(insns, labels, ranges):
    """Rewrite a disassembled range into something the assembler accepts.

    Returns (text, notes). Notes record every rewrite so the caveat travels with the
    number rather than being lost.
    """
    sel = selected(insns, ranges)
    if not sel:
        return "", ["empty range"]
    addrs = {a for a, _ in sel}
    addr_to_label = {v: k for k, v in labels.items() if v is not None}
    notes = []
    lines = []
    dropped_calls = 0
    for a, t in sel:
        if a in addr_to_label:
            lines.append(f".{addr_to_label[a]}:")
        m = re.search(r'<(L\d+)>', t)
        if m:
            tgt = labels.get(m.group(1))
            if tgt is not None and tgt in addrs:
                t = t.replace(f"<{m.group(1)}>", f".{m.group(1)}")
            else:
                # branch leaves the analysed block: retarget to the block end so the
                # instruction still assembles. Changes nothing about resource usage.
                t = t.replace(f"<{m.group(1)}>", ".Lblockend")
                notes.append(f"branch at {a:#x} targets {m.group(1)} outside the range; "
                             f"retargeted to block end")
        # a bl to an absolute address does not assemble; drop the call and say so
        if CALL.match(t.split(" ")[0]):
            dropped_calls += 1
            notes.append(f"call at {a:#x} ({t}) dropped -- mca cannot model the callee")
            continue
        t = re.sub(r'<[^>]*>', '.Lblockend', t)
        lines.append("  " + t)
    lines.append(".Lblockend:")
    lines.append("  nop")
    if dropped_calls:
        notes.append(f"TOTAL {dropped_calls} call(s) dropped: the mca number is the cost "
                     f"of the loop body EXCLUDING the callee, and is therefore a LOWER "
                     f"bound on the real cost of this side")
    return "\n".join(lines) + "\n", notes


def run_mca(asm_text, iterations=100):
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".s", delete=False,
                                     dir=os.path.join(HERE, "tmp")) as fh:
        fh.write(asm_text)
        path = fh.name
    r = subprocess.run([MCA, f"-mcpu={MCPU}", f"-mtriple={TRIPLE}",
                        f"-iterations={iterations}", "-bottleneck-analysis", path],
                       capture_output=True, text=True)
    return r.stdout, r.stderr, path


def parse_mca(out, iterations=100):
    res = {}
    for key, rx in [("iterations", r"Iterations:\s+(\d+)"),
                    ("instructions", r"Instructions:\s+(\d+)"),
                    ("total_cycles", r"Total Cycles:\s+(\d+)"),
                    ("total_uops", r"Total uOps:\s+(\d+)"),
                    ("ipc", r"IPC:\s+([\d.]+)"),
                    ("block_rthroughput", r"Block RThroughput:\s+([\d.]+)"),
                    ("upc", r"uOps Per Cycle:\s+([\d.]+)")]:
        m = re.search(rx, out)
        if m:
            res[key] = float(m.group(1))
    if "total_cycles" in res:
        res["cycles_per_iteration"] = res["total_cycles"] / (res.get("iterations") or iterations)
    m = re.search(r"Cycles with backend pressure increase:\s+([\d.]+)%", out)
    if m:
        res["backend_pressure_pct"] = float(m.group(1))
    bott = re.findall(r"^\s{2}([A-Za-z0-9_.]+(?:,\s*[A-Za-z0-9_.]+)*)\s+\|?\s*\(([\d.]+)%\)",
                      out, re.M)
    if bott:
        res["bottlenecks"] = bott[:5]
    return res


# ----------------------------------------------------------------------- main
def parse_ranges(rs):
    out = []
    for r in rs or []:
        lo, hi = r.split(":")
        out.append((int(lo, 16), int(hi, 16)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["dis", "count", "mca", "syms"])
    ap.add_argument("--binary")
    ap.add_argument("--symbol")
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--size", type=lambda s: int(s, 0), default=0x4000)
    ap.add_argument("--asm")
    ap.add_argument("--out")
    ap.add_argument("--range", action="append")
    ap.add_argument("--iterations", type=int, default=100)
    ap.add_argument("--label", default="block")
    a = ap.parse_args()

    os.makedirs(os.path.join(HERE, "tmp"), exist_ok=True)

    if a.cmd == "syms":
        for i, (addr, name) in enumerate(find_symbols(a.binary, a.symbol)):
            print(f"[{i}] {addr:#x}  {name}")
        return 0

    if a.cmd == "dis":
        hits = find_symbols(a.binary, a.symbol)
        if not hits:
            raise SystemExit(f"no symbol matching {a.symbol}")
        addr, name = hits[a.index]
        out = a.out or os.path.join(HERE, "codegen", "logs", f"{a.label}.asm")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        disassemble(a.binary, addr, a.size, out)
        print(f"symbol : {name}")
        print(f"address: {addr:#x}  matches={len(hits)}")
        print(f"asm    : {out}")
        return 0

    insns, labels = parse_asm(a.asm)
    ranges = parse_ranges(a.range)

    if a.cmd == "count":
        show(count(insns, ranges, a.label))
        return 0

    asm_text, notes = to_mca_asm(insns, labels, ranges)
    out, err, path = run_mca(asm_text, a.iterations)
    if not out:
        print(err[-2000:], file=sys.stderr)
        raise SystemExit("llvm-mca produced no output; the extracted block did not assemble. "
                         f"Block kept at {path} for inspection.")
    res = parse_mca(out, a.iterations)
    print(f"{a.label}:")
    for k in ("instructions", "total_cycles", "cycles_per_iteration", "ipc",
              "block_rthroughput", "total_uops", "backend_pressure_pct"):
        if k in res:
            print(f"  {k:22s} {res[k]}")
    if "bottlenecks" in res:
        print(f"  bottlenecks            {res['bottlenecks']}")
    for n in notes:
        print(f"  NOTE: {n}")
    if a.out:
        with open(a.out, "w") as fh:
            fh.write(out)
        print(f"  full mca report -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
