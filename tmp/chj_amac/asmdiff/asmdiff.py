#!/usr/bin/env python3
"""Before/after codegen comparison for the cursor-layer commit (PREREG-005).

For each binary, finds the key64/RowRefList sequential insert and probe-loop
symbols, disassembles their ranges with llvm-objdump, and compares opcode
histograms. Expectation: the only systematic differences are the collision
walk advance (mask AND -> increment + compare-against-bufSize) and grower
field layout; anything else (new spills, extra per-row loads) is a finding.
"""
import subprocess
import sys
import re
from collections import Counter

NM = "/usr/local/bin/llvm-nm-22"
OBJDUMP = "/usr/local/bin/llvm-objdump-22"

INSERT_PAT = re.compile(
    r"insertFromBlockImplTypeCase<.*HashMapCell<unsigned long, DB::RowRefList, HashCRC32<unsigned long>.*ColumnVector<unsigned long>")
PROBE_PAT = re.compile(
    r"joinRightColumns<.*HashMapCell<unsigned long, DB::RowRefList, HashCRC32<unsigned long>")


def symbols(binary):
    out = subprocess.run([NM, "--defined-only", "--print-size", "--demangle", binary],
                         capture_output=True, text=True, check=True).stdout
    syms = []
    for line in out.splitlines():
        parts = line.split(" ", 3)
        if len(parts) < 4:
            continue
        addr, size, _kind, name = parts
        if "insertFromBlockImplTypeCase" in name or "joinRightColumns" in name:
            try:
                syms.append((int(addr, 16), int(size, 16), name))
            except ValueError:
                continue
    return syms


def pick(syms, pat):
    matches = [s for s in syms if pat.search(s[2])]
    if not matches:
        return None
    return max(matches, key=lambda s: s[1])  # largest carries the steady loop


def histogram(binary, addr, size):
    out = subprocess.run(
        [OBJDUMP, "-d", "--no-leading-addr", "--no-show-raw-insn",
         f"--start-address={hex(addr)}", f"--stop-address={hex(addr + size)}", binary],
        capture_output=True, text=True, check=True).stdout
    ops = Counter()
    total = 0
    for line in out.splitlines():
        line = line.strip()
        m = re.match(r"^([a-z][a-z0-9.]*)\s", line)
        if m and not line.endswith(":"):
            ops[m.group(1)] += 1
            total += 1
    return ops, total


def classify(ops):
    loads = sum(v for k, v in ops.items() if k.startswith(("ldr", "ldp", "ldu")))
    stores = sum(v for k, v in ops.items() if k.startswith(("str", "stp", "stu")))
    branches = sum(v for k, v in ops.items() if k.startswith(("b.", "b", "cbz", "cbnz", "tbz", "tbnz", "bl", "ret", "br")))
    prefetch = sum(v for k, v in ops.items() if k.startswith("prfm"))
    return loads, stores, branches, prefetch


def main():
    before, after = sys.argv[1], sys.argv[2]
    for label, pat in (("INSERT key64/RowRefList", INSERT_PAT), ("PROBE key64/RowRefList", PROBE_PAT)):
        rows = {}
        for tag, binary in (("before", before), ("after", after)):
            sym = pick(symbols(binary), pat)
            if sym is None:
                print(f"{label} [{tag}]: NO SYMBOL MATCH")
                rows[tag] = None
                continue
            addr, size, name = sym
            ops, total = histogram(binary, addr, size)
            loads, stores, branches, prefetch = classify(ops)
            rows[tag] = (ops, total, size, name)
            print(f"{label} [{tag}]: size={size}B insns={total} loads={loads} stores={stores} "
                  f"branches={branches} prfm={prefetch}")
            print(f"   {name[:160]}")
        if rows.get("before") and rows.get("after"):
            b_ops, b_total, _, _ = rows["before"]
            a_ops, a_total, _, _ = rows["after"]
            delta = Counter(a_ops)
            delta.subtract(b_ops)
            changed = {k: v for k, v in delta.items() if v != 0}
            print(f"{label} opcode delta (after - before), insns {b_total} -> {a_total}:")
            for k, v in sorted(changed.items(), key=lambda kv: -abs(kv[1])):
                print(f"   {k:12s} {v:+d}")
        print()


if __name__ == "__main__":
    main()
