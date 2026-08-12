#!/usr/bin/env python3
"""Instruction-level attribution inside RowRefList::insert, both arms side by side.

perf annotate refuses this recording (multi-event group), so the histogram is built from
raw `perf script` sample offsets and joined against objdump output.
"""
from __future__ import annotations

import collections
import re
import subprocess
import sys
from pathlib import Path

D = Path("/mnt/data/uhj_versions_bench/deep")
ASM = Path("/mnt/data/uhj_versions_bench/asm")
SYM = "_ZN2DB10RowRefList6insertEmRNS_5ArenaE"
BIN = {"baseline": "/mnt/data/uhj_versions_bench/bin/clickhouse-baseline",
       "uhj": "/mnt/data/uhj_versions_bench/bin/clickhouse-uhj"}
EVENTS = ["cpu_cycles", "l1d_cache_refill", "ll_cache_miss_rd", "dtlb_walk"]


def disasm(arm: str):
    """offset -> instruction text, from objdump of the symbol."""
    out = subprocess.run(
        ["llvm-objdump-22", "-d", f"--disassemble-symbols={SYM}", "--no-show-raw-insn", BIN[arm]],
        capture_output=True, text=True).stdout
    insns = {}
    base = None
    for line in out.splitlines():
        m = re.match(r"\s*([0-9a-f]+):\s+(.*)", line)
        if not m:
            continue
        addr = int(m.group(1), 16)
        if base is None:
            base = addr
        insns[addr - base] = m.group(2).strip()
    return insns


def static_addr(arm: str) -> int:
    out = subprocess.run(["nm", "--defined-only", BIN[arm]], capture_output=True, text=True).stdout
    for line in out.splitlines():
        parts = line.split()
        if len(parts) == 3 and parts[2] == SYM:
            return int(parts[0], 16)
    raise SystemExit(f"symbol not found in {arm}")


def samples(arm: str):
    """event -> Counter(offset within the function -> samples).

    This perf build has no `symoff` field, so offsets come from the raw IPs. The load bias
    is page-aligned, so the function's runtime start is the largest address <= the lowest
    sampled IP that is congruent to the static address modulo the page size.
    """
    data = D / f"{arm}_q64.deep.perf.data"
    out = subprocess.run(
        ["perf", "script", "-i", str(data), "--no-demangle",
         f"--symbols={SYM}", "-F", "event,ip,sym"],
        capture_output=True, text=True).stdout
    raw = []
    for line in out.splitlines():
        m = re.match(r"\s*(\S+):\s+([0-9a-f]+)\s+", line)
        if m:
            raw.append((m.group(1).rstrip(":"), int(m.group(2), 16)))
    if not raw:
        return {}
    sa = static_addr(arm)
    min_ip = min(ip for _, ip in raw)
    start = min_ip - ((min_ip - sa) % 4096)
    per = collections.defaultdict(collections.Counter)
    for ev, ip in raw:
        per[ev][ip - start] += 1
    return per


def main():
    asm = {a: disasm(a) for a in BIN}
    smp = {a: samples(a) for a in BIN}

    # Sanity: the two disassemblies must agree instruction for instruction.
    off_b, off_u = sorted(asm["baseline"]), sorted(asm["uhj"])
    same_shape = off_b == off_u and all(
        re.sub(r"0x[0-9a-f]+", "", asm["baseline"][o]) == re.sub(r"0x[0-9a-f]+", "", asm["uhj"][o])
        for o in off_b)
    print(f"disassembly identical (modulo branch addresses): {same_shape}; "
          f"{len(off_b)} instructions\n")

    for ev in EVENTS:
        cb, cu = smp["baseline"].get(ev), smp["uhj"].get(ev)
        if not cb or not cu:
            continue
        tb, tu = sum(cb.values()), sum(cu.values())
        print(f"===== {ev}: samples inside RowRefList::insert  baseline={tb}  uhj={tu} =====")
        print(f"{'off':>6}  {'base%':>7} {'uhj%':>7}  {'delta':>7}  instruction")
        rows = []
        for o in off_b:
            pb = 100.0 * cb.get(o, 0) / tb if tb else 0
            pu = 100.0 * cu.get(o, 0) / tu if tu else 0
            if max(pb, pu) >= 1.0:
                rows.append((o, pb, pu, pu - pb, asm["baseline"][o]))
        for o, pb, pu, d, ins in sorted(rows, key=lambda r: -max(r[1], r[2]))[:12]:
            print(f"0x{o:04x}  {pb:6.2f}% {pu:6.2f}%  {d:+6.2f}%  {ins[:60]}")
        print()


if __name__ == "__main__":
    main()
