#!/usr/bin/env python3
"""Count instruction classes over explicit address ranges of a labeled disassembly."""
import re
import sys

LOAD = re.compile(r'^(ldr|ldrb|ldrh|ldrsw|ldrsb|ldrsh|ldp|ldur|ldurb|ldarb|ldar|ldapr|ldaxr|ldxr)\b')
STORE = re.compile(r'^(str|strb|strh|stp|stur|sturb|stlr|stlrb|stxr|stlxr)\b')
BRANCH = re.compile(r'^(b|b\.\w+|cbz|cbnz|tbz|tbnz|br|ret)\b')
CALL = re.compile(r'^(bl|blr)\b')
STACKREF = re.compile(r'\[(sp|x29)\b')


def parse(path):
    out = []
    for line in open(path):
        m = re.match(r'^\s*([0-9a-f]+):\s+(.*)$', line)
        if not m:
            continue
        addr = int(m.group(1), 16)
        text = m.group(2).replace('\t', ' ').strip()
        text = re.sub(r'\s+', ' ', text)
        if text.startswith('nop'):
            continue
        out.append((addr, text))
    return out


def tally(insns, ranges, label):
    sel = [(a, t) for (a, t) in insns if any(lo <= a <= hi for lo, hi in ranges)]
    n = len(sel)
    loads = sum(1 for _, t in sel if LOAD.match(t))
    stores = sum(1 for _, t in sel if STORE.match(t))
    branches = sum(1 for _, t in sel if BRANCH.match(t) and not CALL.match(t))
    calls = sum(1 for _, t in sel if CALL.match(t))
    stack_st = sum(1 for _, t in sel if STORE.match(t) and STACKREF.search(t))
    stack_ld = sum(1 for _, t in sel if LOAD.match(t) and STACKREF.search(t))
    print(f'{label}: insns={n} loads={loads} stores={stores} branches={branches} '
          f'calls={calls} stack_stores={stack_st} stack_reloads={stack_ld}')
    return dict(n=n, loads=loads, stores=stores, branches=branches, calls=calls,
                stack_st=stack_st, stack_ld=stack_ld)


base = parse(sys.argv[1] if len(sys.argv) > 1 else 'base_u64.labeled.txt')
uhj = parse('uhj_u64.labeled.txt')
oi = parse('oi_u64.labeled.txt')

print('--- BASELINE (hash) fillColumns, per non-joined cell, emit path ---')
b_all = [(0x142a1fe0, 0x142a2020), (0x142a2024, 0x142a2038), (0x142a203c, 0x142a204c),
         (0x142a2050, 0x142a2078)]
tally(base, [(0x142a1fe8, 0x142a2020)], '  offset+flag sub-path')
tally(base, [(0x142a1fe8, 0x142a2014)], '  offset only (offsetInternal inlined)')
tally(base, b_all, '  TOTAL per cell')

print('--- UNIFIED fillColumns, per non-joined cell, emit path ---')
u_all = [(0x165e8b24, 0x165e8b28), (0x165e8b34, 0x165e8ba0), (0x165e8ba4, 0x165e8bac),
         (0x165e8bbc, 0x165e8bd4), (0x165e8bf8, 0x165e8c3c), (0x165e8c68, 0x165e8c70)]
tally(uhj, [(0x165e8b48, 0x165e8b88)], '  offset+flag sub-path (caller part)')
tally(uhj, [(0x165e8b48, 0x165e8b64)], '  bucket routing + call')
tally(uhj, u_all, '  caller TOTAL per cell')

print('--- UNIFIED callee RuntimeStorage::offsetInternal, fast path (once_flag set) ---')
oi_fast = [(0x165f0cc0, 0x165f0d28), (0x165f0d48, 0x165f0d7c)]
tally(oi, oi_fast, '  callee fast path')
tally(oi, [(0x165f0cf8, 0x165f0d28)], '  ...of which call_once closure setup + flag check')

print('--- COMBINED unified per cell (caller + callee) ---')
c = tally(uhj, u_all, '  caller')
d = tally(oi, oi_fast, '  callee')
print('  COMBINED: insns=%d loads=%d stores=%d branches=%d calls=%d stack_stores=%d stack_reloads=%d'
      % (c['n'] + d['n'], c['loads'] + d['loads'], c['stores'] + d['stores'],
         c['branches'] + d['branches'], c['calls'] + d['calls'],
         c['stack_st'] + d['stack_st'], c['stack_ld'] + d['stack_ld']))
