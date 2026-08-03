#!/usr/bin/env python3
"""Count instruction classes over explicit address ranges of an emitted disassembly."""
import re
import sys

LOAD = re.compile(r'^(ldr|ldrb|ldrh|ldrsw|ldrsb|ldrsh|ldp|ldur|ldurb|ldarb|ldar|ldapr|ldaxr|ldxr)\b')
STORE = re.compile(r'^(str|strb|strh|stp|stur|sturb|stlr|stlrb|stxr|stlxr)\b')
BRANCH = re.compile(r'^(b|b\.[a-z]+|cbz|cbnz|tbz|tbnz|br|ret)\b')
CALL = re.compile(r'^(bl|blr)\b')
SPILL_RE = re.compile(r'\[(sp|x29)[,\]]')


def parse(path):
    insns = {}
    for line in open(path):
        m = re.match(r'^([0-9a-f]+):\s+(.*)$', line.rstrip('\n'))
        if not m:
            continue
        addr = int(m.group(1), 16)
        text = re.sub(r'\s*//.*$', '', m.group(2)).strip()
        text = re.sub(r'\s+', ' ', text)
        insns[addr] = text
    return insns


def classify(insns, ranges, label):
    total = loads = stores = branches = calls = 0
    spill_st = spill_ld = 0
    nops = 0
    for lo, hi in ranges:
        for addr in range(lo, hi + 4, 4):
            if addr not in insns:
                print(f"  !! missing addr {addr:#x} in {label}", file=sys.stderr)
                continue
            t = insns[addr]
            op = t.split(' ')[0]
            if op == 'nop':
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
    return dict(label=label, insns=total, loads=loads, stores=stores,
                branches=branches, calls=calls, spill_st=spill_st,
                spill_ld=spill_ld, nops_excluded=nops)


def show(d):
    print(f"{d['label']:<34} insns={d['insns']:>3}  loads={d['loads']:>2}  stores={d['stores']:>2} "
          f" branches={d['branches']:>2}  calls={d['calls']}  spill(st/ld)={d['spill_st']}/{d['spill_ld']} "
          f" [nops excluded: {d['nops_excluded']}]")


if __name__ == '__main__':
    D = '/mnt/ch/ClickHouse/tmp/uhj_parity/perf/codegen/logs'

    before = parse(f'{D}/before_unified.asm')
    after = parse(f'{D}/after_unified.asm')
    callee = parse(f'{D}/before_callee.asm')

    # Hot path = one non-joined cell, emitting (flag clear -> collect), advance to the next
    # non-empty cell in the same bucket. Ranges are the prior artifact's, mapped 1:1 by
    # structural correspondence onto the AFTER binary.
    before_caller = [
        (0x165e8b24, 0x165e8b28),   # L14 loop head
        (0x165e8b34, 0x165e8ba0),   # L15 .. collect  (contains offset computation)
        (0x165e8ba4, 0x165e8bac),   # post-collect iterator load
        (0x165e8bbc, 0x165e8bd4),   # L17 advance
        (0x165e8bf8, 0x165e8c3c),   # L19 bucket-exhaustion check
        (0x165e8c68, 0x165e8c70),   # stream filter (num_streams == 1)
    ]
    after_caller = [
        (0x165e8564, 0x165e8568),
        (0x165e8574, 0x165e85dc),   # L15 .. inlined offset .. flags bound check
        (0x165e861c, 0x165e863c),   # L17 .. collect
        (0x165e8640, 0x165e8648),
        (0x165e8658, 0x165e8670),
        (0x165e8698, 0x165e86dc),
        (0x165e8708, 0x165e8710),
    ]
    # BEFORE callee fast path: entry .. once-flag check, then b.eq skips the call
    before_callee = [
        (0x165f0cc0, 0x165f0d28),
        (0x165f0d48, 0x165f0d7c),
    ]

    b_call = classify(before, before_caller, 'BEFORE caller')
    a_call = classify(after, after_caller, 'AFTER  caller')
    b_cal2 = classify(callee, before_callee, 'BEFORE callee (offsetInternal)')

    show(b_call)
    show(b_cal2)
    show(a_call)

    tot_b = {k: b_call[k] + b_cal2[k] for k in
             ('insns', 'loads', 'stores', 'branches', 'calls', 'spill_st', 'spill_ld')}
    print()
    print(f"BEFORE total/cell: {tot_b}")
    print(f"AFTER  total/cell: "
          f"{{'insns': {a_call['insns']}, 'loads': {a_call['loads']}, 'stores': {a_call['stores']}, "
          f"'branches': {a_call['branches']}, 'calls': {a_call['calls']}, "
          f"'spill_st': {a_call['spill_st']}, 'spill_ld': {a_call['spill_ld']}}}")
