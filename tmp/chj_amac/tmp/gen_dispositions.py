#!/usr/bin/env python3
"""Generate fleet/dispositions.json for the 1800-cell universe (Unit 4,
G-coverage), plus the 12 G-hash-inband cells and the 7 non-universe modifier
floor cells (both tracked separately; check_matrix warns-and-ignores
non-universe entries by design).

Disposition rules (MATRIX.md vocabulary, frozen):
  MEASURED          - swept base cells with a valid fleet verdict.
  EXCLUDED-INVALID  - the 17 duration-floor cells (fail-closed per MATRIX
                      caveat 6), lcstr S1 (size floor, caveat 3), asof x lcstr
                      (no such SQL shape), lcstr S5 probe (venue OOM,
                      caveats 4-5).
  PARITY-ONLY       - all strzero cells; unmeasured lcstr cells.
  NOT-CLAIMED       - unmeasured S1 cells (cache-resident, AMAC
                      auto-disengaged; sampled by G-force-engage: +1.22%
                      in-band).
  INFERRED          - everything else, from its block representative
                      (rules: family-repr / group-repr / size-interp /
                      thread-interp, chained with '+').
"""
import json
import pathlib

BASE = pathlib.Path('/mnt/ch/ClickHouse/tmp/chj_amac')
matrix = json.loads((BASE / 'fleet/matrix.json').read_text())
sweep = json.loads((BASE / 'tmp/sweep_parsed.json').read_text())

UNIVERSE = [e['cell'] for e in matrix['universe']['cells']]
HASH_CELLS = matrix['hash_inband']['cells']

FAMREP_BUILD = {'key32': 'key64', 'key64': 'key64', 'null64': 'key64',
                'str': 'str', 'fixstr': 'str', 'k128': 'k256', 'k256': 'k256',
                'mixed': 'mixed'}

def parse(cell):
    fam, rest = cell.split(':')
    parts = rest.split('.')
    return fam, parts[0], parts[1], parts[2], parts[3]  # fam, side, group, size, T

def is_base(cell):
    return len(cell.split(':')[1].split('.')) == 4

# Valid measured base (universe) cells and their verdicts.
measured = {c: v for c, v in sweep.items()
            if is_base(c) and v['verdict'] in ('WIN', 'TIE', 'LOSS')}

def mfmt(c):
    v = sweep[c]
    return (f"{v['verdict']} diff={v['diff_pct']:+.2f}% band={v['band_pct']}% "
            f"runs={v['runs']} (fleet shard {v['shard']}, "
            f"fleet/results/results.shard{v['shard']}.jsonl)")

FORCE_NOTE = ('G-force-engage sampled the class: forced ring at '
              'key64:probe.inner_all.S1.T96 = +1.22% OFF-vs-FORCE, in-band '
              '(band 3%; fleet/results/force_engage.jsonl)')

PARITY_NOTE = ('correctness covered by dual-side parity gate: PARITY OK '
               '(636 cases: 634 compared, 2 matched-error, 0 failed), '
               'parity/gate_u3.log')

# The 17 duration-floor cells -> (evidence tail: surviving sibling / gap).
FLOOR = {
    'k256:build.inner_all.S2.T96': 'surviving sibling k256:build.inner_all.S5.T96 WIN -17.87%',
    'k256:build.inner_all.S3.T96': 'surviving sibling k256:build.inner_all.S5.T96 WIN -17.87%',
    'key64:build.inner_all.S2.T96': 'surviving siblings key64:build.inner_all.S5.T96 TIE +0.48% and key64:build.inner_all.S3.T1 TIE -0.30%',
    'key64:build.inner_all.S3.T48': 'surviving siblings key64:build.inner_all.S5.T96 TIE +0.48% and key64:build.inner_all.S3.T1 TIE -0.30%',
    'key64:build.inner_all.S3.T96': 'surviving siblings key64:build.inner_all.S5.T96 TIE +0.48% and key64:build.inner_all.S3.T1 TIE -0.30%',
    'mixed:build.inner_all.S2.T96': 'surviving sibling mixed:build.inner_all.S5.T96 TIE +0.01%',
    'mixed:build.inner_all.S3.T96': 'surviving sibling mixed:build.inner_all.S5.T96 TIE +0.01%',
    'str:build.inner_all.S2.T96': 'surviving sibling str:build.inner_all.S5.T96 WIN -24.29%',
    'str:build.inner_all.S3.T96': 'surviving sibling str:build.inner_all.S5.T96 WIN -24.29%',
    'str:probe.inner_all.S2.T1': 'candidate arm under floor (168.4 ms); surviving siblings str:probe.inner_all.S2.T48 WIN -14.51% and key64:probe.inner_all.S2.T1 TIE',
    # Non-universe modifier floor cells (tracked separately; check_matrix
    # warns and ignores them).
    'key64:build.inner_all.S3.T96.statson': 'surviving sibling key64:probe.inner_all.S3.T96.statson TIE -2.16% settles stats-on protocol sensitivity; build-side statson point is a NAMED GAP',
    'key64:probe.inner_all.S3.T96.h05': 'sibling key64:probe.inner_all.S3.T96.h50 LOSS +3.47% measured; h=0.05 point is a NAMED GAP (both h05 cells under floor)',
    'str:probe.inner_all.S3.T96.h05': 'sibling str:probe.inner_all.S3.T96.h50 WIN -16.84% measured; h=0.05 point is a NAMED GAP (both h05 cells under floor)',
    'key64:probe.semi_anti.S2.T96.anti': 'surviving sibling key64:probe.semi_anti.S4.T96.anti LOSS +42.35%',
    'str:probe.semi_anti.S2.T96.anti': 'surviving sibling str:probe.semi_anti.S4.T96.anti WIN -20.51%',
    'str:build.inner_all.S3.T96.dup16': 'siblings key64:build.inner_all.S3.T96.dup16 TIE -0.01% and str:build.inner_all.S5.T96 WIN -24.29%; dup16 str build point is a NAMED GAP',
    'str:build.left_all.S3.T96.dup16': 'siblings key64:build.left_all.S3.T96.dup16 TIE -0.24% and str:build.inner_all.S5.T96 WIN -24.29%; dup16 str build point is a NAMED GAP',
}

SIZE_PREF = {'S1': ['S2', 'S3', 'S4', 'S5'], 'S2': ['S2', 'S3', 'S4', 'S5'],
             'S3': ['S3', 'S4', 'S2', 'S5'], 'S4': ['S4', 'S5', 'S3', 'S2'],
             'S5': ['S5', 'S4', 'S3', 'S2']}

def find_probe_rep(fam, group, size, threads):
    """Nearest measured probe representative + rule chain."""
    tpref = [threads] + [t for t in ('T96', 'T48', 'T1') if t != threads]
    for g, grule in ((group, []), ('inner_all', ['group-repr'])):
        if g == 'inner_all' and group == 'inner_all' and grule:
            continue
        for s in SIZE_PREF[size]:
            for t in tpref:
                cand = f'{fam}:probe.{g}.{s}.{t}'
                if cand in measured:
                    rule = list(grule)
                    if s != size:
                        rule.append('size-interp')
                    if t != threads:
                        rule.append('thread-interp')
                    return cand, '+'.join(rule) or 'exact'
    return None, None

def find_build_rep(fam, group, size, threads):
    rf = FAMREP_BUILD[fam]
    rule = []
    if rf != fam:
        rule.append('family-repr')
    if group != 'inner_all':
        rule.append('group-repr')
    if rf == 'key64' and threads == 'T1':
        cand = 'key64:build.inner_all.S3.T1'
        if size != 'S3':
            rule.append('size-interp')
        return cand, '+'.join(rule) or 'exact'
    cand = f'{rf}:build.inner_all.S5.T96'
    if size != 'S5':
        rule.append('size-interp')
    if threads != 'T96':
        rule.append('thread-interp')
    return cand, '+'.join(rule) or 'exact'

disp = {}

for cell in UNIVERSE:
    fam, side, group, size, threads = parse(cell)
    if cell in FLOOR:
        reason = sweep[cell]['reason']
        disp[cell] = {'disposition': 'EXCLUDED-INVALID',
                      'evidence': (f'duration floor: {reason}; fail-closed per '
                                   f'MATRIX.md caveat 6 (fleet shard '
                                   f"{sweep[cell]['shard']}); {FLOOR[cell]}")}
    elif cell in measured:
        disp[cell] = {'disposition': 'MEASURED', 'evidence': mfmt(cell)}
    elif fam == 'strzero':
        disp[cell] = {'disposition': 'PARITY-ONLY',
                      'evidence': ('no perf claim (MATRIX.md: all strzero '
                                   'cells); ' + PARITY_NOTE)}
    elif fam == 'lcstr':
        if group == 'asof':
            disp[cell] = {'disposition': 'EXCLUDED-INVALID',
                          'evidence': ('no such SQL shape: asof x lcstr '
                                       '(MATRIX.md disposition vocabulary '
                                       'example)')}
        elif size == 'S1':
            disp[cell] = {'disposition': 'EXCLUDED-INVALID',
                          'evidence': ('size floor: lcstr map floor ~14 MiB '
                                       '(L3) makes S1 (1 MiB) structurally '
                                       'unreachable; MATRIX.md caveat 3')}
        elif cell == 'lcstr:probe.inner_all.S5.T96':
            disp[cell] = {'disposition': 'EXCLUDED-INVALID',
                          'evidence': ('structurally unreachable size at the '
                                       'fleet venue: baseline arm OOM in '
                                       'warmup (Code 241 '
                                       'MEMORY_LIMIT_EXCEEDED, would use '
                                       '191.45 GiB; fleet/sweep_shard0.log; '
                                       'MATRIX.md caveats 4-5); surviving '
                                       'siblings lcstr S2 TIE +1.98% / S3 '
                                       'WIN -4.46% measured')}
        else:
            disp[cell] = {'disposition': 'PARITY-ONLY',
                          'evidence': ('no perf claim beyond the measured '
                                       'S2/S3 regression guards (lcstr is an '
                                       'expected AMAC-exclusion family, '
                                       'MATRIX.md caveat 4); ' + PARITY_NOTE)}
    elif size == 'S1':
        disp[cell] = {'disposition': 'NOT-CLAIMED',
                      'evidence': ('cache-resident, AMAC auto-disengaged '
                                   '(measured S2 cells show '
                                   'AmacBuildRows=0); ' + FORCE_NOTE)}
    else:
        if side == 'probe':
            src, rule = find_probe_rep(fam, group, size, threads)
        else:
            src, rule = find_build_rep(fam, group, size, threads)
        if src is None:
            raise SystemExit(f'no representative for {cell}')
        note = ''
        if side == 'build' and size in ('S2', 'S3', 'S4') and threads in ('T48', 'T96'):
            note = ('; small-size build cells at high T are duration-floor-'
                    'bound at fleet scale (MATRIX.md caveat 6) -- S5/T1 '
                    'coverage substitutes')
        disp[cell] = {'disposition': 'INFERRED', 'from': src, 'rule': rule,
                      'evidence': f'block representative: {mfmt(src)}{note}'}

# The 12 G-hash-inband cells (non-universe, tracked separately).
for cell in HASH_CELLS:
    v = sweep[cell]
    assert v['verdict'] == 'TIE', cell
    disp[cell] = {'disposition': 'MEASURED',
                  'evidence': ('G-hash-inband (join_algorithm=hash on BOTH '
                               f'arms): {mfmt(cell)}')}

# The 7 non-universe modifier floor cells.
for cell, tail in FLOOR.items():
    if cell not in disp:
        reason = sweep[cell]['reason']
        disp[cell] = {'disposition': 'EXCLUDED-INVALID',
                      'evidence': (f'duration floor: {reason}; fail-closed '
                                   f'per MATRIX.md caveat 6 (fleet shard '
                                   f"{sweep[cell]['shard']}); {tail}")}

out = BASE / 'fleet/dispositions.json'
out.write_text(json.dumps(disp, indent=1) + '\n')

from collections import Counter
uni = Counter(disp[c]['disposition'] for c in UNIVERSE)
print('universe:', dict(uni), 'total', sum(uni.values()))
print('total entries:', len(disp), '(universe 1800 + hash 12 + modifier-floor 7)')
