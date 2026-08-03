#!/usr/bin/env python3
"""Static inertness / ablation-validity proof by symbol-table diff (G5.3, G3.1).

Answers two questions that timing cannot:

  * "Did my edit perturb the comparators?"  -- G5.3. Re-measuring `hash` and
    `parallel_hash` and finding no significant difference is only a failure to detect
    one; at a 5% noise floor it would miss a 4% perturbation of every baseline number in
    the report. A symbol-table diff answers it outright.
  * "Did my ablation remove what it targeted, or relocate it?" -- the prior mission had
    an ablation whose structural check passed and which had merely moved the work.

    python3 symdiff.py --before A --after B [--expect-changed-regex 'DB::Unified::'] \
                       [--byte-compare 'fillFixedBatch' ...]

Reports symbols added, removed, and changed in size, classified by whether they belong
to the region the edit was allowed to touch. `--byte-compare` disassembles named symbols
in both binaries and asserts they are opcode-identical.

**The ICF caveat, handled explicitly.** Identical-code folding means an edit can
*de-fold* a previously shared address: baseline names then appear in the diff with no
baseline source change. So a changed symbol is reported with whether it shared its
address with another name before or after, and a name that only stopped sharing is
classified separately from one whose instructions changed.
"""

from __future__ import annotations

import argparse
import collections
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
NM = os.path.join(HERE, "..", "perf", "bin", "llvm-nm")
OBJDUMP = os.path.join(HERE, "..", "perf", "bin", "llvm-objdump")


def load(binary):
    out = subprocess.run([NM, "--defined-only", "--demangle", "--print-size", binary],
                         capture_output=True, text=True, check=True).stdout
    by_name, by_addr = {}, collections.defaultdict(list)
    for line in out.splitlines():
        m = re.match(r'^([0-9a-f]+)\s+([0-9a-f]+)?\s*([TtWw])\s+(.*)$', line)
        if not m:
            continue
        addr = int(m.group(1), 16)
        size = int(m.group(2), 16) if m.group(2) else 0
        name = m.group(4)
        by_name[name] = (addr, size)
        by_addr[addr].append(name)
    return by_name, by_addr


def body(binary, addr, size):
    txt = subprocess.run([OBJDUMP, "-d", "--no-show-raw-insn",
                          f"--start-address={hex(addr)}", f"--stop-address={hex(addr+size)}",
                          binary], capture_output=True, text=True).stdout
    ops = []
    adrp_regs: set[str] = set()
    for line in txt.splitlines():
        m = re.match(r'^\s*[0-9a-f]+:\s+(.*)$', line)
        if m:
            t = re.sub(r'\s*//.*$', '', m.group(1)).strip()
            # Absolute addresses legitimately move when anything upstream changes size,
            # so they are normalised -- but ONLY when they are long enough to be
            # addresses. An earlier version normalised every `0x...` token and therefore
            # erased `mov w9, #0x2` -> `mov w9, #0x1`, which is *exactly* the ablation
            # A-K1 makes. It reported the ablated constructor as byte-identical to the
            # reference: a validity check that could not see the change it existed to
            # confirm. See WORKLOG F7.
            t = re.sub(r'0x[0-9a-f]{5,}', 'ADDR', t)
            # objdump annotates each operand with the nearest preceding symbol. For a
            # branch or call that names the callee and is meaningful. For `adrp` it
            # names whatever data symbol happens to sit at that page base, so it changes
            # whenever the data section shifts -- with the instruction untouched.
            # Keeping it produced a false positive on
            # DB::ConcurrentHashJoin::addBlockToJoin: 826 instructions, identical size,
            # three differing `adrp` annotations and nothing else. See WORKLOG F7.
            # llvm-objdump separates mnemonic and operands with a TAB, not a
            # space. Splitting on " " never yielded the opcode, so the "keep the
            # <symbol> annotation on branches and calls" rule below silently never
            # fired and callee names were being stripped everywhere. That is
            # conservative for a same-symbol diff -- it can only hide a difference,
            # never invent one -- but it means a changed callee in a baseline symbol
            # would have gone unnoticed, which is exactly what G5.3 must catch.
            # Found by the cross-tree comparison (codegen/X1_crosstree.md).
            op = re.split(r"[\s\t]+", t)[0]
            if not re.match(r'^(b|bl|blr|br|b\.[a-z]+|cbz|cbnz|tbz|tbnz)$', op):
                t = re.sub(r'\s*<[^>]*>', '', t)

            # An address is materialised as `adrp xN, <page>` plus a second instruction supplying
            # the low bits off xN. Both halves move together when the data section shifts, with no
            # code change. The page half is already normalised above by length; the low half is a
            # SHORT hex immediate and must be normalised too -- but only when it belongs to such a
            # pair, so that a genuine small immediate like `mov w9, #0x1` is still compared.
            #
            # The low half is written either as `add xN, xN, #lo` or as a load/store THROUGH the
            # register, `ldr xM, [xN, #lo]` -- which is how a GOT entry is read. An earlier version
            # matched the base register only outside brackets, so it never saw the second form: it
            # reported `DB::HashJoin::addBlockToJoin` as 35 differing instructions in a function of
            # identical length whose source had not changed, every one of them a GOT offset. A
            # validity check that cries wolf is as useless as one that cannot see the change it
            # exists to confirm; both directions have now been wrong once. See WORKLOG F7/F9.
            #
            # A register stays paired until something other than a paired use redefines it, so one
            # `adrp` feeding several loads is handled, while a field offset off a long-lived register
            # (`ldr x10, [x22, #0x48]`) is still compared -- x22 was never set by an `adrp`.
            paired = None
            mreg = re.match(r'^adrp\s+(x\d+)', t)
            if mreg:
                adrp_regs.add(mreg.group(1))
            else:
                through = re.search(r'\[(x\d+),\s*#0x[0-9a-f]+\]', t)
                added = re.match(r'^add\s+x\d+,\s*(x\d+),\s*#0x[0-9a-f]+$', t)
                if through and through.group(1) in adrp_regs:
                    paired = through.group(1)
                elif added and added.group(1) in adrp_regs:
                    paired = added.group(1)
                if paired is not None:
                    t = re.sub(r'#0x[0-9a-f]+', '#RELOC_LO', t)
                # a redefinition of the register by anything else clears the pairing
                mdef = re.match(r'^\w+\s+(x\d+),', t)
                if mdef and mdef.group(1) in adrp_regs and mdef.group(1) != paired:
                    adrp_regs.discard(mdef.group(1))
            ops.append(re.sub(r'\s+', ' ', t))
    return ops


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--before", required=True)
    ap.add_argument("--after", required=True)
    ap.add_argument("--expect-changed-regex", default=r"DB::Unified::")
    ap.add_argument("--byte-compare", action="append", default=[],
                    help="symbols that MUST be opcode-identical")
    ap.add_argument("--expect-differ", action="append", default=[],
                    help="the ablation target: symbols that MUST differ. A validity "
                         "check that cannot see the change it exists to confirm is "
                         "worthless, so this direction is asserted too.")
    a = ap.parse_args()

    bn, ba = load(a.before)
    an, aa = load(a.after)
    allowed = re.compile(a.expect_changed_regex)

    added = set(an) - set(bn)
    removed = set(bn) - set(an)
    resized = {n for n in set(bn) & set(an) if bn[n][1] != an[n][1]}

    # Linker-generated range-extension thunks (`__AArch64ADRPThunk_<callee>`) have no source of
    # their own: the linker emits, drops and resizes them purely because code moved, so any edit
    # that changes the image size churns hundreds of them. Counting them as "outside the region the
    # edit was allowed to touch" made the verdict RED for an edit that had perturbed no baseline
    # instruction, which is a false alarm rather than a finding. They are reported separately rather
    # than dropped, because "the linker rearranged its thunks" is itself worth seeing.
    thunk = re.compile(r'^__AArch64ADRPThunk_')

    def split(names):
        ok = [n for n in names if allowed.search(n)]
        rest = [n for n in names if not allowed.search(n)]
        return ok, [n for n in rest if not thunk.search(n)], [n for n in rest if thunk.search(n)]

    add_ok, add_bad, add_thunk = split(added)
    rem_ok, rem_bad, rem_thunk = split(removed)
    res_ok, res_bad, res_thunk = split(resized)

    # Re-classify "bad" resized/removed names that merely stopped (or started) sharing
    # an address with another name: ICF de-folding, not a code change.
    def icf_only(n):
        b_addr = bn.get(n, (None, None))[0]
        a_addr = an.get(n, (None, None))[0]
        b_share = len(ba.get(b_addr, [])) > 1 if b_addr is not None else False
        a_share = len(aa.get(a_addr, [])) > 1 if a_addr is not None else False
        return b_share != a_share

    res_icf = [n for n in res_bad if icf_only(n)]
    res_real = [n for n in res_bad if n not in res_icf]

    print(f"before: {a.before}")
    print(f"after : {a.after}")
    print(f"text symbols: {len(bn)} -> {len(an)}")
    print(f"allowed-to-change regex: {a.expect_changed_regex!r}")
    print()
    print(f"  added   : {len(added):6d}  ({len(add_ok)} allowed, {len(add_bad)} OUTSIDE, "
          f"{len(add_thunk)} linker thunks)")
    print(f"  removed : {len(removed):6d}  ({len(rem_ok)} allowed, {len(rem_bad)} OUTSIDE, "
          f"{len(rem_thunk)} linker thunks)")
    print(f"  resized : {len(resized):6d}  ({len(res_ok)} allowed, {len(res_real)} OUTSIDE, "
          f"{len(res_icf)} explained by ICF de/re-folding, {len(res_thunk)} linker thunks)")

    for label, names in (("added OUTSIDE", add_bad), ("removed OUTSIDE", rem_bad),
                         ("resized OUTSIDE", res_real)):
        if names:
            print(f"\n  {label} ({len(names)}):")
            for n in sorted(names)[:25]:
                print(f"    {n[:150]}")
            if len(names) > 25:
                print(f"    ... and {len(names)-25} more")

    ok = not (add_bad or rem_bad or res_real)

    for pat in a.expect_differ:
        rx = re.compile(pat)
        hits = [n for n in set(bn) & set(an) if rx.search(n)]
        if not hits:
            print(f"\n  expect-differ {pat!r}: NO MATCHING SYMBOL")
            ok = False
            continue
        differ = [n for n in hits if body(a.before, *bn[n]) != body(a.after, *an[n])]
        print(f"\n  expect-differ {pat!r}: {len(hits)} symbols, {len(differ)} DIFFER "
              f"(required >=1) -> {'OK' if differ else 'FAIL: ablation had no effect'}")
        if not differ:
            ok = False

    for pat in a.byte_compare:
        rx = re.compile(pat)
        hits = [n for n in set(bn) & set(an) if rx.search(n)]
        if not hits:
            print(f"\n  byte-compare {pat!r}: NO MATCHING SYMBOL -- check the pattern")
            ok = False
            continue
        same = diff = 0
        examples = []
        for n in sorted(hits):
            b = body(a.before, *bn[n])
            c = body(a.after, *an[n])
            if b == c and b:
                same += 1
            else:
                diff += 1
                examples.append(n)
        print(f"\n  byte-compare {pat!r}: {len(hits)} symbols, "
              f"{same} opcode-identical, {diff} DIFFER")
        for n in examples[:5]:
            print(f"      DIFFERS: {n[:140]}")
        if diff:
            ok = False

    print(f"\nsymdiff: {'GREEN' if ok else 'RED'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
