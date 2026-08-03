#!/usr/bin/env python3
"""Cross-tree, same-binary codegen parity: baseline `DB::...` vs unified `DB::Unified::...`.

`symdiff.py` answers "is symbol X the same in binary A and binary B". This answers the
other question: "inside ONE binary, does the unified tree emit the same instructions as
the baseline tree for the corresponding function". Same normalisation rules as
`symdiff.body` -- they are load-bearing and were debugged the hard way (see WORKLOG F7):

  * hex tokens of >=5 digits -> ADDR (addresses move), but NOT shorter ones, because a
    short hex is a real immediate and erasing it once hid a real code change;
  * llvm-objdump's `<symbol>` operand annotation is stripped EXCEPT on branch/call
    opcodes, where it names the callee and is meaningful;
  * the short immediate of the `add`/load/store that pairs with a preceding `adrp` into
    the same register is the low half of a relocation and is normalised too.

Plus one rule that only the cross-tree case needs: the callee name inside a branch/call
annotation is resolved through the symbol table to a demangled name and then has
`Unified::` / `DB::Unified::` deleted from it. Without that, every call from unified code
to its own sibling function looks like a difference. The count of annotations normalised
this way is reported, and a branch whose target still differs afterwards is flagged
separately -- that is a real difference (one side calling something the other does not).

Resolving the target through the symbol table rather than trusting objdump's text also
makes the two sides comparable at all: objdump prints mangled names, and a unified
mangled name is not the baseline one with a substring deleted (the substitution indices
shift), so textual deletion on the mangled form would not work.

    python3 xtree.py --base 'HashJoinMethods.*joinBlockImpl' [--unified REGEX] [--list]

Identical-code folding is handled: a target address can carry several names, so the
normalised name set at the address is used, and a pair that differs only in which alias
objdump happened to print is reported as alias-only rather than as a code difference.
"""

from __future__ import annotations

import argparse
import bisect
import collections
import difflib
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
NM = os.path.join(HERE, "..", "perf", "bin", "llvm-nm")
OBJDUMP = os.path.join(HERE, "..", "perf", "bin", "llvm-objdump")
DEFAULT_BINARY = os.path.join(HERE, "bin", "clickhouse.ref")

BRANCH = re.compile(r'^(b|bl|blr|br|b\.[a-z]+|cbz|cbnz|tbz|tbnz)$')
# Only the `Unified` namespace component is deleted, so `DB::Unified::HashJoin` becomes
# `DB::HashJoin` and compares equal to the baseline name. Deleting the whole `DB::Unified::`
# would yield `HashJoin` and make every such callee look like a difference again.
UNIFIED = re.compile(r'\bUnified::')
THUNK = re.compile(r'^__AArch64(ADRP)?Thunk_')


class SymbolTable:
    """Demangled text symbols of one binary, queryable by name and by address."""

    def __init__(self, binary, cache=None):
        self.binary = binary
        text = self._load_text(binary, cache)
        self.by_name = {}
        self.by_addr = collections.defaultdict(list)
        for line in text.splitlines():
            m = re.match(r'^([0-9a-f]+)\s+([0-9a-f]+)?\s*([TtWw])\s+(.*)$', line)
            if not m:
                continue
            addr = int(m.group(1), 16)
            size = int(m.group(2), 16) if m.group(2) else 0
            name = m.group(4)
            self.by_name[name] = (addr, size)
            self.by_addr[addr].append(name)
        self.addrs = sorted(self.by_addr)
        self.sizes = {}
        for name, (addr, size) in self.by_name.items():
            self.sizes[addr] = max(self.sizes.get(addr, 0), size)

    @staticmethod
    def _load_text(binary, cache):
        if cache and os.path.exists(cache):
            with open(cache, errors="replace") as f:
                return f.read()
        out = subprocess.run([NM, "--defined-only", "--demangle", "--print-size", binary],
                             capture_output=True, text=True, check=True).stdout
        if cache:
            with open(cache, "w") as f:
                f.write(out)
        return out

    def names_at(self, addr):
        """Names covering `addr`, plus the offset into the enclosing symbol."""
        i = bisect.bisect_right(self.addrs, addr) - 1
        if i < 0:
            return [], None
        base = self.addrs[i]
        size = self.sizes.get(base, 0)
        if addr != base and size and addr >= base + size:
            return [], None
        return self.by_addr[base], addr - base


_thunk_cache: dict[int, int] = {}


def resolve_thunk(binary, addr):
    """Follow a linker range-extension thunk (`adrp x16; add x16, x16, #lo; br x16`).

    The unified tree is linked tens of megabytes away from the baseline tree, so calls
    that the baseline reaches directly need a thunk from the unified side. That is a
    placement artefact, not a code difference, so the thunk is followed to its target.
    """
    if addr in _thunk_cache:
        return _thunk_cache[addr]
    txt = subprocess.run([OBJDUMP, "-d", "--no-show-raw-insn",
                          f"--start-address={hex(addr)}", f"--stop-address={hex(addr + 16)}",
                          binary], capture_output=True, text=True).stdout
    # A short-range thunk is a single `b <target>`; a long-range one materialises the
    # address with `adrp` + `add` and jumps through the register. Parsing stops at the
    # first terminator, because thunks are packed four to a 16-byte window and reading
    # past the end picks up the *next* thunk's target.
    page = imm = target = None
    for line in txt.splitlines():
        m = re.match(r'^\s*[0-9a-f]+:\s+(\S+)\s*(.*)$', line)
        if not m:
            continue
        op, args = m.group(1), m.group(2)
        if op == "adrp":
            mm = re.search(r'(0x[0-9a-f]+)', args)
            page = int(mm.group(1), 16) if mm else None
        elif op == "add":
            mm = re.search(r'#(0x[0-9a-f]+)', args)
            imm = int(mm.group(1), 16) if mm else None
        elif op == "b":
            mm = re.match(r'(0x[0-9a-f]+)', args)
            target = int(mm.group(1), 16) if mm else None
            break
        elif op == "br":
            target = page + imm if page is not None and imm is not None else None
            break
    _thunk_cache[addr] = target
    return target


def normalised_target(binary, symtab, addr, own=None):
    """Canonical, Unified-stripped rendering of a branch/call target.

    Returns (text, stripped_name_set, n_unified_stripped, n_thunks_followed). ICF can
    put several names on one address, so the whole set is kept for the alias-only
    classification and one deterministic member is used for the comparable text.
    """
    # A branch inside the function under comparison, or a tail call to itself, names the
    # function -- and that name embeds the map type, which legitimately differs between
    # the trees (`TwoLevelHashTableGrower<8ul>, ..., 8` against
    # `HashTableGrowerWithPrecalculation<8ul>, ..., -1`). Rendering it as SELF keeps the
    # control-flow structure comparable instead of marking every loop back-edge as a
    # different call target.
    if own is not None and own[0] <= addr < own[0] + own[1]:
        off = addr - own[0]
        return (f"<SELF+{off:#x}>" if off else "<SELF>"), frozenset({"SELF"}), 0, 0
    names, off = symtab.names_at(addr)
    n_thunk = 0
    if names and all(THUNK.match(n) for n in names):
        inner = resolve_thunk(binary, addr)
        if inner is not None:
            n_thunk = 1
            names, off = symtab.names_at(inner)
    if not names:
        return "<?>", frozenset(), 0, n_thunk
    stripped = {UNIFIED.sub("", n) for n in names}
    n_stripped = sum(1 for n in names if UNIFIED.search(n))
    # Prefer a C++ name over a C alias folded onto the same address by ICF, then the
    # shortest, so the choice is deterministic and readable.
    text = sorted(stripped, key=lambda n: (0 if "::" in n else 1, len(n), n))[0]
    if off:
        text = f"{text}+{off:#x}"
    return f"<{text}>", frozenset(stripped), n_stripped, n_thunk


def body(binary, addr, size, symtab):
    """Normalised instruction list. Rules identical to `symdiff.body` plus target names.

    Also returns the number of branch/call annotations whose target name contained
    `Unified::`, and the per-index target name sets so a differing branch can be
    classified as a real retarget or as an ICF alias-naming artefact.
    """
    txt = subprocess.run([OBJDUMP, "-d", "--no-show-raw-insn",
                          f"--start-address={hex(addr)}", f"--stop-address={hex(addr + size)}",
                          binary], capture_output=True, text=True).stdout
    ops = []
    targets = []
    n_stripped = 0
    n_thunk = 0
    adrp_regs: set[str] = set()
    for line in txt.splitlines():
        m = re.match(r'^\s*[0-9a-f]+:\s+(.*)$', line)
        if not m:
            continue
        t = re.sub(r'\s*//.*$', '', m.group(1)).strip()
        # Whitespace is collapsed HERE, not at the end as in `symdiff.body`, because
        # llvm-objdump separates the mnemonic from the operands with a TAB: splitting on
        # " " yields "bl\t0x1428a1e0" as the opcode, so symdiff's branch exception never
        # actually fired and it stripped callee annotations everywhere. That is only
        # conservative for a same-symbol diff, but for the cross-tree diff the callee
        # name is the whole point.
        t = re.sub(r'\s+', ' ', t)
        op = t.split(" ")[0]
        target = frozenset()
        if BRANCH.match(op):
            # Replace objdump's mangled annotation with the Unified-stripped demangled
            # name resolved from the symbol table, so that a call to a sibling unified
            # function compares equal to the baseline's call to its own sibling.
            mt = re.search(r'\b0x([0-9a-f]+)\s*(<[^>]*>)?\s*$', t)
            if mt:
                text, target, k, kt = normalised_target(binary, symtab,
                                                        int(mt.group(1), 16), (addr, size))
                n_stripped += k
                n_thunk += kt
                t = t[:mt.start()] + text
        else:
            t = re.sub(r'\s*<[^>]*>', '', t)
        t = re.sub(r'0x[0-9a-f]{5,}', 'ADDR', t)

        reg = None
        mreg = re.match(r'^adrp\s+(x\d+)', t)
        if mreg:
            adrp_regs.add(mreg.group(1))
        else:
            mu = re.match(r'^(add|ldr\w*|str\w*|ldp|stp)\s+\w+,\s*(x\d+)', t)
            if mu and mu.group(2) in adrp_regs:
                reg = mu.group(2)
                t = re.sub(r'#0x[0-9a-f]+', '#RELOC_LO', t)
                adrp_regs.discard(reg)
            mdef = re.match(r'^\w+\s+(x\d+),', t)
            if mdef and mdef.group(1) in adrp_regs and reg is None:
                adrp_regs.discard(mdef.group(1))
        ops.append(t)
        targets.append(target)
    return ops, targets, n_stripped, n_thunk


def opcode(insn):
    return insn.split(" ")[0]


def call_histogram(ops):
    """Multiset of `bl` target names (already Unified-stripped by `body`)."""
    h = collections.Counter()
    for x in ops:
        if opcode(x) in ("bl", "b"):
            m = re.search(r'<([^>]*)>', x)
            if m:
                h[re.sub(r'\+0x[0-9a-f]+$', '', m.group(1))] += 1
    return h


def aligned(bops, uops):
    """Edit script between the two instruction sequences.

    A purely positional diff is misleading when one side inserts a couple of
    instructions: everything after the insertion shifts and reports as different. The
    alignment separates "the same code, moved" from "different code".
    """
    sm = difflib.SequenceMatcher(a=bops, b=uops, autojunk=False)
    same = replaced = inserted = deleted = 0
    pairs = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            same += i2 - i1
        elif tag == "replace":
            replaced += max(i2 - i1, j2 - j1)
            for k in range(max(i2 - i1, j2 - j1)):
                b = bops[i1 + k] if i1 + k < i2 else "<none>"
                u = uops[j1 + k] if j1 + k < j2 else "<none>"
                pairs.append((i1 + k, b, u))
        elif tag == "delete":
            deleted += i2 - i1
            pairs += [(i, bops[i], "<none>") for i in range(i1, i2)]
        elif tag == "insert":
            inserted += j2 - j1
            pairs += [(i1, "<none>", uops[j]) for j in range(j1, j2)]
    return same, replaced, inserted, deleted, pairs


def classify(b, u):
    """Why one aligned instruction differs from the other.

    The unified `HashJoin` object has a different member layout and a different stack
    frame, so a large share of the differences are one immediate apart with everything
    else equal. Those say nothing about the algorithm; separating them keeps the
    structural differences visible.
    """
    if b == "<none>" or u == "<none>":
        return "insert/delete"
    strip = lambda t: re.sub(r'(#-?0x[0-9a-f]+|SELF\+0x[0-9a-f]+|#-?\d+)', '#IMM', t)
    if strip(b) == strip(u):
        return "immediate or field offset only"
    if opcode(b) == opcode(u):
        return "same opcode, different operands"
    return "different opcode"


def compare(base, unified):
    """Positional diff plus an opcode-sequence diff that tolerates length changes."""
    bops, btgt, bstrip, bthunk = base
    uops, utgt, ustrip, uthunk = unified
    n = min(len(bops), len(uops))
    positions = [i for i in range(n) if bops[i] != uops[i]]
    positions += list(range(n, max(len(bops), len(uops))))
    retargets, alias_only = [], []
    for i in range(n):
        if bops[i] == uops[i]:
            continue
        if BRANCH.match(opcode(bops[i])) and BRANCH.match(opcode(uops[i])):
            if btgt[i] and utgt[i] and not (btgt[i] & utgt[i]):
                retargets.append(i)
            elif btgt[i] and utgt[i]:
                alias_only.append(i)
    return positions, retargets, alias_only, (bstrip, ustrip), (bthunk, uthunk)


def pick(symtab, regex, want_unified):
    rx = re.compile(regex)
    out = []
    for name, (addr, size) in symtab.by_name.items():
        if not rx.search(name):
            continue
        if bool(UNIFIED.search(name)) != want_unified:
            continue
        out.append((name, addr, size))
    return sorted(out, key=lambda x: (-x[2], x[0]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", default=DEFAULT_BINARY)
    ap.add_argument("--cache", default=os.path.join(HERE, "tmp", "nm_ref_demangled.txt"))
    ap.add_argument("--base", required=True, help="regex for the baseline symbol")
    ap.add_argument("--unified", help="regex for the unified symbol (default: --base)")
    ap.add_argument("--pick-base", type=int, help="index into the baseline candidate list")
    ap.add_argument("--pick-unified", type=int)
    ap.add_argument("--list", action="store_true", help="only list candidates")
    ap.add_argument("--examples", type=int, default=5)
    ap.add_argument("--label", default="")
    ap.add_argument("--dump", help="write both normalised instruction listings here")
    a = ap.parse_args()

    symtab = SymbolTable(a.binary, a.cache)
    bases = pick(symtab, a.base, False)
    unis = pick(symtab, a.unified or a.base, True)

    print(f"binary: {a.binary}")
    if a.label:
        print(f"group : {a.label}")
    print(f"baseline candidates: {len(bases)}   unified candidates: {len(unis)}")
    if a.list or not bases or not unis:
        for tag, lst in (("BASE", bases), ("UNIFIED", unis)):
            for i, (name, addr, size) in enumerate(lst[:40]):
                print(f"  [{tag} {i}] {addr:#x} size={size} {name}")
            if len(lst) > 40:
                print(f"  [{tag}] ... {len(lst) - 40} more")
        if not bases:
            print("COULD NOT RESOLVE: no baseline symbol matches "
                  "(inlined away, or the regex is wrong)")
        if not unis:
            print("COULD NOT RESOLVE: no DB::Unified:: symbol matches "
                  "(inlined away, or the regex is wrong)")
        return 2 if (not bases or not unis) else 0

    def choose(lst, idx):
        if idx is not None:
            return lst[idx]
        return lst[0] if len(lst) == 1 else None

    b = choose(bases, a.pick_base)
    u = choose(unis, a.pick_unified)
    if b is None or u is None:
        # Try name-based pairing: identical after deleting Unified::.
        bmap = {UNIFIED.sub("", n): (n, ad, sz) for n, ad, sz in bases}
        pairs = [(bmap[UNIFIED.sub("", n)], (n, ad, sz))
                 for n, ad, sz in unis if UNIFIED.sub("", n) in bmap]
        if len(pairs) == 1:
            b, u = pairs[0]
        else:
            print(f"AMBIGUOUS: {len(bases)} baseline / {len(unis)} unified candidates, "
                  f"{len(pairs)} exact name pairs. Use --pick-base/--pick-unified "
                  f"or tighten the regex; run with --list to see them.")
            for tag, lst in (("BASE", bases), ("UNIFIED", unis)):
                for i, (name, addr, size) in enumerate(lst[:15]):
                    print(f"  [{tag} {i}] {addr:#x} size={size} {name[:200]}")
            return 3

    bn, ba_, bs = b
    un, ua_, us = u
    print(f"\nBASELINE  {ba_:#x} size={bs}\n  {bn}")
    print(f"UNIFIED   {ua_:#x} size={us}\n  {un}")
    if ba_ == ua_:
        print("\nNOTE: both names resolve to the SAME address -- identical-code folding "
              "already proves the bodies are byte-identical.")

    bb = body(a.binary, ba_, bs, symtab)
    uu = body(a.binary, ua_, us, symtab)
    positions, retargets, alias_only, strips, thunks = compare(bb, uu)

    print(f"\ninstructions: baseline {len(bb[0])}  unified {len(uu[0])}  "
          f"delta {len(uu[0]) - len(bb[0]):+d}")
    print(f"call/branch targets Unified-normalised: baseline {strips[0]}, unified {strips[1]}")
    print(f"call/branch targets reached via a linker range thunk (followed): "
          f"baseline {thunks[0]}, unified {thunks[1]}")
    print(f"identical after normalisation: {'YES' if not positions else 'NO'}")
    if positions:
        print(f"differing positions: {len(positions)} of {max(len(bb[0]), len(uu[0]))}")
        print(f"  of which branch/call with a genuinely different target: {len(retargets)}"
              + (f" -> indices {retargets[:10]}" if retargets else ""))
        print(f"  of which branch/call differing only in ICF alias naming: {len(alias_only)}")
        bopc = collections.Counter(opcode(x) for x in bb[0])
        uopc = collections.Counter(opcode(x) for x in uu[0])
        delta = {k: uopc[k] - bopc[k] for k in set(bopc) | set(uopc) if uopc[k] != bopc[k]}
        if delta:
            print("  opcode histogram delta (unified - baseline), largest first:")
            for k, v in sorted(delta.items(), key=lambda kv: -abs(kv[1]))[:15]:
                print(f"    {k:12s} {v:+d}   (baseline {bopc[k]}, unified {uopc[k]})")
        # A positional diff is weak once the two sides differ in length, so the callee
        # multiset is compared too: a target that only one side calls is a real
        # difference regardless of where it sits.
        bh, uh = call_histogram(bb[0]), call_histogram(uu[0])
        only_b = {k: v for k, v in (bh - uh).items()}
        only_u = {k: v for k, v in (uh - bh).items()}
        print(f"  callee multiset: {sum(bh.values())} baseline calls to {len(bh)} targets, "
              f"{sum(uh.values())} unified calls to {len(uh)} targets; "
              f"{len(only_b)} targets baseline-only, {len(only_u)} unified-only")
        for tag, d in (("BASELINE-ONLY callee", only_b), ("UNIFIED-ONLY callee", only_u)):
            for k, v in sorted(d.items(), key=lambda kv: -kv[1])[:12]:
                print(f"    {tag} x{v}: {k[:170]}")
        same, rep, ins, dele, pairs = aligned(bb[0], uu[0])
        print(f"  aligned edit script: {same} instructions identical and in the same "
              f"relative order, {rep} replaced, {ins} inserted by unified, "
              f"{dele} only in baseline")
        classes = collections.Counter(classify(b, u) for _, b, u in pairs)
        for k, v in classes.most_common():
            print(f"    {v:6d}  {k}")
        structural = [p for p in pairs if classify(*p[1:]) == "different opcode"]
        print(f"\n  first {a.examples} aligned differences that are not just an "
              f"immediate/offset ({len(structural)} such):")
        for i, bi, ui in structural[:a.examples]:
            print(f"    [{i:4d}] base    {bi}")
            print(f"           unified {ui}")
    if a.dump:
        with open(a.dump, "w") as f:
            f.write(f"# BASELINE {bn}\n")
            f.write("".join(f"{i:5d} {x}\n" for i, x in enumerate(bb[0])))
            f.write(f"\n# UNIFIED {un}\n")
            f.write("".join(f"{i:5d} {x}\n" for i, x in enumerate(uu[0])))
        print(f"\nlistings: {a.dump}")
    return 0 if not positions else 1


if __name__ == "__main__":
    sys.exit(main())
