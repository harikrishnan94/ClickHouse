# U1b NFC verification for the `hash` join probe path

Date: 2026-07-28

## Verdict: PASS {#verdict}

No semantic codegen change in any `hash`-path anchor. Every differing instruction across all
anchors is either a benign link-layout shift (GOT slot offsets, string/table address
materializations — all targets verified equivalent) or the intended `slot_ids` load-width change
confined to the routed branch, which the `hash` path provably never executes (skipped by a `cbz`
on the null routed context).

## Binaries {#binaries}

- REFERENCE: `tmp/chj_amac/bins/clickhouse-candidate-5b276c5fb88.bin` (aarch64, BuildID `72a2dbbf…`)
- CANDIDATE: `tmp/chj_amac/bins/uncommitted-u1b.tmp.bin` (aarch64, BuildID `28200a79…`)

Caveat on the premise: the delta between the two builds is NOT just U1b plus comment-only
commits. `git log 5b276c5fb88..HEAD` includes functional routed-side commits (`7dfe941a6d0`
"Add the join slot-routing fold family", `f8d4826722d` "Route parallel_hash slots by the
dedicated fold", plus the U1b working tree in `HashJoinRoutedMethodsImpl.h`). All of that is
routed/`parallel_hash`-side; this comparison therefore verifies the *cumulative* delta is NFC for
the `hash` path, which subsumes the U1b claim. One visible consequence: `RoutedProbeContext`
became a class template (`DB::RoutedProbeContext` -> `DB::RoutedProbeContext<Map>`), renaming the
mangled names of all `joinRightColumnsWithAdditionalFilter` instantiations; pairing below
normalizes that rename.

## Method {#method}

`llvm-nm-22` symbol tables (filtered to `HashJoinMethods|RoutedProbeContext|AmacProbe|joinRoutedBlock`,
demangled with `c++filt`), `llvm-objdump-22` disassembly by address range, normalized diff
(addresses/pages blinded, `#imm` struct offsets kept). Every residual diff line was resolved to
its referenced data: `.rela.dyn` binary search for GOT/`.data.rel.ro` relocation targets
(symbolized with `llvm-symbolizer-22` in both binaries), and program-header vaddr->offset mapping
to compare string literal bytes. Artifacts: `tmp/chj_probe_parity/*.raw`, `*.norm`, `*.syms.dem`.

## Global size sweep (stronger than the 3 anchors) {#global-sweep}

All non-Routed `DB::HashJoinMethods<...>` symbols: 14064 in each binary; after normalizing the
`RoutedProbeContext` rename they pair 1:1 by name with **zero size mismatches** — including all
672 `joinRightColumnsWithAdditionalFilter` instantiations per binary and all `joinRightColumns`
loop instantiations.

## Inventory check {#inventory}

| filter | REF | CAND |
|---|---|---|
| symbols mentioning `RoutedProbeContext` | 1344 | 1344 |
| symbols mentioning `joinRightColumnsWithAdditionalFilter` | 1344 | 1344 |
| `RoutedHashJoinMethods` symbols | 6906 | 6906 |

Routed-side symbol *names* shifted as expected (slot-id types appear in routed signatures); counts
are unchanged. All non-routed names are stable modulo the `RoutedProbeContext<Map>` rename.

## Anchor (a): `HashJoinMethods<Inner, All, MapsAll>::joinBlockImpl(…, ScatteredBlock, …)` {#anchor-a}

REF `0x14653c00` / CAND `0x14642fc0`, size `0xabc` (equal), 687 instructions each.

**Verdict: BENIGN-DIFF.** 10 changed instructions, all `adrp`/`add` address materializations:

- `.data.rel.ro` tables at REF `0x1d5fd1e8`/`0x1d5fb140` vs CAND `0x1d5d5468`/`0x1d5d33c0`
  (variant-dispatch/vtable tables; nearest-symbol annotations are noise). Entries resolved through
  `.rela.dyn` and symbolized in both binaries: **entry-for-entry identical** (variant
  `__dispatcher` instantiations, `RoutedJoinResult` vtable/typeinfo, etc.).
- libc++ hardening abort strings (`%s`, `vector.h:417 …`): **byte-identical content**, shifted
  addresses only.

No instruction-mix, control-flow, or struct-offset change.

## Anchor (b): `joinRightColumnsWithAdditionalFilter` — Inner/All, key64 (`HashMethodOneNumber<…unsigned long…>`, `ResumableHashMap<HashMapTable<UInt64, …RowRefList…>>`) {#anchor-b}

Primary copy REF `0x1466fd40` / CAND `0x1465f100`; second linkonce copy REF `0x15a60d80` /
CAND `0x15a4d940`. Size `0x11f4` (equal), 1149 instructions each; both copies show the identical
18-instruction diff.

**Verdict: BENIGN-DIFF for the `hash` path** (the 2 semantic sites are the intended U1b routed-only
change and are unreachable with a null routed context). Diff classification, exhaustive:

1. **2 slot-id loads** — the U1b `slot_ids` `UInt64*`->`UInt8*` change itself:
   `ldr x10, [x10, x9, lsl #3]` -> `ldrb w10, [x10, x9]` (and the same for `x11/x22`).
   Both sites are double-guarded, e.g. (REF `0x14670320`, CAND `0x1465f6e0`):
   ```
   cbz  x25, <away>        ; x25 = routed context; null on the `hash` path -> whole block skipped
   ldr  x10, [x25]         ; routed->slot_ids
   cbz  x10, <skip-load>   ; null slot_ids -> slot 0
   ldr  x10,[x10,x9,lsl #3]   |   ldrb w10,[x10,x9]     ; REF | CAND — routed rows only
   ldr  x11, [x25, #0x8]   ; routed->maps_by_slot
   ```
   The `hash` algorithm passes `routed = nullptr` (`HashJoinMethodsImpl.h:436` call has the
   defaulted null argument), so the first `cbz` branches away before either load.
2. **8 GOT-slot loads** (`#0x668`->`#0x8d0` etc.): all six distinct slots resolved via `.rela.dyn`
   and symbolized — identical targets in both binaries: `DB::empty_pod_array`,
   `DB::ErrorCodes::LOGICAL_ERROR`, `typeinfo for std::system_error`,
   `std::regex_error::~regex_error`, `typeinfo for DB::Exception`, `DB::Exception::~Exception`.
3. **8 string materializations**: abort messages (`vector.h:417`, `vector.h:412`) and the
   `"Sizes are mismatched. selected_rows.size:{} …"` exception format string — all
   **byte-identical content** at shifted addresses.

## Anchor (c): `HashJoinMethods<Inner, All, MapsAll>::joinRightColumns<…key64…>` hot loops {#anchor-c}

Two instantiations checked (KeyGetter over `unsigned long`, `ResumableHashMap<HashMapTable<UInt64,
…RowRefList…>>`, `AddedColumns<true>`):

- `fast_path=true, need_filter=false`, range selector: REF `0x146752c0` / CAND `0x14664680`,
  size `0x564`, 345 instructions each — 7 changed instructions.
- `fast_path=true, need_filter=true`, range selector: REF `0x14670f40` / CAND `0x14660300`,
  size `0x728`, 458 instructions each — 8 changed instructions.

**Verdict: BENIGN-DIFF.** Every changed instruction is a GOT-slot load or string materialization
from the same already-verified set as anchor (b) (`empty_pod_array`, `system_error` typeinfo pair,
abort strings). **Zero** instruction-mix or control-flow changes: the pure `hash` probe loop is
instruction-identical modulo link layout. (`joinRightColumns` takes no routed context, so not even
the intended change appears here.)

## Overall {#overall}

**PASS.** The `hash` join algorithm's probe path is functionally unchanged: identical symbol
inventory and sizes across all 14064 paired instantiations, and instruction-identical anchors
modulo (i) verified-equivalent link-layout shifts and (ii) the intended routed-branch slot-id
load width, which a null routed context skips entirely.
