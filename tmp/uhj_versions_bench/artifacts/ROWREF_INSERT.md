# Why `RowRefList::insert` costs more under uhj — JOB q64

> **Answered in `UHJ_PREFETCH.md`.** The open question below ("why deref #1 misses
> more") resolved to software prefetch: uhj captures only ~25% of the prefetch benefit
> baseline gets, and with prefetch disabled uhj is 11% *faster*. The slot-layout guess
> at the end of this document was wrong.

Same code, same calls, same misses. uhj spends **+37% memory-stall cycles** servicing an
**identical number of last-level misses**, because the misses move to a dependency position
where they cannot overlap.

All measurements on JOB q64, `collect_hash_table_stats_during_joins=0` on both arms so the
plan (and therefore the work) is identical, 16 vCPU / 32 GiB cgroup.

---

## (a) Both arms do the same work

**Same code.** `RowRefList::insert` is not forked — both binaries export the same mangled
symbol `_ZN2DB10RowRefList6insertEmRNS_5ArenaE` from the shared `src/Interpreters/RowRefs.h`.
Disassembling it from each binary and normalising the branch-target addresses (which differ
only because the function loads at a different address):

```
RESULT: RowRefList::insert is INSTRUCTION-IDENTICAL in both binaries (200 instructions)
```

**Same number of calls.** A uprobe on the symbol, one execution of q64 per arm:

```
baseline   34,494,831   probe_clickhouse:rrl_baseline
uhj        34,494,831   probe_clickhouse:rrl_uhj
```

Exactly equal, to the call.

**Same input.** Both arms run the same plan and build the same relations
(`JoinBuildTableRowCount` = 42,296,370 on both). Every key lands in exactly one row list in
both engines, so the per-key row counts — and therefore the branch each call takes inside
`insert` — are the same multiset. The identical call count is consistent with that.

## (b) Which instructions cost more

Samples inside the function land on four offsets, which are the consumers of two loads —
the two pointer dereferences the function performs:

```
site A — hash cell word  ->  cell node
  0x004c  and  x22, x8, #0xffffffffffff   ; mask the 48-bit node pointer out of the word
  0x0050  ldr  x8, [x22]                  ; load the CELL node header      <-- deref #1
  0x0054  lsr  x9, x8, #8                 ; total_rows   <-- stalls here
  0x0058  ubfx x10, x8, #1, #7            ; size         <-- stalls here
  0x005c  cmp  x10, x9                    ; unchained?
  0x0064  b.ne -> site B

site B — cell node  ->  overflow node (keys with >= 8 rows)
  0x0164  ldr  x24, [x22, #0x38]          ; refs[SLOTS] = newest overflow node
  0x0168  ldr  x8, [x24]                  ; load the OVERFLOW node header  <-- deref #2
  0x016c  ubfx x9, x8, #1, #7             ; size         <-- stalls here
  0x0170  cmp  x9, #0x5                   ; room left?   <-- stalls here
```

Share of samples inside the function, by site:

| event | site A (deref #1) | | site B (deref #2) | |
|---|---|---|---|---|
| | baseline | **uhj** | baseline | **uhj** |
| cpu_cycles | 31.6% | **52.6%** | 62.1% | **42.3%** |
| ll_cache_miss_rd | 27.9% | **49.0%** | 65.4% | **46.2%** |
| dtlb_walk | 41.7% | **66.3%** | 55.5% | **31.3%** |
| l1d_cache_refill | 31.5% | **54.9%** | 62.0% | **40.7%** |

The cost does not appear at new instructions — it moves. Under baseline the misses are
concentrated on the *second* hop; under uhj they shift onto the *first* hop, the
hash-cell → cell-node dereference.

This matters because the two hops are **dependent**: site B's address comes from the node
that site A just loaded. When site A hits, the chain is one miss deep. When site A misses,
the chain is two serialised misses, and the second cannot be issued until the first returns.

## (c) Full counter set, per query

61 (baseline) and 52 (uhj) query iterations counted in a 30s window; every counter divided by
its iteration count. Event groups kept to 6 counters so nothing is multiplexed.

| metric | baseline | uhj | delta |
|---|---:|---:|---:|
| wall time (hot) | 0.402–0.409s | 0.492–0.506s | **+22%** |
| cpu_cycles | 16.35 G | 20.47 G | **+25.2%** |
| inst_retired | 22.44 G | 23.46 G | +4.6% |
| **IPC** | **1.372** | **1.146** | **−16.5%** |
| stall_frontend | 0.84 G (5.1% of cycles) | 0.69 G (3.4%) | −17.5% |
| stall_backend | 11.14 G (68.1%) | 15.26 G (74.5%) | **+37.0%** |
| stall_backend_mem | 9.48 G (58.0%) | 13.02 G (63.6%) | **+37.3%** |
| br_mis_pred_retired | 37.41 M | 37.09 M | −0.9% |
| mem_access | 6.84 G | 7.25 G | +5.9% |
| l1d_cache_refill | 257.7 M | 244.3 M | −5.2% |
| l2d_cache_refill | 140.7 M | 135.0 M | −4.1% |
| **ll_cache_miss_rd** | **215.2 M** | **214.2 M** | **−0.5%** |
| dtlb_walk | 65.2 M | 63.2 M | −3.1% |

**MLP.** There is no direct MLP counter on Neoverse V2, so use memory-stall cycles per
last-level miss, which is inversely proportional to the achieved overlap:

| | baseline | uhj |
|---|---:|---:|
| stall_backend_mem / ll_cache_miss_rd | **44.0 cycles/miss** | **60.8 cycles/miss** |

uhj pays **38% more stall cycles for the same miss**, i.e. roughly 28% less memory-level
parallelism. Everything else is flat: identical misses, identical TLB walks, identical branch
misses, 4.6% more instructions. **Backend-memory-bound, not frontend, not branch, not work.**

Where those misses live (share of the whole query, `perf record` per event):

| event | share inside `RowRefList::insert` |
|---|---|
| ll_cache_miss_rd | 62.3% baseline / 66.9% uhj |
| dtlb_walk | 79.6% baseline / 82.4% uhj |
| cpu_cycles | 47.7% baseline / 57.7% uhj |

The function owns two thirds of the query's last-level misses and four fifths of its page
walks on both arms — it is the whole memory profile of this query.

## What it is not

The obvious hypothesis was uhj's per-slot build scatter (`scatterBlockBySlot`) hurting arena
locality. A `max_threads` sweep refutes it — the gap survives at a single slot:

| max_threads | baseline | uhj | delta |
|---|---:|---:|---:|
| 1 | 5.07s | 6.05s | +19.4% |
| 2 | 2.78s | 3.70s | +33.1% |
| 4 | 1.40s | 1.85s | +32.0% |
| 8 | 0.739s | 0.956s | +29.4% |
| 16 | 0.402s | 0.492s | +22.4% |

A serial build shows the same regression, so the cause is not the number of build slots or
the parallel scatter.

## Conclusion and the open question

The regression is not extra work and not a different algorithm: it is **the same 34.5M
dereferences achieving less memory-level parallelism**, with the miss weight moved from the
overflow hop onto the first hash-cell → cell-node hop.

What is still unresolved is *why* deref #1 misses more under uhj when the miss total is
unchanged. Since it survives at `max_threads=1`, the remaining candidates are the layout of
the `Batch` nodes in the `Arena` relative to the hash cells that point at them — allocation
order, arena chunk growth, or a different map instantiation placing cells differently. The
mangled hot-path symbols already differ in one template argument
(`...HashMapTable` vs `...HashMapTable, 8`), which is worth checking first.

That is where the next investigation should start; it is a build-side layout question, not a
probe-path or algorithm-selection one.

## Reproduce

```bash
ARM=uhj QIDX=64 bash tmp/uhj_versions_bench/count_calls.sh        # (a) call count
ARM=uhj QIDX=64 bash tmp/uhj_versions_bench/deep_metrics.sh       # (c) counters + per-symbol
ARM=uhj QIDX=64 bash tmp/uhj_versions_bench/deep_metrics_norm.sh  # (c) with iteration counts
python3 tmp/uhj_versions_bench/annotate_insert.py                 # (b) instruction level
ARM=uhj QIDX=64 bash tmp/uhj_versions_bench/thread_sweep.sh       # slot-count control
```
