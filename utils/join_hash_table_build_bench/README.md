# Parallel hash-table build microbenchmark

Standalone C++20 binary. Standard library and compiler CRC32C intrinsics only. Measures the
**build/insert** phase of four configurations on 64-bit keys.

Paper: Birler et al., *Simple, Efficient, and Robust Hash Tables for Join Processing*, DaMoN 2024
(https://db.in.tum.de/~birler/papers/hashtable.pdf).

ClickHouse sources this is modelled on: `src/Common/HashTable/HashMap.h`,
`TwoLevelHashTable.h`, `src/Interpreters/ConcurrentHashJoin.cpp`, `src/Interpreters/RowRefs.h`,
`src/Common/HashTable/Hash.h` (`HashCRC32` / `intHashCRC32`).

## Configurations

| Name | What it is |
| --- | --- |
| `unchained` | Umbra unchained. 64-bit directory (48-bit pointer, 16-bit Bloom tag), fill `2^ceil(log2(1.125 n))`, adjacency-array tuples, partitioned build, no atomics in the directory. |
| `chained` | Umbra chained with atomics. Same tagged directory, thread-local slabs, per-row `exchange` + `fetch_or`. No build partitioning. |
| `ch-mutex` | ClickHouse-style: `ch-parts` (default 256) sub-tables, each a HashMap (linear probing, 0.5 load, power-of-two growth, zero-key cell). Duplicates as `RowRefList`. Per-sub-table `std::mutex`. |
| `ch-spin` | Identical to `ch-mutex` except the latch is a 1-byte test-and-test-and-set spin. Isolates the primitive. |

Umbra designs have no mutex or latch. The only extra atomic is a counter to claim a partition (unchained) or nothing beyond the per-row `xchg`/`or` (chained). ClickHouse is the only design with the latch knob.

## Timing boundaries

Phase 1 is untimed: Zipf keys into one array, plus a per-key histogram. Also untimed: `mmap` + first-touch write of directory / tuple storage / hash-map buffers, and the ClickHouse unique-key `reserve` pre-pass (hashes `1..distinct` into sub-tables and sizes each HashMap; this is the documented deviation from stock `reserve` estimates).

The timed region is turning that key array into a probe-ready table:

| Design | `collect` | `count` | `insert` | throughput denominator |
| --- | --- | --- | --- | --- |
| unchained | thread-local 3-level bump, partitioned by top hash bits | merge per-partition counts, exclusive prefix | per-partition owner: slot counts + Bloom OR, exclusive prefix over that directory range, copy into the adjacency array | sum of the three (they are sequential) |
| chained | 0 | 0 | bump tuples in a pre-sized slab and link with atomic `xchg`/`or` | that interval |
| ch-* | max-thread scatter time | 0 | max-thread latch + emplace + `RowRefList` append | **wall clock of the parallel region** (scatter and insert overlap across threads) |

`latch_pct` is `100 * (sum of per-thread acquire ns / threads) / wall_ns`. Failed `try_lock` counts. Unlock does not. Cycle counter is `cntvct_el0` / `cntfrq_el0` on aarch64, `rdtscp` calibrated against `CLOCK_MONOTONIC` on x86-64.

## Choices that move the numbers

- **Umbra `hash64` seeds** are not in the paper. Used `0x9E3779B9` and `0x85EBCA77`. Mixing constant is the paper's `0x2545F4914F6CDD1D`. Hardware CRC32C (`crc32` / `__crc32cd`).
- **Bloom tags**: all `C(16,4)=1820` 4-bit patterns in lexicographic order, padded to 2048 by sampling those patterns with `mt19937(0xC0FFEE01)`.
- **Umbra tuple** is 32 bytes (`key, hash, next, row`), as in the paper's Figure 12 microbenchmark. ClickHouse cells are 16 bytes (`key` + `RowRefList` word) plus 64-byte `Batch` nodes for duplicates. Not homogenized.
- **Level-1 bump** is per-thread `mmap`, not a shared `malloc`. Small chunks are 8 KiB. Enough of that space is mapped and first-touched before the timed collect, so the timed region is partitioning and copies, not page faults.
- **Directory size** is `max(next_pow2(ceil(1.125 n)), parts)` so each unchained partition owns a disjoint slot range.
- **ClickHouse bucket** is `TwoLevelHashTable::getBucketFromHash`: `(hash >> (32 - log2(parts))) & (parts-1)`. For 256 parts that is bits 24–31 of `HashCRC32`.
- **ClickHouse acquire**: `try_lock` around the ring of pending sub-blocks; **blocking `lock` only after a round with no progress**. Stock `ConcurrentHashJoin::addBlockToJoin` yields and retries `try_lock` instead. The latch comparison uses the same discipline for mutex and spin.
- **Latches** are 64-byte aligned. One per sub-table.
- **Thread pin**: thread `i` to CPU `i % ncpu` (`--pin 0` to disable).
- **Huge pages**: `MADV_HUGEPAGE` on anonymous maps.
- **Zipf**: Vose alias table, keys in `1..distinct` (so the HashMap zero-key cell is unused). `s=0` is uniform via unbiased 64-bit reduction.
- **Workload identity**: one generated array per `s`, reused across designs, thread counts, and reps.

## Build and run

```bash
cd utils/join_hash_table_build_bench
make
./join_ht_build_bench                  # default sweep, sized for contention
./join_ht_build_bench --quick          # smoke
./join_ht_build_bench --csv out.csv --s 0,1.5 --threads 1,8,96
```

Needs C++20, pthread, and hardware CRC32C (x86-64 SSE4.2 or aarch64 +crc). `-march=native` is enough on Graviton.

## Sample run (96-core Neoverse-V2, 2026-08-29)

Default sweep: 32e6 rows, 8e6 distinct keys, 5 timed reps after 1 warmup. Full log:
`sample-run-graviton-neoverse-v2-96c.log`.

s=0 speedup vs 1 thread (Figure 12 shape check; unchained flattens earlier, chained keeps taking threads):

| threads | unchained | chained |
| ---: | ---: | ---: |
| 1 | 1.00 | 1.00 |
| 8 | 7.16 | 5.56 |
| 24 | 15.07 | 16.23 |
| 64 | 20.58 | 44.31 |
| 96 | 17.31 | 63.08 |

`ch-mutex` latch share at 96 threads vs Zipf `s` (the uniform ~5–10% figure, then skew):

| s | latch_pct | imbalance (max/mean sub-table rows) |
| ---: | ---: | ---: |
| 0.00 | 9.80 | 1.06 |
| 0.50 | 5.29 | 1.04 |
| 1.00 | 52.58 | 16.19 |
| 1.25 | 77.20 | 56.89 |
| 1.50 | 84.50 | 98.06 |

At s=0, 96 threads, `ch-mutex` 55.0 ms (53.8–71.8) vs `ch-spin` 57.8 ms (53.3–77.1): the intervals overlap, so that is not a result. Under heavy skew both latches serialize on the hot sub-table; spin is noisier, not faster.
