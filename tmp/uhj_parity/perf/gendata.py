#!/usr/bin/env python3
"""Create the benchmark tables once, so no sweep run pays generation cost.

Schema, shared by build and probe tables:
    k  UInt64   the fixed-width join key
    s  String   'k<k>', so a string join matches exactly when the u64 join does
    a  UInt64   intDiv(k, 1000)   } together a two-column composite key that is
    b  UInt64   k % 1000          } unique exactly when k is
    v  UInt64   payload, so the join has a right column to gather

Every key column is a function of `k`, which is what makes the match rate mean
the same thing on all three key-type axes: one probe row either matches on all of
them or on none.

Build tables have UNIQUE keys, so cardinality == row count == hash table size, and
a probe row can match at most one build row.

The keys are deliberately NOT the dense range 0..N-1. `canConvertToFixedHashMap`
(UnifiedHashJoin/HashJoin.cpp:2077) converts a `key32`/`key64` map to a
direct-addressed fixed map when the key range fits in 2^18, and
`enable_join_fixed_hash_table_conversion` defaults to true. Dense keys would
therefore have turned every small-cardinality cell into a comparison of two
direct-addressed arrays while it was labelled a hash-join comparison, and left the
small-cardinality hash path with no coverage at all. Multiplying by an odd 64-bit
constant spreads the keys over the whole `UInt64` range, so no conversion can fire
at any cardinality. See WORKLOG E4; the conversion itself is measured separately
as the `dense` key type, because it is a real place `unified_hash` wins.

Probe tables mix matching and non-matching rows at a fixed rate. Matching keys are
scattered by an odd multiplier rather than taken in order, so the probe walks the
hash table randomly; sequential probe keys would flatter every cache and hide
exactly the differences this mission is looking for.
"""

from __future__ import annotations

import sys
import time

from harness import CARDS, MATCH_RATES, PROBE_ROWS, ROWS_PER_KEY, run_query, scalar

# Odd 64-bit multiplier (Knuth). Used twice, for two different jobs:
#  - SPREAD turns a dense ordinal into a key scattered over the whole UInt64
#    range, which is what keeps the fixed-hash-map conversion from firing.
#  - the same constant scatters which ordinals a probe row matches, so the probe
#    walks the hash table in random order rather than sequentially. Sequential
#    probe keys would flatter every cache and hide the differences being measured.
SCATTER = 6364136223846793005

DDL_COLS = "k UInt64, s String, a UInt64, b UInt64, v UInt64"

def probe_rows_for(card_name: str) -> int:
    """small/medium run only at 1 thread; large runs at 16/64."""
    return PROBE_ROWS[1] if card_name in ("small", "medium") else PROBE_ROWS[16]


def key_expr(ordinal: str) -> str:
    """Map a dense ordinal to the actual join key.

    Spreading over the whole UInt64 range is what stops
    `canConvertToFixedHashMap` from turning a hash-join comparison into a
    direct-addressed-array comparison at small cardinality (WORKLOG E4).
    """
    return f"({ordinal}) * {SCATTER}"


def derived(kexpr: str, vexpr: str = "k * 7 + 1") -> str:
    """All key columns are functions of `k`, which is what makes the match rate
    mean the same thing on the u64, string and composite axes."""
    return (f"{kexpr} AS k, concat('k', toString(k)) AS s, "
            f"intDiv(k, 1000) AS a, k % 1000 AS b, {vexpr} AS v")


def create_build(card_name: str, card_rows: int) -> None:
    """Build table: `card_rows` distinct keys, ROWS_PER_KEY rows each.

    `ordinal = intDiv(number, ROWS_PER_KEY)` makes each key appear exactly
    ROWS_PER_KEY times while `v` still differs per row, so the rows are
    distinguishable and MapsAll actually chains.
    """
    t = f"b_{card_name}"
    total = card_rows * ROWS_PER_KEY
    ordinal = f"intDiv(number, {ROWS_PER_KEY})"
    run_query(f"DROP TABLE IF EXISTS {t}")
    run_query(f"CREATE TABLE {t} ({DDL_COLS}) ENGINE = MergeTree ORDER BY tuple()")
    run_query(
        f"INSERT INTO {t} SELECT {derived(key_expr(ordinal), 'number')} "
        f"FROM numbers_mt({total})",
        {"max_insert_threads": 32, "max_threads": 32},
    )


def create_probe(card_name: str, card_rows: int, match_name: str, rows: int) -> None:
    t = f"p_{card_name}_{match_name}"
    rate = MATCH_RATES[match_name]
    # `number % 10 < rate*10` gives an exact, reproducible match fraction.
    # Matching rows pick a build ordinal pseudo-randomly (so the probe walks the
    # table in random order); non-matching rows pick an ordinal beyond the build
    # table's range. Both then go through the same ordinal -> key mapping, which is
    # what makes the match rate identical on the u64, string and composite axes.
    hits = int(round(rate * 10))
    ordinal = (f"if(number % 10 < {hits}, "
               f"(number * {SCATTER}) % {card_rows}, "
               f"{card_rows} + number)")
    kexpr = key_expr(ordinal)
    run_query(f"DROP TABLE IF EXISTS {t}")
    run_query(f"CREATE TABLE {t} ({DDL_COLS}) ENGINE = MergeTree ORDER BY tuple()")
    run_query(
        f"INSERT INTO {t} SELECT {derived(kexpr)} FROM numbers_mt({rows})",
        {"max_insert_threads": 32, "max_threads": 32},
    )


def main() -> int:
    plan = []
    for card_name, card_rows in CARDS.items():
        plan.append(("build", card_name, card_rows, None, card_rows))
        for match_name in MATCH_RATES:
            plan.append(("probe", card_name, card_rows, match_name,
                         probe_rows_for(card_name)))

    for what, card_name, card_rows, match_name, rows in plan:
        t0 = time.time()
        if what == "build":
            create_build(card_name, rows)
            name = f"b_{card_name}"
        else:
            create_probe(card_name, card_rows, match_name, rows)
            name = f"p_{card_name}_{match_name}"
        print(f"{name:24s} rows={rows:>12,}  {time.time() - t0:6.1f}s", flush=True)

    print("\n=== verification ===", flush=True)
    ok = True
    for card_name, card_rows in CARDS.items():
        got = int(scalar(f"SELECT count() FROM b_{card_name} FORMAT TSV"))
        uniq = int(scalar(f"SELECT uniqExact(k) FROM b_{card_name} FORMAT TSV"))
        # Asserting rows == card * ROWS_PER_KEY AND distinct == card is what keeps
        # the RightAny promotion from coming back unnoticed (WORKLOG E5.2).
        good = (got == card_rows * ROWS_PER_KEY and uniq == card_rows)
        ok &= good
        print(f"b_{card_name}: rows={got:,} distinct_k={uniq:,} "
              f"rows_per_key={got/uniq if uniq else 0:.1f} {'OK' if good else 'FAIL'}")
        probe_rows = probe_rows_for(card_name)
        for match_name, rate in MATCH_RATES.items():
            t = f"p_{card_name}_{match_name}"
            n = int(scalar(f"SELECT count() FROM {t} FORMAT TSV"))
            # Actual match fraction, measured against the build table rather than
            # assumed from the generating expression.
            m = int(scalar(
                f"SELECT count() FROM {t} AS l "
                f"WHERE l.k IN (SELECT k FROM b_{card_name}) FORMAT TSV"))
            frac = m / n if n else 0.0
            good = (n == probe_rows) and abs(frac - rate) < 0.01
            ok &= good
            print(f"{t}: rows={n:,} match_frac={frac:.4f} (target {rate}) "
                  f"{'OK' if good else 'FAIL'}")

    print(f"\nGENDATA_VERDICT={'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
