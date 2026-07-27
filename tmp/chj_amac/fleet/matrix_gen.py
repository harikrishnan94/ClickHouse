#!/usr/bin/env python3
"""matrix_gen.py -- emit fleet/matrix.json: the FULL 1800-cell universe
(2 sides x 10 families x 6 groups x 5 sizes x 3 threads), each with the
disposition placeholder 'UNDISPOSITIONED', plus the 94-cell measured plan
(the approved plan's 9 blocks, encoded as data with a rationale per block)
and the 12 hash-inband cells as a separate list.

DECISION (documented for check_matrix): the universe is the 1800 BASE cells
only. Modifier cells (.dup16/.h50/.h05/.jun/.statson) and the 12 .hash
algo-override cells are NOT universe members -- they are auxiliary measured
evidence (they support INFERRED dispositions of base cells / gate the grower
change) and are listed under measured_plan / hash_inband. check_matrix on
empty dispositions therefore prints '1800 undispositioned'.

Block counts follow the approved plan (mission-amac-zany-hammock.md, item 8):
probe grid 27; size ladder 6; thread ladder 12; kind/strictness 24; build 14;
dup-heavy 4; hit-rate 4; join_use_nulls 1; stats-on 2 -- total 94 distinct
cells (assert-checked). Hash in-band: {key64,str,k256} x {S2,S4} x T{1,96}.
"""

from __future__ import annotations

import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import fleet_ab  # noqa: E402

DEFAULT_OUT = pathlib.Path(__file__).resolve().parent / "matrix.json"


def _cell(family, side, group, size, threads, **mods) -> str:
    return fleet_ab.Cell(family, side, group, size, threads, **mods).cell_id


def measured_blocks() -> list[dict]:
    blocks = []

    # 1. probe grid (27): 9 of the 10 families (all but mixed, which the
    # thread ladder carries) x S3 x all threads, probe side, inner_all.
    cells = [
        _cell(f, "probe", "inner_all", "S3", t)
        for f in ("key32", "key64", "str", "strzero", "fixstr", "k128", "k256", "null64", "lcstr")
        for t in (1, 48, 96)
    ]
    blocks.append({
        "name": "probe_grid",
        "rationale": "Primary claim surface: probe-side AMAC lookup on the DRAM-resident S3 map "
                     "across every key family and the full thread axis.",
        "cells": cells,
    })

    # 2. size ladder (6): completes key64 to the full 5-size ladder at T96 and
    # anchors str at the extremes (S3 rungs come from the probe grid).
    cells = [_cell("key64", "probe", "inner_all", s, 96) for s in ("S1", "S2", "S4", "S5")]
    cells += [_cell("str", "probe", "inner_all", s, 96) for s in ("S1", "S5")]
    blocks.append({
        "name": "size_ladder",
        "rationale": "Map-residency ladder (L2 -> L3 -> DRAM): AMAC's win must grow with map "
                     "size; S1 bounds the small-map regression risk.",
        "cells": cells,
    })

    # 3. thread ladder (12): key64 off-S3 sizes at T1/T48 (completing the
    # 5x3 key64 grid) plus the mixed family's full thread axis.
    cells = [_cell("key64", "probe", "inner_all", s, t)
             for s in ("S1", "S2", "S4", "S5") for t in (1, 48)]
    cells += [_cell("mixed", "probe", "inner_all", "S3", t) for t in (1, 48, 96)]
    cells += [_cell("str", "probe", "inner_all", "S5", 48)]
    blocks.append({
        "name": "thread_ladder",
        "rationale": "Thread scaling off the S3 anchor: key64 becomes a full 5-size x 3-thread "
                     "grid; mixed (composite numeric+string key) gets its thread axis here.",
        "cells": cells,
    })

    # 4. kind/strictness (24): every non-inner group on the two anchor
    # families at S3; T1 sanity on the numeric family.
    cells = [
        _cell(f, "probe", g, "S3", t)
        for g in ("left_all", "rf_all", "any", "semi_anti", "asof")
        for f in ("key64", "str")
        for t in (48, 96)
    ]
    cells += [_cell("key64", "probe", g, "S3", 1)
              for g in ("left_all", "rf_all", "any", "semi_anti")]
    blocks.append({
        "name": "kind_strictness",
        "rationale": "The routed probe must not regress non-INNER kinds/strictnesses (LEFT/FULL/"
                     "ANY/SEMI/ASOF instantiations) at the size where AMAC engages.",
        "cells": cells,
    })

    # 5. build (14): build-side cells for the build-insert ring claim.
    cells = [_cell(f, "build", "inner_all", s, 96)
             for f in ("key64", "str", "k256", "fixstr") for s in ("S2", "S4")]
    cells += [_cell(f, "build", "inner_all", "S3", 48) for f in ("key64", "str")]
    cells += [_cell("key64", "build", "inner_all", s, 96) for s in ("S1", "S5")]
    cells += [_cell(f, "build", "inner_all", "S3", 96) for f in ("null64", "lcstr")]
    blocks.append({
        "name": "build",
        "rationale": "Build-insert ring claim: build-heavy cells (small probe) across key kinds "
                     "and sizes; S1/S5 bound the ladder, T48 checks mid-parallelism.",
        "cells": cells,
    })

    # 6. dup-heavy (4): 16x duplicated build keys -- chain-following stress
    # for ring disassembly (both sides).
    cells = [_cell(f, side, "inner_all", "S3", 96, dup16=True)
             for f in ("key64", "str") for side in ("probe", "build")]
    blocks.append({
        "name": "dup_heavy",
        "rationale": "Duplicate chains (16x) stress collision-chain walking in the AMAC ring "
                     "and RowRef list handling on both build and probe.",
        "cells": cells,
    })

    # 7. hit-rate (4): miss-dominated probes change the lookup's branch mix.
    cells = [_cell(f, "probe", "inner_all", "S3", 96, hit_pct=p)
             for f in ("key64", "str") for p in (50, 5)]
    blocks.append({
        "name": "hit_rate",
        "rationale": "h=0.5/0.05: misses terminate lookups early and shift the ring's "
                     "prefetch-to-work ratio; must not regress miss-heavy probes.",
        "cells": cells,
    })

    # 8. join_use_nulls (1): output-nullability flag on the nullable family.
    blocks.append({
        "name": "join_use_nulls",
        "rationale": "join_use_nulls=1 changes output column wrapping, not the lookup; one "
                     "sentinel cell proves in-band.",
        "cells": [_cell("null64", "probe", "inner_all", "S3", 96, jun=True)],
    })

    # 9. stats-on (2): the runtime-stats collector interacts with build sizing.
    blocks.append({
        "name": "stats_on",
        "rationale": "collect_hash_table_stats_during_joins=1 alters initial map sizing "
                     "(fewer growths); both sides sampled once.",
        "cells": [
            _cell("key64", "probe", "inner_all", "S3", 96, statson=True),
            _cell("key64", "build", "inner_all", "S3", 96, statson=True),
        ],
    })

    return blocks


def measured_cell_ids() -> list[str]:
    ids: list[str] = []
    for block in measured_blocks():
        ids.extend(block["cells"])
    unique = list(dict.fromkeys(ids))
    assert len(ids) == len(unique), "measured blocks overlap; the 94-cell count would be wrong"
    assert len(unique) == 94, f"measured plan must have exactly 94 cells, got {len(unique)}"
    for cid in unique:
        fleet_ab.parse_cell(cid)  # must round-trip the grammar
    return unique


def hash_inband_cell_ids() -> list[str]:
    # Requester decision 4: the tail-padded grower also changes plain `hash`;
    # 12 in-band A/B cells with join_algorithm='hash' on both arms.
    ids = [
        _cell(f, "probe", "inner_all", s, t, algo="hash")
        for f in ("key64", "str", "k256")
        for s in ("S2", "S4")
        for t in (1, 96)
    ]
    assert len(ids) == 12
    return ids


def universe_cell_ids() -> list[str]:
    ids = [
        _cell(f, side, g, s, t)
        for side in fleet_ab.SIDES
        for f in fleet_ab.FAMILIES
        for g in fleet_ab.GROUPS
        for s in fleet_ab.SIZES
        for t in fleet_ab.THREADS
    ]
    assert len(ids) == 1800, f"universe must be 1800 cells, got {len(ids)}"
    return ids


def build_matrix() -> dict:
    universe = universe_cell_ids()
    blocks = measured_blocks()
    measured = measured_cell_ids()
    hash_cells = hash_inband_cell_ids()
    return {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "generator": "tmp/chj_amac/fleet/matrix_gen.py",
        "notes": (
            "Universe = base cells only (no modifiers). Modifier and .hash cells are "
            "auxiliary measured evidence, not universe members; see module docstring. "
            "Dispositions live in fleet/dispositions.json (cell -> {disposition, evidence, "
            "from, rule}); allowed dispositions: MEASURED, INFERRED, PARITY-ONLY, "
            "EXCLUDED-INVALID, NOT-CLAIMED."
        ),
        "universe": {
            "axes": {
                "sides": list(fleet_ab.SIDES),
                "families": list(fleet_ab.FAMILIES),
                "groups": list(fleet_ab.GROUPS),
                "sizes": list(fleet_ab.SIZES),
                "threads": list(fleet_ab.THREADS),
            },
            "count": len(universe),
            "cells": [{"cell": c, "disposition": "UNDISPOSITIONED"} for c in universe],
        },
        "measured_plan": {
            "count": len(measured),
            "blocks": blocks,
            "cells": measured,
        },
        "hash_inband": {
            "rationale": "G-hash-inband gate: the tail-padded grower rebind also changes "
                         "join_algorithm='hash'; these 12 A/B cells (hash on BOTH arms) must "
                         "all verdict in-band (TIE).",
            "count": len(hash_cells),
            "cells": hash_cells,
        },
    }


def main() -> int:
    out = DEFAULT_OUT
    if len(sys.argv) > 1:
        out = pathlib.Path(sys.argv[1])
    matrix = build_matrix()
    out.write_text(json.dumps(matrix, indent=1) + "\n")
    print(f"wrote {out}")
    print(
        f"MATRIX_GEN RESULT: universe={matrix['universe']['count']} "
        f"measured={matrix['measured_plan']['count']} "
        f"hash_inband={matrix['hash_inband']['count']} -> OK"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
