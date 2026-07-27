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

The 9 blocks transcribe MATRIX.md's "Measured subset" table, which is the
authoritative freeze -- on any conflict MATRIX.md wins and THIS file gets
fixed (MATRIX.md's own rule). Block counts 27/6/12/24/14/4/4/1/2 = 94
distinct cells, assert-checked per block and in total. Block 4 lists the
semi_anti group twice (LEFT SEMI and LEFT ANTI); the ANTI instantiation is
the `.anti` cell modifier. Hash in-band: {key64,str,k256} x {S2,S4} x T{1,96}.
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


# Expected cell count per MATRIX.md block, in table order.
MATRIX_MD_BLOCK_COUNTS = (27, 6, 12, 24, 14, 4, 4, 1, 2)


def measured_blocks() -> list[dict]:
    # Every block below is a 1:1 transcription of a MATRIX.md table row;
    # rationale strings are that row's rationale column.
    blocks = []

    # 1. Probe core grid (27): 9 families (no strzero: PARITY-ONLY by
    # disposition) x {S2,S3,S5} x T96, probe side, inner_all.
    cells = [
        _cell(f, "probe", "inner_all", s, 96)
        for f in ("key32", "key64", "str", "fixstr", "k128", "k256", "null64", "lcstr", "mixed")
        for s in ("S2", "S3", "S5")
    ]
    blocks.append({
        "name": "probe_core_grid",
        "rationale": "family x map-residency are the first-order AMAC axes; T96 is the headline",
        "cells": cells,
    })

    # 2. Size-ladder completion (6).
    cells = [_cell(f, "probe", "inner_all", s, 96)
             for f in ("key64", "str", "k256") for s in ("S1", "S4")]
    blocks.append({
        "name": "size_ladder",
        "rationale": "full 5-point ladder on 3 sentinel families locates the engagement knee",
        "cells": cells,
    })

    # 3. Thread ladder (12).
    cells = [_cell(f, "probe", "inner_all", s, t)
             for f in ("key64", "str", "k256") for s in ("S2", "S4") for t in (1, 48)]
    blocks.append({
        "name": "thread_ladder",
        "rationale": "ring + ordered-probe costs scale with lanes; T1 isolates single-slot",
        "cells": cells,
    })

    # 4. Kind/strictness (24): six instantiation rows -- the semi_anti group
    # appears twice, as LEFT SEMI (plain) and LEFT ANTI (.anti modifier).
    # MATRIX.md labels the rf_all point RIGHT ALL and the any point INNER ANY;
    # fleet_ab instantiates rf_all as FULL JOIN (superset of RIGHT) and any as
    # ANY LEFT JOIN -- the documented decisions at fleet_ab.GROUP_JOIN_CLAUSE.
    variants = (("left_all", False), ("rf_all", False), ("any", False),
                ("semi_anti", False), ("semi_anti", True), ("asof", False))
    cells = [
        _cell(f, "probe", g, s, 96, anti=anti)
        for g, anti in variants
        for f in ("key64", "str")
        for s in ("S2", "S4")
    ]
    blocks.append({
        "name": "kind_strictness",
        "rationale": "one measured point per instantiation group per sentinel family per "
                     "residency class",
        "cells": cells,
    })

    # 5. Build side (14).
    cells = [_cell(f, "build", "inner_all", s, 96)
             for f in ("key64", "str", "k256", "mixed") for s in ("S2", "S3", "S5")]
    cells += [_cell("key64", "build", "inner_all", "S3", t) for t in (1, 48)]
    blocks.append({
        "name": "build",
        "rationale": "build events must not regress; kind is second-order on build (shared "
                     "insert path)",
        "cells": cells,
    })

    # 6. Duplicate-heavy build (4): dup=16, build side, {inner_all,left_all}.
    cells = [_cell(f, "build", g, "S3", 96, dup16=True)
             for f in ("key64", "str") for g in ("inner_all", "left_all")]
    blocks.append({
        "name": "dup_heavy",
        "rationale": "duplicate chains change ring occupancy and `RowRefList` appends",
        "cells": cells,
    })

    # 7. Hit-rate (4): h in {0.5, 0.05}.
    cells = [_cell(f, "probe", "inner_all", "S3", 96, hit_pct=p)
             for f in ("key64", "str") for p in (50, 5)]
    blocks.append({
        "name": "hit_rate",
        "rationale": "miss-dominated probes stress the ring differently",
        "cells": cells,
    })

    # 8. join_use_nulls=1 (1).
    blocks.append({
        "name": "join_use_nulls",
        "rationale": "nullable output path interacts with ordered gather",
        "cells": [_cell("key64", "probe", "left_all", "S3", 96, jun=True)],
    })

    # 9. Stats-on sensitivity (2).
    blocks.append({
        "name": "stats_on",
        "rationale": "protocol-sensitivity check for the stats-off measurement decision",
        "cells": [
            _cell("key64", "build", "inner_all", "S3", 96, statson=True),
            _cell("key64", "probe", "inner_all", "S3", 96, statson=True),
        ],
    })

    return blocks


def measured_cell_ids() -> list[str]:
    blocks = measured_blocks()
    for block, want in zip(blocks, MATRIX_MD_BLOCK_COUNTS):
        assert len(block["cells"]) == want, \
            f"block {block['name']}: {len(block['cells'])} cells, MATRIX.md says {want}"
    ids: list[str] = []
    for block in blocks:
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
