#!/usr/bin/env python3
"""Emit calibration.json from the final ladder plan + measurements.jsonl.

Expected map bytes come straight from measurements.jsonl for "measured"
entries; "model" entries use exact plateau values (regime empirically
confirmed at neighbouring points); "analytic-extrapolation" (S5) values are
plateau + fitted smooth terms, one grower step beyond the observed range.
"""

import json

MiB = 1024 * 1024
GiB = 1024 * MiB
TARGETS = {"S1": MiB, "S2": 32 * MiB, "S3": GiB, "S4": 4 * GiB, "S5": 16 * GiB}
TOL = 0.30

meas = {}
measured_points = []
for line in open("measurements.jsonl"):
    r = json.loads(line)
    meas[(r["family"], r["D"])] = r
    measured_points.append({
        "family": r["family"], "build_rows": r["D"],
        "map_bytes": r["map_bytes"], "peak_memory": r["peak_memory"],
        "table_bytes": r["table_bytes"], "build_us": r["build_us"],
    })


def m(fam, d):
    return meas[(fam, d)]["map_bytes"]


# str/strzero S5: 16 GiB of cells + the smooth extra extrapolated at the S4 rate.
str_extra_rate = (m("str", 48000000) - 4 * GiB) / 48000000.0
str_s5 = int(16 * GiB + str_extra_rate * 192000000)

# lcstr S5: linear fit through the two widest-spaced calibration points.
lc_slope = (m("lcstr", 85000000) - m("lcstr", 500000)) / (85000000 - 500000)
lc_icept = m("lcstr", 500000) - lc_slope * 500000
lcstr_s5 = int(lc_icept + lc_slope * 340000000)

# family -> size -> (build_rows, expected_map_bytes, source)
LADDER = {
    "key32": {
        "S1": (24000, m("key32", 24000), "measured"),
        "S2": (260000, m("key32", 260000), "measured"),
        "S3": (24000000, m("key32", 24000000), "measured"),
        "S4": (96000000, 4 * GiB, "model"),
        "S5": (384000000, 16 * GiB, "analytic-extrapolation"),
    },
    "key64": {
        "S1": (24000, m("key64", 24000), "measured"),
        "S2": (260000, m("key64", 260000), "measured"),
        "S3": (24000000, m("key64", 24000000), "measured"),
        "S4": (96000000, m("key64", 96000000), "measured"),
        "S5": (384000000, 16 * GiB, "analytic-extrapolation"),
    },
    "null64": {
        "S1": (27000, m("null64", 27000), "measured"),
        "S2": (290000, m("null64", 290000), "measured"),
        "S3": (26700000, m("null64", 26700000), "measured"),
        "S4": (107000000, 4 * GiB, "model"),
        "S5": (427000000, 16 * GiB, "analytic-extrapolation"),
    },
    "str": {
        "S1": (24000, m("str", 24000), "measured"),
        "S2": (260000, m("str", 260000), "measured"),
        "S3": (12000000, m("str", 12000000), "measured"),
        "S4": (48000000, m("str", 48000000), "measured"),
        "S5": (192000000, str_s5, "analytic-extrapolation"),
    },
    "strzero": {
        "S1": (24000, m("strzero", 24000), "measured"),
        "S2": (260000, m("strzero", 260000), "measured"),
        "S3": (12000000, m("strzero", 12000000), "measured"),
        "S4": (48000000, m("str", 48000000), "model"),
        "S5": (192000000, str_s5, "analytic-extrapolation"),
    },
    "fixstr": {
        "S1": (24000, m("fixstr", 24000), "measured"),
        "S2": (260000, m("fixstr", 260000), "measured"),
        "S3": (12000000, m("fixstr", 12000000), "measured"),
        "S4": (48000000, 3 * GiB, "model"),
        "S5": (192000000, 12 * GiB, "analytic-extrapolation"),
    },
    "k128": {
        "S1": (24000, m("k128", 24000), "measured"),
        "S2": (260000, m("k128", 260000), "measured"),
        "S3": (12000000, m("k128", 12000000), "measured"),
        "S4": (48000000, 3 * GiB, "model"),
        "S5": (192000000, 12 * GiB, "analytic-extrapolation"),
    },
    "k256": {
        "S1": (24000, m("k256", 24000), "measured"),
        "S2": (260000, m("k256", 260000), "measured"),
        "S3": (12000000, m("k256", 12000000), "measured"),
        "S4": (48000000, 5 * GiB, "model"),
        "S5": (192000000, 20 * GiB, "analytic-extrapolation"),
    },
    "mixed": {
        "S1": (24000, m("mixed", 24000), "measured"),
        "S2": (260000, m("mixed", 260000), "measured"),
        "S3": (12000000, m("mixed", 12000000), "measured"),
        "S4": (48000000, 3 * GiB, "model"),
        "S5": (192000000, 12 * GiB, "analytic-extrapolation"),
    },
    "lcstr": {
        "S1": (100000, m("lcstr", 100000), "measured"),
        "S2": (500000, m("lcstr", 500000), "measured"),
        "S3": (21000000, m("lcstr", 21000000), "measured"),
        "S4": (85000000, m("lcstr", 85000000), "measured"),
        "S5": (340000000, lcstr_s5, "analytic-extrapolation"),
    },
}

GENERATORS = {
    "key32": "toUInt32(number) AS k",
    "key64": "number AS k",
    "str": "concat(lpad(hex(number), 16, '0'), lpad(toString(number % 100000000), 8, '0')) AS k",
    "strzero": "concat(unhex('00'), lpad(hex(number), 16, '0'), lpad(toString(number % 10000000), 7, '0')) AS k",
    "fixstr": "toFixedString(lpad(hex(number), 16, '0'), 16) AS k",
    "k128": "number AS k1, bitXor(number, 12345) AS k2",
    "k256": "number AS k1, number + 1 AS k2, number + 2 AS k3, number + 3 AS k4",
    "null64": "if(number % 10 = 0, NULL, number) AS k",
    "lcstr": "toLowCardinality(concat('s', lpad(toString(number % 100000), 6, '0'))) AS k",
    "mixed": "toUInt32(number) AS k1, concat(lpad(hex(number), 16, '0'), lpad(toString(number % 100000000), 8, '0')) AS k2",
}

out = {}
for fam, sizes in LADDER.items():
    fam_out = {}
    for s, (rows, exp_bytes, source) in sizes.items():
        dev = exp_bytes / TARGETS[s] - 1.0
        fam_out[s] = {
            "build_rows": rows,
            "dup": 1,
            "expected_map_bytes": int(exp_bytes),
            "deviation_vs_target": round(dev, 4),
            "in_tolerance": abs(dev) <= TOL,
            "source": source,
        }
    out[fam] = fam_out

out["meta"] = {
    "method": (
        "Aggregate hash-map bytes = HashJoin::getTotalByteCount of the single data-carrying "
        "slot after ConcurrentHashJoin's build-finish merge (parsed from the per-shard "
        "'Join data is built' trace lines of binary a05f3ee81ff), cross-checked against "
        "query_log.memory_usage. With a count()-only probe no right-table columns are stored, "
        "so this equals map buffers + join pool. Structure on this binary: build scatters over "
        "32 slots (max_threads=32), then merges into ONE two-level map (256 buckets, "
        "min 256 cells/bucket, max load factor 0.5, grower steps x4 per resize up to 64Ki "
        "cells/bucket and x2 above). Cell sizes (measured): 16 B (key32/key64/null64), "
        "24 B (k128/mixed/fixstr; FixedString(16) packs into keys128), 32 B (str/strzero/lcstr; "
        "string keys are NOT arena-copied), 40 B (k256). Targets interpreted as binary units "
        "(1 MiB / 32 MiB / 1 GiB / 4 GiB / 16 GiB). build_rows sit mid-plateau of the bucket "
        "resize staircase for robustness."
    ),
    "binary": "clickhouse-baseline-a05f3ee81ff.bin",
    "binary_sha256": "0d32ef1c96e6d378aa20d3ab3063b1dfede0753c075ee94781ba5ec2779d88f4",
    "server_version": "26.8.1.1 (GIT_HASH a05f3ee81ff8411759637fa367aad62e72726e71)",
    "probe_settings": (
        "join_algorithm='parallel_hash', max_threads=32, collect_hash_table_stats_during_joins=0, "
        "max_bytes_before_external_join=0, max_bytes_ratio_before_external_join=0, "
        "parallel_hash_join_threshold=1"
    ),
    "generators": GENERATORS,
    "tolerance": TOL,
    "measured_points": measured_points,
    "caveats": [
        "Achievable aggregate map sizes are quantized to the bucket-resize staircase (x4 steps below 64Ki cells/bucket, x2 above); a target between plateaus cannot be hit for fixed-width keys with dup=1.",
        "S1 (1 MiB) is reachable only for key32/key64/null64 (floor = 256 buckets x 256 cells). Floors: 1.5 MiB for 24 B cells (+50%), ~2 MiB for 32 B string cells (+~105%), 2.5 MiB for k256 (+150%), ~14 MiB for lcstr at its 100k-distinct minimum (+1300%). Their S1 entries are the closest achievable point, OUT of tolerance.",
        "S2 (32 MiB) is out of tolerance for key32/key64/null64: only 16 MiB (-50%) or 64 MiB (+100%) exist; the ladder uses 16 MiB.",
        "fixstr/k128/mixed sit at -25% and k256 at +25% for S2..S5 (nearest plateaus); within tolerance but systematically offset.",
        "lcstr bytes are the join-reported counter, which includes per-block LowCardinality dictionary copies referenced from stored blocks and shared with the source table; incremental process memory is roughly half (e.g. peak 592 MiB vs 1019 MiB reported at S3). Model: bytes ~= 9.2 MiB + 50.4 B/row (linear fit residual <1% over 0.5M..85M rows). SHAKY family: which counter matters depends on what the benchmark measures.",
        "lcstr build_rows are total rows (distinct keys fixed at 100k by the family definition); rows beyond 100k append to RowRefList batches, dup field stays 1 (no extra multiplier).",
        "S5 is analytic extrapolation: one grower step (x2) beyond the largest empirically observed bucket size (2^20 cells); the x2-step regime was observed at 2^17..2^20, so risk is low. No S5 point was run.",
        "str/strzero carry a small smooth extra of ~1.0-1.8 B/row on top of cell plateaus (included in expected_map_bytes for measured points; S5 uses the S4 rate, uncertainty <2% of total).",
        "During the build phase all 32 per-slot two-level skeletons are transiently allocated (~32 MiB floor for 16 B cells, up to ~78 MiB for 40 B cells) and freed after the merge; query peak memory can therefore never be below ~34 MiB regardless of D.",
        "Peak query memory exceeds map bytes by insert/scatter transients (e.g. str S4: peak 6.6 GiB vs map 4.06 GiB); size external-memory/limit settings from peak, not map bytes.",
        "Calibration holds for count()-style probes where no right-table columns are output; selecting right columns adds stored-block bytes on top.",
        "This binary predates the two-level-machinery removal at branch HEAD (commits 69bf5c26c9f/0d06aaf2933); binaries built at HEAD have a different (pure per-slot scatter) layout and need recalibration.",
    ],
}

with open("calibration.json", "w") as f:
    json.dump(out, f, indent=2)

print("wrote calibration.json:", len(LADDER), "families,", len(measured_points), "measured points")
print(f"str S5 = {str_s5} ({str_s5/GiB:.3f} GiB), lcstr S5 = {lcstr_s5} ({lcstr_s5/GiB:.3f} GiB)")
print(f"lcstr fit: {lc_icept/MiB:.2f} MiB + {lc_slope:.2f} B/row")
for fam, sizes in out.items():
    if fam == "meta":
        continue
    row = [f"{fam:8s}"]
    for s in ("S1", "S2", "S3", "S4", "S5"):
        e = sizes[s]
        flag = "" if e["in_tolerance"] else "*"
        row.append(f"{s}={e['build_rows']:>9d} ({e['expected_map_bytes']/MiB:8.1f}MiB {e['deviation_vs_target']:+6.1%}{flag})")
    print(" ".join(row))
