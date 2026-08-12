#!/usr/bin/env bash
# Create every synthetic table the map-family measurements need: m_D4, m_D8, m_D9, m_D13, m_D16.
#
# Idempotent. A table whose row count already matches is left alone; anything else is truncated and
# refilled. Run this once and the five measurement scripts can assume the data is there -- though
# each of them calls the same ensure_synth_maps itself, so none of them depends on this having run.
#
# Built with clickhouse-baseline, the merge-base build. The branch build reads parts written by the
# merge base; the reverse is not guaranteed, and both arms read these tables.
#
# Only one script in this directory may run at a time: stop_server kills everything in the
# uhj_versions_bench cgroup, and the lock in _maps_common.sh enforces it.
#
# Environment knobs (all optional):
#   D9_PROBE_ROWS   D9_BUILD_ROWS   LC_PROBE_ROWS   LC_BUILD_ROWS   LC_SWEEP_ROWS   KEYS_BUILD_ROWS
set -euo pipefail

# shellcheck source=_maps_common.sh
. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_maps_common.sh"

maps_take_lock
maps_init setup

trap 'stop_server' EXIT

start_server baseline
ensure_synth_maps

echo
echo "=== ${SYNTH_DB} contents ========================================================="
client --query "
    SELECT table, sum(rows) AS rows, formatReadableSize(sum(bytes_on_disk)) AS on_disk
    FROM system.parts WHERE active AND database = '${SYNTH_DB}'
    GROUP BY table ORDER BY table FORMAT PrettyCompactMonoBlock" || true
client --query "
    SELECT formatReadableSize(sum(bytes_on_disk)) AS total_on_disk
    FROM system.parts WHERE active AND database = '${SYNTH_DB}' FORMAT TSV"

# The dictionary a LowCardinality column actually carries at read time is what D8, D13 and D16 all
# scale with, and it is a property of the part rather than of the INSERT. Print it so the runner can
# confirm that OPTIMIZE FINAL left one part per table with one dictionary of the intended size,
# rather than several parts whose dictionaries each hold a fraction of the values.
echo
echo "=== LowCardinality dictionary sizes actually stored ============================="
client --query "
    SELECT table, count() AS parts, sum(rows) AS rows
    FROM system.parts WHERE active AND database = '${SYNTH_DB}' AND table LIKE 'lc\\_%'
    GROUP BY table ORDER BY table FORMAT PrettyCompactMonoBlock" || true
for t in lc_probe_d16 lc_probe_d1k lc_probe_d100k lc_probe_d1m \
         lc_sweep_d1k lc_sweep_d10k lc_sweep_d100k lc_sweep_d1m \
         lc_build_w16_d1k lc_build_w48_d1k; do
    printf '%-20s uniqExact(k) = %s\n' "${t}" \
        "$(client --query "SELECT uniqExact(k) FROM ${SYNTH_DB}.${t}" 2>/dev/null || echo '?')"
done

realworld_report "${OUT}/realworld_exposure.txt"
echo
cat "${OUT}/realworld_exposure.txt"

stop_server
trap - EXIT
echo
echo "SETUP_SYNTH_DONE"
