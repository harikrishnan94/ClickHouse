#!/bin/bash
# Re-run every gate in this campaign from a clean shell and print each exit code.
#
# The scorer changed after the first gate runs (TSV whitespace flattening, the
# --fail-on-order-effect and --expect-unit-set-seen flags, NO-VERDICT surfacing
# for units the harness abandons), so the recorded exit codes have to be
# re-established against the final scorer rather than inherited. Any line whose
# exit code differs from the one in REPORT.md is a defect.
#
# Usage: rerun_all_gates.sh   (from the campaign directory)
set -u
cd "$(dirname "$0")"
BASE=$(cat bins.arm 2>/dev/null || echo bins/clickhouse-baseline-a05f3ee81ff.bin)

pass=0; fail=0
run() {
    local label=$1; shift
    local expect=$1; shift
    "$@" > tmp/gate_out.txt 2>&1
    local rc=$?
    if [ "$rc" = "$expect" ]
    then printf 'ok   %-58s exit %s (expected %s)\n' "$label" "$rc" "$expect"; pass=$((pass+1))
    else printf 'FAIL %-58s exit %s (expected %s)\n' "$label" "$rc" "$expect"; fail=$((fail+1))
         sed -n '1,12p' tmp/gate_out.txt | sed 's/^/       /'
    fi
}
mkdir -p tmp

S="python3 probe_ab_report.py"
FA="results/fleet_abba/results.shard*.jsonl"
FB="results/fleet_baab/results.shard*.jsonl"

echo "=== Unit 0 ==="
run "G0-a fleet A/A (all TIE both metrics)"        0 $S --results "results/aa_fleet/results.shard*.jsonl" --arm-a aaA --arm-b aaB --metric both --aa-control --quiet-report
run "G0-a power: --band-override 0 must go RED"    1 $S --results "results/aa_fleet/results.shard*.jsonl" --arm-a aaA --arm-b aaB --metric both --aa-control --band-override 0 --quiet-report
run "G0-a jbmt synthetic 5-run (RED)"              1 $S --results results/aa_jbmt/results.jsonl --arm-a aaA --arm-b aaB --metric both --aa-control --quiet-report
run "G0-a jbmt synthetic 11-run (RED)"             1 $S --results results/aa_jbmt11/results.jsonl --arm-a aaA --arm-b aaB --metric both --aa-control --quiet-report
run "G0-a jbmt synthetic 11-run swapped (RED)"     1 $S --results results/aa_jbmt11_swap/results.jsonl --arm-a aaA --arm-b aaB --metric both --aa-control --quiet-report
run "G0-a jbmt synthetic 11-run quiet (RED)"       1 $S --results results/aa_jbmt11_quiet/results.jsonl --arm-a aaA --arm-b aaB --metric both --aa-control --quiet-report
run "G0-a jbmt real 9005/9007 (GREEN)"             0 $S --results results/aa_real/results.jsonl --arm-a aaA --arm-b aaB --metric both --aa-control --quiet-report
run "G0-a jbmt real measured pair 9005/9006"       0 $S --results results/aa_real_pair/results.jsonl --arm-a aaA --arm-b aaB --metric both --aa-control --quiet-report
run "G0-b decomposition, fleet ABBA"               0 $S --results "$FA" --arm-a baseline --arm-b candidate --check-decomposition --quiet-report
run "G0-b decomposition, fleet BAAB"               0 $S --results "$FB" --arm-a baseline --arm-b candidate --check-decomposition --quiet-report
run "G0-b decomposition, real tier a"              0 $S --results results/jbmt_real_a/results.jsonl --arm-a baseline --arm-b candidate --check-decomposition --quiet-report
run "G0-b decomposition, real tier b"              0 $S --results results/jbmt_real_b/results.jsonl --arm-a baseline --arm-b candidate --check-decomposition --quiet-report
run "G0-c only parallel_hash, fleet ABBA"          0 $S --results "$FA" --arm-a baseline --arm-b candidate --check-path-event --quiet-report
run "G0-c only parallel_hash, fleet BAAB"          0 $S --results "$FB" --arm-a baseline --arm-b candidate --check-path-event --quiet-report
run "G0-c only parallel_hash, real tier a"         0 $S --results results/jbmt_real_a/results.jsonl --arm-a baseline --arm-b candidate --check-path-event --quiet-report
run "G0-c only parallel_hash, real tier b"         0 $S --results results/jbmt_real_b/results.jsonl --arm-a baseline --arm-b candidate --check-path-event --quiet-report

echo "=== Unit 1 ==="
run "G1 coverage ABBA (RED at 78/94)"              1 $S --results "$FA" --arm-a baseline --arm-b candidate --metric both --expect-cells 94 --quiet-report
run "G1 coverage BAAB (RED at 78/94)"              1 $S --results "$FB" --arm-a baseline --arm-b candidate --metric both --expect-cells 94 --quiet-report
run "G1-set measured cells == 94-cell plan, ABBA"  0 $S --results "$FA" --arm-a baseline --arm-b candidate --metric both --expect-unit-set reports/fleet_plan94.json:cell --expect-unit-set-seen --quiet-report
run "G1-set measured cells == 94-cell plan, BAAB"  0 $S --results "$FB" --arm-a baseline --arm-b candidate --metric both --expect-unit-set reports/fleet_plan94.json:cell --expect-unit-set-seen --quiet-report
run "G1-b order effect (empty list, exit 0)"       0 $S --results "$FA" --compare-order "$FB" --arm-a baseline --arm-b candidate --metric both --quiet-report
run "G1-b enforced (--fail-on-order-effect)"       0 $S --results "$FA" --compare-order "$FB" --arm-a baseline --arm-b candidate --metric both --fail-on-order-effect --quiet-report

echo "=== Unit 3 ==="
run "G3 coverage tier a (RED at 368/376)"          1 $S --results results/jbmt_real_a/results.jsonl --arm-a baseline --arm-b candidate --metric both --expect-cells 376 --quiet-report
run "G3 coverage tier b (RED at 365/376)"          1 $S --results results/jbmt_real_b/results.jsonl --arm-a baseline --arm-b candidate --metric both --expect-cells 376 --quiet-report

echo "=== scorer power ==="
run "scorer_selftest (every gate can go red)"      0 python3 scorer_selftest.py

echo
echo "GATE RERUN SUMMARY: $pass as expected, $fail unexpected"
[ "$fail" = 0 ] || exit 1
