#!/usr/bin/env bash
# Gate 2 — formal syntax, safety, refinement, progress, and non-vacuity.
#
# Exits 0 only when ALL of the following hold:
#   1. SANY parses src/Interpreters/RadixHashJoin/WaveJoinProbe.tla cleanly.
#   2. TLC finds no error on the positive configurations:
#        MC_Normal     (PL>1, budget-sealed wave + EOF final partial wave)
#        MC_PL1        (PL=1, refine stages skipped by the same machine)
#        MC_MultiWave  (two budget-sealed waves, 3 workers, duplicate results)
#        MC_Fail       (two distinct-error probe faults that can race,
#                       scan/pre faults, external cancellation)
#        MC_CancelRace (cancellation racing NORMAL completion across two
#                       waves: Seal, barriers, CompleteWave, EOFSeal,
#                       FinishInput, second-wave drain with history)
#      Every positive configuration checks the full safety battery; the
#      completing configurations explicitly check FinalRefinement (verified
#      textually below) plus Termination and ParticipationLive.
#   3. The negative work-conservation witness MC_NoSteal (dedicated scanner
#      crew + leaf affinity) makes TLC report a TEMPORAL VIOLATION — the
#      participation property is falsifiable, not a tautology.
#   4. Two mutation witnesses on scratch copies of the spec:
#      - first-exception-wins: FaultProbe overwriting `primary`
#        UNCONDITIONALLY (last-exception-wins) must make TLC report a
#        violation of PrimaryStable on MC_Fail;
#      - budget admission: removing Reserve's `st.mem < BUDGET` guard must
#        make TLC report a violation of MemBound on MC_MultiWave (the one
#        config whose total input bytes exceed BUDGET + MaxBlockBytes —
#        established by the round-1 independent verifier).
#      Either mutant surviving means the property is vacuous: gate red.
#   5. The three reachability configurations FAIL as expected, proving the
#      cooperative states are actually exercised:
#        MC_ReachOwn      — a state where EVERY worker owns a drain job
#        MC_ReachInflight — two concurrent in-flight budget reservations
#        MC_ReachCross    — budget crossed while another lane is in flight
#
# Any other outcome (including a PASSING witness) exits nonzero.

set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
JAVA="${JAVA:-$HOME/.local/bin/java}"
JAR="$ROOT/tmp/tla-install/tla2tools-1.7.4.jar"
SPEC="$ROOT/src/Interpreters/RadixHashJoin/WaveJoinProbe.tla"
TLADIR="$ROOT/tmp/wave-join-cooperative/tla"
TIMEOUT_S="${TLC_TIMEOUT:-1800}"

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

fail_count=0
step() { printf '\n=== %s ===\n' "$1"; }
verdict() { # $1 = PASS|FAIL, $2 = label
    if [ "$1" = PASS ]; then printf -- '--> PASS: %s\n' "$2";
    else printf -- '--> FAIL: %s\n' "$2"; fail_count=$((fail_count + 1)); fi
}

[ -f "$JAR" ] || { echo "missing $JAR"; exit 2; }
[ -f "$SPEC" ] || { echo "missing $SPEC"; exit 2; }

cp "$SPEC" "$WORK/"
cp "$TLADIR"/MC_*.tla "$TLADIR"/MC_*.cfg "$WORK/"
cmp -s "$SPEC" "$WORK/WaveJoinProbe.tla" || { echo "work copy differs from production spec"; exit 2; }

step "SANY: parse + semantic analysis of WaveJoinProbe.tla"
sany_out="$WORK/sany.out"
(cd "$WORK" && "$JAVA" -cp "$JAR" tla2sany.SANY WaveJoinProbe.tla) >"$sany_out" 2>&1
sany_rc=$?
cat "$sany_out"
if [ $sany_rc -eq 0 ] && grep -q "Semantic processing of module WaveJoinProbe" "$sany_out" \
   && ! grep -Eq '\*\*\* Errors|Fatal errors' "$sany_out"; then
    verdict PASS "SANY"
else
    verdict FAIL "SANY"
fi

step "Textual: completing configurations explicitly check FinalRefinement"
fr_ok=1
for cfg in MC_Normal MC_PL1 MC_MultiWave MC_CancelRace; do
    grep -q '^[[:space:]]*FinalRefinement$' "$TLADIR/$cfg.cfg" || { echo "FinalRefinement missing from $cfg.cfg"; fr_ok=0; }
done
[ $fr_ok -eq 1 ] && verdict PASS "FinalRefinement is a checked invariant in MC_Normal, MC_PL1, MC_MultiWave, MC_CancelRace" \
                 || verdict FAIL "FinalRefinement missing from a completing configuration"

run_tlc() { # $1 = cfg name, $2 = root module, $3... = extra TLC flags
    local cfg="$1" mod="$2"
    shift 2
    local out="$WORK/$cfg.out"
    (cd "$WORK" && timeout "$TIMEOUT_S" "$JAVA" -XX:+UseParallelGC -Xmx8g -cp "$JAR" \
        tlc2.TLC -workers auto -config "$cfg.cfg" -metadir "$WORK/meta_$cfg" "$@" "$mod") >"$out" 2>&1
    echo "$out"
}

for cfg in MC_Normal MC_PL1 MC_MultiWave MC_Fail MC_CancelRace; do
    step "TLC positive: $cfg (expect: no error)"
    out="$(run_tlc "$cfg" "$cfg")"
    grep -E "Model checking completed|states generated|Error:|violated|Finished in" "$out" | head -20
    if grep -q "Model checking completed. No error has been found" "$out"; then
        verdict PASS "$cfg"
    else
        echo "---- full TLC output ($cfg) ----"; cat "$out"
        verdict FAIL "$cfg"
    fi
done

step "TLC negative witness: MC_NoSteal (expect: temporal violation of ParticipationLive)"
# -deadlock DISABLES TLC's deadlock detector for this run only: the broken
# eligibility wedges the drain (leaf 0 unclaimable, worker 1 idle forever),
# and with stuttering allowed the PARTICIPATION property itself — not the
# deadlock detector — must produce the counterexample.  The positive runs
# above keep deadlock checking on.
out="$(run_tlc MC_NoSteal MC_NoSteal -deadlock)"
grep -E "Temporal properties were violated|Model checking completed|Error:|states generated" "$out" | head -10
if grep -q "Temporal properties were violated" "$out"; then
    verdict PASS "MC_NoSteal produced the expected counterexample"
else
    echo "---- full TLC output (MC_NoSteal) ----"; cat "$out"
    verdict FAIL "MC_NoSteal did NOT fail as expected (participation property would be vacuous)"
fi

step "TLC mutation witness: last-exception-wins mutant (expect: PrimaryStable violated)"
# Mutate ONLY FaultProbe's primary update in a scratch copy: drop the
# first-wins guard so a second distinct-error fault overwrites primary.
# MC_Fail has two concurrently-ownable failing leaves with distinct errors
# (eL1 vs eOther), so the mutant must fail PrimaryStable; if it passes, the
# property is vacuous and this gate goes red.
MUT="$WORK/mutation"
mkdir -p "$MUT"
sed 's/!.primary = IF st.primary = NoError THEN ErrorOf(l) ELSE @,/!.primary = ErrorOf(l),/' \
    "$WORK/WaveJoinProbe.tla" > "$MUT/WaveJoinProbe.tla"
cp "$WORK/MC_Fail.tla" "$WORK/MC_Fail.cfg" "$MUT/"
if cmp -s "$WORK/WaveJoinProbe.tla" "$MUT/WaveJoinProbe.tla"; then
    verdict FAIL "mutation did not apply (sed pattern found nothing)"
else
    mut_out="$MUT/MC_Fail_mutant.out"
    (cd "$MUT" && timeout "$TIMEOUT_S" "$JAVA" -XX:+UseParallelGC -Xmx8g -cp "$JAR" \
        tlc2.TLC -workers auto -config MC_Fail.cfg -metadir "$MUT/meta" MC_Fail) >"$mut_out" 2>&1
    grep -E "Action property.*violated|Temporal properties were violated|Model checking completed|Error:" "$mut_out" | head -5
    if grep -Eq "Action property PrimaryStable is violated|Temporal properties were violated" "$mut_out"; then
        verdict PASS "last-exception-wins mutant fails PrimaryStable as expected"
    else
        echo "---- full TLC output (mutant) ----"; cat "$mut_out"
        verdict FAIL "mutant did NOT fail (PrimaryStable would be vacuous)"
    fi
fi

step "TLC mutation witness: budget-ignoring Reserve mutant (expect: MemBound violated)"
MUT2="$WORK/mutation2"
mkdir -p "$MUT2"
sed '/\/\\ st.mem < BUDGET$/d' "$WORK/WaveJoinProbe.tla" > "$MUT2/WaveJoinProbe.tla"
cp "$WORK/MC_MultiWave.tla" "$WORK/MC_MultiWave.cfg" "$MUT2/"
if cmp -s "$WORK/WaveJoinProbe.tla" "$MUT2/WaveJoinProbe.tla"; then
    verdict FAIL "budget mutation did not apply (sed pattern found nothing)"
else
    mut2_out="$MUT2/MC_MultiWave_mutant.out"
    (cd "$MUT2" && timeout "$TIMEOUT_S" "$JAVA" -XX:+UseParallelGC -Xmx8g -cp "$JAR" \
        tlc2.TLC -workers auto -config MC_MultiWave.cfg -metadir "$MUT2/meta" MC_MultiWave) >"$mut2_out" 2>&1
    grep -E "Invariant .* is violated|Model checking completed|Error:" "$mut2_out" | head -5
    if grep -q "Invariant MemBound is violated" "$mut2_out"; then
        verdict PASS "budget-ignoring mutant fails MemBound as expected"
    else
        echo "---- full TLC output (budget mutant) ----"; cat "$mut2_out"
        verdict FAIL "budget mutant did NOT fail (MemBound would be vacuous)"
    fi
fi

reach() { # $1 = cfg, $2 = invariant name
    step "TLC reachability: $1 (expect: invariant $2 violated)"
    local out
    out="$(run_tlc "$1" MC_Normal)"
    grep -E "Invariant .* is violated|Model checking completed|states generated" "$out" | head -5
    if grep -q "Invariant $2 is violated" "$out"; then
        verdict PASS "$1: cooperative state reachable"
    else
        echo "---- full TLC output ($1) ----"; cat "$out"
        verdict FAIL "$1: expected violation of $2 not observed"
    fi
}
reach MC_ReachOwn NeverFullOwnership
reach MC_ReachInflight NeverTwoInflight
reach MC_ReachCross NeverCrossWithInflight

step "Summary"
if [ $fail_count -eq 0 ]; then
    echo "ALL FORMAL CHECKS BEHAVED AS EXPECTED (positives green, witnesses red)."
    exit 0
else
    echo "$fail_count check(s) did not behave as expected."
    exit 1
fi
