# uhj-parity vs merge-base on ClickBench versions (emulated c7a.4xlarge)
Rounds compared: 1 (baseline files 1, uhj 1)
Noise rule: NO RESULT if |rel_delta| <= max(5%, 1 stdev / mean)

## coffeeshop
geomean baseline=4.7269s uhj=4.8614s delta=+2.84%
queries: 17; null/error=0; NO_RESULT=16; regressions=1; improvements=0
### Regressions (outside noise band)
- coffeeshop/q5: 2.3860s -> 2.5070s (+5.1%, noise±5.0%)

## tpch
geomean baseline=0.5742s uhj=0.4700s delta=-18.14%
queries: 22; null/error=1; NO_RESULT=17; regressions=3; improvements=1
### Regressions (outside noise band)
- tpch/q11: 0.1180s -> 0.2710s (+129.7%, noise±5.0%)
- tpch/q21: 1.7000s -> 3.8710s (+127.7%, noise±5.0%)
- tpch/q7: 0.4490s -> 0.6120s (+36.3%, noise±30.8%)
### Improvements (outside noise band)
- tpch/q8: 352.5910s -> 0.7120s (-99.8%, noise±5.0%)
### Null / failed queries (excluded from averages)
- tpch/q5

## tpcds
geomean baseline=0.2678s uhj=0.2782s delta=+3.85%
queries: 103; null/error=3; NO_RESULT=78; regressions=17; improvements=5
### Regressions (outside noise band)
- tpcds/q54: 0.3720s -> 27.6420s (+7330.6%, noise±10.5%)
- tpcds/q88: 0.0290s -> 0.1010s (+248.3%, noise±5.0%)
- tpcds/q32: 0.4140s -> 0.6670s (+61.1%, noise±5.5%)
- tpcds/q28: 0.3770s -> 0.4970s (+31.8%, noise±5.0%)
- tpcds/q84: 0.2990s -> 0.3550s (+18.7%, noise±5.0%)
- tpcds/q33: 0.0730s -> 0.0830s (+13.7%, noise±9.4%)
- tpcds/q45: 0.0170s -> 0.0190s (+11.8%, noise±5.0%)
- tpcds/q43: 0.0810s -> 0.0900s (+11.1%, noise±11.0%)
- tpcds/q1: 0.0990s -> 0.1090s (+10.1%, noise±5.4%)
- tpcds/q60: 0.1510s -> 0.1620s (+7.3%, noise±5.0%)
- tpcds/q3: 0.0280s -> 0.0300s (+7.1%, noise±5.0%)
- tpcds/q10: 0.2240s -> 0.2390s (+6.7%, noise±5.0%)
- tpcds/q48: 0.0150s -> 0.0160s (+6.7%, noise±5.0%)
- tpcds/q69: 0.3750s -> 0.3990s (+6.4%, noise±5.0%)
- tpcds/q56: 0.0470s -> 0.0500s (+6.4%, noise±5.0%)
- tpcds/q44: 0.0710s -> 0.0750s (+5.6%, noise±5.0%)
- tpcds/q20: 0.0790s -> 0.0830s (+5.1%, noise±5.0%)
### Improvements (outside noise band)
- tpcds/q97: 39.3380s -> 0.4730s (-98.8%, noise±5.2%)
- tpcds/q95: 0.0740s -> 0.0420s (-43.2%, noise±28.2%)
- tpcds/q26: 1.6360s -> 1.4280s (-12.7%, noise±5.3%)
- tpcds/q27: 1.6370s -> 1.4300s (-12.6%, noise±5.0%)
- tpcds/q79: 1.9450s -> 1.7460s (-10.2%, noise±5.0%)
### Null / failed queries (excluded from averages)
- tpcds/q5
- tpcds/q14
- tpcds/q15

## job
geomean baseline=0.0760s uhj=0.0845s delta=+11.14%
queries: 113; null/error=0; NO_RESULT=65; regressions=41; improvements=7
### Regressions (outside noise band)
- job/q2: 0.0170s -> 0.1180s (+594.1%, noise±5.0%)
- job/q4: 0.0170s -> 0.1130s (+564.7%, noise±5.0%)
- job/q23: 0.0940s -> 0.4930s (+424.5%, noise±5.0%)
- job/q64: 0.1170s -> 0.5060s (+332.5%, noise±5.0%)
- job/q68: 0.0540s -> 0.1570s (+190.7%, noise±13.5%)
- job/q88: 0.0630s -> 0.1640s (+160.3%, noise±59.0%)
- job/q57: 0.1160s -> 0.2870s (+147.4%, noise±5.0%)
- job/q90: 0.0850s -> 0.1800s (+111.8%, noise±62.8%)
- job/q106: 0.1040s -> 0.1890s (+81.7%, noise±46.2%)
- job/q105: 0.1100s -> 0.1990s (+80.9%, noise±5.0%)
- job/q89: 0.0900s -> 0.1560s (+73.3%, noise±5.0%)
- job/q108: 0.1190s -> 0.1990s (+67.2%, noise±5.0%)
- job/q43: 0.0350s -> 0.0550s (+57.1%, noise±14.8%)
- job/q22: 0.0320s -> 0.0480s (+50.0%, noise±5.0%)
- job/q110: 0.0350s -> 0.0470s (+34.3%, noise±5.0%)
- job/q109: 0.0340s -> 0.0450s (+32.4%, noise±10.3%)
- job/q5: 0.0410s -> 0.0540s (+31.7%, noise±5.0%)
- job/q18: 0.0350s -> 0.0450s (+28.6%, noise±9.8%)
- job/q6: 0.0390s -> 0.0500s (+28.2%, noise±5.0%)
- job/q21: 0.0400s -> 0.0510s (+27.5%, noise±5.0%)
- job/q1: 0.0230s -> 0.0290s (+26.1%, noise±25.3%)
- job/q20: 0.0300s -> 0.0370s (+23.3%, noise±5.1%)
- job/q30: 0.0680s -> 0.0820s (+20.6%, noise±16.2%)
- job/q59: 0.2340s -> 0.2780s (+18.8%, noise±5.0%)
- job/q8: 0.0550s -> 0.0650s (+18.2%, noise±5.0%)
- job/q61: 0.0680s -> 0.0800s (+17.6%, noise±6.8%)
- job/q102: 0.2120s -> 0.2480s (+17.0%, noise±5.0%)
- job/q56: 0.2270s -> 0.2640s (+16.3%, noise±5.0%)
- job/q75: 0.1180s -> 0.1370s (+16.1%, noise±5.0%)
- job/q53: 0.0760s -> 0.0880s (+15.8%, noise±10.2%)
- job/q58: 0.2500s -> 0.2880s (+15.2%, noise±5.0%)
- job/q74: 0.0620s -> 0.0710s (+14.5%, noise±11.9%)
- job/q48: 0.1130s -> 0.1290s (+14.2%, noise±11.6%)
- job/q3: 0.0220s -> 0.0250s (+13.6%, noise±5.0%)
- job/q93: 0.1270s -> 0.1430s (+12.6%, noise±5.0%)
- job/q13: 0.0220s -> 0.0240s (+9.1%, noise±5.0%)
- job/q101: 0.1890s -> 0.2020s (+6.9%, noise±5.0%)
- job/q37: 0.0640s -> 0.0680s (+6.3%, noise±5.0%)
- job/q16: 0.0170s -> 0.0180s (+5.9%, noise±5.0%)
- job/q100: 0.2070s -> 0.2190s (+5.8%, noise±5.0%)
- job/q98: 0.1500s -> 0.1580s (+5.3%, noise±5.0%)
### Improvements (outside noise band)
- job/q41: 0.1950s -> 0.0500s (-74.4%, noise±5.0%)
- job/q26: 2.0910s -> 0.6830s (-67.3%, noise±5.1%)
- job/q63: 0.1540s -> 0.0680s (-55.8%, noise±5.0%)
- job/q7: 0.0360s -> 0.0280s (-22.2%, noise±5.0%)
- job/q55: 0.0690s -> 0.0590s (-14.5%, noise±5.0%)
- job/q33: 0.1170s -> 0.1020s (-12.8%, noise±5.0%)
- job/q65: 0.1050s -> 0.0960s (-8.6%, noise±5.0%)

## Fidelity (baseline vs published master on c7a.4xlarge)
published machine=c7a.4xlarge version=26.7.1.498
- coffeeshop: published=4.9485s baseline=4.7269s gap=-4.5% FIDELITY OK
- tpch: published=0.3879s baseline=0.5742s gap=+48.0% FIDELITY FAILED — absolute comparisons unreliable
- tpcds: published=0.2432s baseline=0.2678s gap=+10.1% FIDELITY FAILED — absolute comparisons unreliable
- job: published=0.0731s baseline=0.0760s gap=+4.0% FIDELITY OK

Suspected cause of fidelity gap: host is ARM Neoverse-V2 (no SMT) emulating AMD EPYC c7a.4xlarge (Zen 4, SMT). A/B deltas remain valid.

wrote /mnt/data/uhj_versions_bench/report/REPORT.md and report.json
