# join_bench_mt A/B report: baseline vs candidate (ratio = candidate/baseline; ratio > 1 and 'win' mean baseline better)

347 result rows (347 multi-arm); statuses: {'OK': 347}
binaries: {'baseline': ['0d32ef1c96e6'], 'candidate': ['06d804546e0f']}
lead arm distribution (ABAB leader): {'candidate': 181, 'baseline': 166}

## wall / parallel_hash
n=347: {'loss': 119, 'win': 175, 'tie': 53}; median ratio 1.057, p10 0.727, p90 1.689

| group | n | win | tie | loss | median ratio |
|---|---|---|---|---|---|
| LEGACY | 347 | 175 | 53 | 119 | 1.057 |

worst losses: D32000000_K7_mb16_mp1_h1.0_bp8_pp8_T96=0.57, D32000000_K7_mb8_mp1_h1.0_bp8_pp8_T96=0.60, D8000000_K7_mb16_mp1_h1.0_bp8_pp8_T96=0.64, D32000000_K3_mb8_mp1_h1.0_bp8_pp8_T96=0.64, D32000000_K2_mb1_mp1_h0.05_bp8_pp8_T96=0.65, D128000000_K6_mb1_mp1_h1.0_bp8_pp8_T96=0.66, D32000000_K7_mb1_mp1_h1.0_bp8_pp8_T96=0.66, D32000000_K5_mb1_mp1_h0.25_bp8_pp8_T96=0.66, D32000000_K2_mb1_mp1_h0.25_bp8_pp8_T96=0.66, D32000000_K3_mb16_mp1_h1.0_bp8_pp8_T96=0.66, D32000000_K6_mb1_mp1_h1.0_bp8_pp8_T96=0.67, D32000000_K5_mb1_mp1_h0.5_bp8_pp8_T96=0.68, D128000000_K2_mb1_mp1_h1.0_bp8_pp8_T96=0.68, D32000000_K7_mb1_mp1_h0.75_bp8_pp8_T96=0.69, D32000000_K7_mb2_mp1_h1.0_bp8_pp8_T96=0.69

## memory / parallel_hash
n=347: {'win': 81, 'tie': 261, 'loss': 5}; median ratio 1.034, p10 1.015, p90 1.096

| group | n | win | tie | loss | median ratio |
|---|---|---|---|---|---|
| LEGACY | 347 | 81 | 261 | 5 | 1.034 |

worst losses: D65536_K0_mb1_mp16_h1.0_bp8_pp8_T96=0.25, D65536_K0_mb1_mp64_h1.0_bp8_pp8_T96=0.45, D262144_K0_mb1_mp16_h1.0_bp8_pp8_T96=0.56, D2000000_K0_mb1_mp1_h1.0_bp8_pp8_T96=0.86, D8000000_K2_mb1_mp1_h1.0_bp8_pp8_T96=0.92
