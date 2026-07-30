### Unit 3 tier a (jbmt real) — complete: generated from `reports/jbmt_real_a.tsv`

Units in file: 376 · scored: 368 · NO-VERDICT: 8

| | `probe_cost` | `projection_cost` |
| --- | --- | --- |
| verdicts | 368 | 368 |
| **WIN / TIE / LOSS** | **161 / 51 / 156** | **43 / 127 / 198** |
| aggregate | 230,541.8 ms → 210,259.1 ms (**-8.8 %**) | 2,284,639.8 ms → 2,347,582.1 ms (**+2.8 %**) |
| median per-unit delta | **+0.4 %** | **+4.5 %** |

**Recorded, never a verdict** (this campaign verdicts only the two metrics above):

| measured quantity | arm A | arm B | delta |
| --- | --- | --- | --- |
| `ConcurrentHashJoinProbeMicroseconds` (the probe total the two metrics sum to) | 2,515,295.9 ms | 2,557,844.4 ms | **+1.69 %** |
| wall clock (`query_duration_ms`) | 120,936.0 ms | 126,330.0 ms | **+4.46 %** |

Per-unit wall clock: **287 of 368 units slower**, 39 faster, 42 equal.

**Concentration of the `probe_cost` aggregate** (net -20,282.7 ms):

| improving units | their d(`probe_cost`) | share of net | their d(probe total) | their d(wall) |
| --- | --- | --- | --- | --- |
| top 5 | -10,499.7 ms | 51.8 % | **3,413.9 ms** | **646.0 ms** |
| top 20 | -20,636.9 ms | 101.7 % | **4,794.8 ms** | **1,108.0 ms** |

**Worst `probe_cost` regressions** (156 total):

| unit | probe A (ms) | probe B (ms) | delta | band | dispatch A→B | lookup A→B | probe total A→B |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `tpcds__customer_c_customer_sk__catalog_returns_cr_returning_customer_sk__T96__tiera` | 20.9 | 65.0 | **+211.7 %** | 5.0 % | 0.0 → 0.1 | 20.9 → 65.0 | 487.8 → 530.1 |
| `job__movie_keyword_movie_id__title_id__filtered__T96__tiera` | 37.2 | 95.4 | **+156.8 %** | 17.3 % | 0.0 → 0.2 | 37.2 → 95.2 | 67.5 → 126.8 |
| `tpcds__customer_c_customer_sk__catalog_returns_cr_returning_customer_sk__T16__tiera` | 7.5 | 17.3 | **+129.6 %** | 19.8 % | 0.0 → 0.1 | 7.5 → 17.2 | 392.8 → 440.6 |
| `stackoverflow__postlinks_RelatedPostId__posts_Id__T96__tiera` | 764.1 | 1,741.7 | **+128.0 %** | 19.4 % | 0.0 → 0.3 | 764.1 → 1,741.5 | 1,572.4 → 2,715.6 |
| `job__movie_keyword_movie_id__title_id__T96__tiera` | 66.0 | 135.4 | **+105.2 %** | 9.4 % | 0.0 → 0.2 | 66.0 → 135.2 | 178.5 → 250.5 |
| `job__movie_keyword_movie_id__movie_companies_movie_id__T96__tiera` | 70.6 | 137.3 | **+94.5 %** | 8.8 % | 0.0 → 0.2 | 70.6 → 137.0 | 943.8 → 1,056.7 |
| `stackoverflow__postlinks_RelatedPostId__posts_Id__T16__tiera` | 147.2 | 281.5 | **+91.3 %** | 3.1 % | 0.0 → 0.2 | 147.2 → 281.3 | 673.9 → 1,110.6 |
| `job__movie_companies_movie_id__title_id__T96__tiera` | 79.7 | 133.0 | **+66.8 %** | 6.4 % | 0.0 → 0.2 | 79.7 → 132.8 | 166.4 → 225.2 |

**Worst `projection_cost` regressions** (198 total):

| unit | projection A (ms) | projection B (ms) | delta | band | dispatch A→B | lookup A→B | probe total A→B |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `stackoverflow__postlinks_RelatedPostId__posts_Id__T16__tiera` | 525.6 | 829.8 | **+57.9 %** | 3.0 % | 0.0 → 0.2 | 147.2 → 281.3 | 673.9 → 1,110.6 |
| `tpcds__household_demographics_hd_income_band_sk__income_band_ib_income_band_sk__T96__tiera` | 0.1 | 0.1 | **+56.0 %** | 19.2 % | 0.0 → 0.0 | 0.1 → 0.1 | 0.2 → 0.2 |
| `tpcds__customer_address_ca_state__store_s_state__filtered__T16__tiera` | 6.7 | 9.9 | **+48.2 %** | 5.8 % | 0.0 → 0.0 | 0.9 → 1.2 | 7.6 → 11.2 |
| `tpcds__customer_address_ca_state__store_s_state__T16__tiera` | 12.3 | 17.0 | **+37.4 %** | 8.8 % | 0.0 → 0.0 | 1.0 → 1.5 | 13.4 → 18.4 |
| `job__movie_companies_movie_id__title_id__filtered__T96__tiera` | 28.8 | 39.5 | **+37.3 %** | 6.9 % | 0.0 → 0.2 | 41.4 → 53.6 | 70.0 → 92.9 |
| `tpcds__web_sales_ws_bill_customer_sk__customer_c_customer_sk__T16__tiera` | 1,605.7 | 2,192.7 | **+36.6 %** | 3.0 % | 0.0 → 1.2 | 273.6 → 215.6 | 1,879.3 → 2,409.5 |
| `tpcds__catalog_sales_cs_ship_customer_sk__customer_c_customer_sk__T16__tiera` | 3,246.1 | 4,395.9 | **+35.4 %** | 3.0 % | 0.0 → 2.4 | 559.3 → 432.8 | 3,806.1 → 4,835.7 |
| `tpcds__store_returns_sr_cdemo_sk__customer_demographics_cd_demo_sk__T16__tiera` | 579.8 | 783.6 | **+35.2 %** | 3.0 % | 0.0 → 0.5 | 130.4 → 117.6 | 710.0 → 903.5 |

**Units whose two metrics move in opposite directions: 142** (never netted; each appears in both lists).

**NO-VERDICT units (8), with the harness's own reason:**

| unit | reason |
| --- | --- |
| `tpcds__catalog_sales_cs_bill_customer_sk__store_returns_sr_customer_sk__T16__tiera` | harness voided: jbmt unit status OVER_BUDGET: arm baseline warmup 0 took 206.0s > unit-time-budget 30.0s; unit skipped before any timed run |
| `tpcds__catalog_sales_cs_bill_customer_sk__store_returns_sr_customer_sk__T96__tiera` | harness voided: jbmt unit status OVER_BUDGET: arm baseline warmup 0 took 63.6s > unit-time-budget 30.0s; unit skipped before any timed run |
| `tpcds__inventory_inv_item_sk__catalog_sales_cs_item_sk__T16__tiera` | harness voided: jbmt unit status OVER_BUDGET: arm baseline warmup 0 took 54.4s > unit-time-budget 30.0s; unit skipped before any timed run |
| `tpcds__store_sales_ss_item_sk__catalog_sales_cs_item_sk__T16__tiera` | harness voided: jbmt unit status OVER_BUDGET: arm candidate warmup 0 took 149.5s > unit-time-budget 30.0s; unit skipped before any timed run |
| `tpcds__store_sales_ss_item_sk__catalog_sales_cs_item_sk__T96__tiera` | harness voided: jbmt unit status OVER_BUDGET: arm candidate warmup 0 took 36.0s > unit-time-budget 30.0s; unit skipped before any timed run |
| `tpcds__store_sales_ss_item_sk__web_sales_ws_item_sk__T16__tiera` | harness voided: jbmt unit status OVER_BUDGET: arm baseline warmup 0 took 75.0s > unit-time-budget 30.0s; unit skipped before any timed run |
| `tpch__customer_c_nationkey__supplier_s_nationkey__T16__tiera` | harness voided: jbmt unit status OVER_BUDGET: arm candidate warmup 0 took 570.2s > unit-time-budget 30.0s; unit skipped before any timed run |
| `tpch__customer_c_nationkey__supplier_s_nationkey__T96__tiera` | harness voided: jbmt unit status OVER_BUDGET: arm candidate warmup 0 took 137.1s > unit-time-budget 30.0s; unit skipped before any timed run |
