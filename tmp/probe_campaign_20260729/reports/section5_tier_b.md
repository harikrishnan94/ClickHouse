### Unit 3 tier b (jbmt real) — complete: generated from `reports/jbmt_real_b.tsv`

Units in file: 376 · scored: 365 · NO-VERDICT: 11

| | `probe_cost` | `projection_cost` |
| --- | --- | --- |
| verdicts | 365 | 365 |
| **WIN / TIE / LOSS** | **164 / 50 / 151** | **49 / 126 / 190** |
| aggregate | 493,179.2 ms → 449,846.8 ms (**-8.8 %**) | 1,867,031.9 ms → 2,054,156.9 ms (**+10.0 %**) |
| median per-unit delta | **-0.4 %** | **+4.0 %** |

**Recorded, never a verdict** (this campaign verdicts only the two metrics above):

| measured quantity | arm A | arm B | delta |
| --- | --- | --- | --- |
| `ConcurrentHashJoinProbeMicroseconds` (the probe total the two metrics sum to) | 2,359,996.8 ms | 2,504,216.7 ms | **+6.11 %** |
| wall clock (`query_duration_ms`) | 139,824.0 ms | 151,024.0 ms | **+8.01 %** |

Per-unit wall clock: **273 of 365 units slower**, 50 faster, 42 equal.

**Concentration of the `probe_cost` aggregate** (net -43,332.4 ms):

| improving units | their d(`probe_cost`) | share of net | their d(probe total) | their d(wall) |
| --- | --- | --- | --- | --- |
| top 5 | -20,324.3 ms | 46.9 % | **-10,566.1 ms** | **-247.0 ms** |
| top 20 | -43,828.8 ms | 101.1 % | **73,052.7 ms** | **4,988.0 ms** |

**Worst `probe_cost` regressions** (151 total):

| unit | probe A (ms) | probe B (ms) | delta | band | dispatch A→B | lookup A→B | probe total A→B |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `job__movie_keyword_movie_id__title_id__filtered__T96__tierb` | 38.5 | 100.2 | **+159.9 %** | 18.1 % | 0.0 → 0.2 | 38.5 → 99.9 | 68.9 → 132.0 |
| `tpcds__customer_c_customer_sk__catalog_returns_cr_returning_customer_sk__T96__tierb` | 173.0 | 422.1 | **+144.0 %** | 5.3 % | 0.0 → 0.2 | 173.0 → 421.9 | 1,245.3 → 1,513.9 |
| `tpcds__customer_c_customer_sk__catalog_returns_cr_returning_customer_sk__T16__tierb` | 16.5 | 37.6 | **+127.7 %** | 20.2 % | 0.0 → 0.2 | 16.5 → 37.4 | 831.1 → 972.2 |
| `job__movie_keyword_movie_id__title_id__T96__tierb` | 66.1 | 130.2 | **+96.9 %** | 10.7 % | 0.0 → 0.2 | 66.1 → 130.0 | 181.1 → 245.1 |
| `job__movie_keyword_movie_id__movie_companies_movie_id__T96__tierb` | 72.5 | 137.0 | **+88.8 %** | 10.6 % | 0.0 → 0.2 | 72.5 → 136.7 | 958.1 → 1,051.9 |
| `stackoverflow__postlinks_RelatedPostId__posts_Id__T96__tierb` | 1,974.0 | 3,641.0 | **+84.5 %** | 7.4 % | 0.0 → 0.5 | 1,974.0 → 3,640.5 | 3,973.1 → 5,885.7 |
| `job__movie_companies_movie_id__title_id__T96__tierb` | 78.8 | 141.5 | **+79.5 %** | 7.3 % | 0.0 → 0.2 | 78.8 → 141.3 | 164.2 → 237.4 |
| `job__name_id__person_info_person_id__T96__tierb` | 285.7 | 479.3 | **+67.8 %** | 7.1 % | 0.0 → 0.5 | 285.7 → 478.8 | 787.9 → 997.9 |

**Worst `projection_cost` regressions** (190 total):

| unit | projection A (ms) | projection B (ms) | delta | band | dispatch A→B | lookup A→B | probe total A→B |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `stackoverflow__postlinks_RelatedPostId__posts_Id__T16__tierb` | 1,137.8 | 1,914.9 | **+68.3 %** | 3.0 % | 0.0 → 0.4 | 401.5 → 634.6 | 1,539.3 → 2,549.6 |
| `tpcds__household_demographics_hd_income_band_sk__income_band_ib_income_band_sk__T96__tierb` | 0.1 | 0.1 | **+51.8 %** | 22.4 % | 0.0 → 0.0 | 0.1 → 0.1 | 0.2 → 0.2 |
| `tpch__lineitem_l_partkey__part_p_partkey__T16__tierb` | 104,775.6 | 152,068.6 | **+45.1 %** | 3.0 % | 0.1 → 34.0 | 11,479.0 → 9,924.3 | 116,213.9 → 162,035.5 |
| `stackoverflow__comments_UserId__users_Id__T16__tierb` | 13,792.0 | 19,309.6 | **+40.0 %** | 3.0 % | 0.0 → 5.2 | 3,574.5 → 2,430.1 | 17,366.5 → 21,754.1 |
| `tpch__orders_o_custkey__customer_c_custkey__T16__tierb` | 12,702.5 | 17,780.6 | **+40.0 %** | 3.0 % | 0.0 → 8.6 | 2,699.0 → 2,098.7 | 15,409.2 → 19,887.5 |
| `tpcds__customer_address_ca_state__store_s_state__filtered__T16__tierb` | 14.1 | 19.8 | **+39.9 %** | 3.0 % | 0.0 → 0.0 | 1.9 → 2.3 | 16.0 → 22.1 |
| `tpch__orders_o_custkey__customer_c_custkey__left__T16__tierb` | 13,150.4 | 18,347.3 | **+39.5 %** | 3.0 % | 0.0 → 8.7 | 2,902.5 → 2,181.3 | 16,056.5 → 20,557.8 |
| `stackoverflow__posts_OwnerUserId__users_Id__T16__tierb` | 11,117.2 | 15,406.7 | **+38.6 %** | 3.0 % | 0.0 → 4.9 | 2,863.2 → 2,153.5 | 13,982.1 → 17,563.8 |

**Units whose two metrics move in opposite directions: 149** (never netted; each appears in both lists).

**NO-VERDICT units (11), with the harness's own reason:**

| unit | reason |
| --- | --- |
| `tpcds__catalog_sales_cs_bill_customer_sk__store_returns_sr_customer_sk__T16__tierb` | harness voided: jbmt unit status OVER_BUDGET: arm baseline warmup 0 took 600.1s > unit-time-budget 30.0s (and failed: rc=159 stderr=Received exception from server (version 26.8.1): Code: 159. DB::Exception: Received from localhost:9005. D); unit skipped before any timed run |
| `tpcds__catalog_sales_cs_bill_customer_sk__store_returns_sr_customer_sk__T96__tierb` | harness voided: jbmt unit status OVER_BUDGET: arm baseline warmup 0 took 270.7s > unit-time-budget 30.0s; unit skipped before any timed run |
| `tpcds__catalog_sales_cs_item_sk__store_returns_sr_item_sk__T16__tierb` | harness voided: jbmt unit status OVER_BUDGET: arm candidate warmup 0 took 32.8s > unit-time-budget 30.0s; unit skipped before any timed run |
| `tpcds__inventory_inv_item_sk__catalog_sales_cs_item_sk__T16__tierb` | harness voided: jbmt unit status OVER_BUDGET: arm baseline warmup 0 took 144.0s > unit-time-budget 30.0s; unit skipped before any timed run |
| `tpcds__inventory_inv_item_sk__catalog_sales_cs_item_sk__T96__tierb` | harness voided: jbmt unit status OVER_BUDGET: arm baseline warmup 0 took 36.3s > unit-time-budget 30.0s; unit skipped before any timed run |
| `tpcds__store_sales_ss_item_sk__catalog_sales_cs_item_sk__T16__tierb` | harness voided: jbmt unit status OVER_BUDGET: arm candidate warmup 0 took 321.9s > unit-time-budget 30.0s; unit skipped before any timed run |
| `tpcds__store_sales_ss_item_sk__catalog_sales_cs_item_sk__T96__tierb` | harness voided: jbmt unit status OVER_BUDGET: arm candidate warmup 0 took 76.4s > unit-time-budget 30.0s; unit skipped before any timed run |
| `tpcds__store_sales_ss_item_sk__web_sales_ws_item_sk__T16__tierb` | harness voided: jbmt unit status OVER_BUDGET: arm baseline warmup 0 took 161.7s > unit-time-budget 30.0s; unit skipped before any timed run |
| `tpcds__store_sales_ss_item_sk__web_sales_ws_item_sk__T96__tierb` | harness voided: jbmt unit status OVER_BUDGET: arm baseline warmup 0 took 38.7s > unit-time-budget 30.0s; unit skipped before any timed run |
| `tpch__customer_c_nationkey__supplier_s_nationkey__T16__tierb` | harness voided: jbmt unit status OVER_BUDGET: arm candidate warmup 0 took 600.1s > unit-time-budget 30.0s (and failed: rc=159 stderr=Received exception from server (version 26.8.1): Code: 159. DB::Exception: Received from localhost:9006. D); unit skipped before any timed run |
| `tpch__customer_c_nationkey__supplier_s_nationkey__T96__tierb` | harness voided: jbmt unit status OVER_BUDGET: arm candidate warmup 0 took 600.1s > unit-time-budget 30.0s (and failed: rc=159 stderr=Received exception from server (version 26.8.1): Code: 159. DB::Exception: Received from localhost:9006. D); unit skipped before any timed run |
