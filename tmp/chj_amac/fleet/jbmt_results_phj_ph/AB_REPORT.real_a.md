# join_bench_mt A/B report: baseline vs candidate (ratio = candidate/baseline; ratio > 1 and 'win' mean baseline better)

376 result rows (376 multi-arm); statuses: {'OK': 375, 'INVALID': 1}
- INVALID: ['tpch__customer_c_nationkey__supplier_s_nationkey__T16__tiera']
binaries: {'baseline': ['0d32ef1c96e6'], 'candidate': ['06d804546e0f']}
lead arm distribution (ABAB leader): {'baseline': 172, 'candidate': 204}

## wall / parallel_hash
n=375: {'win': 211, 'loss': 26, 'tie': 138}; median ratio 1.071, p10 0.974, p90 1.583

| group | n | win | tie | loss | median ratio |
|---|---|---|---|---|---|
| real-coffeeshop | 6 | 2 | 4 | 0 | 1.030 |
| real-job | 126 | 73 | 43 | 10 | 1.091 |
| real-stackoverflow | 12 | 6 | 0 | 6 | 1.034 |
| real-tpcds | 198 | 112 | 81 | 5 | 1.064 |
| real-tpch | 33 | 18 | 10 | 5 | 1.074 |

worst losses: stackoverflow__badges_UserId__users_Id__T96__tiera=0.80, stackoverflow__postlinks_RelatedPostId__posts_Id__T96__tiera=0.82, tpch__lineitem_l_orderkey__orders_o_orderkey__T96__tiera=0.83, stackoverflow__votes_PostId__posts_Id__T96__tiera=0.83, tpcds__customer_c_customer_sk__catalog_returns_cr_returning_customer_sk__T96__tiera=0.84, stackoverflow__comments_PostId__posts_Id__T96__tiera=0.85, tpch__partsupp_ps_partkey__part_p_partkey__T96__tiera=0.85, job__movie_companies_movie_id__title_id__T96__tiera=0.86, job__name_id__person_info_person_id__T96__tiera=0.87, job__movie_keyword_movie_id__title_id__T96__tiera=0.89, stackoverflow__posts_OwnerUserId__users_Id__T96__tiera=0.89, tpcds__customer_address_ca_county__store_s_county__filtered__T96__tiera=0.89, tpcds__web_returns_wr_reason_sk__reason_r_reason_sk__T96__tiera=0.90, stackoverflow__comments_UserId__users_Id__T96__tiera=0.91, job__title_id__aka_title_movie_id__T96__tiera=0.92

## memory / parallel_hash
n=375: {'win': 132, 'tie': 177, 'loss': 66}; median ratio 1.034, p10 0.522, p90 1.137

| group | n | win | tie | loss | median ratio |
|---|---|---|---|---|---|
| real-coffeeshop | 6 | 2 | 4 | 0 | 1.042 |
| real-job | 126 | 47 | 52 | 27 | 1.037 |
| real-stackoverflow | 12 | 0 | 12 | 0 | 1.034 |
| real-tpcds | 198 | 67 | 97 | 34 | 1.029 |
| real-tpch | 33 | 16 | 12 | 5 | 1.053 |

worst losses: tpch__nation_n_regionkey__region_r_regionkey__T96__tiera=0.04, job__movie_link_link_type_id__link_type_id__T96__tiera=0.04, tpcds__household_demographics_hd_income_band_sk__income_band_ib_income_band_sk__T96__tiera=0.04, job__complete_cast_movie_id__movie_link_movie_id__T96__tiera=0.07, tpcds__customer_address_ca_county__store_s_county__T96__tiera=0.07, tpcds__customer_address_ca_county__store_s_county__filtered__T96__tiera=0.07, tpcds__customer_address_ca_zip__store_s_zip__filtered__T96__tiera=0.09, tpcds__customer_address_ca_zip__store_s_zip__T96__tiera=0.09, job__complete_cast_status_id__comp_cast_type_id__T96__tiera=0.10, job__complete_cast_subject_id__comp_cast_type_id__T96__tiera=0.10, tpcds__customer_address_ca_state__store_s_state__filtered__T96__tiera=0.16, job__movie_keyword_movie_id__movie_link_movie_id__T96__tiera=0.16, job__movie_keyword_keyword_id__keyword_id__filtered__T96__tiera=0.17, job__movie_info_idx_info_type_id__info_type_id__filtered__T96__tiera=0.18, tpch__nation_n_regionkey__region_r_regionkey__T16__tiera=0.18
