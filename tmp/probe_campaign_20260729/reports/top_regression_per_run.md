# Per-run separation of the largest `probe_cost` regressions (tier a)

A median can be moved by one bad run. These are the raw per-run `probe_cost` values of
both arms, sorted, for the largest reported regressions, plus the leave-one-out range of
the delta. If the two arms overlapped, or one run carried the delta, it would show here.

## `tpcds__customer_c_customer_sk__catalog_returns_cr_returning_customer_sk__T96__tiera`

- arm A per-run probe_cost (us): `[18767, 19301, 20073, 20587, 20779, 20866, 21531, 21544, 21642, 21939, 22227]`
- arm B per-run probe_cost (us): `[60000, 63875, 64130, 64924, 65038, 65041, 65579, 66671, 66843, 66858, 73830]`
- distributions: **DISJOINT (no overlap)**
- delta at the median: **+211.7 %**; leave-one-out range **+208.1 % .. +212.4 %**
- lead arm: `baseline` (timed runs strictly interleaved ABAB)

## `job__movie_keyword_movie_id__title_id__filtered__T96__tiera`

- arm A per-run probe_cost (us): `[34136, 34572, 35629, 36835, 36888, 37170, 37647, 38491, 49359, 50409, 51884]`
- arm B per-run probe_cost (us): `[74155, 75924, 84925, 90927, 92703, 95438, 104945, 105683, 106472, 114128, 116436]`
- distributions: **DISJOINT (no overlap)**
- delta at the median: **+156.8 %**; leave-one-out range **+154.0 % .. +167.8 %**
- lead arm: `baseline` (timed runs strictly interleaved ABAB)

## `stackoverflow__postlinks_RelatedPostId__posts_Id__T16__tiera`

- arm A per-run probe_cost (us): `[141316, 141494, 142224, 145765, 147101, 147151, 147715, 149274, 151338, 152048, 157031]`
- arm B per-run probe_cost (us): `[265572, 270615, 274272, 276359, 279026, 281487, 283250, 285735, 288236, 289053, 292435]`
- distributions: **DISJOINT (no overlap)**
- delta at the median: **+91.3 %**; leave-one-out range **+90.5 % .. +91.5 %**
- lead arm: `candidate` (timed runs strictly interleaved ABAB)

## `stackoverflow__postlinks_RelatedPostId__posts_Id__T96__tiera`

- arm A per-run probe_cost (us): `[732805, 736434, 755900, 756433, 762654, 764073, 859273, 954237, 983893, 1038127, 1197889]`
- arm B per-run probe_cost (us): `[1369163, 1537427, 1698674, 1729578, 1735193, 1741736, 1764584, 1776114, 1779977, 1796952, 1804922]`
- distributions: **DISJOINT (no overlap)**
- delta at the median: **+128.0 %**; leave-one-out range **+115.8 % .. +127.7 %**
- lead arm: `candidate` (timed runs strictly interleaved ABAB)

