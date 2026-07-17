# Measurement and correctness {#join-mergetree-bench-measurement}

`--time` supplies wall latency for each warmup or timed query. Final query-scoped
`--print-profile-events` totals provide the partitioned/parallel phase counters. The shown
`ProfileEvents` counters come from the timed run whose wall latency is closest to the
computed median; with an even run count, the median may not be the latency of any literal
sample. Every timed run is still checked to assert the requested execution path. The
benchmark does not subtract cumulative `system.events` snapshots because the snapshot queries
contaminate those counters. It also does not use `RealTimeMicroseconds` as latency because
that event sums time across worker threads rather than measuring wall time.

Every benchmark query pins `join_algorithm` to `partitioned_hash` or `parallel_hash`,
`max_threads` to `--threads`, `query_plan_join_swap_table = false`, `enable_analyzer = 1`,
`enable_join_runtime_filters = 0`, `max_bytes_before_external_join = 0`,
`max_bytes_ratio_before_external_join = 0`, and `max_memory_usage` to `--max-memory`. Before
timing, exact probe, build, and joined row-count assertions must pass. Every event used by
fallback detection is asserted to exist in the binary up front (`system.events` with
`system_events_show_zero_values = 1`); a missing event fails the run with exit code 2, so an
absent counter can never silently satisfy a zero check. Fallback detection then requires
nonzero `PartitionedHashJoin*` build/probe activity (and a zero `parallel_hash` build
counter) for `partitioned_hash`, and a nonzero `ConcurrentHashJoinBuildMicroseconds` with
zero partitioned-path activity for `parallel_hash`.

The final join projection contains only the requested payload columns, never the join key.
When both payload counts are zero it projects `toUInt8(0) AS matched`.

Cross-algorithm result verification is a separate sorted `ORDER BY ALL FORMAT Hash` query.
It is skipped when expected output exceeds `--verify-max-output-rows` (10,000,000 by default)
or when `--no-verify` is explicit; count assertions and path checks still run.

Normal output consists of readable per-point tables and a summary on stdout; the tool creates
no result files. Fatal setup and validation diagnostics go to stderr. For `run`, completed
benchmark failures such as invalid counts, fallback detection, measurement or verification
errors, and hash mismatches return 1; setup or loaded-state validation errors return 2.
