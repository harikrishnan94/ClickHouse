# bench_rhj_vs_chj: a stand-alone driver that links the real ClickHouse `dbms` and exercises the genuine
# RadixHashJoin (`radix_hash`) and ConcurrentHashJoin (`parallel_hash`) build/probe code paths directly
# through the IJoin API, for the reference query
#     SELECT count() FROM <probe> INNER JOIN <build> USING (k0)
# It is a research/reproduction harness only and does not modify the production join path.
clickhouse_add_executable(bench_rhj_vs_chj bench_rhj_vs_chj.cpp)
target_link_libraries(bench_rhj_vs_chj PRIVATE dbms)

# bench_build_bandwidth: a stand-alone driver that isolates the build-side data-movement pipelines of the
# RadixHashJoin / ConcurrentHashJoin build (CHJ zero-copy dispatch replica vs RHJ scatterToLeaves vs a
# sequential key+ref memcpy) over the same pre-generated blocks, to quantify the true RHJ scatter overhead
# and the true CHJ zero-copy advantage. Research/reproduction harness only; production join path untouched.
clickhouse_add_executable(bench_build_bandwidth bench_build_bandwidth.cpp)
target_link_libraries(bench_build_bandwidth PRIVATE dbms)

# bench_scatter_perf: phase-attributed PMU instrumentation (perf_event_open groups + getrusage) of the RHJ
# build-side data movement, wrapping the genuine BuildSide::scatterToLeaves and a memcpy baseline under
# identical groups to explain the T_scatter vs T_memcpy gap. Research/diagnostic harness only.
clickhouse_add_executable(bench_scatter_perf bench_scatter_perf.cpp)
target_link_libraries(bench_scatter_perf PRIVATE dbms)
