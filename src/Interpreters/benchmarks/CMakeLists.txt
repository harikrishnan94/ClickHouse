# bench_rhj_vs_chj: a stand-alone driver that links the real ClickHouse `dbms` and exercises the genuine
# RadixHashJoin (`radix_hash`) and ConcurrentHashJoin (`parallel_hash`) build/probe code paths directly
# through the IJoin API, for the reference query
#     SELECT count() FROM <probe> INNER JOIN <build> USING (k0)
# It is a research/reproduction harness only and does not modify the production join path.
clickhouse_add_executable(bench_rhj_vs_chj bench_rhj_vs_chj.cpp)
target_link_libraries(bench_rhj_vs_chj PRIVATE dbms)
