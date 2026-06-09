# RadixHashJoin benchmarks: a single gtest-free executable holding every RHJ benchmark (build/scatter
# and leaf-HT build + probe), each driving the production path. Select a bench by argv (run with --list).
clickhouse_add_executable(bench_radix_hash_join
    bench_radix_hash_join.cpp
    bench_radix_hash_scatter.cpp)
target_link_libraries(bench_radix_hash_join PRIVATE dbms)
