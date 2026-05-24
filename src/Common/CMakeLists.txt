if (ENABLE_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()

if (ENABLE_EXAMPLES)
    add_subdirectory(examples)
endif()

if (ENABLE_MYSQL)
    add_subdirectory(mysqlxx)
endif ()

if (ENABLE_TESTS)
    clickhouse_add_executable(bench_radix_shuffle_column_primitives tests/bench_radix_shuffle_column_primitives.cpp)
    target_link_libraries(bench_radix_shuffle_column_primitives PRIVATE dbms)

    clickhouse_add_executable(bench_radix_partition tests/bench_radix_partition.cpp)
    target_link_libraries(bench_radix_partition PRIVATE dbms)

    clickhouse_add_executable(bench_radix_mem_sweep tests/bench_radix_mem_sweep.cpp)
    target_link_libraries(bench_radix_mem_sweep PRIVATE dbms)
endif()
