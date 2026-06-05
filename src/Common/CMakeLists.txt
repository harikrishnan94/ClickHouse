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
    # RadixHashJoin P2: column-major SWWC/NT key+ref scatter calibration harness. Measures realistic
    # ns/row/pass (fresh output mmap + first-touch faults inside the timed region) across a P sweep
    # (BITS_PER_PASS recalibration) and a key-width sweep, 16 threads.
    clickhouse_add_executable(bench_radix_sweep_native tests/bench_radix_sweep_native.cpp)
    target_link_libraries(bench_radix_sweep_native PRIVATE dbms)
endif ()
