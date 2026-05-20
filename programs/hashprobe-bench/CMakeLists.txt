# hashprobe-bench — in-tree HashJoin probe-path benchmark harness (issue #104581).
#
# Builds a standalone binary that drives HashJoin / ConcurrentHashJoin directly
# (no QueryPipeline, no InterpreterSelectQuery) to isolate and replay probe cost
# under sweeping configurations.
#
# Non-substituting diffs to src/ are enumerated in NON_SUBSTITUTING_DIFF.allowlist.

option(ENABLE_HASHPROBE_BENCH "Build hashprobe-bench HashJoin probe-path harness" ON)

if (NOT ENABLE_HASHPROBE_BENCH)
    message(STATUS "Not building hashprobe-bench (ENABLE_HASHPROBE_BENCH=OFF)")
    return()
endif()


clickhouse_add_executable(hashprobe-bench
    main.cpp
)

target_include_directories(hashprobe-bench PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(hashprobe-bench PRIVATE
    clickhouse_common_io
    dbms
    clickhouse_parsers
    boost::program_options
)

# HARNESS_OBSERVABILITY gates read-only const accessors added to HashJoin.h so
# they are statically unreachable from non-harness builds (A1, non-substituting def).
target_compile_definitions(hashprobe-bench PRIVATE HARNESS_OBSERVABILITY=1)


# ── Track E: Instrumentation Module ─────────────────────────────────────────────
target_sources(hashprobe-bench PRIVATE
    instrumentation/hw_counters.cpp
    instrumentation/cache_mode.cpp
    instrumentation/per_block_log.cpp
)




# ── Track D: Oracle Module ─────────────────────────────────────────────────────
target_sources(hashprobe-bench PRIVATE
    oracle/oracle_sql.cpp
    oracle/oracle.cpp
)
target_link_libraries(hashprobe-bench PRIVATE OpenSSL::Crypto)

# ── Track A.1: Key Generator ──────────────────────────────────────────────────
set(GENERATOR_SOURCES
    generator/key_generator.cpp
    generator/block_builder.cpp
    generator/native_writer.cpp
)

target_sources(hashprobe-bench PRIVATE ${GENERATOR_SOURCES})
target_link_libraries(hashprobe-bench PRIVATE pcg_random)

message(STATUS "hashprobe-bench: Track A generator sources added")

# ── Track C: Probe Driver ──────────────────────────────────────────────────────
target_sources(hashprobe-bench PRIVATE
    driver/probe_driver.cpp
)

message(STATUS "hashprobe-bench: Track C probe driver sources added")

# ── Track B: Build Driver ──────────────────────────────────────────────────────
target_sources(hashprobe-bench PRIVATE
    driver/build_driver.cpp
)

message(STATUS "hashprobe-bench: Track B build driver sources added")

# ── Track H: Verification Wrappers ──────────────────────────────────────────────
target_sources(hashprobe-bench PRIVATE
    verification/verifier.cpp
    verification/e_l0.cpp
    verification/e_l1.cpp
    verification/e_l2.cpp
    verification/e_auto.cpp
)

message(STATUS "hashprobe-bench: Track H verification sources added")

# ── Track G: Sweep Manager ────────────────────────────────────────────────────
target_sources(hashprobe-bench PRIVATE
    sweep/sweep_manager.cpp
)

target_link_libraries(hashprobe-bench PRIVATE ch_contrib::nlohmann_json)

message(STATUS "hashprobe-bench: Track G sweep manager sources added")

# ── G.4: Compile-time build metadata (I2, I3) ────────────────────────────────
execute_process(
    COMMAND git -C "${CMAKE_SOURCE_DIR}" log -1 --format=%H
    OUTPUT_VARIABLE GIT_COMMIT_HASH
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
)
if(NOT GIT_COMMIT_HASH)
    set(GIT_COMMIT_HASH "unknown")
endif()

target_compile_definitions(hashprobe-bench PRIVATE
    GIT_COMMIT_HASH="${GIT_COMMIT_HASH}"
    BUILD_TYPE_STR="${CMAKE_BUILD_TYPE}"
)

message(STATUS "hashprobe-bench: GIT_COMMIT_HASH=${GIT_COMMIT_HASH}  BUILD_TYPE=${CMAKE_BUILD_TYPE}")

# ── Partitioned Hash Join module ──────────────────────────────────────────────
target_sources(hashprobe-bench PRIVATE
    partitioned/phj_run.cpp
)
message(STATUS "hashprobe-bench: PHJ partitioned module added")

# ── Partitioned Hash Join POC ─────────────────────────────────────────────────
clickhouse_add_executable(partitioned-join-poc
    partitioned_join_poc.cpp
)
target_include_directories(partitioned-join-poc PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}
)
target_link_libraries(partitioned-join-poc PRIVATE
    clickhouse_common_io
    dbms
    pcg_random
)
target_compile_definitions(partitioned-join-poc PRIVATE HARNESS_OBSERVABILITY=1)
target_compile_options(partitioned-join-poc PRIVATE -march=native)
message(STATUS "partitioned-join-poc: added")
