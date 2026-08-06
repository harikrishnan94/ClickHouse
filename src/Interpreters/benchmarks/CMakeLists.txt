clickhouse_add_executable(unified_hash_join_probe_loop unified_hash_join_probe_loop.cpp)
target_link_libraries (unified_hash_join_probe_loop PRIVATE
    ch_contrib::gbenchmark_all
    dbms)
