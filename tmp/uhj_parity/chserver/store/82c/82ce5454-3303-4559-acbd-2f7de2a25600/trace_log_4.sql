ATTACH TABLE _ UUID '976a9f2c-3850-4425-919d-375bee2734a3'
(
    `hostname` LowCardinality(String) COMMENT 'Hostname of the server executing the query.',
    `event_date` Date COMMENT 'Date of sampling moment.',
    `event_time` DateTime COMMENT 'Timestamp of the sampling moment.',
    `event_time_microseconds` DateTime64(6) COMMENT 'Timestamp of the sampling moment with microseconds precision.',
    `timestamp_ns` UInt64 COMMENT 'Timestamp of the sampling moment in nanoseconds.',
    `revision` UInt32 COMMENT 'ClickHouse server build revision. When connecting to the server by `clickhouse-client`, you see a string similar to `Connected to ClickHouse server version 19.18.1.`. This field contains the `revision`, but not the `version` of a server.',
    `trace_type` Enum8('Real' = 0, 'CPU' = 1, 'Memory' = 2, 'MemorySample' = 3, 'MemoryPeak' = 4, 'ProfileEvent' = 5, 'JemallocSample' = 6, 'MemoryAllocatedWithoutCheck' = 7, 'Instrumentation' = 8) COMMENT 'Trace type: `Real` represents collecting stack traces by wall-clock time. `CPU` represents collecting stack traces by CPU time. `Memory` represents collecting allocations and deallocations when memory allocation exceeds the subsequent watermark. `MemorySample` represents collecting random allocations and deallocations. `MemoryPeak` represents collecting updates of peak memory usage. `ProfileEvent` represents collecting of increments of profile events. `JemallocSample` represents collecting of jemalloc samples. `MemoryAllocatedWithoutCheck` represents collection of significant allocations (>16MiB) that is done with ignoring any memory limits (for ClickHouse developers only).`Instrumentation` represents traces collected by the instrumentation performed through XRay.',
    `cpu_id` UInt64 COMMENT 'CPU identifier.',
    `thread_id` UInt64 COMMENT 'Thread identifier.',
    `thread_name` LowCardinality(String) COMMENT 'Thread name.',
    `query_id` String COMMENT 'Query identifier that can be used to get details about a query that was running from the query_log system table.',
    `trace` Array(UInt64) COMMENT 'Stack trace at the moment of sampling. For profiler-collected trace types, on ELF platforms except FreeBSD, addresses inside the main ClickHouse binary are stored as physical file offsets, and other addresses are virtual memory addresses inside the ClickHouse server process. Instrumentation trace rows are an exception: they store raw virtual memory addresses.',
    `size` Int64 COMMENT 'For trace types Memory, MemorySample, MemoryAllocatedWithoutCheck or MemoryPeak is the amount of memory allocated, for other trace types is 0.',
    `ptr` UInt64 COMMENT 'The address of the allocated chunk.',
    `memory_context` Enum8('Unknown' = -1, 'Global' = 0, 'User' = 1, 'Process' = 2, 'Thread' = 3, 'Max' = 4) COMMENT 'Memory Tracker context (only for Memory/MemoryPeak): `Unknown` context is not defined for this trace_type. `Global` represents server context. `User` represents user/merge context. `Process` represents process (i.e. query) context. `Thread` represents thread (thread of particular process) context. `Max` this is a special value means that memory tracker is not blocked (for blocked_context column). ',
    `memory_blocked_context` Enum8('Unknown' = -1, 'Global' = 0, 'User' = 1, 'Process' = 2, 'Thread' = 3, 'Max' = 4) COMMENT 'Context for which memory tracker is blocked (for ClickHouse developers only): `Unknown` context is not defined for this trace_type. `Global` represents server context. `User` represents user/merge context. `Process` represents process (i.e. query) context. `Thread` represents thread (thread of particular process) context. `Max` this is a special value means that memory tracker is not blocked (for blocked_context column). ',
    `event` LowCardinality(String) COMMENT 'For trace type ProfileEvent is the name of updated profile event, for other trace types is an empty string.',
    `increment` Int64 COMMENT 'For trace type ProfileEvent is the amount of increment of profile event, for other trace types is 0.',
    `symbols` Array(LowCardinality(String)) COMMENT 'If the symbolization is enabled, contains demangled symbol names, corresponding to the `trace`. Symbolization can be enabled or disabled in the `symbolize` setting under `trace_log` in the server configuration file; the setting applies to profiler-collected trace types, while rows with the `Instrumentation` trace type are symbolized regardless of it. Symbolization is supported on ELF platforms (such as Linux) and macOS; on FreeBSD this column is always empty.',
    `lines` Array(LowCardinality(String)) COMMENT 'If the symbolization is enabled, contains strings with file names with line numbers, corresponding to the `trace`. The `symbolize` setting applies to profiler-collected trace types, while rows with the `Instrumentation` trace type are symbolized regardless of it. Symbolization is supported on ELF platforms (such as Linux) and macOS; on FreeBSD this column is always empty. Source locations are best-effort: they require debug info (a `.dSYM` bundle on macOS) and, on ELF platforms, are resolved only for frames inside the main ClickHouse binary; unresolved frames have empty entries.',
    `function_id` Nullable(Int32) COMMENT 'For trace type Instrumentation, ID assigned to the function in xray_instr_map section of elf-binary.',
    `function_name` Nullable(String) COMMENT 'For trace type Instrumentation, name of the instrumented function.',
    `handler` Nullable(String) COMMENT 'For trace type Instrumentation, handler of the instrumented function.',
    `entry_type` Nullable(Enum8('Entry' = 0, 'Exit' = 1)) COMMENT 'For trace type Instrumentation, entry type of the instrumented function.',
    `duration_nanoseconds` Nullable(UInt64) COMMENT 'For trace type Instrumentation, time the function was running for in nanoseconds.',
    `build_id` String ALIAS 'D72DAF8D1208B7097DB3B2084FC071409DD8844D',
    INDEX event_time_index event_time TYPE minmax GRANULARITY 1,
    INDEX event_time_microseconds_index event_time_microseconds TYPE minmax GRANULARITY 1,
    INDEX query_id_index query_id TYPE bloom_filter(0.001) GRANULARITY 1
)
ENGINE = MergeTree
PARTITION BY event_date
ORDER BY event_time
SETTINGS index_granularity = 8192, table_readonly = true
COMMENT 'Contains stack traces collected by the sampling query profiler.\n\nIt is safe to truncate or drop this table at any time.'
