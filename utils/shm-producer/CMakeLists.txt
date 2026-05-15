if (NOT OS_LINUX)
    message (STATUS "Not building shm-producer: depends on InProcessProducer which is Linux-only")
    return ()
endif ()

# Standalone helper for the zero-copy SHM source feature (plan task T4.1).
# Lives under utils/ so the in-process test producer does not ship inside the main
# clickhouse multi-call binary. When ENABLE_TESTS=ON the InProcessProducer translation
# unit is already part of dbms (see src/CMakeLists.txt), so we just link against dbms.
# When ENABLE_TESTS=OFF dbms does not contain it, so we compile it directly into this
# target. The two paths are mutually exclusive to avoid duplicate-symbol link errors
# (dbms is a STATIC library).

set (SHM_PRODUCER_SOURCES main.cpp)

if (NOT ENABLE_TESTS)
    list (APPEND SHM_PRODUCER_SOURCES
        ${ClickHouse_SOURCE_DIR}/src/Storages/SharedMemorySource/TestProducer/InProcessProducer.cpp
    )
endif ()

clickhouse_add_executable (shm-producer ${SHM_PRODUCER_SOURCES})

target_link_libraries (shm-producer PRIVATE
    dbms
    boost::program_options
)

# Install the binary into the standard bin directory so the integration test
# suite (tests/integration/test_shm_table_function) can resolve it via
# `command -v shm-producer` inside the test container, mirroring how every
# other ClickHouse-installed binary (clickhouse, clickhouse-keeper, ...) lands
# on PATH. Without this, the integration suite has no path to the binary and
# the start_cluster fixture in test.py fails LOUDLY (pytest.fail) per the
# AC1/AC4/AC5/AC6/AC7/AC10 coverage requirement.
#
# NOTE on the docker-compose mount gap: the integration runner mounts the
# host's `clickhouse` binary into the container as a single-file bind
# (`server_bin_path:/usr/bin/clickhouse:ro`, see
# tests/integration/helpers/cluster.py); it does NOT mount the entire build
# output directory, so this install() alone does not get the binary into the
# container. Wiring the same single-file bind for shm-producer is a separate
# follow-up in cluster.py (out of scope here); this install() is the
# packaging-side prerequisite for that follow-up.
install (TARGETS shm-producer
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    COMPONENT clickhouse
)
