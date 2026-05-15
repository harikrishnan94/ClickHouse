/// shm-producer — standalone driver around InProcessProducer for the zero-copy
/// SHM source feature (plan task T4.1). Generates deterministic AC1 data
/// (id UInt64, v1 UInt64, v2 UInt64, s1 String, s2 String) into POSIX SHM via the
/// SHM-adoption wire ABI and exposes scenarios covering the AC6 producer-misbehaviour
/// matrix plus the AC10 republish-under-retain point.
///
/// Lives in utils/ rather than programs/ so it does not ship inside the multi-call
/// clickhouse binary; the in-process test producer is then kept out of the dbms
/// library on release builds (src/CMakeLists.txt gates it on ENABLE_TESTS).

#if defined(OS_LINUX)

#include <Storages/SharedMemorySource/TestProducer/InProcessProducer.h>
#include <Storages/SharedMemorySource/Wire/ControlSocket.h>
#include <Storages/SharedMemorySource/Wire/Layout.h>
#include <Common/Exception.h>

#include <boost/program_options.hpp>

#include <city.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <random>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <unistd.h>

namespace
{

namespace po = boost::program_options;
using DB::InProcessProducer;
using DB::SharedMemoryWire::PADDING_FOR_SIMD;

/// Set by SIGTERM/SIGINT so the main thread exits and the InProcessProducer dtor
/// gets to unlink the SHM object and the control socket.
std::atomic<int> g_shutdown_signal{0};

void shutdownHandler(int signum) noexcept
{
    g_shutdown_signal.store(signum, std::memory_order_release);
}

/// Deterministic AC1 row generator. One std::mt19937_64 drives every draw in a
/// fixed per-row order — v1, v2, s1_len ∈ [0,31], s1_chars[…], s2_len ∈ [0,255],
/// s2_chars[…] — so (seed, start_row_id, rows_in_block) uniquely determines the
/// block bytes; test harnesses can re-derive expected values from the seed alone.
/// id is not drawn — id[i] = start_row_id + i (AC1 contract).
///
/// Trailing PADDING_FOR_SIMD bytes are appended to every value buffer as a safety
/// margin against any SIMD-style overread past `value_count`. The producer copies
/// exactly value_count / offset_count bytes; the padding never goes on the wire.
struct BlockBuilder
{
    std::mt19937_64 rng;
    std::vector<uint64_t> id_buf, v1_buf, v2_buf, s1_offsets, s2_offsets;
    std::vector<uint8_t>  s1_chars, s2_chars;

    explicit BlockBuilder(uint64_t seed) : rng(seed) {}

    void buildOneBlock(uint64_t start_row_id, uint64_t rows_in_block)
    {
        id_buf.clear(); v1_buf.clear(); v2_buf.clear();
        s1_chars.clear(); s1_offsets.clear();
        s2_chars.clear(); s2_offsets.clear();
        id_buf.reserve(rows_in_block + 8);
        v1_buf.reserve(rows_in_block + 8);
        v2_buf.reserve(rows_in_block + 8);
        s1_offsets.reserve(rows_in_block + 8);
        s2_offsets.reserve(rows_in_block + 8);

        for (uint64_t i = 0; i < rows_in_block; ++i)
        {
            id_buf.push_back(start_row_id + i);
            v1_buf.push_back(rng());
            v2_buf.push_back(rng());
            const size_t s1_len = static_cast<size_t>(rng() & 0x1FULL);
            for (size_t j = 0; j < s1_len; ++j)
                s1_chars.push_back(static_cast<uint8_t>(rng() & 0xFFULL));
            s1_offsets.push_back(s1_chars.size());
            const size_t s2_len = static_cast<size_t>(rng() & 0xFFULL);
            for (size_t j = 0; j < s2_len; ++j)
                s2_chars.push_back(static_cast<uint8_t>(rng() & 0xFFULL));
            s2_offsets.push_back(s2_chars.size());
        }

        constexpr size_t pad_u64 = (PADDING_FOR_SIMD + sizeof(uint64_t) - 1) / sizeof(uint64_t);
        id_buf.resize(id_buf.size() + pad_u64, 0);
        v1_buf.resize(v1_buf.size() + pad_u64, 0);
        v2_buf.resize(v2_buf.size() + pad_u64, 0);
        s1_offsets.resize(s1_offsets.size() + pad_u64, 0);
        s2_offsets.resize(s2_offsets.size() + pad_u64, 0);
        s1_chars.resize(s1_chars.size() + PADDING_FOR_SIMD, 0);
        s2_chars.resize(s2_chars.size() + PADDING_FOR_SIMD, 0);
    }

    std::vector<InProcessProducer::ColumnPayload> makePayloads(uint64_t rows_in_block) const
    {
        const size_t s1_bytes = (rows_in_block == 0) ? 0 : s1_offsets[rows_in_block - 1];
        const size_t s2_bytes = (rows_in_block == 0) ? 0 : s2_offsets[rows_in_block - 1];
        return {
            {id_buf.data(), rows_in_block, nullptr, 0},
            {v1_buf.data(), rows_in_block, nullptr, 0},
            {v2_buf.data(), rows_in_block, nullptr, 0},
            {s1_chars.data(), s1_bytes, s1_offsets.data(), rows_in_block},
            {s2_chars.data(), s2_bytes, s2_offsets.data(), rows_in_block},
        };
    }
};

/// Sleep in short slices so the SIGTERM/SIGINT flag is observed promptly.
void sleepUntilShutdown(std::chrono::seconds budget)
{
    using clock = std::chrono::steady_clock;
    const auto deadline = clock::now() + budget;
    while (g_shutdown_signal.load(std::memory_order_acquire) == 0 && clock::now() < deadline)
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
}

/// AC1 reference path. Replays BlockBuilder's per-row rng draws — v1, v2, s1_len,
/// s1_chars[…], s2_len, s2_chars[…] — with a single std::mt19937_64 seeded with
/// `seed`, then computes the AC1 8-tuple in process: count, sum(id), sum(v1),
/// sum(v2), sum(cityHash64(s1)), sum(cityHash64(s2)), sum(length(s1)),
/// sum(length(s2)). id is not drawn (id[i] = i, matching BlockBuilder's
/// id_buf.push_back(start_row_id + i) over the single sweep start_row_id = 0).
/// Output is TAB-separated on one line so the integration test can compare it
/// byte-for-byte against `node.query(AC1_QUERY_TEMPLATE).strip().split('\\t')`.
/// CityHash_v1_0_2::CityHash64 is the same implementation ClickHouse's SQL
/// cityHash64 function dispatches to for String arguments — see
/// src/Functions/FunctionsHashing.h::ImplCityHash64::apply (cited from
/// the spec: system.md AC1 "compared against a reference path").
void printReferenceValues(uint64_t rows, uint64_t seed)
{
    std::mt19937_64 rng(seed);
    uint64_t sum_id = 0;
    uint64_t sum_v1 = 0;
    uint64_t sum_v2 = 0;
    uint64_t sum_hash_s1 = 0;
    uint64_t sum_hash_s2 = 0;
    uint64_t sum_len_s1 = 0;
    uint64_t sum_len_s2 = 0;
    std::string s1_buf;
    std::string s2_buf;
    for (uint64_t i = 0; i < rows; ++i)
    {
        sum_id += i;
        sum_v1 += rng();
        sum_v2 += rng();
        const size_t s1_len = static_cast<size_t>(rng() & 0x1FULL);
        s1_buf.resize(s1_len);
        for (size_t j = 0; j < s1_len; ++j)
            s1_buf[j] = static_cast<char>(rng() & 0xFFULL);
        sum_hash_s1 += CityHash_v1_0_2::CityHash64(s1_buf.data(), s1_len);
        sum_len_s1 += s1_len;
        const size_t s2_len = static_cast<size_t>(rng() & 0xFFULL);
        s2_buf.resize(s2_len);
        for (size_t j = 0; j < s2_len; ++j)
            s2_buf[j] = static_cast<char>(rng() & 0xFFULL);
        sum_hash_s2 += CityHash_v1_0_2::CityHash64(s2_buf.data(), s2_len);
        sum_len_s2 += s2_len;
    }
    std::cout << rows << '\t' << sum_id << '\t' << sum_v1 << '\t' << sum_v2 << '\t'
              << sum_hash_s1 << '\t' << sum_hash_s2 << '\t' << sum_len_s1 << '\t'
              << sum_len_s2 << '\n';
}

int run(int argc, char ** argv)
{
    po::options_description desc("shm-producer options");
    desc.add_options()
        ("help,h", "print help")
        ("name", po::value<std::string>()->required(),
            "SHM object name (leading '/' added automatically)")
        ("rows", po::value<uint64_t>()->required(), "total rows to publish")
        ("seed", po::value<uint64_t>()->required(), "random seed (deterministic)")
        ("ring-depth", po::value<uint32_t>()->default_value(4), "K, ring buffer depth")
        ("data-region-size", po::value<size_t>()->default_value(16U * 1024U * 1024U),
            "data region size in bytes")
        ("rows-per-block", po::value<uint64_t>()->default_value(4096),
            "rows per published block")
        ("scenario", po::value<std::string>()->default_value("normal"),
            "normal|crash-after|stall-after|abort-mid-publish|block-framing-invalid|"
            "mid-publication-crash|republish-after-retain|socket-missing")
        ("scenario-arg", po::value<uint64_t>()->default_value(10),
            "scenario parameter (e.g. blocks before crash/stall/abort)")
        ("print-reference-values",
            "compute the AC1 reference 8-tuple (count, sum_id, sum_v1, sum_v2, "
            "sum(cityHash64(s1)), sum(cityHash64(s2)), sum(length(s1)), "
            "sum(length(s2))) from (--rows, --seed) and exit; bypasses --name "
            "and --scenario requirements (no SHM segment is created)");

    po::variables_map vm;
    try
    {
        po::store(po::parse_command_line(argc, argv, desc), vm);
        if (vm.contains("help")) { std::cout << desc << "\n"; return 0; }
        if (vm.contains("print-reference-values"))
        {
            if (!vm.contains("rows") || !vm.contains("seed"))
                throw std::runtime_error("shm-producer: --print-reference-values requires --rows and --seed");
            printReferenceValues(vm["rows"].as<uint64_t>(), vm["seed"].as<uint64_t>());
            return 0;
        }
        po::notify(vm);
    }
    catch (const std::exception & e)
    {
        std::cerr << "shm-producer: " << e.what() << "\n\n" << desc << "\n";
        return 1;
    }

    struct sigaction sa{};
    /// glibc defines `sa_handler` as a recursive macro (`__sigaction_handler.sa_handler`),
    /// which triggers `-Wdisabled-macro-expansion`.
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdisabled-macro-expansion"
    sa.sa_handler = &shutdownHandler;
#pragma clang diagnostic pop
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = SA_RESTART;
    ::sigaction(SIGTERM, &sa, nullptr);
    ::sigaction(SIGINT, &sa, nullptr);

    InProcessProducer::Config cfg{
        .shm_name = vm["name"].as<std::string>(),
        .ring_depth_k = vm["ring-depth"].as<uint32_t>(),
        .schema = {{"id", "UInt64"}, {"v1", "UInt64"}, {"v2", "UInt64"},
                   {"s1", "String"}, {"s2", "String"}},
        .data_region_size = vm["data-region-size"].as<size_t>(),
    };

    InProcessProducer producer(std::move(cfg));
    while (!producer.isReady())
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    std::cout << "shm-producer ready. shm_name=" << producer.shmName() << "\n"
              << std::flush;

    const uint64_t total_rows = vm["rows"].as<uint64_t>();
    const uint64_t rows_per_block = vm["rows-per-block"].as<uint64_t>();
    const std::string scenario = vm["scenario"].as<std::string>();
    const uint64_t scenario_arg = vm["scenario-arg"].as<uint64_t>();

    /// AC6 row "readiness-fd locator unresolvable" -> SHM_ATTACH_FAILED. The SHM
    /// segment + handshake are valid; we just unlink the control-socket path so a
    /// consumer's connect() to /tmp/clickhouse_shm_<name>.sock returns ENOENT.
    /// unlink() removes only the filesystem entry — the bound listen fd stays open
    /// inside the producer (no consumer ever reaches accept), and the dtor still
    /// gets to unmap + shm_unlink the SHM on SIGTERM.
    if (scenario == "socket-missing")
    {
        const auto sock_path = DB::controlSocketPathForShmName(producer.shmName());
        if (::unlink(sock_path.c_str()) != 0)
            std::cerr << "scenario=socket-missing: unlink(" << sock_path << ") failed: "
                      << ::strerror(errno) << "\n" << std::flush;
        else
            std::cerr << "scenario=socket-missing: unlinked " << sock_path << "\n" << std::flush;
        std::cout << "shm-producer: socket-missing scenario active. Awaiting orchestrator signal.\n"
                  << std::flush;
        sleepUntilShutdown(std::chrono::hours(24));
        return 0;
    }

    BlockBuilder builder(vm["seed"].as<uint64_t>());
    uint64_t rows_published = 0;
    uint64_t blocks_published = 0;
    bool stalled = false;

    while (rows_published < total_rows
            && g_shutdown_signal.load(std::memory_order_acquire) == 0)
    {
        const uint64_t rows_in_block = std::min(rows_per_block, total_rows - rows_published);

        if (scenario == "crash-after" && blocks_published == scenario_arg)
        {
            std::cerr << "scenario=crash-after: ungraceful exit after " << blocks_published
                      << " blocks\n" << std::flush;
            InProcessProducer::forceUngracefulExit();
        }
        /// AC6 row "producer crash mid-publication" -> SHM_BLOCK_FRAMING_INVALID.
        /// After `scenario_arg` complete publishes we use the test-only escape hatch
        /// `setSlotStateForTesting(slot, WRITING)` to leave the NEXT slot in WRITING
        /// (emulating a producer that flipped EMPTY->WRITING and crashed before
        /// WRITING->PUBLISHED), then `_exit(1)`. The brief sleep before exiting gives
        /// the integration-test consumer time to attach via the live control socket
        /// AND drain the prior PUBLISHED blocks BEFORE the peer-end close fires; that
        /// makes the resulting POLLHUP land on an ESTABLISHED connection so
        /// `PollableShmSource::checkProducerDeath` runs, finds the WRITING slot, and
        /// throws SHM_BLOCK_FRAMING_INVALID (NOT ECONNREFUSED at connect(), which is
        /// SHM_ATTACH_FAILED). Mirrors the 2 s sleep in
        /// gtest_pollable_shm_source.cpp::ProducerCrashMidPublicationYields*.
        if (scenario == "mid-publication-crash" && blocks_published == scenario_arg)
        {
            const uint32_t ring_depth_k = vm["ring-depth"].as<uint32_t>();
            const uint32_t target_slot = static_cast<uint32_t>(blocks_published % ring_depth_k);
            std::cerr << "scenario=mid-publication-crash: setting slot " << target_slot
                      << " to WRITING then ungraceful exit after " << blocks_published
                      << " blocks\n" << std::flush;
            producer.setSlotStateForTesting(target_slot,
                DB::SharedMemoryWire::SlotState::WRITING);
            std::this_thread::sleep_for(std::chrono::milliseconds(2000));
            InProcessProducer::forceUngracefulExit();
        }
        if (scenario == "stall-after" && blocks_published == scenario_arg)
        {
            std::cerr << "scenario=stall-after: stalling after " << blocks_published
                      << " blocks\n" << std::flush;
            producer.stallProducer();
            stalled = true;
            break;
        }
        if (scenario == "republish-after-retain" && blocks_published == 1)
        {
            std::cerr << "scenario=republish-after-retain: waiting for slot-0 retain to release\n"
                      << std::flush;
            producer.waitForRetainToRelease(0);
        }

        builder.buildOneBlock(rows_published, rows_in_block);
        const auto payloads = builder.makePayloads(rows_in_block);

        if (scenario == "abort-mid-publish" && blocks_published == scenario_arg)
        {
            std::cerr << "scenario=abort-mid-publish: OffsetOverflow malformation at block "
                      << blocks_published << "\n" << std::flush;
            producer.publishMalformedBlock(payloads, rows_in_block,
                InProcessProducer::Malformation::OffsetOverflow);
        }
        else if (scenario == "block-framing-invalid" && blocks_published == scenario_arg)
        {
            std::cerr << "scenario=block-framing-invalid: BadSlotIdentity malformation at block "
                      << blocks_published << "\n" << std::flush;
            producer.publishMalformedBlock(payloads, rows_in_block,
                InProcessProducer::Malformation::BadSlotIdentity);
        }
        else
        {
            producer.publishBlock(payloads, rows_in_block);
        }

        rows_published += rows_in_block;
        ++blocks_published;
    }

    /// EOS only on a clean run. crash-after never reaches here. stall-after must NOT signal
    /// EOS (consumer must observe the stall). abort-mid-publish does signal EOS after the
    /// malformed block; the consumer is expected to fail well before observing EOS.
    if (!stalled && g_shutdown_signal.load(std::memory_order_acquire) == 0)
    {
        try { producer.signalEndOfStream(); }
        catch (const std::exception & e)
        {
            std::cerr << "shm-producer: signalEndOfStream failed: " << e.what() << "\n";
        }
    }

    std::cout << "shm-producer: published " << blocks_published << " blocks ("
              << rows_published << " rows). Awaiting orchestrator signal.\n" << std::flush;

    /// Stay alive until SIGTERM/SIGINT so the consumer can drain the ring at its own pace.
    /// republish-after-retain caps the wait because that integration test is short-lived.
    sleepUntilShutdown(scenario == "republish-after-retain"
        ? std::chrono::seconds(30) : std::chrono::hours(24));
    return 0;
}

}

int main(int argc, char ** argv)
{
    try { return run(argc, argv); }
    catch (...)
    {
        std::cerr << "shm-producer fatal: "
                  << DB::getCurrentExceptionMessage(/*with_stacktrace=*/true) << "\n";
        return 1;
    }
}

#else

#include <iostream>
int main(int, char **)
{
    std::cerr << "shm-producer is only supported on Linux\n";
    return 1;
}

#endif
