#include <gtest/gtest.h>

#if defined(OS_LINUX)

#    include <Storages/SharedMemorySource/Source/PollableShmSource.h>
#    include <Storages/SharedMemorySource/TestProducer/InProcessProducer.h>
#    include <Storages/SharedMemorySource/Wire/Layout.h>

#    include <Columns/ColumnString.h>
#    include <Columns/ColumnsNumber.h>
#    include <Columns/IColumn.h>
#    include <Core/Block.h>
#    include <DataTypes/DataTypeString.h>
#    include <DataTypes/DataTypesNumber.h>
#    include <Processors/Chunk.h>
#    include <Processors/Executors/PullingPipelineExecutor.h>
#    include <Processors/IProcessor.h>
#    include <Processors/ISource.h>
#    include <QueryPipeline/QueryPipeline.h>
#    include <Common/Exception.h>

#    include <base/errnoToString.h>

#    include <fmt/format.h>

#    include <fcntl.h>
#    include <unistd.h>
#    include <sys/mman.h>
#    include <sys/stat.h>
#    include <sys/wait.h>

#    include <atomic>
#    include <cerrno>
#    include <chrono>
#    include <cstdint>
#    include <exception>
#    include <memory>
#    include <string>
#    include <string_view>
#    include <thread>
#    include <vector>


using namespace DB;
using namespace DB::SharedMemoryWire;

namespace DB::ErrorCodes
{
    extern const int SHM_BLOCK_FRAMING_INVALID;
    extern const int SHM_PRODUCER_STALL;
    extern const int SHM_PRODUCER_DEATH_BEFORE_EOS;
}

namespace
{

std::string uniqueShmName(const char * tag)
{
    return fmt::format("test_src_{}_{}", tag, ::getpid());
}

InProcessProducer::Config defaultConfig(const char * tag, uint32_t k = 4)
{
    InProcessProducer::Config cfg;
    cfg.shm_name = uniqueShmName(tag);
    cfg.ring_depth_k = k;
    cfg.schema = {{"id", "UInt64"}, {"s", "String"}};
    cfg.data_region_size = 256 * 1024;
    return cfg;
}

SharedHeader makeHeader()
{
    Block b;
    b.insert({std::make_shared<DataTypeUInt64>()->createColumn(), std::make_shared<DataTypeUInt64>(), "id"});
    b.insert({std::make_shared<DataTypeString>()->createColumn(), std::make_shared<DataTypeString>(), "s"});
    return std::make_shared<const Block>(std::move(b));
}

std::shared_ptr<PollableShmSource> makeSource(const std::string & shm_name, UInt64 stall_ms = 60'000)
{
    /// `requested_column_names` matches the full schema in these tests (no projection).
    return std::make_shared<PollableShmSource>(
        makeHeader(),
        shm_name,
        std::vector<DataTypePtr>{std::make_shared<DataTypeUInt64>(), std::make_shared<DataTypeString>()},
        std::vector<String>{"id", "s"},
        std::vector<String>{"id", "s"},
        stall_ms);
}

InProcessProducer::ColumnPayload uint64Payload(const std::vector<uint64_t> & v)
{
    return {v.data(), v.size(), nullptr, 0};
}

InProcessProducer::ColumnPayload stringPayload(const std::vector<uint8_t> & chars, const std::vector<uint64_t> & offs)
{
    return {chars.data(), chars.size(), offs.data(), offs.size()};
}

}


/// I6 (Pollable contract): producer publishes N blocks + EOS; the source drains
/// them through the PullingPipelineExecutor (the same pipeline harness that
/// production queries use). Asserts the chunks come out in producer-publication
/// order with the right shape. The AC3 pointer-identity check is owned by T5.1.
TEST(PollableShmSource, DrainsAllPublishedBlocks)
{
    constexpr size_t n_blocks = 20;
    InProcessProducer producer(defaultConfig("drain"));
    ASSERT_TRUE(producer.isReady());

    /// Producer thread publishes a stream of small blocks; it WILL block on the
    /// ring being full (K=4) until the consumer drops the retain — that's
    /// exactly the cooperation point we want to exercise here.
    std::thread producer_thread(
        [&]()
        {
            for (size_t b = 0; b < n_blocks; ++b)
            {
                const std::vector<uint64_t> ids = {b, b + 100, b + 200};
                const std::vector<uint8_t> chars = {'a', 'b', 'c'};
                const std::vector<uint64_t> offs = {1, 2, 3};
                producer.publishBlock({uint64Payload(ids), stringPayload(chars, offs)}, ids.size());
            }
            producer.signalEndOfStream();
        });

    auto src = makeSource(producer.shmName());
    QueryPipeline pipeline(src);
    PullingPipelineExecutor executor(pipeline);

    size_t chunks = 0;
    Chunk chunk;
    while (executor.pull(chunk))
    {
        if (chunk.hasRows())
        {
            ASSERT_EQ(chunk.getNumRows(), 3u) << "block " << chunks;
            ASSERT_EQ(chunk.getNumColumns(), 2u);
            const auto & cols = chunk.getColumns();
            const auto * id_col = typeid_cast<const ColumnUInt64 *>(cols[0].get());
            ASSERT_NE(id_col, nullptr);
            EXPECT_EQ(id_col->getData()[0], chunks);
            EXPECT_EQ(id_col->getData()[1], chunks + 100);
            EXPECT_EQ(id_col->getData()[2], chunks + 200);
            ++chunks;
        }
    }
    producer_thread.join();

    ASSERT_EQ(chunks, n_blocks);
}


/// I6: a single attach + drain through PullingPipelineExecutor, with the
/// producer publishing BEFORE the consumer attaches. Verifies the async path
/// is wired correctly (the source goes Async until the eventfd wakes it).
TEST(PollableShmSource, AsyncWakeOnPublication)
{
    InProcessProducer producer(defaultConfig("async"));

    auto src = makeSource(producer.shmName());
    QueryPipeline pipeline(src);
    PullingPipelineExecutor executor(pipeline);

    /// Publish from a worker so the executor is forced into Async at least once.
    std::thread producer_thread(
        [&]()
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            const std::vector<uint64_t> ids = {77};
            const std::vector<uint8_t> chars = {'q'};
            const std::vector<uint64_t> offs = {1};
            producer.publishBlock({uint64Payload(ids), stringPayload(chars, offs)}, 1);
            producer.signalEndOfStream();
        });

    std::vector<Chunk> chunks;
    Chunk chunk;
    while (executor.pull(chunk))
    {
        if (chunk.hasRows())
            chunks.emplace_back(std::move(chunk));
    }
    producer_thread.join();

    ASSERT_EQ(chunks.size(), 1u);
    EXPECT_EQ(chunks[0].getNumRows(), 1u);
}


/// I9 (Cancellation is bounded): producer stalls (never publishes); a cancel
/// from another thread terminates the executor well before the I12 stall budget
/// (60s default), without producer cooperation.
TEST(PollableShmSource, CancelMidStreamUnblocks)
{
    InProcessProducer producer(defaultConfig("cancel"));
    auto src = makeSource(producer.shmName(), /*stall_ms=*/60'000);
    QueryPipeline pipeline(src);
    PullingPipelineExecutor executor(pipeline);

    /// pull() blocks waiting on the source (which is in Async). Cancel from
    /// another thread.
    std::atomic<bool> stopped{false};
    std::thread cancel_thread(
        [&]()
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            executor.cancel();
            stopped.store(true, std::memory_order_release);
        });

    const auto t0 = std::chrono::steady_clock::now();
    Chunk chunk;
    while (executor.pull(chunk))
    {
        // drain whatever (likely nothing) leaks out before cancel takes effect
    }
    const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t0).count();

    cancel_thread.join();
    EXPECT_TRUE(stopped.load(std::memory_order_acquire));
    /// Keep this comfortably under the 60-second stall budget while allowing
    /// debug/sanitizer builds more scheduling and instrumentation slack.
#    if defined(DEBUG_OR_SANITIZER_BUILD)
    constexpr auto cancellation_bound_ms = 10'000;
#    else
    constexpr auto cancellation_bound_ms = 2'000;
#    endif
    EXPECT_LT(elapsed, cancellation_bound_ms);
}


/// Meta Q4: multiple stalled sources should all be woken by cancellation
/// without waiting for the per-source stall timeout or producer cooperation.
TEST(PollableShmSource, CancelManyStalledSourcesUnblocks)
{
    constexpr size_t n_sources = 10;

    std::vector<std::unique_ptr<InProcessProducer>> producers;
    std::vector<std::shared_ptr<PollableShmSource>> sources;
    std::vector<std::unique_ptr<QueryPipeline>> pipelines;
    std::vector<std::unique_ptr<PullingPipelineExecutor>> executors;
    producers.reserve(n_sources);
    sources.reserve(n_sources);
    pipelines.reserve(n_sources);
    executors.reserve(n_sources);

    for (size_t i = 0; i < n_sources; ++i)
    {
        const std::string tag = fmt::format("cancel_many_{}", i);
        auto producer = std::make_unique<InProcessProducer>(defaultConfig(tag.c_str()));
        ASSERT_TRUE(producer->isReady());

        sources.emplace_back(makeSource(producer->shmName(), /*stall_ms=*/60'000));
        pipelines.emplace_back(std::make_unique<QueryPipeline>(sources.back()));
        executors.emplace_back(std::make_unique<PullingPipelineExecutor>(*pipelines.back()));
        producers.emplace_back(std::move(producer));
    }

    std::atomic<size_t> entered{0};
    std::atomic<size_t> exited{0};
    std::vector<std::exception_ptr> errors(n_sources);
    std::vector<std::thread> workers;
    workers.reserve(n_sources);

    for (size_t i = 0; i < n_sources; ++i)
    {
        workers.emplace_back(
            [&, i]()
            {
                entered.fetch_add(1, std::memory_order_release);
                try
                {
                    Chunk chunk;
                    while (executors[i]->pull(chunk))
                    {
                    }
                }
                catch (...)
                {
                    errors[i] = std::current_exception();
                }
                exited.fetch_add(1, std::memory_order_release);
            });
    }

    const auto enter_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (entered.load(std::memory_order_acquire) < n_sources && std::chrono::steady_clock::now() < enter_deadline)
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    if (entered.load(std::memory_order_acquire) != n_sources)
    {
        for (auto & executor : executors)
            executor->cancel();
        for (auto & worker : workers)
            worker.join();
        FAIL() << "Only " << entered.load(std::memory_order_acquire) << " of " << n_sources << " workers entered pull()";
    }

    /// Give each worker a brief chance to reach the Async wait before measuring
    /// cancel-to-unblock latency across the whole group.
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    const auto t0 = std::chrono::steady_clock::now();
    for (auto & executor : executors)
        executor->cancel();
    for (auto & worker : workers)
        worker.join();
    const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t0).count();

    for (size_t i = 0; i < errors.size(); ++i)
    {
        if (!errors[i])
            continue;
        try
        {
            std::rethrow_exception(errors[i]);
        }
        catch (const DB::Exception & e)
        {
            FAIL() << "worker " << i << " threw DB::Exception code " << e.code() << ": " << e.message();
        }
        catch (const std::exception & e)
        {
            FAIL() << "worker " << i << " threw std::exception: " << e.what();
        }
        catch (...)
        {
            FAIL() << "worker " << i << " threw an unknown exception";
        }
    }

    ASSERT_EQ(exited.load(std::memory_order_acquire), n_sources);
#    if defined(DEBUG_OR_SANITIZER_BUILD)
    constexpr auto cancellation_bound_ms = 20'000;
#    else
    constexpr auto cancellation_bound_ms = 5'000;
#    endif
    EXPECT_LT(elapsed, cancellation_bound_ms);
}


/// I12 (Stall is bounded): with a short stall budget and no producer
/// publication, the source surfaces SHM_PRODUCER_STALL via the executor.
TEST(PollableShmSource, StallFiresWithinBudget)
{
    InProcessProducer producer(defaultConfig("stall"));
    constexpr UInt64 budget_ms = 200;
    auto src = makeSource(producer.shmName(), budget_ms);
    QueryPipeline pipeline(src);
    PullingPipelineExecutor executor(pipeline);

    const auto t0 = std::chrono::steady_clock::now();
    bool threw = false;
    try
    {
        Chunk chunk;
        while (executor.pull(chunk))
        {
            // empty
        }
    }
    catch (const DB::Exception & e)
    {
        EXPECT_EQ(e.code(), ErrorCodes::SHM_PRODUCER_STALL);
        threw = true;
    }
    const auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t0).count();
    EXPECT_TRUE(threw);
    EXPECT_LT(elapsed_ms, 2'000);
}


/// `shm-block-stream.md` I11 + precondition 25: producer-death-before-EOS
/// surfaces as SHM_PRODUCER_DEATH_BEFORE_EOS via POLLHUP on the control socket.
TEST(PollableShmSource, ProducerDeathBeforeEosThrows)
{
    const std::string shm_name = uniqueShmName("death");

    pid_t pid = ::fork();
    ASSERT_GE(pid, 0);
    if (pid == 0)
    {
        InProcessProducer producer({shm_name, 4, {{"id", "UInt64"}, {"s", "String"}}, 256 * 1024});
        const std::vector<uint64_t> ids = {1, 2};
        const std::vector<uint8_t> chars = {'x', 'y'};
        const std::vector<uint64_t> offs = {1, 2};
        producer.publishBlock({uint64Payload(ids), stringPayload(chars, offs)}, 2);
        producer.publishBlock({uint64Payload(ids), stringPayload(chars, offs)}, 2);
        /// Sleep so the parent has a chance to attach + drain before we exit.
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        InProcessProducer::forceUngracefulExit();
    }

    /// Give the child time to populate the handshake.
    std::this_thread::sleep_for(std::chrono::milliseconds(200));

    /// Make the source with a generous stall budget so SHM_PRODUCER_STALL doesn't fire first.
    auto src = makeSource(shm_name, /*stall_ms=*/30'000);
    QueryPipeline pipeline(src);
    PullingPipelineExecutor executor(pipeline);

    /// Phase 1: drain the two complete PUBLISHED blocks WHILE child is alive.
    /// This establishes the control-socket connection before the child's fd
    /// table is closed; otherwise the lazy attach would fail with
    /// SHM_ATTACH_FAILED/ECONNREFUSED instead of exercising the POLLHUP path.
    Chunk chunk;
    size_t drained = 0;
    size_t pull_iterations = 0;
    while (drained < 2 && executor.pull(chunk))
    {
        if (chunk.hasRows())
            ++drained;
        ++pull_iterations;
        ASSERT_LT(pull_iterations, 16u) << "drain loop overran — pipeline harness regression";
    }
    ASSERT_EQ(drained, 2u);

    /// Phase 2: wait for the child to exit so POLLHUP is visible on the
    /// already-established control-socket connection.
    int status = 0;
    ::waitpid(pid, &status, 0);

    /// Phase 3: subsequent pull should observe producer-death-before-EOS.
    bool threw_death = false;
    try
    {
        while (executor.pull(chunk))
        {
        }
    }
    catch (const DB::Exception & e)
    {
        EXPECT_EQ(e.code(), ErrorCodes::SHM_PRODUCER_DEATH_BEFORE_EOS);
        threw_death = true;
    }
    EXPECT_TRUE(threw_death);
}

TEST(PollableShmSource, ProducerDeathWhileAsyncWakesExecutor)
{
    const std::string shm_name = uniqueShmName("death_async");

    pid_t pid = ::fork();
    ASSERT_GE(pid, 0);
    if (pid == 0)
    {
        InProcessProducer producer({shm_name, 4, {{"id", "UInt64"}, {"s", "String"}}, 256 * 1024});
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        InProcessProducer::forceUngracefulExit();
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(200));

    auto src = makeSource(shm_name, /*stall_ms=*/30'000);
    QueryPipeline pipeline(src);
    PullingPipelineExecutor executor(pipeline);

    bool threw_death = false;
    int observed_code = 0;
    try
    {
        Chunk chunk;
        while (executor.pull(chunk))
        {
        }
    }
    catch (const DB::Exception & e)
    {
        observed_code = e.code();
        threw_death = (e.code() == ErrorCodes::SHM_PRODUCER_DEATH_BEFORE_EOS);
    }

    int status = 0;
    ::waitpid(pid, &status, 0);
    EXPECT_TRUE(threw_death) << "Expected SHM_PRODUCER_DEATH_BEFORE_EOS, got code " << observed_code;
}


/// AC6 mid-publication-crash branch (F6). The wire distinguishes:
///   - "producer crash mid-publication (before a complete block is published)"
///     → SHM_BLOCK_FRAMING_INVALID, and
///   - "producer dying after publishing a complete block but before signalling
///     end-of-stream" → SHM_PRODUCER_DEATH_BEFORE_EOS.
/// The discriminator is whether ANY slot is in WRITING when POLLHUP is observed.
/// The child publishes 2 complete blocks (slots 0, 1 → PUBLISHED), then uses
/// the test-only `setSlotStateForTesting` helper to drive slot 2 from EMPTY to
/// WRITING — emulating a crash AFTER the producer's E→W transition but BEFORE
/// the W→P that would have completed publication — and `_exit(1)`s.
///
/// Timing constraint (lessons from the analogous ProducerDeathBeforeEosThrows
/// test): the parent's `ensureAttached()` is lazy in `prepare()`, and a
/// `connect()` to a Unix-domain socket whose listener has died returns
/// ECONNREFUSED (SHM_ATTACH_FAILED). To exercise the in-flight POLLHUP path
/// instead of the attach-failure path, the parent must drain the two
/// PUBLISHED blocks WHILE the child is still alive (so the control-socket
/// connection is established). After the child's subsequent `_exit(1)`, the
/// established connection fd surfaces POLLHUP, and the next `prepare()` ->
/// `checkProducerDeath()` scans the slot table, sees slot 2 in WRITING, and
/// throws SHM_BLOCK_FRAMING_INVALID — NOT SHM_PRODUCER_DEATH_BEFORE_EOS.
TEST(PollableShmSource, ProducerCrashMidPublicationYieldsBlockFramingInvalid)
{
    const std::string shm_name = uniqueShmName("midcrash");

    pid_t pid = ::fork();
    ASSERT_GE(pid, 0);
    if (pid == 0)
    {
        InProcessProducer producer({shm_name, 4, {{"id", "UInt64"}, {"s", "String"}}, 256 * 1024});
        const std::vector<uint64_t> ids = {1, 2};
        const std::vector<uint8_t> chars = {'x', 'y'};
        const std::vector<uint64_t> offs = {1, 2};
        /// Slot 0 → PUBLISHED, sequence=1.
        producer.publishBlock({uint64Payload(ids), stringPayload(chars, offs)}, 2);
        /// Slot 1 → PUBLISHED, sequence=1.
        producer.publishBlock({uint64Payload(ids), stringPayload(chars, offs)}, 2);
        /// Slot 2 → WRITING via the test escape hatch, simulating a producer that
        /// started E→W and crashed before W→P. The helper bumps the transition
        /// counter alongside the state store so the consumer's precondition-24
        /// monotonicity check stays satisfied.
        producer.setSlotStateForTesting(2, SlotState::WRITING);
        /// Sleep long enough for the parent to attach AND drain both PUBLISHED
        /// blocks BEFORE we exit. The drain itself takes <50ms in practice; 2s
        /// is generous slack for CI noise. We intentionally do NOT
        /// signalEndOfStream — the _exit(1) is the crash signal.
        std::this_thread::sleep_for(std::chrono::milliseconds(2000));
        InProcessProducer::forceUngracefulExit();
    }

    /// Give the child time to populate the handshake AND publish both blocks.
    std::this_thread::sleep_for(std::chrono::milliseconds(200));

    /// Make the source with a generous stall budget so SHM_PRODUCER_STALL doesn't fire first.
    auto src = makeSource(shm_name, /*stall_ms=*/30'000);
    QueryPipeline pipeline(src);
    PullingPipelineExecutor executor(pipeline);

    /// Phase 1: drain the two complete PUBLISHED blocks WHILE child is alive.
    /// This is what establishes the parent's control-socket connection (lazy
    /// `ensureAttached` runs on the first pull). The drain is bounded by a
    /// counter to keep CI fast and detect harness regressions.
    Chunk chunk;
    std::vector<Chunk> drained;
    size_t pull_iterations = 0;
    while (drained.size() < 2 && executor.pull(chunk))
    {
        if (chunk.hasRows())
            drained.emplace_back(std::move(chunk));
        ++pull_iterations;
        ASSERT_LT(pull_iterations, 16u) << "drain loop overran — pipeline harness regression";
    }
    ASSERT_EQ(drained.size(), 2u);

    /// Phase 2: wait for child to crash. Established control-socket connection
    /// will surface POLLHUP after the child's fd table is reaped.
    int status = 0;
    ::waitpid(pid, &status, 0);

    /// Phase 3: subsequent pull must throw SHM_BLOCK_FRAMING_INVALID.
    /// The third pull's tryGenerate finds no PUBLISHED slot, transitions the
    /// source into async state, and the next prepare() invokes
    /// checkProducerDeath which detects POLLHUP, scans the slot table, sees
    /// slot 2 in WRITING, and throws.
    bool threw_block_framing_invalid = false;
    int observed_code = 0;
    try
    {
        while (executor.pull(chunk))
        {
        }
    }
    catch (const DB::Exception & e)
    {
        observed_code = e.code();
        threw_block_framing_invalid = (e.code() == ErrorCodes::SHM_BLOCK_FRAMING_INVALID);
    }
    EXPECT_TRUE(threw_block_framing_invalid) << "Expected SHM_BLOCK_FRAMING_INVALID, got code " << observed_code;
}


/// Producer that exits AFTER EOS is not an error; the source must finish cleanly.
TEST(PollableShmSource, ProducerDeathAfterEosOk)
{
    const std::string shm_name = uniqueShmName("clean");

    pid_t pid = ::fork();
    ASSERT_GE(pid, 0);
    if (pid == 0)
    {
        InProcessProducer producer({shm_name, 4, {{"id", "UInt64"}, {"s", "String"}}, 256 * 1024});
        const std::vector<uint64_t> ids = {1};
        const std::vector<uint8_t> chars = {'a'};
        const std::vector<uint64_t> offs = {1};
        producer.publishBlock({uint64Payload(ids), stringPayload(chars, offs)}, 1);
        producer.signalEndOfStream();
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        InProcessProducer::forceUngracefulExit();
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    auto src = makeSource(shm_name, /*stall_ms=*/30'000);
    QueryPipeline pipeline(src);
    PullingPipelineExecutor executor(pipeline);

    /// Drain the complete stream while the child is still alive so the
    /// control socket is established before `_exit(1)`. Producer death AFTER
    /// EOS is not an error.
    size_t chunks = 0;
    Chunk chunk;
    while (executor.pull(chunk))
    {
        if (chunk.hasRows())
            ++chunks;
    }
    ASSERT_EQ(chunks, 1u);

    int status = 0;
    ::waitpid(pid, &status, 0);
}


/// system spec I5 + shm-block-stream spec I11: an adopted Chunk must remain safe to
/// read AFTER the PollableShmSource that produced it has been destroyed. The
/// consumer's SHM mapping is pinned for the lifetime of every RetainToken alias,
/// so any column the user holds keeps the mapping address-valid until the column
/// itself drops. This verifies the shared-ptr-capture behaviour wired through
/// `PollableShmSource::drainSlot`.
///
/// Failure mode without the fix: `source` destruction unmaps the consumer's
/// region, leaving the held chunk's columns pointing at unmapped memory; the
/// reads below SIGSEGV/SIGBUS, and the RetainToken deleter's
/// `slot_capture->retain_refcount` write faults likewise.
TEST(PollableShmSource, ChunkOutlivesSource)
{
    Chunk held_chunk;

    {
        InProcessProducer producer(defaultConfig("outlives"));
        ASSERT_TRUE(producer.isReady());

        const std::vector<uint64_t> ids = {11, 22, 33};
        const std::vector<uint8_t> chars = {'p', 'q', 'r'};
        const std::vector<uint64_t> offs = {1, 2, 3};
        producer.publishBlock({uint64Payload(ids), stringPayload(chars, offs)}, ids.size());
        producer.signalEndOfStream();

        auto src = makeSource(producer.shmName());
        QueryPipeline pipeline(src);
        PullingPipelineExecutor executor(pipeline);

        Chunk chunk;
        while (executor.pull(chunk))
        {
            if (chunk.hasRows())
            {
                held_chunk = std::move(chunk);
                break;
            }
        }
        ASSERT_GT(held_chunk.getNumRows(), 0u);

        /// `executor`, `pipeline`, `src` and `producer` are all torn down at the
        /// end of this scope. The source's shared_ptr<SharedMemoryRegion> reference
        /// drops here; the only remaining references live inside the held chunk's
        /// adopted-column RetainTokens.
    }

    /// Source is gone; the chunk must still be safely readable because each
    /// adopted column's RetainToken closure captured a shared_ptr alias to the
    /// SHM region, pinning the mapping past source destruction.
    EXPECT_EQ(held_chunk.getNumRows(), 3u);
    ASSERT_EQ(held_chunk.getNumColumns(), 2u);

    const auto & cols = held_chunk.getColumns();
    {
        const auto * id_col = typeid_cast<const ColumnUInt64 *>(cols[0].get());
        ASSERT_NE(id_col, nullptr);
        const auto & data = id_col->getData();
        ASSERT_EQ(data.size(), 3u);
        EXPECT_EQ(data[0], 11u);
        EXPECT_EQ(data[1], 22u);
        EXPECT_EQ(data[2], 33u);
    }
    {
        const auto * s_col = typeid_cast<const ColumnString *>(cols[1].get());
        ASSERT_NE(s_col, nullptr);
        ASSERT_EQ(s_col->size(), 3u);
        EXPECT_EQ(s_col->getDataAt(0), std::string_view("p", 1));
        EXPECT_EQ(s_col->getDataAt(1), std::string_view("q", 1));
        EXPECT_EQ(s_col->getDataAt(2), std::string_view("r", 1));
    }

    /// On test exit, held_chunk's dtor releases its columns → last RetainToken
    /// alias drops → deleter lambda runs (the slot's retain_refcount-- write
    /// goes into the still-mapped region) → region_capture's shared_ptr drops →
    /// refcount reaches zero → ~SharedMemoryRegion unmaps and closes the fd.
}


/// Finding 2: the producer that keeps its accept-side socket open for its lifetime must
/// NOT trip the consumer's `checkProducerDeath()` POLLHUP false-positive. We publish 5
/// blocks + EOS and drain them through a normal pipeline; the only acceptable
/// terminating condition is clean EOS, with no `SHM_PRODUCER_DEATH_BEFORE_EOS` raised
/// (which is exactly what the immediately-close accept loop would surface — Finding 2).
///
/// We use K=8 (slot ring strictly larger than n_blocks+1=6) so the producer never has to
/// wait on slot reuse; combined with the per-loop chunk-overwrite below this keeps the
/// drain pipeline single-buffered and exercises only the producer-lifetime / POLLHUP
/// path under test, not slot reuse (which RingFullBlocksUntilRelease covers separately).
TEST(PollableShmSource, LongLivedProducerDoesNotTriggerDeath)
{
    constexpr size_t n_blocks = 5;
    InProcessProducer producer(defaultConfig("longlived", /*k=*/8));
    ASSERT_TRUE(producer.isReady());

    std::thread producer_thread(
        [&]()
        {
            for (size_t b = 0; b < n_blocks; ++b)
            {
                const std::vector<uint64_t> ids = {b + 1};
                const std::vector<uint8_t> chars = {static_cast<uint8_t>('a' + b)};
                const std::vector<uint64_t> offs = {1};
                producer.publishBlock({uint64Payload(ids), stringPayload(chars, offs)}, 1);
            }
            producer.signalEndOfStream();
        });

    /// Generous stall budget so we don't race with SHM_PRODUCER_STALL; the assertion
    /// we care about is that no SHM_PRODUCER_DEATH_BEFORE_EOS fires while the producer
    /// is alive and serving the connection.
    auto src = makeSource(producer.shmName(), /*stall_ms=*/30'000);
    QueryPipeline pipeline(src);
    PullingPipelineExecutor executor(pipeline);

    /// Record the first id we observe per chunk into a side buffer. We intentionally
    /// do NOT accumulate the chunks themselves — every `pull(chunk)` overwrites the
    /// previous chunk, dropping its RetainToken and freeing the slot for producer
    /// reuse. The assertion target is the POLLHUP-false-positive path, not whether
    /// chunks can be retained across pulls.
    std::vector<uint64_t> ids_seen;
    Chunk chunk;
    while (executor.pull(chunk))
    {
        if (chunk.hasRows())
        {
            const auto * id_col = typeid_cast<const ColumnUInt64 *>(chunk.getColumns()[0].get());
            ASSERT_NE(id_col, nullptr);
            ids_seen.push_back(id_col->getData()[0]);
        }
    }
    producer_thread.join();

    ASSERT_EQ(ids_seen.size(), n_blocks);
    for (size_t b = 0; b < n_blocks; ++b)
        EXPECT_EQ(ids_seen[b], b + 1);
}

TEST(PollableShmSource, AdoptedColumnStringProtectIsNoop)
{
    InProcessProducer producer(defaultConfig("string_protect"));
    ASSERT_TRUE(producer.isReady());

    const std::vector<uint64_t> ids = {1, 2};
    const std::vector<uint8_t> chars = {'a', 'b', 'c'};
    const std::vector<uint64_t> offs = {1, 3};
    producer.publishBlock({uint64Payload(ids), stringPayload(chars, offs)}, ids.size());
    producer.signalEndOfStream();

    auto src = makeSource(producer.shmName(), /*stall_ms=*/30'000);
    QueryPipeline pipeline(src);
    PullingPipelineExecutor executor(pipeline);

    Chunk adopted_chunk;
    Chunk chunk;
    while (executor.pull(chunk))
    {
        if (chunk.hasRows())
        {
            adopted_chunk = std::move(chunk);
            break;
        }
    }

    ASSERT_EQ(adopted_chunk.getNumRows(), ids.size());
    ASSERT_EQ(adopted_chunk.getNumColumns(), 2u);

    const auto & cols = adopted_chunk.getColumns();
    ASSERT_NE(typeid_cast<const ColumnString *>(cols[1].get()), nullptr);
    EXPECT_NO_THROW(const_cast<IColumn *>(cols[1].get())->protect());
}


/// Extends ChunkOutlivesSource to K=4 with multiple retained chunks. The strict
/// requirement here is that every slot's release-side write — the consumer's deleter
/// transitioning state→EMPTY in addition to retain_refcount-- — lands in *still-mapped*
/// memory even after the source is gone and chunks are released out of order. Each
/// chunk's RetainToken closure captures its own `region_capture` shared_ptr alias, so
/// the SHM mapping survives until the LAST chunk drops.
///
/// Publish layout: K=4 slots, 3 data blocks (slots 0..2 PUBLISHED) + EOS (slot 3).
/// Holding 3 data chunks across 3 distinct slots while the producer publishes EOS
/// fits the ring without forcing the producer's slot-reuse wait — this test exercises
/// the *post-drop* state-machine writes from the consumer's deleter under the new
/// release contract (Findings 1+3), not the ring-full backpressure path (that's
/// RingFullBlocksUntilRelease).
TEST(PollableShmSource, ChunkOutlivesSourceWithMultipleSlots)
{
    constexpr uint32_t k = 4;
    constexpr uint32_t n_data = k - 1; // 3 data blocks; EOS lands on slot K-1.
    std::vector<Chunk> held;
    held.reserve(n_data);

    {
        InProcessProducer producer(defaultConfig("multislot_outlives", k));
        ASSERT_TRUE(producer.isReady());

        std::thread producer_thread(
            [&]
            {
                for (uint32_t b = 0; b < n_data; ++b)
                {
                    const std::vector<uint64_t> ids = {b * 1000ULL + 1, b * 1000ULL + 2};
                    const std::vector<uint8_t> chars = {static_cast<uint8_t>('A' + b), static_cast<uint8_t>('a' + b)};
                    const std::vector<uint64_t> offs = {1, 2};
                    producer.publishBlock({uint64Payload(ids), stringPayload(chars, offs)}, 2);
                }
                producer.signalEndOfStream();
            });

        auto src = makeSource(producer.shmName());
        QueryPipeline pipeline(src);
        PullingPipelineExecutor executor(pipeline);

        Chunk chunk;
        while (executor.pull(chunk))
        {
            /// hasRows() filters out the zero-row EOS chunk — drop it immediately so the
            /// EOS slot's retain releases and we don't carry an extra alias through to
            /// the post-source phase. Data chunks (rows>0) accumulate in `held`.
            if (chunk.hasRows())
                held.emplace_back(std::move(chunk));
        }
        ASSERT_EQ(held.size(), n_data);
        producer_thread.join();

        /// src, pipeline, executor, producer all torn down here. The source's own
        /// shared_ptr<SharedMemoryRegion> reference drops; the only remaining
        /// references live inside `held[*]`'s adopted-column RetainTokens.
    }

    /// Read every chunk after the source and producer have gone. The columns'
    /// underlying SHM bytes are still mapped because each chunk's RetainToken
    /// closure pins its own region_capture alias.
    for (uint32_t b = 0; b < n_data; ++b)
    {
        ASSERT_EQ(held[b].getNumRows(), 2u);
        ASSERT_EQ(held[b].getNumColumns(), 2u);
        const auto * id_col = typeid_cast<const ColumnUInt64 *>(held[b].getColumns()[0].get());
        ASSERT_NE(id_col, nullptr);
        EXPECT_EQ(id_col->getData()[0], b * 1000ULL + 1);
        EXPECT_EQ(id_col->getData()[1], b * 1000ULL + 2);
    }

    /// Reverse-order drop. Each release runs the RetainToken deleter, which writes
    /// both the retain_refcount decrement AND (on last-alias drop, which is right
    /// here) the SlotState::EMPTY store into the slot. Both writes go into still-
    /// mapped memory because the about-to-drop `held[i]` still has a region_capture
    /// alias; only the chunks earlier in the loop have already released theirs.
    while (!held.empty())
        held.pop_back();
    /// All RetainToken aliases are gone now; the last alias's deleter ran and
    /// dropped the region_capture shared_ptr, which unmapped and closed the SHM.
}


/// `shm-block-stream.md` AC10 + `adoption-layer.md` AC3 retain integrity under
/// producer reuse — exercised at the C++ boundary because the integration test
/// at `tests/integration/test_shm_table_function/test.py` AC10 sub-test can
/// only check chunk *equality across queries*, not byte-stability of a held
/// chunk against a CONCURRENT republish attempt against the same slot.
///
/// Setup uses K=1 so the producer has exactly one slot in the ring; any second
/// publishBlock MUST wait for the consumer to release the first block before
/// it can begin writing — the wire's `state == EMPTY` reuse gate
/// (shm-block-stream.md §Publication state machine + §Backpressure) is the
/// hard contract the producer's reuse loop polls (`InProcessProducer::
/// publishBlockImpl`, `wait_for_state_empty` step).
///
/// The test asserts that *while* the producer is parked in that reuse wait
/// (because the consumer holds chunk 1's RetainToken), the columns of chunk 1
/// re-read the SAME bytes the producer published — i.e. the retain contract
/// pins those bytes against republish-overwrite. After dropping chunk 1 the
/// producer's blocked publish completes, and chunk 2 carries the new bytes.
TEST(PollableShmSource, Ac10HeldChunkBytesStableThroughRetain)
{
    auto cfg = defaultConfig("ac10_byte_stable", /*k=*/1);
    InProcessProducer producer(std::move(cfg));
    ASSERT_TRUE(producer.isReady());

    /// Block-1 payload. Bytes are chosen so block-2 below differs at every
    /// position — so the post-republish read can reliably distinguish them.
    const std::vector<uint64_t> ids_1 = {0x1111111111111111ULL, 0x2222222222222222ULL};
    const std::vector<uint8_t> chars_1 = {'a', 'a', 'a'};
    const std::vector<uint64_t> offs_1 = {2, 3}; // row0="aa", row1="a"

    /// Block-2 payload — different bytes; the producer will eventually overwrite
    /// slot 0 with these bytes once chunk 1 is released. publishBlock(block 2)
    /// BLOCKS in `wait_for_state_empty` until that release happens.
    const std::vector<uint64_t> ids_2 = {0xFFFFFFFFFFFFFFFFULL, 0xEEEEEEEEEEEEEEEEULL};
    const std::vector<uint8_t> chars_2 = {'b', 'c', 'd'};
    const std::vector<uint64_t> offs_2 = {1, 3}; // row0="b", row1="cd"

    std::thread producer_thread(
        [&]()
        {
            producer.publishBlock({uint64Payload(ids_1), stringPayload(chars_1, offs_1)}, ids_1.size());
            /// Will block in wait_for_state_empty until chunk 1 is dropped.
            producer.publishBlock({uint64Payload(ids_2), stringPayload(chars_2, offs_2)}, ids_2.size());
            /// Will block again on slot 0 (K=1) until chunk 2 is dropped.
            producer.signalEndOfStream();
        });

    auto src = makeSource(producer.shmName(), /*stall_ms=*/30'000);
    QueryPipeline pipeline(src);
    PullingPipelineExecutor executor(pipeline);

    /// Pull chunk 1 and hold it.
    Chunk held;
    {
        Chunk chunk;
        while (executor.pull(chunk))
        {
            if (chunk.hasRows())
            {
                held = std::move(chunk);
                break;
            }
        }
    }
    ASSERT_EQ(held.getNumRows(), ids_1.size());
    ASSERT_EQ(held.getNumColumns(), 2u);

    const auto * id_col = typeid_cast<const ColumnUInt64 *>(held.getColumns()[0].get());
    const auto * s_col = typeid_cast<const ColumnString *>(held.getColumns()[1].get());
    ASSERT_NE(id_col, nullptr);
    ASSERT_NE(s_col, nullptr);

    /// Byte-snapshot the underlying buffers via full copies. The post-republish
    /// re-read below compares against these byte-for-byte.
    const std::vector<uint64_t> id_snap(id_col->getData().begin(), id_col->getData().end());
    const std::vector<uint8_t> chars_snap(s_col->getChars().begin(), s_col->getChars().end());
    const std::vector<uint64_t> offs_snap(s_col->getOffsets().begin(), s_col->getOffsets().end());

    /// Sanity: snapshot equals what the producer published — otherwise we're
    /// snapshotting wrong memory and the stability check below would be vacuous.
    ASSERT_EQ(id_snap, ids_1);

    /// Give the producer thread time to reach its publishBlock(block-2) reuse-wait.
    /// 100 ms is generous; the AC10 retain contract guarantees the publish CANNOT
    /// make progress while `held` retains slot 0. The chunk's bytes must be
    /// observably stable for the entire duration of that wait — that is exactly
    /// what AC10 promises.
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    const std::vector<uint64_t> id_now(id_col->getData().begin(), id_col->getData().end());
    const std::vector<uint8_t> chars_now(s_col->getChars().begin(), s_col->getChars().end());
    const std::vector<uint64_t> offs_now(s_col->getOffsets().begin(), s_col->getOffsets().end());
    EXPECT_EQ(id_now, id_snap) << "AC10: held chunk's UInt64 column bytes mutated while producer was blocked on retain";
    EXPECT_EQ(chars_now, chars_snap) << "AC10: held chunk's String chars bytes mutated while producer was blocked on retain";
    EXPECT_EQ(offs_now, offs_snap) << "AC10: held chunk's String offsets bytes mutated while producer was blocked on retain";

    /// Drop chunk 1 → slot 0 deleter stores EMPTY → producer thread's
    /// publishBlock(block 2) unblocks, writes block-2 bytes into slot 0, and
    /// release-stores PUBLISHED.
    held = Chunk{};

    /// Pull chunk 2 and assert it carries block-2 bytes (proving the new bytes
    /// did land in slot 0 once our retain was gone, i.e., the producer's
    /// blocked publish actually completed).
    Chunk chunk2;
    bool got_block2 = false;
    while (executor.pull(chunk2))
    {
        if (chunk2.hasRows())
        {
            got_block2 = true;
            break;
        }
    }
    ASSERT_TRUE(got_block2);
    const auto * id2_col = typeid_cast<const ColumnUInt64 *>(chunk2.getColumns()[0].get());
    ASSERT_NE(id2_col, nullptr);
    ASSERT_EQ(id2_col->getData().size(), ids_2.size());
    EXPECT_EQ(id2_col->getData()[0], ids_2[0]) << "after release, slot 0 did not carry block-2's id_2[0]";
    EXPECT_EQ(id2_col->getData()[1], ids_2[1]) << "after release, slot 0 did not carry block-2's id_2[1]";

    /// Drop chunk 2 and drain EOS so the producer's signalEndOfStream call
    /// (also blocked on slot-0 reuse) can complete and the thread can join.
    chunk2 = Chunk{};
    Chunk tail;
    while (executor.pull(tail))
    {
    }
    producer_thread.join();
}


/// F4 strict precondition-24 detection: a producer that skips the WRITING state
/// and transitions a slot directly EMPTY → PUBLISHED with the `transition_counter`
/// bumped by only 1 (instead of the legal +2 for E→W→P) violates the publication
/// state machine. The consumer's cycle-position check
/// `(prev_pos + delta) % 3 == obs_pos` MUST catch the mismatch and raise
/// SHM_BLOCK_FRAMING_INVALID.
///
/// We can't reuse `InProcessProducer::setSlotStateForTesting` for this because
/// it bumps the counter alongside the state store — it would land us in a
/// CONSISTENT skip (counter delta matches the cyclic walk to PUBLISHED via
/// EMPTY→WRITING→PUBLISHED if it bumped by 2). Instead we open the SHM
/// directly RW from the test thread and inject the malformation by hand:
/// counter += 1 (mirroring just the E→? counter bump), then state = PUBLISHED.
/// The producer is alive (so the control socket / eventfd path is wired) but
/// is NOT publishing — `publishBlock` would itself drive slot 0 through E→W→P
/// and would block on the slot we corrupted.
TEST(PollableShmSource, ProducerSkipsWritingStateThrowsBlockFramingInvalid)
{
    InProcessProducer producer(defaultConfig("skipwriting"));
    ASSERT_TRUE(producer.isReady());

    /// Open the producer-created SHM RW and mmap it just to inject the
    /// malformation on slot 0. This bypasses InProcessProducer entirely so
    /// the counter increment count is fully under the test's control.
    const int rw_fd = ::shm_open(producer.shmName().c_str(), O_RDWR, 0);
    ASSERT_GE(rw_fd, 0) << "shm_open: " << errnoToString(errno);

    struct stat stat_buf{};
    ASSERT_EQ(::fstat(rw_fd, &stat_buf), 0) << errnoToString(errno);
    const size_t rw_size = static_cast<size_t>(stat_buf.st_size);
    void * rw_map = ::mmap(nullptr, rw_size, PROT_READ | PROT_WRITE, MAP_SHARED, rw_fd, 0);
    ASSERT_NE(rw_map, MAP_FAILED) << "mmap: " << errnoToString(errno);

    /// Skip WRITING: bump `transition_counter` by ONLY +1 (legal E→W→P needs
    /// +2 producer-side bumps) and then release-store state=PUBLISHED. From
    /// the consumer's POV: prev_state=EMPTY(0), prev_counter=0, obs_state=
    /// PUBLISHED(2), counter_delta=1. (0+1)%3 = 1 (WRITING) != 2 (PUBLISHED)
    /// → SHM_BLOCK_FRAMING_INVALID. The cycle-walk math is what makes the
    /// skip determinable.
    const auto * hs = static_cast<const HandshakeRegion *>(rw_map);
    auto * slot0 = reinterpret_cast<SlotEntry *>(static_cast<char *>(rw_map) + hs->slot_table_offset);
    slot0->transition_counter.fetch_add(1, std::memory_order_release);
    slot0->state.store(static_cast<uint32_t>(SlotState::PUBLISHED), std::memory_order_release);

    auto src = makeSource(producer.shmName(), /*stall_ms=*/30'000);
    QueryPipeline pipeline(src);
    PullingPipelineExecutor executor(pipeline);

    bool threw_block_framing_invalid = false;
    int observed_code = 0;
    std::string observed_msg;
    try
    {
        Chunk chunk;
        while (executor.pull(chunk))
        {
        }
    }
    catch (const DB::Exception & e)
    {
        observed_code = e.code();
        observed_msg = e.message();
        threw_block_framing_invalid = (e.code() == ErrorCodes::SHM_BLOCK_FRAMING_INVALID);
    }

    /// Detach the RW alias BEFORE assertions so an EXPECT failure doesn't leak
    /// the mapping/fd via gtest's longjmp-on-FAIL behaviour. The producer's own
    /// mapping is still live; its dtor unmaps and `shm_unlink`s.
    ::munmap(rw_map, rw_size);
    ::close(rw_fd);

    EXPECT_TRUE(threw_block_framing_invalid) << "Expected SHM_BLOCK_FRAMING_INVALID, got code " << observed_code << ": "
                                             << observed_msg;
    /// Diagnostic content: the message must identify the precondition-24 /
    /// state-machine class, not the generic monotonicity branch (which would
    /// hide a real skip in CI triage).
    EXPECT_NE(observed_msg.find("state-machine"), std::string::npos) << observed_msg;
    EXPECT_NE(observed_msg.find("precondition 24"), std::string::npos) << observed_msg;
}

TEST(PollableShmSource, MisalignedDescriptorOffsetThrowsBlockFramingInvalid)
{
    InProcessProducer producer(defaultConfig("misaligned_desc_offset"));
    ASSERT_TRUE(producer.isReady());

    const std::vector<uint64_t> ids = {1, 2, 3};
    const std::vector<uint8_t> chars = {'a', 'b', 'c'};
    const std::vector<uint64_t> offs = {1, 2, 3};
    producer.publishMalformedBlock(
        {uint64Payload(ids), stringPayload(chars, offs)},
        ids.size(),
        InProcessProducer::Malformation::MisalignedDescriptorOffset);

    auto src = makeSource(producer.shmName(), /*stall_ms=*/30'000);
    QueryPipeline pipeline(src);
    PullingPipelineExecutor executor(pipeline);

    bool threw_block_framing_invalid = false;
    int observed_code = 0;
    std::string observed_msg;
    try
    {
        Chunk chunk;
        while (executor.pull(chunk))
        {
        }
    }
    catch (const DB::Exception & e)
    {
        observed_code = e.code();
        observed_msg = e.message();
        threw_block_framing_invalid = (e.code() == ErrorCodes::SHM_BLOCK_FRAMING_INVALID);
    }

    EXPECT_TRUE(threw_block_framing_invalid) << "Expected SHM_BLOCK_FRAMING_INVALID, got code " << observed_code << ": "
                                             << observed_msg;
    EXPECT_NE(observed_msg.find("per_column_descriptors_offset"), std::string::npos) << observed_msg;
    EXPECT_NE(observed_msg.find("alignof(ColumnDescriptor)"), std::string::npos) << observed_msg;
}

#endif
