#include <gtest/gtest.h>

#if defined(OS_LINUX)

#include <Storages/SharedMemorySource/TestProducer/InProcessProducer.h>
#include <Storages/SharedMemorySource/Wire/SharedMemoryRegion.h>
#include <Storages/SharedMemorySource/Wire/ControlSocket.h>
#include <Storages/SharedMemorySource/Wire/Layout.h>

#include <fmt/format.h>

#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>


using namespace DB;
using namespace DB::SharedMemoryWire;

namespace
{
    std::string uniqueShmName(const char * tag)
    { return fmt::format("test_inproc_{}_{}", tag, ::getpid()); }

    const SlotEntry * consumerSlotAt(const SharedMemoryRegion & region, uint32_t i)
    {
        const auto & hs = region.handshake();
        const auto * base = static_cast<const char *>(region.data()) + hs.slot_table_offset;
        return reinterpret_cast<const SlotEntry *>(base + i * hs.slot_table_stride);
    }

    InProcessProducer::ColumnPayload uint64Payload(const std::vector<uint64_t> & v)
    { return {v.data(), v.size(), nullptr, 0}; }

    InProcessProducer::ColumnPayload stringPayload(
        const std::vector<uint8_t> & chars, const std::vector<uint64_t> & offs)
    { return {chars.data(), chars.size(), offs.data(), offs.size()}; }

    InProcessProducer::Config defaultConfig(const char * tag, uint32_t k = 4)
    {
        InProcessProducer::Config cfg;
        cfg.shm_name = uniqueShmName(tag);
        cfg.ring_depth_k = k;
        cfg.schema = {{"id", "UInt64"}, {"s", "String"}};
        cfg.data_region_size = 64 * 1024;
        return cfg;
    }

    /// Mutable handle on a slot's retain_refcount: lets the test simulate a consumer that has
    /// adopted the block (refcount > 0) before the producer reuses the slot. Real consumers
    /// will own this via the T1.4 RetainToken; here we just bump it directly.
    std::atomic<uint64_t> * mutableRefcount(const SharedMemoryRegion & region, uint32_t i)
    {
        return const_cast<std::atomic<uint64_t> *>(&consumerSlotAt(region, i)->retain_refcount);
    }

    /// Simulate the consumer-side release that PollableShmSource's RetainToken deleter
    /// performs: decrement the retain_refcount and, when the LAST alias drops, release-
    /// store SlotState::EMPTY into the slot. After this call the producer's reuse wait
    /// (`state == EMPTY`) is satisfied — without the state store, the producer would
    /// hang because Findings 1 + 3 changed the producer's wait condition.
    /// Also bump `transition_counter` on the P→E edge to mirror the deleter's
    /// precondition-24 protocol (Layout.h SlotEntry).
    void simulateConsumerRelease(const SharedMemoryRegion & region, uint32_t i) noexcept
    {
        const auto * slot = consumerSlotAt(region, i);
        auto * refcount = const_cast<std::atomic<uint64_t> *>(&slot->retain_refcount);
        auto * state = const_cast<std::atomic<uint32_t> *>(&slot->state);
        auto * counter = const_cast<std::atomic<uint64_t> *>(&slot->transition_counter);
        if (refcount->fetch_sub(1, std::memory_order_acq_rel) == 1)
        {
            counter->fetch_add(1, std::memory_order_release);
            state->store(static_cast<uint32_t>(SlotState::EMPTY), std::memory_order_release);
        }
    }
}

/// Full T1.1 + T1.2 round-trip: consumer attaches via SharedMemoryRegion + receives the
/// eventfd via ControlSocketClient, walks the slot table, and recovers producer-written bytes.
/// This is the cross-component integration proof the brief requires.
TEST(InProcessProducer, AttachReceiveBytesReproduceContent)
{
    InProcessProducer producer(defaultConfig("attach"));
    ASSERT_TRUE(producer.isReady());

    int conn_fd = -1;
    int received_evfd = ControlSocketClient::connectAndReceiveEventFd(
        controlSocketPathForShmName(producer.shmName()), conn_fd);
    ASSERT_GE(received_evfd, 0);

    auto region = SharedMemoryRegion::attach(producer.shmName());
    ASSERT_NE(region, nullptr);
    const auto & hs = region->handshake();
    EXPECT_EQ(hs.magic.load(std::memory_order_acquire), SHM_MAGIC);
    EXPECT_EQ(hs.abi_version, SHM_ABI_VERSION_1);
    EXPECT_EQ(hs.ring_depth_k, 4u);
    EXPECT_EQ(hs.schema_count, 2u);

    const std::vector<uint64_t> ids = {10, 20, 30};
    const std::vector<uint8_t>  chars = {'a','b','c','d','e','f'};
    const std::vector<uint64_t> offs  = {1, 3, 6};
    producer.publishBlock({uint64Payload(ids), stringPayload(chars, offs)}, ids.size());
    producer.signalEndOfStream();

    uint64_t drained = 0;
    ASSERT_EQ(::read(received_evfd, &drained, sizeof(drained)), static_cast<ssize_t>(sizeof(drained)));
    EXPECT_GE(drained, 2u);

    const auto * slot0 = consumerSlotAt(*region, 0);
    ASSERT_EQ(slot0->state.load(std::memory_order_acquire),
              static_cast<uint32_t>(SlotState::PUBLISHED));
    EXPECT_EQ(slot0->slot_index, 0u);
    EXPECT_EQ(slot0->row_count, ids.size());
    EXPECT_EQ(slot0->eos_marker.load(std::memory_order_acquire), 0u);
    EXPECT_EQ(slot0->sequence.load(std::memory_order_acquire), 1u);

    const auto * data_base = static_cast<const char *>(region->data()) + hs.data_region_offset;
    const auto * descs = reinterpret_cast<const ColumnDescriptor *>(
        data_base + slot0->per_column_descriptors_offset);

    EXPECT_EQ(descs[0].type, static_cast<uint32_t>(WireColumnType::UInt64));
    EXPECT_EQ(descs[0].value_count, ids.size());
    EXPECT_EQ(descs[0].value_padding, PADDING_FOR_SIMD);
    const auto * ids_view = reinterpret_cast<const uint64_t *>(data_base + descs[0].value_offset);
    for (size_t r = 0; r < ids.size(); ++r) EXPECT_EQ(ids_view[r], ids[r]);

    EXPECT_EQ(descs[1].type, static_cast<uint32_t>(WireColumnType::String));
    EXPECT_EQ(descs[1].value_count, chars.size());
    EXPECT_EQ(descs[1].offsets_count, offs.size());
    EXPECT_EQ(descs[1].offsets_padding, PADDING_FOR_SIMD);
    const auto * chars_view = reinterpret_cast<const uint8_t *>(data_base + descs[1].value_offset);
    EXPECT_EQ(0, std::memcmp(chars_view, chars.data(), chars.size()));
    const auto * offs_view = reinterpret_cast<const uint64_t *>(data_base + descs[1].offsets_offset);
    for (size_t r = 0; r < offs.size(); ++r) EXPECT_EQ(offs_view[r], offs[r]);

    /// offsets[-1] sentinel — consumer's ColumnString::offsetAt(0) reads it.
    EXPECT_EQ(*(offs_view - 1), 0u);

    /// EOS lands in slot 1 (next round-robin position) as a zero-row block.
    const auto * slot1 = consumerSlotAt(*region, 1);
    EXPECT_EQ(slot1->state.load(std::memory_order_acquire),
              static_cast<uint32_t>(SlotState::PUBLISHED));
    EXPECT_EQ(slot1->row_count, 0u);
    EXPECT_EQ(slot1->eos_marker.load(std::memory_order_acquire), 1u);

    ::close(received_evfd);
    ::close(conn_fd);
}

/// K=2; publish 2 blocks (ring full). The 3rd publish must block until the consumer
/// transitions slot 0 back to EMPTY. Under the new release contract (Findings 1 + 3)
/// the producer's wait polls `state == EMPTY`, not `retain_refcount == 0` — simulating
/// "consumer drops retain" therefore requires the test to refcount-- AND state→EMPTY
/// (the producer would otherwise hang on a still-PUBLISHED slot whose refcount is 0).
TEST(InProcessProducer, RingFullBlocksUntilRelease)
{
    InProcessProducer producer(defaultConfig("ringfull", /*k=*/2));
    const std::vector<uint64_t> ids = {1};
    const std::vector<uint8_t>  chars = {'x'};
    const std::vector<uint64_t> offs = {1};
    producer.publishBlock({uint64Payload(ids), stringPayload(chars, offs)}, 1);
    producer.publishBlock({uint64Payload(ids), stringPayload(chars, offs)}, 1);

    auto region = SharedMemoryRegion::attach(producer.shmName());
    /// Consumer attaches and retains slot 0 — refcount 0 → 1; slot stays PUBLISHED.
    auto * refcount = mutableRefcount(*region, 0);
    refcount->fetch_add(1, std::memory_order_acq_rel);

    std::atomic<bool> third_done{false};
    std::thread pub([&] {
        producer.publishBlock({uint64Payload(ids), stringPayload(chars, offs)}, 1);
        third_done.store(true, std::memory_order_release);
    });

    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    /// Still blocked: state is PUBLISHED (consumer hasn't released), so the producer
    /// can't reuse the slot under the new wait condition.
    EXPECT_FALSE(third_done.load(std::memory_order_acquire));

    /// Consumer's RetainToken-deleter equivalent: decrement refcount and (because this
    /// IS the last alias) release-store state→EMPTY. The producer's wait now passes.
    simulateConsumerRelease(*region, 0);
    pub.join();
    EXPECT_TRUE(third_done.load(std::memory_order_acquire));
    /// Slot 0 carries the third publish (slot 0 reused, sequence=2).
    EXPECT_EQ(consumerSlotAt(*region, 0)->sequence.load(std::memory_order_acquire), 2u);
    EXPECT_EQ(consumerSlotAt(*region, 0)->state.load(std::memory_order_acquire),
              static_cast<uint32_t>(SlotState::PUBLISHED));
}

/// AC10: K=1. Producer publishes block. Consumer attaches and acquires the retain
/// (refcount 0→1). Producer's republish attempt must block on the publication state
/// machine — under the new release contract (Findings 1 + 3) the producer's reuse
/// wait polls `state == EMPTY`, which only flips when the consumer's RetainToken
/// deleter runs its `if (fetch_sub == 1) state.store(EMPTY)` step. The simulated
/// drop here mirrors that deleter. Old bytes remain visible while the retain is live.
TEST(InProcessProducer, Ac10RepublishUnderRetainCooperates)
{
    InProcessProducer producer(defaultConfig("ac10", /*k=*/1));
    const std::vector<uint64_t> v_old = {0xAA'AA'AA'AA};
    const std::vector<uint8_t>  c_old = {'o'};
    const std::vector<uint64_t> o_old = {1};
    producer.publishBlock({uint64Payload(v_old), stringPayload(c_old, o_old)}, 1);

    auto region = SharedMemoryRegion::attach(producer.shmName());
    const auto * slot = consumerSlotAt(*region, 0);
    /// Step "consumer retains block 0": refcount 0 → 1. State stays PUBLISHED.
    auto * refcount = mutableRefcount(*region, 0);
    refcount->fetch_add(1, std::memory_order_acq_rel);

    const auto * data_base = static_cast<const char *>(region->data())
                           + region->handshake().data_region_offset;
    const auto * descs_old = reinterpret_cast<const ColumnDescriptor *>(
        data_base + slot->per_column_descriptors_offset);
    EXPECT_EQ(*reinterpret_cast<const uint64_t *>(data_base + descs_old[0].value_offset), v_old[0]);

    const std::vector<uint64_t> v_new = {0xBB'BB'BB'BB};
    const std::vector<uint8_t>  c_new = {'n'};
    const std::vector<uint64_t> o_new = {1};
    std::atomic<bool> republished{false};
    std::thread pub([&] {
        /// Producer attempts to republish slot 0 — blocks because state != EMPTY
        /// (consumer holds the retain and has not yet transitioned the slot).
        producer.publishBlock({uint64Payload(v_new), stringPayload(c_new, o_new)}, 1);
        republished.store(true, std::memory_order_release);
    });

    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    EXPECT_FALSE(republished.load(std::memory_order_acquire));
    /// While retain is live, bytes through the held pointer are unchanged (AC10).
    EXPECT_EQ(*reinterpret_cast<const uint64_t *>(data_base + descs_old[0].value_offset), v_old[0]);

    /// "Consumer drops retain → state goes EMPTY → producer's wait-loop succeeds."
    /// simulateConsumerRelease does both writes (refcount-- AND state→EMPTY on last
    /// alias drop), mirroring PollableShmSource's RetainToken deleter exactly.
    simulateConsumerRelease(*region, 0);
    pub.join();
    EXPECT_TRUE(republished.load(std::memory_order_acquire));

    const auto * descs_new = reinterpret_cast<const ColumnDescriptor *>(
        data_base + slot->per_column_descriptors_offset);
    EXPECT_EQ(*reinterpret_cast<const uint64_t *>(data_base + descs_new[0].value_offset), v_new[0]);
    EXPECT_EQ(slot->sequence.load(std::memory_order_acquire), 2u);
}

/// EOS is observable through the slot's eos_marker (per §End-of-stream); further publishes
/// throw.
TEST(InProcessProducer, EosObservable)
{
    InProcessProducer producer(defaultConfig("eos"));
    const std::vector<uint64_t> ids = {7};
    const std::vector<uint8_t>  c = {'q'};
    const std::vector<uint64_t> o = {1};
    producer.publishBlock({uint64Payload(ids), stringPayload(c, o)}, 1);
    producer.signalEndOfStream();

    auto region = SharedMemoryRegion::attach(producer.shmName());
    EXPECT_EQ(consumerSlotAt(*region, 0)->eos_marker.load(std::memory_order_acquire), 0u);
    EXPECT_EQ(consumerSlotAt(*region, 1)->eos_marker.load(std::memory_order_acquire), 1u);
    EXPECT_EQ(consumerSlotAt(*region, 1)->row_count, 0u);
    EXPECT_THROW(producer.publishBlock({uint64Payload(ids), stringPayload(c, o)}, 1),
                 std::runtime_error);
}

/// Ctor validates ring_depth_k, schema count, schema string lengths.
TEST(InProcessProducer, RejectsBadConfig)
{
    auto construct = [](InProcessProducer::Config c) { InProcessProducer p(std::move(c)); };
    auto bad_k0 = defaultConfig("badk0");          bad_k0.ring_depth_k = 0;
    auto bad_kmax = defaultConfig("badkmax");      bad_kmax.ring_depth_k = IMPL_MAX_K + 1;
    auto bad_schema = defaultConfig("badschema");  bad_schema.schema.clear();
    auto bad_name = defaultConfig("badname");      bad_name.schema[0].first = std::string(SCHEMA_ENTRY_STR_MAX, 'x');

    EXPECT_THROW(construct(std::move(bad_k0)), std::runtime_error);
    EXPECT_THROW(construct(std::move(bad_kmax)), std::runtime_error);
    EXPECT_THROW(construct(std::move(bad_schema)), std::runtime_error);
    EXPECT_THROW(construct(std::move(bad_name)), std::runtime_error);
}

#endif
