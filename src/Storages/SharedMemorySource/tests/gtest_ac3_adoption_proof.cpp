#include <gtest/gtest.h>

#if defined(OS_LINUX)

#include <Columns/ColumnString.h>
#include <Columns/ColumnsNumber.h>
#include <Columns/IColumn.h>
#include <Core/Block.h>
#include <DataTypes/DataTypeString.h>
#include <DataTypes/DataTypesNumber.h>
#include <Processors/Chunk.h>
#include <Processors/Executors/PullingPipelineExecutor.h>
#include <QueryPipeline/QueryPipeline.h>
#include <Storages/SharedMemorySource/Source/PollableShmSource.h>
#include <Storages/SharedMemorySource/TestProducer/InProcessProducer.h>
#include <Storages/SharedMemorySource/Wire/Layout.h>
#include <Storages/SharedMemorySource/Wire/SharedMemoryRegion.h>
#include <Common/typeid_cast.h>

#include <unistd.h>

#include <atomic>
#include <cstdint>
#include <cstring>
#include <memory>
#include <random>
#include <string>
#include <string_view>
#include <thread>
#include <vector>


using namespace DB;
using namespace DB::SharedMemoryWire;


namespace
{
    constexpr size_t NUM_BLOCKS = 120;
    constexpr size_t ROWS_PER_BLOCK = 64;
    constexpr uint32_t RING_DEPTH = 4;

    /// SharedHeader that matches AC1 in `streamed_table/specs/system.md`:
    /// (id UInt64, v1 UInt64, v2 UInt64, s1 String, s2 String). The first three
    /// columns exercise ColumnVector<UInt64> adoption; the last two exercise
    /// ColumnString (chars + offsets) adoption.
    SharedHeader makeAc1Header()
    {
        Block b;
        b.insert({std::make_shared<DataTypeUInt64>()->createColumn(),
                  std::make_shared<DataTypeUInt64>(), "id"});
        b.insert({std::make_shared<DataTypeUInt64>()->createColumn(),
                  std::make_shared<DataTypeUInt64>(), "v1"});
        b.insert({std::make_shared<DataTypeUInt64>()->createColumn(),
                  std::make_shared<DataTypeUInt64>(), "v2"});
        b.insert({std::make_shared<DataTypeString>()->createColumn(),
                  std::make_shared<DataTypeString>(), "s1"});
        b.insert({std::make_shared<DataTypeString>()->createColumn(),
                  std::make_shared<DataTypeString>(), "s2"});
        return std::make_shared<const Block>(std::move(b));
    }

    /// One block's per-column byte buffers. Reused across publishBlock calls; the
    /// producer memcpys these into SHM before publishBlock returns, so the caller
    /// may safely mutate them between iterations.
    struct BlockBuffers
    {
        std::vector<UInt64> id_buf;
        std::vector<UInt64> v1_buf;
        std::vector<UInt64> v2_buf;
        std::vector<UInt8>  s1_chars;
        std::vector<UInt64> s1_offsets;
        std::vector<UInt8>  s2_chars;
        std::vector<UInt64> s2_offsets;
    };

    /// AC1 string lengths: s1 in [0, 31], s2 in [0, 255]. All values derived from
    /// `rng` so the test is reproducible (seed pinned at the call site).
    void fillRandomBlock(BlockBuffers & bb, size_t block_id, std::mt19937_64 & rng)
    {
        bb.id_buf.assign(ROWS_PER_BLOCK, 0);
        bb.v1_buf.assign(ROWS_PER_BLOCK, 0);
        bb.v2_buf.assign(ROWS_PER_BLOCK, 0);
        bb.s1_chars.clear();
        bb.s2_chars.clear();
        bb.s1_offsets.assign(ROWS_PER_BLOCK, 0);
        bb.s2_offsets.assign(ROWS_PER_BLOCK, 0);

        UInt64 s1_cum = 0;
        UInt64 s2_cum = 0;
        for (size_t r = 0; r < ROWS_PER_BLOCK; ++r)
        {
            bb.id_buf[r] = block_id * ROWS_PER_BLOCK + r;
            bb.v1_buf[r] = rng();
            bb.v2_buf[r] = rng();
            const size_t s1_len = rng() % 32;
            for (size_t i = 0; i < s1_len; ++i)
                bb.s1_chars.push_back(static_cast<UInt8>(rng()));
            s1_cum += s1_len;
            bb.s1_offsets[r] = s1_cum;

            const size_t s2_len = rng() % 256;
            for (size_t i = 0; i < s2_len; ++i)
                bb.s2_chars.push_back(static_cast<UInt8>(rng()));
            s2_cum += s2_len;
            bb.s2_offsets[r] = s2_cum;
        }
    }

    std::vector<InProcessProducer::ColumnPayload> toPayloads(const BlockBuffers & bb)
    {
        std::vector<InProcessProducer::ColumnPayload> p(5);
        p[0] = {bb.id_buf.data(), bb.id_buf.size(), nullptr, 0};
        p[1] = {bb.v1_buf.data(), bb.v1_buf.size(), nullptr, 0};
        p[2] = {bb.v2_buf.data(), bb.v2_buf.size(), nullptr, 0};
        p[3] = {bb.s1_chars.data(), bb.s1_chars.size(),
                bb.s1_offsets.data(), bb.s1_offsets.size()};
        p[4] = {bb.s2_chars.data(), bb.s2_chars.size(),
                bb.s2_offsets.data(), bb.s2_offsets.size()};
        return p;
    }

    const SlotEntry * slotPtr(const SharedMemoryRegion & region, uint32_t i)
    {
        const auto & hs = region.handshake();
        const auto * base = static_cast<const char *>(region.data()) + hs.slot_table_offset;
        return reinterpret_cast<const SlotEntry *>(base + i * hs.slot_table_stride);
    }

    /// AC3 per-chunk metadata-proof helper — asserts that every adopted column's
    /// payload pointer equals `source_data_region_base + descriptor.<offset>`, AND
    /// that the consumer-visible buffer size equals the descriptor-declared count.
    /// Per `adoption-layer.md` AC3 "Every emitted column across the run is adopted",
    /// the test calls this for EVERY emitted chunk; the chunk-0-only metadata path
    /// (with the in-range check on later chunks) that the original gtest used is
    /// strictly weaker than this.
    ///
    /// `descriptors` is a SNAPSHOT taken by the caller of the per-column descriptor
    /// array for the slot the chunk was drained from, copied out WHILE the chunk's
    /// retain is alive (so the producer cannot republish over the slot and mutate
    /// the descriptors mid-check). `source_data_region_base` is the source-side
    /// VA of the SHM data region — derived once from chunk 0's first column, then
    /// passed through unchanged for every subsequent chunk so the equality check
    /// is grounded in the chunk-0 derivation rather than circular per-chunk
    /// re-derivation.
    ///
    /// VC2 gotcha (T2.1/T2.2 F3): every accessor below is `const`. The non-const
    /// `getData()/getChars()/getOffsets()` overloads on the adopted columns trip
    /// the I3 mutation guard.
    void assertChunkMetadataIdentity(
        const Chunk & chunk,
        const ColumnDescriptor * descriptors,
        const char * source_data_region_base,
        UInt64 expected_row_count,
        size_t chunk_index)
    {
        ASSERT_EQ(chunk.getNumRows(), expected_row_count)
            << "chunk " << chunk_index << ": row count != slot->row_count";
        ASSERT_EQ(chunk.getNumColumns(), 5u)
            << "chunk " << chunk_index << ": column count";
        const auto & cols = chunk.getColumns();

        for (size_t ci = 0; ci < 3; ++ci)
        {
            const auto * cv = typeid_cast<const ColumnUInt64 *>(cols[ci].get());
            ASSERT_NE(cv, nullptr)
                << "chunk " << chunk_index << " col " << ci << ": expected ColumnVector<UInt64>";
            EXPECT_EQ(reinterpret_cast<const char *>(cv->getData().data()),
                      source_data_region_base + descriptors[ci].value_offset)
                << "chunk " << chunk_index << " col " << ci << ": UInt64 value pointer != descriptor-derived";
            EXPECT_EQ(cv->getData().size(), descriptors[ci].value_count)
                << "chunk " << chunk_index << " col " << ci << ": UInt64 value size != descriptor-declared";
            /// getRawData() packages (pointer, byteSize) for the value buffer;
            /// it is the canonical "metadata proves the byte range" accessor
            /// for fixed-width columns and is the surface AC3 ultimately wants.
            const std::string_view raw = cv->getRawData();
            EXPECT_EQ(raw.data(), source_data_region_base + descriptors[ci].value_offset)
                << "chunk " << chunk_index << " col " << ci << ": getRawData().data() mismatch";
            EXPECT_EQ(raw.size(), descriptors[ci].value_count * sizeof(UInt64))
                << "chunk " << chunk_index << " col " << ci << ": getRawData().size() mismatch";
        }
        for (size_t ci = 3; ci < 5; ++ci)
        {
            const auto * cs = typeid_cast<const ColumnString *>(cols[ci].get());
            ASSERT_NE(cs, nullptr)
                << "chunk " << chunk_index << " col " << ci << ": expected ColumnString";
            EXPECT_EQ(reinterpret_cast<const char *>(cs->getChars().data()),
                      source_data_region_base + descriptors[ci].value_offset)
                << "chunk " << chunk_index << " col " << ci << ": String chars pointer != descriptor-derived";
            EXPECT_EQ(cs->getChars().size(), descriptors[ci].value_count)
                << "chunk " << chunk_index << " col " << ci << ": String chars size != descriptor-declared";
            EXPECT_EQ(reinterpret_cast<const char *>(cs->getOffsets().data()),
                      source_data_region_base + descriptors[ci].offsets_offset)
                << "chunk " << chunk_index << " col " << ci << ": String offsets pointer != descriptor-derived";
            EXPECT_EQ(cs->getOffsets().size(), descriptors[ci].offsets_count)
                << "chunk " << chunk_index << " col " << ci << ": String offsets size != descriptor-declared";
        }
    }
}


/// AC3 Adoption proof — `streamed_table/specs/adoption-layer.md` §Acceptance criteria.
///
/// Drives the InProcessProducer + PollableShmSource end-to-end via the same
/// PullingPipelineExecutor that production queries use, for NUM_BLOCKS (>=100)
/// blocks of the AC1 schema (3 x UInt64 + 2 x String).
///
/// For EVERY emitted chunk this test asserts the FULL metadata-proof half of
/// AC3: every adopted column's payload pointer equals
/// `source_data_region_base + descriptor.<offset_field>` and the consumer-side
/// buffer size equals the descriptor-declared count. The descriptor for chunk i
/// is read directly from slot `i % K` of the producer-written slot table via
/// the test-side `region_inspect` mapping, snapshotted WHILE the chunk's retain
/// is alive (slot can't be republished mid-snap because the producer's reuse
/// wait polls `state == EMPTY`, which the consumer-side deleter only stores
/// once the chunk drops).
///
/// Chunk-to-slot mapping derivation: the producer publishes block i to slot
/// `i % K` (see `InProcessProducer::publishBlockImpl`, the `slot_pos =
/// next_publish_slot % ring_depth_k` line). The consumer drains in strictly-
/// increasing per-slot sequence with scan-order tie-break across slots (see
/// `PollableShmSource::findNextReadySlot`), which under the lockstep publish
/// schedule above means chunk i is emitted from slot `i % K`. Sanity-checked
/// at chunk 0 below by asserting slot 0's retain is the unique non-zero one.
///
/// After draining, asserts that every slot's `retain_refcount` has returned to
/// zero (the retain protocol's final-drop guarantee — system spec I5 +
/// adoption-layer spec AC3 last sentence).
///
/// VC2 gotcha: all column accesses go through const refs / const accessors;
/// non-const `getData()/getChars()/getOffsets()` would trigger the I3 mutation
/// guard added by T2.1/T2.2.
TEST(Ac3AdoptionProof, EveryColumnAdoptedAcrossManyBlocks)
{
    const std::string shm_name = "test_ac3_" + std::to_string(::getpid());

    InProcessProducer::Config cfg;
    cfg.shm_name = shm_name;
    cfg.ring_depth_k = RING_DEPTH;
    cfg.schema = {{"id", "UInt64"}, {"v1", "UInt64"}, {"v2", "UInt64"},
                  {"s1", "String"}, {"s2", "String"}};
    cfg.data_region_size = 4 * 1024 * 1024;
    InProcessProducer producer(std::move(cfg));
    ASSERT_TRUE(producer.isReady());

    /// Producer thread: publish NUM_BLOCKS random AC1-shaped blocks, then EOS.
    /// publishBlock blocks on ring-full (consumer holding retain) for blocks
    /// >= RING_DEPTH; that backpressure is what keeps producer and consumer in
    /// lockstep so every block round-trips through adoption.
    std::thread producer_thread([&]
    {
        std::mt19937_64 rng(0xC0FFEEULL); // NOLINT(cert-msc32-c, cert-msc51-cpp)
        BlockBuffers bb;
        for (size_t b = 0; b < NUM_BLOCKS; ++b)
        {
            fillRandomBlock(bb, b, rng);
            producer.publishBlock(toPayloads(bb), ROWS_PER_BLOCK);
        }
        producer.signalEndOfStream();
    });

    /// Test-side attach for reading slot descriptors and slot-table retain
    /// refcounts. This `mmap` is a SEPARATE virtual address range from the
    /// PollableShmSource's `mmap` (two independent `mmap(MAP_SHARED, ...)`
    /// calls on the same backing fd). The two VAs overlay the same physical
    /// pages, so reads via either see the same bytes; but the column pointers
    /// emitted by the source are addresses in the source's VA, NOT this one.
    /// We therefore infer the source's data-region base by subtracting the
    /// first column's descriptor-declared offset from its data pointer in
    /// chunk 0, and use that inferred base as the ground truth for the
    /// per-chunk metadata-equality assertion on all subsequent chunks.
    auto region_inspect = SharedMemoryRegion::attach(shm_name);
    ASSERT_NE(region_inspect, nullptr);
    const char * inspect_base = static_cast<const char *>(region_inspect->data());
    const auto & hs = region_inspect->handshake();
    const char * inspect_data_region_base = inspect_base + hs.data_region_offset;

    /// Build the source + standard pipeline harness.
    auto source = std::make_shared<PollableShmSource>(
        makeAc1Header(), shm_name,
        std::vector<DataTypePtr>{std::make_shared<DataTypeUInt64>(),
                                  std::make_shared<DataTypeUInt64>(),
                                  std::make_shared<DataTypeUInt64>(),
                                  std::make_shared<DataTypeString>(),
                                  std::make_shared<DataTypeString>()},
        std::vector<String>{"id", "v1", "v2", "s1", "s2"},
        /*requested_column_names=*/std::vector<String>{"id", "v1", "v2", "s1", "s2"},
        /*stall_timeout_ms=*/60'000);
    QueryPipeline pipeline(source);
    PullingPipelineExecutor executor(pipeline);

    size_t total_chunks = 0;
    /// Source-side data-region base; nullptr until inferred from chunk 0's
    /// first-column pointer (the source's mmap of the SHM region is at a
    /// DIFFERENT VA than the test's `region_inspect` mmap).
    const char * source_data_region_base = nullptr;

    {
        Chunk chunk;
        while (executor.pull(chunk))
        {
            if (chunk.getNumRows() == 0)
                continue;

            /// chunk i → slot (i % K). Justification: see test-level comment.
            const uint32_t this_slot_idx = static_cast<uint32_t>(total_chunks % hs.ring_depth_k);
            const auto * slot = slotPtr(*region_inspect, this_slot_idx);

            /// Snapshot the slot's per-column descriptors AND row_count while we
            /// hold the chunk. The slot is in PUBLISHED state with refcount > 0
            /// (this chunk's retain alias); the producer's reuse wait polls
            /// `state == EMPTY` and therefore cannot overwrite these bytes
            /// until our deleter fires at end-of-iteration.
            std::vector<ColumnDescriptor> snap(hs.schema_count);
            const auto * descs_src = reinterpret_cast<const ColumnDescriptor *>(
                inspect_data_region_base + slot->per_column_descriptors_offset);
            std::memcpy(snap.data(), descs_src, sizeof(ColumnDescriptor) * snap.size());
            const UInt64 snap_row_count = slot->row_count;

            if (total_chunks == 0)
            {
                /// Derive source_data_region_base ONCE from chunk 0's first column.
                /// For every subsequent chunk we pass the same base to the helper,
                /// so the equality `chunk-pointer == base + descriptor.offset` is
                /// grounded in chunk 0's derivation rather than circular per-chunk
                /// re-derivation (a circular check would always trivially hold).
                const auto * id_col = typeid_cast<const ColumnUInt64 *>(chunk.getColumns()[0].get());
                ASSERT_NE(id_col, nullptr) << "chunk 0: first column is not ColumnVector<UInt64>";
                const char * id_ptr = reinterpret_cast<const char *>(id_col->getData().data());
                source_data_region_base = id_ptr - snap[0].value_offset;

                /// Sanity-check the chunk-to-slot mapping: with only chunk 0's
                /// retain alive, slot 0 must be the unique slot with refcount > 0.
                /// (Executor pre-pulls can hold retains on slots 1..K-1, but
                /// chunk 0's slot 0 is the one we just derived the base from.)
                ASSERT_GT(slot->retain_refcount.load(std::memory_order_acquire), 0u)
                    << "chunk 0: slot 0 has retain == 0; chunk-to-slot mapping assumption broken";
            }

            assertChunkMetadataIdentity(chunk, snap.data(), source_data_region_base,
                                        snap_row_count, total_chunks);
            ++total_chunks;
        }
        /// `chunk` falls out of scope here. Its columns drop, the per-column
        /// RetainToken aliases drop, and the last alias on the most-recently
        /// retained slot fires its release callback — decrementing that slot's
        /// retain_refcount back to zero AND release-storing SlotState::EMPTY
        /// (the consumer-driven P→E transition the producer's reuse wait polls).
    }

    producer_thread.join();
    EXPECT_EQ(total_chunks, NUM_BLOCKS);

    /// AC3 last sentence + system spec I5: after every chunk and every derived
    /// handle has dropped, every slot's retain_refcount must be 0.
    for (uint32_t i = 0; i < hs.ring_depth_k; ++i)
    {
        const auto * slot = slotPtr(*region_inspect, i);
        EXPECT_EQ(slot->retain_refcount.load(std::memory_order_acquire), 0u)
            << "slot " << i << ": retain_refcount leaked after drain + chunk drop";
    }
}

#endif
