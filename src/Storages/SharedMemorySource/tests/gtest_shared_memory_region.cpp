#include <gtest/gtest.h>

#include <Storages/SharedMemorySource/Wire/SharedMemoryRegion.h>
#include <Storages/SharedMemorySource/Wire/Layout.h>

#include <Common/Exception.h>

#include <base/errnoToString.h>
#include <base/types.h>

#include <fmt/format.h>

#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#include <atomic>
#include <cerrno>


namespace DB::ErrorCodes
{
    extern const int SHM_ATTACH_FAILED;
    extern const int SHM_HANDSHAKE_INVALID;
}

using namespace DB;
using namespace DB::SharedMemoryWire;

namespace
{

/// RAII helper: creates and prepares a POSIX SHM region producer-side (RW),
/// unlinks the name in the destructor so prior-run crashes don't poison the
/// next run. Names include the pid to avoid parallel-test collisions.
class TestShm
{
public:
    String name;
    void * mapping = nullptr;
    size_t mapping_size = 0;
    int fd = -1;

    explicit TestShm(const String & suffix)
        : name(fmt::format("/ch_shmregion_gt_{}_{}", suffix, ::getpid()))
    { ::shm_unlink(name.c_str()); }

    void create(size_t bytes)
    {
        fd = ::shm_open(name.c_str(), O_RDWR | O_CREAT | O_EXCL, 0600);
        ASSERT_GE(fd, 0) << "shm_open: " << errnoToString(errno);
        ASSERT_EQ(::ftruncate(fd, static_cast<off_t>(bytes)), 0) << errnoToString(errno);
        mapping_size = bytes;
        if (bytes > 0)
        {
            mapping = ::mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
            ASSERT_NE(mapping, MAP_FAILED) << "mmap: " << errnoToString(errno);
        }
    }

    HandshakeRegion * hs() const noexcept { return static_cast<HandshakeRegion *>(mapping); }

    void writeValidLayout(uint32_t k = 4, uint32_t schema_count = 2, size_t data_bytes = 1024) const
    {
        auto * h = hs();
        h->abi_version = SHM_ABI_VERSION_1;
        h->ring_depth_k = k;
        h->schema_count = schema_count;
        h->reserved_pad32 = 0;
        h->slot_table_offset = sizeof(HandshakeRegion);
        h->slot_table_stride = sizeof(SlotEntry);
        h->schema_table_offset = sizeof(HandshakeRegion) + k * sizeof(SlotEntry);
        h->schema_table_size = schema_count * sizeof(SchemaEntry);
        h->data_region_offset = h->schema_table_offset + h->schema_table_size;
        h->data_region_size = data_bytes;
    }

    /// Release-store of magic = producer's last write (`shm-block-stream.md` §Memory ordering).
    void sealMagic(uint64_t magic_value = SHM_MAGIC) const
    { hs()->magic.store(magic_value, std::memory_order_release); }

    ~TestShm()
    {
        if (mapping != nullptr && mapping != MAP_FAILED)
            ::munmap(mapping, mapping_size);
        if (fd >= 0)
            ::close(fd);
        ::shm_unlink(name.c_str());
    }
};

size_t defaultTotalSize(uint32_t k = 4, uint32_t schema_count = 2, size_t data_bytes = 1024)
{
    return sizeof(HandshakeRegion) + k * sizeof(SlotEntry)
        + schema_count * sizeof(SchemaEntry) + data_bytes;
}

}

#define EXPECT_ATTACH_THROWS(name_, expected_code) \
    do { \
        try { \
            auto region = SharedMemoryRegion::attach(name_); \
            FAIL() << "Expected error code " << (expected_code) << ", got success"; \
        } catch (const Exception & e) { \
            EXPECT_EQ(e.code(), (expected_code)) << e.message(); \
        } \
    } while (0)


TEST(SharedMemoryRegion, AttachValidHandshakeReturnsRegionWithMatchingFields)
{
    TestShm shm("valid");
    shm.create(defaultTotalSize());
    shm.writeValidLayout(/*k=*/4, /*schema_count=*/2);
    shm.sealMagic();

    auto region = SharedMemoryRegion::attach(shm.name);
    ASSERT_NE(region, nullptr);
    EXPECT_EQ(region->name(), shm.name);
    EXPECT_GE(region->fd(), 0);
    EXPECT_EQ(region->size(), defaultTotalSize());
    EXPECT_NE(region->data(), nullptr);

    const auto & hs = region->handshake();
    EXPECT_EQ(hs.magic.load(std::memory_order_acquire), SHM_MAGIC);
    EXPECT_EQ(hs.abi_version, SHM_ABI_VERSION_1);
    EXPECT_EQ(hs.ring_depth_k, 4u);
    EXPECT_EQ(hs.schema_count, 2u);
    EXPECT_EQ(hs.slot_table_offset, sizeof(HandshakeRegion));
    EXPECT_EQ(hs.slot_table_stride, sizeof(SlotEntry));
}

TEST(SharedMemoryRegion, AttachNonexistentObjectRaisesAttachFailed)
{
    String missing = fmt::format("/ch_shmregion_gt_missing_{}", ::getpid());
    ::shm_unlink(missing.c_str());
    EXPECT_ATTACH_THROWS(missing, ErrorCodes::SHM_ATTACH_FAILED);
}

TEST(SharedMemoryRegion, AttachTooSmallObjectRaisesHandshakeInvalid)
{
    TestShm shm("toosmall");
    /// Size strictly smaller than sizeof(HandshakeRegion) — rejected before mmap.
    shm.create(sizeof(HandshakeRegion) / 2);
    EXPECT_ATTACH_THROWS(shm.name, ErrorCodes::SHM_HANDSHAKE_INVALID);
}

TEST(SharedMemoryRegion, AttachWrongMagicRaisesHandshakeInvalid)
{
    TestShm shm("wrongmagic");
    shm.create(defaultTotalSize());
    shm.writeValidLayout();
    /// Precondition 1 (`pollable-shm-source.md` §Producer-side preconditions row 1).
    shm.sealMagic(0xDEADBEEFDEADBEEFULL);
    EXPECT_ATTACH_THROWS(shm.name, ErrorCodes::SHM_HANDSHAKE_INVALID);
}

TEST(SharedMemoryRegion, AttachUnsupportedAbiVersionRaisesHandshakeInvalid)
{
    TestShm shm("badabi");
    shm.create(defaultTotalSize());
    shm.writeValidLayout();
    shm.hs()->abi_version = 99;
    shm.sealMagic();
    EXPECT_ATTACH_THROWS(shm.name, ErrorCodes::SHM_HANDSHAKE_INVALID);
}

TEST(SharedMemoryRegion, AttachZeroRingDepthRaisesHandshakeInvalid)
{
    TestShm shm("kzero");
    shm.create(defaultTotalSize());
    shm.writeValidLayout();
    shm.hs()->ring_depth_k = 0;
    shm.sealMagic();
    EXPECT_ATTACH_THROWS(shm.name, ErrorCodes::SHM_HANDSHAKE_INVALID);
}

TEST(SharedMemoryRegion, AttachOversizedRingDepthRaisesHandshakeInvalid)
{
    TestShm shm("kover");
    /// Layout as if K were valid (so size math doesn't trip first), then poke K past max.
    shm.create(defaultTotalSize(/*k=*/4));
    shm.writeValidLayout(/*k=*/4);
    shm.hs()->ring_depth_k = IMPL_MAX_K + 1;
    shm.sealMagic();
    EXPECT_ATTACH_THROWS(shm.name, ErrorCodes::SHM_HANDSHAKE_INVALID);
}

TEST(SharedMemoryRegion, AttachDataRegionOverflowsSizeRaisesHandshakeInvalid)
{
    TestShm shm("dataoverflow");
    shm.create(defaultTotalSize());
    shm.writeValidLayout();
    /// Precondition 7 (`pollable-shm-source.md` row 7): regions must fit in SHM.
    shm.hs()->data_region_size = shm.mapping_size;
    shm.sealMagic();
    EXPECT_ATTACH_THROWS(shm.name, ErrorCodes::SHM_HANDSHAKE_INVALID);
}


/// F9 (handshake validity, precondition 7 extension): slot_table_stride must
/// be a multiple of alignof(SlotEntry). A stride that's >= sizeof(SlotEntry)
/// (passes the existing size check) but not a multiple of alignof(SlotEntry)
/// would yield misaligned atomic accesses inside SlotEntry on strict-align
/// architectures.
TEST(SharedMemoryRegion, BadSlotTableStrideAlignmentRejected)
{
    TestShm shm("badstridealign");
    /// Generous size so the unaligned stride doesn't trip the range check first
    /// (4*129 > 4*sizeof(SlotEntry)=4*64 but still fits with room to spare).
    shm.create(defaultTotalSize(/*k=*/4, /*schema_count=*/2, /*data_bytes=*/1024));
    shm.writeValidLayout(/*k=*/4, /*schema_count=*/2);
    /// 129 satisfies `>= sizeof(SlotEntry) == 64` but 129 % 64 == 1.
    shm.hs()->slot_table_stride = sizeof(SlotEntry) + 1;
    shm.sealMagic();
    EXPECT_ATTACH_THROWS(shm.name, ErrorCodes::SHM_HANDSHAKE_INVALID);
}


/// F9: slot_table_offset must be a multiple of alignof(SlotEntry). The
/// producer's normal layout page-aligns it (well above 64); poke a +1 here to
/// exercise the consumer's strictness without disturbing other invariants.
TEST(SharedMemoryRegion, BadSlotTableOffsetAlignmentRejected)
{
    TestShm shm("badslotoffalign");
    shm.create(defaultTotalSize(/*k=*/4, /*schema_count=*/2, /*data_bytes=*/1024));
    shm.writeValidLayout(/*k=*/4, /*schema_count=*/2);
    /// sizeof(HandshakeRegion) is 128, divisible by alignof(SlotEntry)=64;
    /// adding 1 makes it unaligned. The shift cascades into schema/data
    /// offsets in the helper but writeValidLayout already wrote those — we
    /// only nudge the slot offset to isolate the alignment violation.
    shm.hs()->slot_table_offset = sizeof(HandshakeRegion) + 1;
    shm.sealMagic();
    EXPECT_ATTACH_THROWS(shm.name, ErrorCodes::SHM_HANDSHAKE_INVALID);
}


/// F9: schema_table_size MUST exactly equal `schema_count * sizeof(SchemaEntry)`.
/// A truncated size hides legitimate schema entries past the declared span; an
/// inflated size lets a non-conforming producer pack arbitrary bytes (or other
/// regions) inside the schema span and still pass overlap checks.
TEST(SharedMemoryRegion, BadSchemaTableSizeRejected)
{
    TestShm shm("badschematablesize");
    /// schema_count=3 in the test, but schema_table_size will be set to
    /// 2*sizeof(SchemaEntry) — exactly the case called out in F9.
    constexpr uint32_t real_schema_count = 3;
    shm.create(defaultTotalSize(/*k=*/4, real_schema_count, /*data_bytes=*/1024));
    shm.writeValidLayout(/*k=*/4, real_schema_count, /*data_bytes=*/1024);
    shm.hs()->schema_table_size = 2 * sizeof(SchemaEntry);
    shm.sealMagic();
    EXPECT_ATTACH_THROWS(shm.name, ErrorCodes::SHM_HANDSHAKE_INVALID);
}
