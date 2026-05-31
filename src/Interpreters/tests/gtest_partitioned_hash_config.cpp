#include <Columns/ColumnsNumber.h>
#include <Interpreters/PartitionedHashConfig.h>

#include <random>
#include <vector>

#include <gtest/gtest.h>

using namespace DB;

namespace
{

PartitionConfig makeConfig(std::optional<UInt64> rhs_rows, size_t key_bytes, size_t payload_bytes, size_t l2_bytes)
{
    PartitionConfigInputs in;
    in.rhs_rows_estimation = rhs_rows;
    in.key_bytes = key_bytes;
    in.payload_bytes = payload_bytes;
    in.cell_bytes = 48;
    in.l2_bytes = l2_bytes;
    in.max_partitions_per_pass = 64;
    return derivePartitionConfig(in);
}

}

/// Hash determinism: the same key column hashed twice yields identical hashes (and identical leaves).
TEST(PartitionedHashConfig, HashDeterminism)
{
    auto col = ColumnUInt64::create();
    std::mt19937_64 rng(12345); // NOLINT(cert-msc51-cpp): deterministic test input is intended
    for (size_t i = 0; i < 4096; ++i)
        col->insertValue(rng());
    const size_t n = col->size();

    std::vector<UInt32> h1(n);
    std::vector<UInt32> h2(n);
    col->computeHashInto(0, n, h1.data(), /*initial=*/true);
    col->computeHashInto(0, n, h2.data(), /*initial=*/true);

    PartitionConfig cfg = makeConfig(100ULL * 1000 * 1000, 8, 56, 2 * 1024 * 1024);
    for (size_t i = 0; i < n; ++i)
    {
        EXPECT_EQ(h1[i], h2[i]);
        EXPECT_EQ(cfg.leafForHash(h1[i]), cfg.leafForHash(h2[i]));
    }
}

/// Build/probe routing identity: two independently-derived configs from identical inputs route every
/// key to the same pids and leaf (spec invariant #6 — same hash + same schedule on both sides).
TEST(PartitionedHashConfig, BuildProbeRoutingIdentity)
{
    PartitionConfig build_cfg = makeConfig(100ULL * 1000 * 1000, 8, 56, 2 * 1024 * 1024);
    PartitionConfig probe_cfg = makeConfig(100ULL * 1000 * 1000, 8, 56, 2 * 1024 * 1024);
    ASSERT_EQ(build_cfg.pass_bits, probe_cfg.pass_bits);
    ASSERT_EQ(build_cfg.total_leaves, probe_cfg.total_leaves);

    std::mt19937_64 rng(99); // NOLINT(cert-msc51-cpp): deterministic test input is intended
    for (size_t i = 0; i < 100000; ++i)
    {
        auto h = static_cast<UInt32>(rng());
        EXPECT_EQ(build_cfg.leafForHash(h), probe_cfg.leafForHash(h));
        for (size_t p = 0; p < build_cfg.numPasses(); ++p)
            EXPECT_EQ(build_cfg.pidForPass(h, p), probe_cfg.pidForPass(h, p));
    }
}

/// Bit-window disjointness + coverage: windows are contiguous & non-overlapping, leaf in range, and
/// leaf == top totalBits() of the hash == the fold of the per-pass pids.
TEST(PartitionedHashConfig, BitWindowCoverageAndFold)
{
    PartitionConfig cfg = makeConfig(100ULL * 1000 * 1000, 8, 56, 2 * 1024 * 1024);
    const UInt8 total_bits = cfg.totalBits();

    /// Sum of per-pass bits == total bits, each pass <= 6.
    UInt8 sum = 0;
    for (auto b : cfg.pass_bits)
    {
        EXPECT_LE(b, 6);
        EXPECT_GE(b, 1);
        sum = static_cast<UInt8>(sum + b);
    }
    EXPECT_EQ(sum, total_bits);

    /// Windows: shift decreases by exactly pass_bits[i] each pass (contiguous, non-overlapping).
    UInt32 expected_shift = PartitionConfig::HASH_BITS;
    for (size_t p = 0; p < cfg.numPasses(); ++p)
    {
        expected_shift -= cfg.pass_bits[p];
        EXPECT_EQ(cfg.shiftForPass(p), expected_shift);
    }

    std::mt19937_64 rng(7); // NOLINT(cert-msc51-cpp): deterministic test input is intended
    for (size_t i = 0; i < 100000; ++i)
    {
        auto h = static_cast<UInt32>(rng());
        const size_t leaf = cfg.leafForHash(h);
        EXPECT_LT(leaf, cfg.total_leaves);
        EXPECT_EQ(leaf, static_cast<size_t>(h >> (PartitionConfig::HASH_BITS - total_bits)));

        /// Reconstruct leaf from the per-pass pids (fold).
        size_t folded = 0;
        for (size_t p = 0; p < cfg.numPasses(); ++p)
            folded = (folded << cfg.pass_bits[p]) | cfg.pidForPass(h, p);
        EXPECT_EQ(folded, leaf);
    }
}

/// Config derivation: power-of-two leaves, even <=6-bit distribution, worked examples from spec §5.4.
TEST(PartitionedHashConfig, DerivationWorkedExamples)
{
    const size_t l2 = 2 * 1024 * 1024;
    const UInt64 rows = 100ULL * 1000 * 1000;

    /// Q7: 8 build cols (key + 7 payload), right_row_bytes = 64 -> 8192 leaves, schedule 5,4,4.
    {
        PartitionConfig cfg = makeConfig(rows, /*key*/ 8, /*payload*/ 56, l2);
        EXPECT_EQ(cfg.total_leaves, 8192u);
        EXPECT_EQ(cfg.totalBits(), 13);
        std::vector<UInt8> expected{5, 4, 4};
        EXPECT_EQ(cfg.pass_bits, expected);
    }

    /// Q1: 1 build col (key + 1 payload), right_row_bytes = 16 -> 4096 leaves, schedule 6,6.
    {
        PartitionConfig cfg = makeConfig(rows, /*key*/ 8, /*payload*/ 8, l2);
        EXPECT_EQ(cfg.total_leaves, 4096u);
        EXPECT_EQ(cfg.totalBits(), 12);
        std::vector<UInt8> expected{6, 6};
        EXPECT_EQ(cfg.pass_bits, expected);
    }
}

/// Default leaf count when the estimate is absent (spec §5.2 -> 256 leaves -> 4,4).
TEST(PartitionedHashConfig, DefaultLeavesWhenEstimateAbsent)
{
    PartitionConfig cfg = makeConfig(std::nullopt, 8, 56, 2 * 1024 * 1024);
    EXPECT_EQ(cfg.total_leaves, PHJ_DEFAULT_LEAVES);
    EXPECT_EQ(cfg.totalBits(), 8);
    std::vector<UInt8> expected{4, 4};
    EXPECT_EQ(cfg.pass_bits, expected);
}

/// MAX_LEAVES clamp: a hugely over-estimated right side cannot shrink leaves past the cap.
TEST(PartitionedHashConfig, MaxLeavesClamp)
{
    /// Absurd estimate -> would derive far more than PHJ_MAX_LEAVES leaves; must clamp.
    PartitionConfig cfg = makeConfig(1ULL << 60, 8, 56, 256 * 1024);
    EXPECT_LE(cfg.total_leaves, PHJ_MAX_LEAVES);
    EXPECT_EQ(cfg.total_leaves, size_t{1} << cfg.totalBits());
    for (auto b : cfg.pass_bits)
        EXPECT_LE(b, 6);
}

/// roundUpPow2 basic behaviour.
TEST(PartitionedHashConfig, RoundUpPow2)
{
    EXPECT_EQ(roundUpPow2(0), 1u);
    EXPECT_EQ(roundUpPow2(1), 1u);
    EXPECT_EQ(roundUpPow2(3), 4u);
    EXPECT_EQ(roundUpPow2(4), 4u);
    EXPECT_EQ(roundUpPow2(6677), 8192u);
    EXPECT_EQ(roundUpPow2(3816), 4096u);
}
