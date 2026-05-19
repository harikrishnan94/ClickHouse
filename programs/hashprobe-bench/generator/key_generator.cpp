/// hashprobe-bench/generator/key_generator.cpp

#include "generator/key_generator.h"

#include <pcg_random.hpp>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>

namespace DB::HashProbeBench
{

// ─── KeyGenerator ─────────────────────────────────────────────────────────────

uint64_t KeyGenerator::maxKeyValue() const
{
    return (p_.shape.width == 32) ? ((uint64_t(1) << 31) - 1) : ((uint64_t(1) << 63) - 1);
}

double KeyGenerator::getBuildRowToKeyRatio() const
{
    if (p_.build_distinct_keys == 0)
        return std::numeric_limits<double>::infinity();
    return static_cast<double>(p_.build_rows) / static_cast<double>(p_.build_distinct_keys);
}

KeyGenerator::KeyGenerator(Params p) : p_(std::move(p))
{
    if (p_.build_distinct_keys == 0)
        throw std::invalid_argument("KeyGenerator: build_distinct_keys must be >= 1");
    if (p_.build_rows == 0)
        throw std::invalid_argument("KeyGenerator: build_rows must be >= 1");
    if (p_.probe_rows == 0)
        throw std::invalid_argument("KeyGenerator: probe_rows must be >= 1");
    if (p_.match_rate < 0.0 || p_.match_rate > 1.0)
        throw std::invalid_argument("KeyGenerator: match_rate must be in [0,1]");
    if (p_.null_fraction < 0.0 || p_.null_fraction > 1.0)
        throw std::invalid_argument("KeyGenerator: null_fraction must be in [0,1]");
    if (p_.shape.n == 0 || (p_.shape.n != 1 && p_.shape.n != 2 && p_.shape.n != 4))
        throw std::invalid_argument("KeyGenerator: shape.n must be 1, 2, or 4");
    if (p_.shape.width != 32 && p_.shape.width != 64)
        throw std::invalid_argument("KeyGenerator: shape.width must be 32 or 64");

    // A.2: ALL requires at least one duplicate build key (ratio > 1)
    if (p_.strictness == StrictnessConfig::ALL && p_.build_distinct_keys >= p_.build_rows)
        throw std::invalid_argument(
            "KeyGenerator (ALL): build_distinct_keys must be < build_rows "
            "(got build_distinct_keys=" + std::to_string(p_.build_distinct_keys) +
            " build_rows=" + std::to_string(p_.build_rows) + ")");

    // A.2: RIGHTANY requires at most one build row per distinct key
    if (p_.strictness == StrictnessConfig::RIGHTANY && p_.build_distinct_keys != p_.build_rows)
        throw std::invalid_argument(
            "KeyGenerator (RIGHTANY): build_distinct_keys must equal build_rows "
            "(got build_distinct_keys=" + std::to_string(p_.build_distinct_keys) +
            " build_rows=" + std::to_string(p_.build_rows) + ")");


    if (p_.build_distinct_keys > maxKeyValue())
        throw std::invalid_argument(
            "KeyGenerator: build_distinct_keys=" + std::to_string(p_.build_distinct_keys) +
            " exceeds maxKeyValue=" + std::to_string(maxKeyValue()) +
            " for width=" + std::to_string(p_.shape.width));
}

// ─── BuildIterator ─────────────────────────────────────────────────────────────

namespace
{

class BuildIterator final : public KeyGenerator::Iterator
{
public:
    explicit BuildIterator(const KeyGenerator::Params & p) : p_(p), pos_(0)
    {
        const uint64_t B = p_.build_distinct_keys;
        const uint64_t N = p_.build_rows;
        key_indices_.resize(N);

        // First B slots: one of each distinct key (ensures full coverage)
        for (uint64_t i = 0; i < B; ++i)
            key_indices_[i] = i;

        // Remaining N-B slots: random from [0, B) — creates duplicates for ALL
        if (N > B)
        {
            pcg64 rng_dup(p_.seed);
            std::uniform_int_distribution<uint64_t> dist(0, B - 1);
            for (uint64_t i = B; i < N; ++i)
                key_indices_[i] = dist(rng_dup);
        }

        // Shuffle the build order
        pcg64 rng_shuf(p_.seed + 100);
        std::shuffle(key_indices_.begin(), key_indices_.end(), rng_shuf);

        rng_null_ = pcg64(p_.seed + 200);
        null_dist_ = std::bernoulli_distribution(p_.null_fraction);
    }

    bool next(KeyGenerator::KeyRow & out) override
    {
        if (pos_ >= key_indices_.size())
            return false;

        const uint32_t n = p_.shape.n;
        out.key_values.resize(n);
        out.null_mask.resize(n);
        out.payload           = pos_;
        out.from_build_universe = true;

        const uint64_t tuple_idx = key_indices_[pos_++];

        for (uint32_t col = 0; col < n; ++col)
        {
            // +1 shift: build keys live in [1, B], keeping 0 out of the hash table.
            out.key_values[col] = tuple_idx + 1;
            if (p_.null_fraction > 0.0 && null_dist_(rng_null_))
            {
                out.null_mask[col]   = 1;
                out.key_values[col]  = 0;
            }
            else
            {
                out.null_mask[col] = 0;
            }
        }
        return true;
    }

private:
    KeyGenerator::Params          p_;
    std::vector<uint64_t>         key_indices_;
    uint64_t                      pos_;
    pcg64                         rng_null_;
    std::bernoulli_distribution   null_dist_;
};

// ─── ProbeIterator ─────────────────────────────────────────────────────────────

class ProbeIterator final : public KeyGenerator::Iterator
{
public:
    explicit ProbeIterator(const KeyGenerator::Params & p) : p_(p), pos_(0)
    {
        const uint64_t B       = p_.build_distinct_keys;
        const uint64_t N       = p_.probe_rows;
        const uint64_t max_val = maxKeyValueForWidth(p_.shape.width);

        const uint64_t match_count   = static_cast<uint64_t>(std::round(p_.match_rate * static_cast<double>(N)));
        const uint64_t nomatch_count = N - match_count;

        key_indices_.resize(N);
        is_match_.resize(N, 0);

        // Match rows: keys uniformly from [1, B]
        if (match_count > 0 && B > 0)
        {
            pcg64 rng(p_.seed + 300);
            std::uniform_int_distribution<uint64_t> dist(1, B);
            for (uint64_t i = 0; i < match_count; ++i)
            {
                key_indices_[i] = dist(rng);
                is_match_[i]    = 1;
            }
        }

        // No-match rows: keys uniformly from [B+1, max_val]
        if (nomatch_count > 0)
        {
            pcg64 rng(p_.seed + 400);
            const uint64_t lo = B + 1;
            const uint64_t hi = max_val;
            if (lo <= hi)
            {
                std::uniform_int_distribution<uint64_t> dist(lo, hi);
                for (uint64_t i = match_count; i < N; ++i)
                    key_indices_[i] = dist(rng);
            }
            else
            {
                // Degenerate: entire key space is the build universe
                for (uint64_t i = match_count; i < N; ++i)
                    key_indices_[i] = 0; // key space [1, max_val] fully occupied; 0 is out of range
            }
        }

        // In-place Fisher-Yates shuffle on two parallel arrays (O(1) extra memory)
        {
            pcg64 rng(p_.seed + 500);
            for (uint64_t i = N - 1; i > 0; --i)
            {
                // Modulo bias is < N/2^64 < 6e-13, negligible for tests
                uint64_t j = rng() % (i + 1);
                std::swap(key_indices_[i], key_indices_[j]);
                std::swap(is_match_[i],    is_match_[j]);
            }
        }

        rng_null_ = pcg64(p_.seed + 600);
        null_dist_ = std::bernoulli_distribution(p_.null_fraction);
    }

    bool next(KeyGenerator::KeyRow & out) override
    {
        if (pos_ >= key_indices_.size())
            return false;

        const uint32_t n = p_.shape.n;
        out.key_values.resize(n);
        out.null_mask.resize(n);
        out.payload             = pos_;
        out.from_build_universe = (is_match_[pos_] != 0);

        const uint64_t tuple_idx = key_indices_[pos_++];

        for (uint32_t col = 0; col < n; ++col)
        {
            // +1 shift: probe keys in [1, B] (match) or [B+1, max_val] (no-match).
            out.key_values[col] = tuple_idx;
            if (p_.null_fraction > 0.0 && null_dist_(rng_null_))
            {
                out.null_mask[col]  = 1;
                out.key_values[col] = 0;
            }
            else
            {
                out.null_mask[col] = 0;
            }
        }
        return true;
    }

private:
    static uint64_t maxKeyValueForWidth(uint32_t width)
    {
        return (width == 32) ? ((uint64_t(1) << 31) - 1) : ((uint64_t(1) << 63) - 1);
    }

    KeyGenerator::Params         p_;
    std::vector<uint64_t>        key_indices_;
    std::vector<uint8_t>         is_match_;
    uint64_t                     pos_;
    pcg64                        rng_null_;
    std::bernoulli_distribution  null_dist_;
};

} // anonymous namespace

std::unique_ptr<KeyGenerator::Iterator> KeyGenerator::buildIterator() const
{
    return std::make_unique<BuildIterator>(p_);
}

std::unique_ptr<KeyGenerator::Iterator> KeyGenerator::probeIterator() const
{
    return std::make_unique<ProbeIterator>(p_);
}

} // namespace DB::HashProbeBench
