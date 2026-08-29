#pragma once

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

/// Zipf(s) over ranks 1..n. s = 0 is uniform. Alias table (Vose) for s > 0.
struct Zipf
{
    uint64_t n = 1;
    double s = 0;
    std::vector<uint32_t> alias;
    std::vector<double> prob;

    Zipf() = default;

    Zipf(uint64_t n_, double s_)
        : n(n_)
        , s(s_)
    {
        if (n == 0)
            throw std::runtime_error("Zipf n must be > 0");
        if (s < 0)
            throw std::runtime_error("Zipf s must be >= 0");
        if (s == 0.0)
            return;
        build_alias();
    }

    /// Unbiased in 1..n from a 64-bit random word.
    uint64_t sample(uint64_t rnd) const
    {
        if (s == 0.0)
            return static_cast<uint64_t>((__uint128_t(rnd) * n) >> 64) + 1;

        const uint32_t nn = static_cast<uint32_t>(n);
        const uint32_t i = static_cast<uint32_t>((__uint128_t(static_cast<uint32_t>(rnd)) * nn) >> 32);
        const double u = static_cast<double>(rnd >> 11) * (1.0 / 9007199254740992.0); /// 2^53
        if (u < prob[i])
            return static_cast<uint64_t>(i) + 1;
        return static_cast<uint64_t>(alias[i]) + 1;
    }

private:
    void build_alias()
    {
        const size_t nn = static_cast<size_t>(n);
        std::vector<double> p(nn);
        double z = 0;
        for (size_t i = 0; i < nn; ++i)
        {
            p[i] = std::pow(static_cast<double>(i + 1), -s);
            z += p[i];
        }
        for (size_t i = 0; i < nn; ++i)
            p[i] = p[i] * static_cast<double>(nn) / z;

        alias.assign(nn, 0);
        prob.assign(nn, 0);
        std::vector<uint32_t> small;
        std::vector<uint32_t> large;
        small.reserve(nn);
        large.reserve(nn);
        for (uint32_t i = 0; i < nn; ++i)
        {
            if (p[i] < 1.0)
                small.push_back(i);
            else
                large.push_back(i);
        }
        while (!small.empty() && !large.empty())
        {
            const uint32_t l = small.back();
            small.pop_back();
            const uint32_t g = large.back();
            large.pop_back();
            prob[l] = p[l];
            alias[l] = g;
            p[g] = (p[g] + p[l]) - 1.0;
            if (p[g] < 1.0)
                small.push_back(g);
            else
                large.push_back(g);
        }
        while (!large.empty())
        {
            prob[large.back()] = 1.0;
            alias[large.back()] = large.back();
            large.pop_back();
        }
        while (!small.empty())
        {
            prob[small.back()] = 1.0;
            alias[small.back()] = small.back();
            small.pop_back();
        }
    }
};

inline uint64_t splitmix64(uint64_t & state)
{
    state += 0x9E3779B97F4A7C15ull;
    uint64_t z = state;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}
