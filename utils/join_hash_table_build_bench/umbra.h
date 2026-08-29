#pragma once

#include "hashes.h"
#include "platform.h"

#include <algorithm>
#include <atomic>
#include <barrier>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

/// 32-byte tuples, matching the paper's Figure 12 microbenchmark.
struct UmbraTuple
{
    uint64_t key;
    uint64_t hash;
    UmbraTuple * next;
    uint64_t row;
};
static_assert(sizeof(UmbraTuple) == 32);

inline constexpr size_t kLargeChunk = 2ull << 20;
inline constexpr size_t kSmallChunk = 8ull << 10;

struct ChunkSpan
{
    char * begin;
    char * end;
};

struct Bump
{
    char * base = nullptr;
    char * cur = nullptr;
    char * end = nullptr;
    std::vector<ChunkSpan> spans;

    size_t free_space() const { return static_cast<size_t>(end - cur); }

    void freeze()
    {
        if (base && cur > base)
            spans.push_back({base, cur});
        base = cur = end = nullptr;
    }

    void add_space(char * p, size_t n)
    {
        freeze();
        base = cur = p;
        end = p + n;
    }

    void * alloc(size_t n)
    {
        char * p = cur;
        cur += n;
        return p;
    }
};

struct ThreadCollector
{
    Bump level2;
    std::vector<Bump> level3;
    std::vector<uint64_t> counts;
    std::vector<std::pair<void *, size_t>> larges;

    void init(size_t parts)
    {
        level3.assign(parts, {});
        counts.assign(parts, 0);
        larges.clear();
        level2 = {};
    }

    void refill(size_t part)
    {
        if (level2.free_space() < kSmallChunk)
        {
            void * p = map_anon(kLargeChunk);
            larges.push_back({p, kLargeChunk});
            level2.add_space(static_cast<char *>(p), kLargeChunk);
        }
        char * small = static_cast<char *>(level2.alloc(kSmallChunk));
        level3[part].add_space(small, kSmallChunk);
    }

    void consume(const UmbraTuple & t, uint32_t log_parts)
    {
        const size_t part = t.hash >> (64u - log_parts);
        if (level3[part].free_space() < sizeof(UmbraTuple))
            refill(part);
        *static_cast<UmbraTuple *>(level3[part].alloc(sizeof(UmbraTuple))) = t;
        counts[part] += 1;
    }

    void freeze_all()
    {
        level2.freeze();
        for (auto & b : level3)
            b.freeze();
    }

    void release()
    {
        for (auto & [p, n] : larges)
            unmap_anon(p, n);
        larges.clear();
    }
};

template <typename F>
void for_each_tuple(const std::vector<ChunkSpan> & spans, F && f)
{
    for (const auto & s : spans)
    {
        for (char * p = s.begin; p < s.end; p += sizeof(UmbraTuple))
            f(*reinterpret_cast<UmbraTuple *>(p));
    }
}

struct StageNs
{
    uint64_t collect = 0;
    uint64_t count = 0;
    uint64_t insert = 0;
    uint64_t latch = 0;
    uint64_t wall = 0;
    uint64_t total() const { return wall ? wall : (collect + count + insert); }
};

struct UnchainedTable
{
    uint64_t * dir_raw = nullptr;
    uint64_t * directory = nullptr;
    UmbraTuple * storage = nullptr;
    size_t dir_slots = 0;
    size_t n_tuples = 0;
    size_t dir_bytes = 0;
    size_t storage_bytes = 0;
    uint32_t shift = 0;

    void reset()
    {
        unmap_anon(dir_raw, dir_bytes);
        unmap_anon(storage, storage_bytes);
        dir_raw = nullptr;
        directory = nullptr;
        storage = nullptr;
        dir_slots = 0;
        n_tuples = 0;
        dir_bytes = 0;
        storage_bytes = 0;
        shift = 0;
    }

    ~UnchainedTable() { reset(); }
    UnchainedTable() = default;
    UnchainedTable(const UnchainedTable &) = delete;
    UnchainedTable & operator=(const UnchainedTable &) = delete;
};

/// 2^ceil(log2(1.125 n)), at least `parts` so collection partitions own disjoint directory ranges.
inline size_t umbra_dir_slots(uint64_t n, size_t parts)
{
    const uint64_t need = n == 0 ? 1 : (n * 9 + 7) / 8;
    return next_pow2(std::max(need, static_cast<uint64_t>(parts)));
}

inline StageNs build_unchained(UnchainedTable & table, const uint64_t * keys, uint64_t n, size_t threads, size_t parts, bool pin)
{
    if ((parts & (parts - 1)) != 0)
        throw std::runtime_error("umbra parts must be a power of two");

    const uint32_t log_parts = ceil_log2_u64(parts);
    const size_t dir_slots = umbra_dir_slots(n, parts);
    const uint32_t kbits = ceil_log2_u64(dir_slots);
    const uint32_t shift = 64u - kbits;

    table.reset();
    table.dir_slots = dir_slots;
    table.n_tuples = n;
    table.shift = shift;
    table.dir_bytes = (dir_slots + 1) * sizeof(uint64_t);
    table.storage_bytes = n * sizeof(UmbraTuple);
    table.dir_raw = static_cast<uint64_t *>(map_anon(table.dir_bytes));
    table.storage = static_cast<UmbraTuple *>(map_anon(table.storage_bytes));
    prefault_write(table.dir_raw, table.dir_bytes);
    prefault_write(table.storage, table.storage_bytes);
    table.directory = table.dir_raw + 1;
    table.dir_raw[0] = reinterpret_cast<uintptr_t>(table.storage) << 16;

    std::vector<ThreadCollector> collectors(threads);
    std::barrier bar(static_cast<std::ptrdiff_t>(threads));
    std::atomic<size_t> next_part{0};
    std::vector<uint64_t> part_counts(parts, 0);
    std::vector<uint64_t> part_off(parts, 0);
    std::vector<std::vector<ChunkSpan>> part_spans(parts);
    uint64_t t_collect0 = 0;
    uint64_t t_collect1 = 0;
    uint64_t t_count0 = 0;
    uint64_t t_count1 = 0;
    uint64_t t_copy0 = 0;
    uint64_t t_copy1 = 0;

    parallel_for(
        threads,
        pin,
        [&](size_t tid)
        {
            collectors[tid].init(parts);
            {
                const uint64_t begin = n * tid / threads;
                const uint64_t end = n * (tid + 1) / threads;
                const size_t need = static_cast<size_t>(end - begin) * sizeof(UmbraTuple) + parts * kSmallChunk + kLargeChunk;
                void * p = map_anon(need);
                collectors[tid].larges.push_back({p, need});
                prefault_write(p, need);
                collectors[tid].level2.add_space(static_cast<char *>(p), need);
            }
            bar.arrive_and_wait();
            if (tid == 0)
                t_collect0 = ns_now();
            bar.arrive_and_wait();

            const uint64_t begin = n * tid / threads;
            const uint64_t end = n * (tid + 1) / threads;
            for (uint64_t i = begin; i < end; ++i)
            {
                UmbraTuple t{};
                t.key = keys[i];
                t.hash = umbra_hash64(keys[i]);
                t.next = nullptr;
                t.row = i;
                collectors[tid].consume(t, log_parts);
            }
            collectors[tid].freeze_all();
            bar.arrive_and_wait();
            if (tid == 0)
                t_collect1 = ns_now();

            if (tid == 0)
            {
                t_count0 = ns_now();
                for (size_t t = 0; t < threads; ++t)
                {
                    for (size_t p = 0; p < parts; ++p)
                    {
                        part_counts[p] += collectors[t].counts[p];
                        const auto & spans = collectors[t].level3[p].spans;
                        part_spans[p].insert(part_spans[p].end(), spans.begin(), spans.end());
                    }
                }
                uint64_t acc = 0;
                for (size_t p = 0; p < parts; ++p)
                {
                    part_off[p] = acc;
                    acc += part_counts[p];
                }
                t_count1 = ns_now();
            }
            bar.arrive_and_wait();
            if (tid == 0)
                t_copy0 = ns_now();
            bar.arrive_and_wait();

            for (;;)
            {
                const size_t p = next_part.fetch_add(1, std::memory_order_relaxed);
                if (p >= parts)
                    break;

                uint64_t * directory = table.directory;
                for_each_tuple(
                    part_spans[p],
                    [&](const UmbraTuple & t)
                    {
                        const uint64_t slot = t.hash >> shift;
                        directory[slot] += sizeof(UmbraTuple) << 16;
                        directory[slot] |= umbra_tag(t.hash);
                    });

                uintptr_t cur = reinterpret_cast<uintptr_t>(table.storage) + part_off[p] * sizeof(UmbraTuple);
                const uint64_t start = (static_cast<uint64_t>(p) << kbits) / parts;
                const uint64_t end_slot = (static_cast<uint64_t>(p + 1) << kbits) / parts;
                for (uint64_t i = start; i < end_slot; ++i)
                {
                    const uint64_t val = directory[i] >> 16;
                    directory[i] = (cur << 16) | (directory[i] & 0xFFFFull);
                    cur += val;
                }

                for_each_tuple(
                    part_spans[p],
                    [&](const UmbraTuple & t)
                    {
                        const uint64_t slot = t.hash >> shift;
                        auto * target = reinterpret_cast<UmbraTuple *>(directory[slot] >> 16);
                        *target = t;
                        directory[slot] += sizeof(UmbraTuple) << 16;
                    });
            }
            bar.arrive_and_wait();
            if (tid == 0)
                t_copy1 = ns_now();
        });

    for (auto & c : collectors)
        c.release();

    StageNs stages;
    stages.collect = t_collect1 - t_collect0;
    stages.count = t_count1 - t_count0;
    stages.insert = t_copy1 - t_copy0;
    stages.wall = stages.collect + stages.count + stages.insert;
    return stages;
}

inline void validate_unchained(const UnchainedTable & table, const uint32_t * hist, uint64_t distinct)
{
    std::vector<uint32_t> got(distinct + 1, 0);
    for (size_t i = 0; i < table.n_tuples; ++i)
    {
        const uint64_t k = table.storage[i].key;
        if (k == 0 || k > distinct)
            throw std::runtime_error("unchained: key out of range");
        got[k] += 1;
    }
    for (uint64_t k = 1; k <= distinct; ++k)
    {
        if (got[k] != hist[k])
            throw std::runtime_error("unchained: histogram mismatch");
    }
}

struct ChainedTable
{
    std::atomic<uint64_t> * directory = nullptr;
    size_t dir_slots = 0;
    uint32_t shift = 0;
    size_t dir_bytes = 0;
    std::vector<std::pair<void *, size_t>> slabs;

    void reset()
    {
        unmap_anon(directory, dir_bytes);
        directory = nullptr;
        for (auto & [p, nbytes] : slabs)
            unmap_anon(p, nbytes);
        slabs.clear();
        dir_slots = 0;
        dir_bytes = 0;
    }

    ~ChainedTable() { reset(); }
    ChainedTable() = default;
    ChainedTable(const ChainedTable &) = delete;
    ChainedTable & operator=(const ChainedTable &) = delete;
};

inline StageNs build_chained(ChainedTable & table, const uint64_t * keys, uint64_t n, size_t threads, size_t /*parts*/, bool pin)
{
    const size_t dir_slots = umbra_dir_slots(n, 1);
    const uint32_t kbits = ceil_log2_u64(dir_slots);
    const uint32_t shift = 64u - kbits;

    table.reset();
    table.dir_slots = dir_slots;
    table.shift = shift;
    table.dir_bytes = dir_slots * sizeof(std::atomic<uint64_t>);
    table.directory = static_cast<std::atomic<uint64_t> *>(map_anon(table.dir_bytes));
    prefault_write(table.directory, table.dir_bytes);

    std::vector<UmbraTuple *> bases(threads, nullptr);
    table.slabs.resize(threads);
    for (size_t t = 0; t < threads; ++t)
    {
        const uint64_t begin = n * t / threads;
        const uint64_t end = n * (t + 1) / threads;
        const size_t bytes = static_cast<size_t>(end - begin) * sizeof(UmbraTuple);
        void * p = map_anon(bytes);
        prefault_write(p, bytes);
        table.slabs[t] = {p, bytes};
        bases[t] = static_cast<UmbraTuple *>(p);
    }

    std::barrier bar(static_cast<std::ptrdiff_t>(threads));
    uint64_t t0 = 0;
    uint64_t t1 = 0;

    parallel_for(
        threads,
        pin,
        [&](size_t tid)
        {
            bar.arrive_and_wait();
            if (tid == 0)
                t0 = ns_now();
            bar.arrive_and_wait();

            const uint64_t begin = n * tid / threads;
            const uint64_t end = n * (tid + 1) / threads;
            UmbraTuple * slab = bases[tid];
            auto * directory = table.directory;
            for (uint64_t i = begin; i < end; ++i)
            {
                UmbraTuple * tup = &slab[i - begin];
                tup->key = keys[i];
                tup->hash = umbra_hash64(keys[i]);
                tup->row = i;
                const uint64_t slot = tup->hash >> shift;
                const uint64_t new_entry = reinterpret_cast<uintptr_t>(tup) << 16;
                const uint64_t prev = directory[slot].exchange(new_entry, std::memory_order_acq_rel);
                tup->next = reinterpret_cast<UmbraTuple *>(prev >> 16);
                directory[slot].fetch_or((prev & 0xFFFFull) | umbra_tag(tup->hash), std::memory_order_relaxed);
            }

            bar.arrive_and_wait();
            if (tid == 0)
                t1 = ns_now();
        });

    StageNs stages;
    stages.insert = t1 - t0;
    stages.wall = stages.insert;
    return stages;
}

inline void validate_chained(const ChainedTable & table, const uint32_t * hist, uint64_t distinct)
{
    std::vector<uint32_t> got(distinct + 1, 0);
    uint64_t seen = 0;
    uint64_t n = 0;
    for (const auto & [p, bytes] : table.slabs)
        n += bytes / sizeof(UmbraTuple);

    for (size_t s = 0; s < table.dir_slots; ++s)
    {
        const uint64_t entry = table.directory[s].load(std::memory_order_relaxed);
        auto * cur = reinterpret_cast<UmbraTuple *>(entry >> 16);
        while (cur)
        {
            if (cur->key == 0 || cur->key > distinct)
                throw std::runtime_error("chained: key out of range");
            got[cur->key] += 1;
            ++seen;
            cur = cur->next;
        }
    }
    if (seen != n)
        throw std::runtime_error("chained: row count mismatch");
    for (uint64_t k = 1; k <= distinct; ++k)
    {
        if (got[k] != hist[k])
            throw std::runtime_error("chained: histogram mismatch");
    }
}
