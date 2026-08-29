#pragma once

#include "hashes.h"
#include "latch.h"
#include "platform.h"
#include "umbra.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <vector>

struct RowRef
{
    static constexpr uint32_t INLINE_FLAG = 0x80000000u;
    static constexpr uint64_t ENCODED_INLINE = 1ull << 63;

    static uint64_t encode(uint32_t block_no, uint32_t row_no) { return (static_cast<uint64_t>(block_no | INLINE_FLAG) << 32) | row_no; }
};

struct Batch
{
    static constexpr size_t SLOTS = 6;
    uint64_t is_range : 1;
    uint64_t size : 7;
    uint64_t total_rows : 56;
    uint64_t refs[SLOTS + 1];
};
static_assert(sizeof(Batch) == 64);

struct Arena
{
    static constexpr size_t kChunk = 64ull << 10;
    std::vector<std::pair<void *, size_t>> chunks;
    char * cur = nullptr;
    char * end = nullptr;

    Batch * alloc_batch()
    {
        constexpr size_t n = sizeof(Batch);
        if (cur + n > end)
        {
            const size_t cap = std::max(n, kChunk);
            void * p = map_anon(cap);
            chunks.push_back({p, cap});
            cur = static_cast<char *>(p);
            end = cur + cap;
        }
        auto * b = reinterpret_cast<Batch *>(cur);
        cur += n;
        std::memset(b, 0, n);
        return b;
    }

    void release()
    {
        for (auto & [p, nbytes] : chunks)
            unmap_anon(p, nbytes);
        chunks.clear();
        cur = end = nullptr;
    }
};

inline constexpr uint64_t kPtrMask = (1ull << 48) - 1;
inline constexpr uint32_t kCountShift = 48;
inline constexpr uint32_t kCountSat = 0x7FFFu;
inline constexpr size_t kMaxLocal = 1 + Batch::SLOTS;

inline void set_list_word(uint64_t & word, Batch * b, uint64_t total)
{
    const uint64_t ptr = reinterpret_cast<uint64_t>(b);
    const uint64_t count = total < kCountSat ? total : kCountSat;
    word = ptr | (count << kCountShift);
}

inline void list_insert(uint64_t & word, uint64_t ref, Arena & pool)
{
    if (word == 0)
    {
        word = ref;
        return;
    }
    if (word & RowRef::ENCODED_INLINE)
    {
        Batch * b = pool.alloc_batch();
        b->size = 2;
        b->total_rows = 2;
        b->refs[0] = word;
        b->refs[1] = ref;
        set_list_word(word, b, 2);
        return;
    }

    auto * b = reinterpret_cast<Batch *>(word & kPtrMask);
    const uint64_t new_total = b->total_rows + 1;
    if (b->size == b->total_rows)
    {
        if (b->size < kMaxLocal)
        {
            b->refs[b->size] = ref;
            b->size = b->size + 1;
        }
        else
        {
            Batch * n = pool.alloc_batch();
            n->size = 2;
            n->refs[1] = b->refs[Batch::SLOTS];
            n->refs[2] = ref;
            b->refs[Batch::SLOTS] = reinterpret_cast<uint64_t>(n);
            b->size = kMaxLocal - 1;
        }
    }
    else
    {
        auto * newest = reinterpret_cast<Batch *>(b->refs[Batch::SLOTS]);
        if (newest->size < Batch::SLOTS)
        {
            newest->refs[newest->size + 1] = ref;
            newest->size = newest->size + 1;
        }
        else
        {
            Batch * n = pool.alloc_batch();
            n->size = 1;
            n->refs[0] = reinterpret_cast<uint64_t>(newest);
            n->refs[1] = ref;
            b->refs[Batch::SLOTS] = reinterpret_cast<uint64_t>(n);
        }
    }
    b->total_rows = new_total;
    set_list_word(word, b, new_total);
}

inline uint32_t list_rows(uint64_t word)
{
    if (word & RowRef::ENCODED_INLINE)
        return 1;
    if (word == 0)
        return 0;
    const uint32_t c = static_cast<uint32_t>((word >> kCountShift) & kCountSat);
    if (c != kCountSat)
        return c;
    return static_cast<uint32_t>(reinterpret_cast<Batch *>(word & kPtrMask)->total_rows);
}

struct HashMapCell
{
    uint64_t key = 0;
    uint64_t mapped = 0;
};

/// ClickHouse HashMap: open addressing, linear probing, TwoLevelHashTableGrower (0.5 load, +2 degrees until 15).
struct HashMap
{
    HashMapCell * buf = nullptr;
    size_t buf_bytes = 0;
    uint8_t size_degree = 8;
    size_t mask = 255;
    size_t max_fill = 128;
    size_t m_size = 0;
    bool has_zero = false;
    HashMapCell zero{};

    static constexpr uint8_t kInitial = 8;

    void refresh()
    {
        mask = (1ull << size_degree) - 1;
        max_fill = 1ull << (size_degree - 1);
    }

    void alloc_buf()
    {
        buf_bytes = (1ull << size_degree) * sizeof(HashMapCell);
        buf = static_cast<HashMapCell *>(map_anon(buf_bytes));
    }

    HashMap()
    {
        refresh();
        alloc_buf();
    }

    HashMap(const HashMap &) = delete;
    HashMap & operator=(const HashMap &) = delete;

    HashMap(HashMap && o) noexcept
        : buf(o.buf)
        , buf_bytes(o.buf_bytes)
        , size_degree(o.size_degree)
        , mask(o.mask)
        , max_fill(o.max_fill)
        , m_size(o.m_size)
        , has_zero(o.has_zero)
        , zero(o.zero)
    {
        o.buf = nullptr;
        o.buf_bytes = 0;
    }

    HashMap & operator=(HashMap && o) noexcept
    {
        if (this != &o)
        {
            unmap_anon(buf, buf_bytes);
            buf = o.buf;
            buf_bytes = o.buf_bytes;
            size_degree = o.size_degree;
            mask = o.mask;
            max_fill = o.max_fill;
            m_size = o.m_size;
            has_zero = o.has_zero;
            zero = o.zero;
            o.buf = nullptr;
            o.buf_bytes = 0;
        }
        return *this;
    }

    ~HashMap() { unmap_anon(buf, buf_bytes); }

    void set_for(size_t num_elems)
    {
        if (num_elems <= 1)
            size_degree = kInitial;
        else if (kInitial > static_cast<size_t>(std::log2(static_cast<double>(num_elems - 1))) + 2)
            size_degree = kInitial;
        else
            size_degree = static_cast<uint8_t>(std::log2(static_cast<double>(num_elems - 1))) + 2;
        refresh();
    }

    void reserve(size_t n)
    {
        if (n == 0)
            return;
        set_for(n);
        unmap_anon(buf, buf_bytes);
        alloc_buf();
        prefault_write(buf, buf_bytes);
        m_size = 0;
        has_zero = false;
        zero = {};
    }

    size_t find_cell(uint64_t key, size_t hash) const
    {
        size_t p = hash & mask;
        while (buf[p].key != 0 && buf[p].key != key)
            p = (p + 1) & mask;
        return p;
    }

    void grow(HashMapCell *& it, uint64_t key, size_t hash)
    {
        HashMapCell * old = buf;
        const size_t old_bytes = buf_bytes;
        const size_t old_n = 1ull << size_degree;
        size_degree = static_cast<uint8_t>(size_degree + (size_degree >= 15 ? 1 : 2));
        refresh();
        alloc_buf();
        for (size_t i = 0; i < old_n; ++i)
        {
            if (old[i].key != 0)
            {
                const size_t h = ch_hash64(old[i].key);
                size_t p = h & mask;
                while (buf[p].key != 0)
                    p = (p + 1) & mask;
                buf[p] = old[i];
            }
        }
        unmap_anon(old, old_bytes);
        it = &buf[find_cell(key, hash)];
    }

    void emplace(uint64_t key, size_t hash, HashMapCell *& it, bool & inserted)
    {
        if (key == 0)
        {
            it = &zero;
            inserted = !has_zero;
            if (inserted)
            {
                has_zero = true;
                ++m_size;
            }
            return;
        }
        const size_t p = find_cell(key, hash);
        it = &buf[p];
        if (buf[p].key != 0)
        {
            inserted = false;
            return;
        }
        buf[p].key = key;
        inserted = true;
        ++m_size;
        if (m_size > max_fill)
            grow(it, key, hash);
    }
};

template <typename Latch>
struct ChTable
{
    size_t parts = 0;
    std::vector<HashMap> maps;
    std::vector<Arena> arenas;
    std::unique_ptr<Latch[]> latches;

    void reset()
    {
        maps.clear();
        for (auto & a : arenas)
            a.release();
        arenas.clear();
        latches.reset();
        parts = 0;
    }
};

inline void reserve_ch_unique(std::vector<HashMap> & maps, uint64_t distinct, uint32_t parts_mask)
{
    std::vector<size_t> uniq(maps.size(), 0);
    for (uint64_t k = 1; k <= distinct; ++k)
        uniq[ch_bucket(ch_hash64(k), parts_mask)] += 1;
    for (size_t b = 0; b < maps.size(); ++b)
        maps[b].reserve(uniq[b]);
}

inline void insert_slice(HashMap & map, Arena & arena, const uint64_t * keys, const uint32_t * rows, uint32_t n, uint32_t block_no)
{
    for (uint32_t i = 0; i < n; ++i)
    {
        const uint64_t key = keys[i];
        const size_t hash = ch_hash64(key);
        HashMapCell * it = nullptr;
        bool inserted = false;
        map.emplace(key, hash, it, inserted);
        const uint64_t ref = RowRef::encode(block_no, rows[i]);
        if (inserted)
            it->mapped = ref;
        else
            list_insert(it->mapped, ref, arena);
    }
}

/// Per-block scatter then try-lock insert. Full lock only after a round with no progress.
template <typename Latch>
StageNs build_ch(
    ChTable<Latch> & table, const uint64_t * keys, uint64_t n, size_t threads, size_t parts, size_t block_size, uint64_t distinct, bool pin)
{
    if ((parts & (parts - 1)) != 0)
        throw std::runtime_error("ch parts must be a power of two");
    if (block_size == 0)
        throw std::runtime_error("block size must be > 0");

    const uint32_t parts_mask = static_cast<uint32_t>(parts - 1);
    table.reset();
    table.parts = parts;
    table.maps = std::vector<HashMap>(parts);
    table.arenas = std::vector<Arena>(parts);
    table.latches = std::make_unique<Latch[]>(parts);
    reserve_ch_unique(table.maps, distinct, parts_mask);

    std::barrier bar(static_cast<std::ptrdiff_t>(threads));
    std::vector<uint64_t> scatter_ns(threads, 0);
    std::vector<uint64_t> insert_ns(threads, 0);
    std::vector<uint64_t> latch_cycles(threads, 0);
    uint64_t t_wall0 = 0;
    uint64_t t_wall1 = 0;

    parallel_for(
        threads,
        pin,
        [&](size_t tid)
        {
            std::vector<uint64_t> packed_key(block_size);
            std::vector<uint32_t> packed_row(block_size);
            std::vector<uint32_t> bucket(block_size);
            std::vector<uint32_t> counts(parts);
            std::vector<uint32_t> starts(parts);

            bar.arrive_and_wait();
            if (tid == 0)
                t_wall0 = ns_now();
            bar.arrive_and_wait();

            const uint64_t begin = n * tid / threads;
            const uint64_t end = n * (tid + 1) / threads;
            const uint32_t block_no = static_cast<uint32_t>(tid);

            for (uint64_t off = begin; off < end; off += block_size)
            {
                const uint32_t nblock = static_cast<uint32_t>(std::min<uint64_t>(block_size, end - off));

                const uint64_t s0 = ns_now();
                std::fill(counts.begin(), counts.end(), 0);
                for (uint32_t i = 0; i < nblock; ++i)
                {
                    const uint64_t h = ch_hash64(keys[off + i]);
                    const uint32_t b = ch_bucket(h, parts_mask);
                    bucket[i] = b;
                    counts[b] += 1;
                }
                uint32_t acc = 0;
                for (size_t b = 0; b < parts; ++b)
                {
                    starts[b] = acc;
                    acc += counts[b];
                }
                for (uint32_t i = 0; i < nblock; ++i)
                {
                    const uint32_t b = bucket[i];
                    const uint32_t dst = starts[b]++;
                    packed_key[dst] = keys[off + i];
                    packed_row[dst] = static_cast<uint32_t>(off + i - begin);
                }
                /// starts now at exclusive ends; rebuild exclusive starts
                acc = 0;
                for (size_t b = 0; b < parts; ++b)
                {
                    const uint32_t c = counts[b];
                    starts[b] = acc;
                    acc += c;
                }
                scatter_ns[tid] += ns_now() - s0;

                const uint64_t i0 = ns_now();
                uint32_t remaining = 0;
                for (size_t b = 0; b < parts; ++b)
                    remaining += counts[b] != 0;

                while (remaining != 0)
                {
                    uint32_t acquired = 0;
                    for (size_t b = 0; b < parts; ++b)
                    {
                        if (counts[b] == 0)
                            continue;
                        TimedLatch<Latch> latch{&table.latches[b], &latch_cycles[tid]};
                        if (!latch.try_lock())
                            continue;
                        insert_slice(
                            table.maps[b],
                            table.arenas[b],
                            packed_key.data() + starts[b],
                            packed_row.data() + starts[b],
                            counts[b],
                            block_no);
                        latch.unlock();
                        counts[b] = 0;
                        --remaining;
                        ++acquired;
                    }
                    if (acquired == 0 && remaining != 0)
                    {
                        for (size_t b = 0; b < parts; ++b)
                        {
                            if (counts[b] == 0)
                                continue;
                            TimedLatch<Latch> latch{&table.latches[b], &latch_cycles[tid]};
                            latch.lock();
                            insert_slice(
                                table.maps[b],
                                table.arenas[b],
                                packed_key.data() + starts[b],
                                packed_row.data() + starts[b],
                                counts[b],
                                block_no);
                            latch.unlock();
                            counts[b] = 0;
                            --remaining;
                            break;
                        }
                    }
                }
                insert_ns[tid] += ns_now() - i0;
            }

            bar.arrive_and_wait();
            if (tid == 0)
                t_wall1 = ns_now();
        });

    StageNs stages;
    for (size_t t = 0; t < threads; ++t)
    {
        stages.collect = std::max(stages.collect, scatter_ns[t]);
        stages.insert = std::max(stages.insert, insert_ns[t]);
    }
    uint64_t latch_cyc = 0;
    for (size_t t = 0; t < threads; ++t)
        latch_cyc += latch_cycles[t];
    stages.latch = cycles_to_ns(latch_cyc);
    stages.wall = t_wall1 - t_wall0;
    return stages;
}

template <typename Latch>
void validate_ch(const ChTable<Latch> & table, const uint32_t * hist, uint64_t distinct)
{
    std::vector<uint32_t> got(distinct + 1, 0);
    uint64_t seen = 0;
    for (size_t b = 0; b < table.maps.size(); ++b)
    {
        const HashMap & map = table.maps[b];
        auto add_cell = [&](const HashMapCell & c)
        {
            if (c.key > distinct)
                throw std::runtime_error("ch: key out of range");
            const uint32_t r = list_rows(c.mapped);
            got[c.key] += r;
            seen += r;
        };
        if (map.has_zero && map.zero.mapped)
            add_cell(map.zero);
        const size_t ncells = 1ull << map.size_degree;
        for (size_t i = 0; i < ncells; ++i)
        {
            if (map.buf[i].key != 0)
                add_cell(map.buf[i]);
        }
    }
    uint64_t expect = 0;
    for (uint64_t k = 1; k <= distinct; ++k)
        expect += hist[k];
    if (seen != expect)
        throw std::runtime_error("ch: row count mismatch");
    for (uint64_t k = 1; k <= distinct; ++k)
    {
        if (got[k] != hist[k])
            throw std::runtime_error("ch: histogram mismatch");
    }
}
