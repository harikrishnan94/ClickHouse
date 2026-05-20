/// hashprobe-bench/partitioned/phj_run.cpp
///
/// Partitioned Hash Join sweep implementation.
/// See phj_run.h for the high-level description.

#include "phj_run.h"

#include <Columns/IColumnPrefetch.h>
#include "../driver/build_driver.h"
#include "../driver/probe_driver.h"
#include "../instrumentation/hw_counters.h"

#include <immintrin.h>
#include <Columns/ColumnNullable.h>
#include <Columns/ColumnsNumber.h>
#include <DataTypes/DataTypesNumber.h>

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cstring>
#include <ctime>
#include <mutex>
#include <numeric>
#include <thread>
#include <vector>
#include <pthread.h>

namespace DB::HashProbeBench
{

// ── Constants ─────────────────────────────────────────────────────────────────
static constexpr int L2_BYTES = 2 * 1024 * 1024; // 2 MiB L2 per core

// ── Splitmix64 for partition hashing ─────────────────────────────────────────
static inline uint64_t phj_mix64(uint64_t x) noexcept
{
    x ^= x >> 30;
    x *= UINT64_C(0xbf58476d1ce4e5b9);
    x ^= x >> 27;
    x *= UINT64_C(0x94d049bb133111eb);
    x ^= x >> 31;
    return x;
}

// ── CPU timer ─────────────────────────────────────────────────────────────────
static uint64_t cpuNs() noexcept
{
    struct timespec ts{};
    clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ts);
    return static_cast<uint64_t>(ts.tv_sec) * 1'000'000'000ULL + static_cast<uint64_t>(ts.tv_nsec);
}
static uint64_t wallNs() noexcept
{
    struct timespec ts{};
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
    return static_cast<uint64_t>(ts.tv_sec) * 1'000'000'000ULL + static_cast<uint64_t>(ts.tv_nsec);
}

// ── Auto-P computation ────────────────────────────────────────────────────────
int computeAutoPPartitions(const ConfigType & cfg, uint64_t build_rows)
{
    const int key_bytes = (cfg.key_width == KeyWidth::W64) ? 8 : 4;
    // key columns + 1 payload column (both sides need data in partition buffers)
    const int row_bytes = static_cast<int>(cfg.key_columns) * key_bytes + 8;
    const int rows_per_part = L2_BYTES / row_bytes;
    const uint64_t raw_P = (build_rows + static_cast<uint64_t>(rows_per_part) - 1) / static_cast<uint64_t>(rows_per_part);

    int P = 64;
    while (static_cast<uint64_t>(P) < raw_P)
        P <<= 1;
    return P;
}

// ── Block partitioner (batched SIMD scatter) ──────────────────────────────────
//
// Replaces IColumn::permute with the same approach as the standalone POC:
//   Phase 1: SIMD hash (8 uint64s / ZMM) → pids[]
//   Phase 2: Histogram → hist[P]
//   Phase 3: Pre-allocate P output MutableColumns (resize to hist[p])
//            Set live write pointers ptrs[p] = first free slot
//   Phase 4: Column-first scatter — one sequential pass per column:
//              for j in 0..rows-1: *ptrs[pids[j]]++ = src_data[j]
//            Avoids IColumn::permute virtual dispatch and index-list construction.
//
// Type dispatch (fast paths → fall back to permute for unknown types):
//   ColumnUInt64          → raw uint64_t* scatter
//   ColumnUInt32          → raw uint32_t* scatter
//   ColumnNullable(T)     → scatter nested T + null-map (uint8_t*)

// ── SIMD vectorized hash for UInt64 key column ────────────────────────────────
__attribute__((target("avx512f,avx512dq"))) static inline __m512i phj_simd_mix64(__m512i x) noexcept
{
    const __m512i M1 = _mm512_set1_epi64(static_cast<int64_t>(UINT64_C(0xbf58476d1ce4e5b9)));
    const __m512i M2 = _mm512_set1_epi64(static_cast<int64_t>(UINT64_C(0x94d049bb133111eb)));
    x = _mm512_xor_epi64(x, _mm512_srli_epi64(x, 30));
    x = _mm512_mullo_epi64(x, M1);
    x = _mm512_xor_epi64(x, _mm512_srli_epi64(x, 27));
    x = _mm512_mullo_epi64(x, M2);
    x = _mm512_xor_epi64(x, _mm512_srli_epi64(x, 31));
    return x;
}

// Hash a UInt64 column into pids[] (first key: initialise; extra keys: XOR in).
__attribute__((target("avx512f,avx512dq"))) static void
hashU64IntoIds(const uint64_t * __restrict__ data, size_t rows, uint64_t mask, uint32_t * __restrict__ pids, bool first_key)
{
    const __m512i vmask = _mm512_set1_epi64(static_cast<int64_t>(mask));
    const __m256i vmask32 = _mm256_set1_epi32(static_cast<int32_t>(mask));
    size_t i = 0;
    for (; i + 8 <= rows; i += 8)
    {
        __m512i k = _mm512_loadu_si512(data + i);
        __m512i h = phj_simd_mix64(k);
        __m256i h32 = _mm512_cvtepi64_epi32(_mm512_and_epi64(h, vmask));
        if (first_key)
        {
            _mm256_storeu_si256(reinterpret_cast<__m256i *>(pids + i), h32);
        }
        else
        {
            __m256i cur = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(pids + i));
            __m256i xrd = _mm256_and_si256(_mm256_xor_si256(cur, h32), vmask32);
            _mm256_storeu_si256(reinterpret_cast<__m256i *>(pids + i), xrd);
        }
    }
    for (; i < rows; ++i)
    {
        const uint32_t h = static_cast<uint32_t>(phj_mix64(data[i]) & mask);
        pids[i] = first_key ? h : static_cast<uint32_t>((pids[i] ^ h) & mask);
    }
}

// Hash a UInt32 column into pids[] (scalar; UInt32 keys are rare).
static void hashU32IntoIds(const uint32_t * __restrict__ data, size_t rows, uint64_t mask, uint32_t * __restrict__ pids, bool first_key)
{
    for (size_t i = 0; i < rows; ++i)
    {
        const uint32_t h = static_cast<uint32_t>(phj_mix64(data[i]) & mask);
        pids[i] = first_key ? h : static_cast<uint32_t>((pids[i] ^ h) & mask);
    }
}

// Column-first scatter of a uint64_t column (live pointer pattern from POC).
static void scatterU64(
    const uint64_t * __restrict__ src,
    uint64_t ** __restrict__ ptrs, // ptrs[p] = next write position in partition p
    const uint32_t * pids,
    size_t rows)
{
    for (size_t j = 0; j < rows; ++j)
        *ptrs[pids[j]]++ = src[j];
}

// Column-first scatter of a uint32_t column.
static void scatterU32(const uint32_t * __restrict__ src, uint32_t ** __restrict__ ptrs, const uint32_t * pids, size_t rows)
{
    for (size_t j = 0; j < rows; ++j)
        *ptrs[pids[j]]++ = src[j];
}

// Column-first scatter of a uint8_t column (null maps).
static void scatterU8(const uint8_t * __restrict__ src, uint8_t ** __restrict__ ptrs, const uint32_t * pids, size_t rows)
{
    for (size_t j = 0; j < rows; ++j)
        *ptrs[pids[j]]++ = src[j];
}

void partitionBlock(const Block & src, int P, const ConfigType & cfg, bool is_build_side, std::vector<Block> & dest)
{
    const size_t rows = src.rows();
    if (rows == 0)
        return;

    const uint64_t mask = static_cast<uint64_t>(P - 1);
    const bool is_w64 = (cfg.key_width == KeyWidth::W64);
    const size_t num_cols = src.columns();

    // ── Phase 1: SIMD hash key columns → pids[] ──────────────────────────────
    std::vector<uint32_t> pids(rows);
    for (uint32_t ki = 0; ki < cfg.key_columns; ++ki)
    {
        const std::string name = is_build_side ? ("b_k" + std::to_string(ki)) : ("k" + std::to_string(ki));
        const IColumn * col = src.getByName(name).column.get();
        const bool is_null = (dynamic_cast<const ColumnNullable *>(col) != nullptr);
        if (is_null)
            col = &static_cast<const ColumnNullable *>(col)->getNestedColumn();

        const bool first = (ki == 0);
        if (!is_null && is_w64)
            hashU64IntoIds(static_cast<const ColumnUInt64 &>(*col).getData().data(), rows, mask, pids.data(), first);
        else if (!is_null && !is_w64)
            hashU32IntoIds(static_cast<const ColumnUInt32 &>(*col).getData().data(), rows, mask, pids.data(), first);
        else
        {
            // Nullable: treat NULLs as key=0 (maps to partition 0).
            for (size_t j = 0; j < rows; ++j)
            {
                uint64_t kv
                    = is_w64 ? static_cast<const ColumnUInt64 &>(*col).getData()[j] : static_cast<const ColumnUInt32 &>(*col).getData()[j];
                uint32_t h = static_cast<uint32_t>(phj_mix64(kv) & mask);
                pids[j] = first ? h : static_cast<uint32_t>((pids[j] ^ h) & mask);
            }
        }
    }

    // ── Phase 2: Histogram ────────────────────────────────────────────────────
    std::vector<uint32_t> hist(static_cast<size_t>(P), 0u);
    for (uint32_t p : pids)
        hist[static_cast<size_t>(p)]++;

    // ── Phase 3: Pre-allocate output MutableColumns and set live write ptrs ──
    // MutableColumns[P][num_cols]
    std::vector<MutableColumns> out(static_cast<size_t>(P));
    for (int p = 0; p < P; ++p)
    {
        if (!hist[static_cast<size_t>(p)])
            continue;
        out[static_cast<size_t>(p)] = src.cloneEmptyColumns();
    }

    // ── Phase 4: Column-first scatter ────────────────────────────────────────
    // One sequential pass per column; live pointer per partition.
    // (pointer arrays below are heap-allocated vectors; no stack limit concern)

    for (size_t ci = 0; ci < num_cols; ++ci)
    {
        const IColumn * src_col_ptr = src.getByPosition(ci).column.get();
        const ColumnNullable * src_null = dynamic_cast<const ColumnNullable *>(src_col_ptr);
        const IColumn * src_nested = src_null ? &src_null->getNestedColumn() : src_col_ptr;
        const ColumnUInt8 * src_nm = src_null ? &src_null->getNullMapColumn() : nullptr;

        const ColumnUInt64 * col_u64 = dynamic_cast<const ColumnUInt64 *>(src_nested);
        const ColumnUInt32 * col_u32 = dynamic_cast<const ColumnUInt32 *>(src_nested);

        if (col_u64)
        {
            // ── Fast path: UInt64 (or Nullable(UInt64)) ────────────────────
            std::vector<uint64_t *> ptrs(static_cast<size_t>(P), nullptr);
            for (int p = 0; p < P; ++p)
            {
                if (!hist[static_cast<size_t>(p)])
                    continue;
                IColumn * dst = out[static_cast<size_t>(p)][ci].get();
                if (src_null)
                {
                    // The mutable output is also ColumnNullable; get its nested.
                    auto & dst_u64 = static_cast<ColumnUInt64 &>(static_cast<ColumnNullable &>(*dst).getNestedColumn());
                    dst_u64.getData().resize(hist[static_cast<size_t>(p)]);
                    ptrs[p] = dst_u64.getData().data();
                }
                else
                {
                    auto & dst_u64 = static_cast<ColumnUInt64 &>(*dst);
                    dst_u64.getData().resize(hist[static_cast<size_t>(p)]);
                    ptrs[p] = dst_u64.getData().data();
                }
            }
            scatterU64(col_u64->getData().data(), ptrs.data(), pids.data(), rows);

            // Also scatter the null map if nullable.
            if (src_null && src_nm)
            {
                std::vector<uint8_t *> nm_ptrs(static_cast<size_t>(P), nullptr);
                for (int p = 0; p < P; ++p)
                {
                    if (!hist[static_cast<size_t>(p)])
                        continue;
                    auto & dst_nm
                        = static_cast<ColumnUInt8 &>(static_cast<ColumnNullable &>(*out[static_cast<size_t>(p)][ci]).getNullMapColumn());
                    dst_nm.getData().resize(hist[static_cast<size_t>(p)]);
                    nm_ptrs[p] = reinterpret_cast<uint8_t *>(dst_nm.getData().data());
                }
                scatterU8(reinterpret_cast<const uint8_t *>(src_nm->getData().data()), nm_ptrs.data(), pids.data(), rows);
            }
        }
        else if (col_u32)
        {
            // ── Fast path: UInt32 ──────────────────────────────────────────
            std::vector<uint32_t *> ptrs(static_cast<size_t>(P), nullptr);
            for (int p = 0; p < P; ++p)
            {
                if (!hist[static_cast<size_t>(p)])
                    continue;
                IColumn * dst = out[static_cast<size_t>(p)][ci].get();
                auto & dst_u32 = src_null ? static_cast<ColumnUInt32 &>(static_cast<ColumnNullable &>(*dst).getNestedColumn())
                                          : static_cast<ColumnUInt32 &>(*dst);
                dst_u32.getData().resize(hist[static_cast<size_t>(p)]);
                ptrs[p] = dst_u32.getData().data();
            }
            scatterU32(col_u32->getData().data(), ptrs.data(), pids.data(), rows);

            if (src_null && src_nm)
            {
                std::vector<uint8_t *> nm_ptrs(static_cast<size_t>(P), nullptr);
                for (int p = 0; p < P; ++p)
                {
                    if (!hist[static_cast<size_t>(p)])
                        continue;
                    auto & dst_nm
                        = static_cast<ColumnUInt8 &>(static_cast<ColumnNullable &>(*out[static_cast<size_t>(p)][ci]).getNullMapColumn());
                    dst_nm.getData().resize(hist[static_cast<size_t>(p)]);
                    nm_ptrs[p] = reinterpret_cast<uint8_t *>(dst_nm.getData().data());
                }
                scatterU8(reinterpret_cast<const uint8_t *>(src_nm->getData().data()), nm_ptrs.data(), pids.data(), rows);
            }
        }
        else
        {
            // ── Fallback: unknown column type — use IColumn::permute ────────
            IColumn::Permutation perm;
            for (size_t j = 0; j < rows; ++j)
                perm.push_back(j); // identity; will be filtered per-p

            // Build per-partition permutations on demand.
            std::vector<IColumn::Permutation> pp(static_cast<size_t>(P));
            for (int p = 0; p < P; ++p)
                if (hist[static_cast<size_t>(p)])
                    pp[static_cast<size_t>(p)].reserve(hist[static_cast<size_t>(p)]);
            for (size_t j = 0; j < rows; ++j)
                pp[pids[j]].push_back(j);

            for (int p = 0; p < P; ++p)
            {
                if (pp[static_cast<size_t>(p)].empty())
                    continue;
                out[static_cast<size_t>(p)][ci]
                    = src_col_ptr->permute(pp[static_cast<size_t>(p)], pp[static_cast<size_t>(p)].size())->assumeMutable();
            }
        }
    }

    // ── Phase 5: Assemble output Blocks ───────────────────────────────────────
    for (int p = 0; p < P; ++p)
    {
        if (!hist[static_cast<size_t>(p)])
            continue;
        dest[static_cast<size_t>(p)] = src.cloneWithColumns(std::move(out[static_cast<size_t>(p)]));
    }
}

// ── Streaming accumulator: scatter all blocks into persistent MutableColumns ───
//
// Avoids the O(P × num_cols × num_blocks) column-object allocations of the
// per-block partitionBlock approach.  Instead: one MutableColumns per partition,
// grown with resize() calls per block (amortised O(1) with vector doubling).
//
// Flow: for each input block call accumulateInto(), then call finalizeAccum()
// to extract the P output Blocks.
struct PartAccum
{
    MutableColumns cols; // one mutable column per schema column; empty = uninitialized
    size_t n_rows = 0; // total rows accumulated
};

// Initialise (or extend) one input block's rows into per-partition accumulators.
// `schema` is any Block with the right column names and types (used for the first
// call to create the MutableColumns).
__attribute__((target("avx512f,avx512dq"))) static void accumulateInto(
    const Block & src,
    int P,
    const ConfigType & cfg,
    bool is_build_side,
    const Block & schema,
    std::vector<PartAccum> & accum) // size == P
{
    const size_t rows = src.rows();
    if (rows == 0)
        return;
    const uint64_t mask = static_cast<uint64_t>(P - 1);
    const bool is_w64 = (cfg.key_width == KeyWidth::W64);
    const size_t ncols = schema.columns();

    // ── Compute pids (SIMD hash) ──────────────────────────────────────────────
    std::vector<uint32_t> pids(rows);
    for (uint32_t ki = 0; ki < cfg.key_columns; ++ki)
    {
        const std::string name = is_build_side ? ("b_k" + std::to_string(ki)) : ("k" + std::to_string(ki));
        const IColumn * col = src.getByName(name).column.get();
        const bool is_null = (dynamic_cast<const ColumnNullable *>(col) != nullptr);
        if (is_null)
            col = &static_cast<const ColumnNullable *>(col)->getNestedColumn();
        const bool first = (ki == 0);
        if (!is_null && is_w64)
            hashU64IntoIds(static_cast<const ColumnUInt64 &>(*col).getData().data(), rows, mask, pids.data(), first);
        else if (!is_null)
            hashU32IntoIds(static_cast<const ColumnUInt32 &>(*col).getData().data(), rows, mask, pids.data(), first);
        else
            for (size_t j = 0; j < rows; ++j)
            {
                uint64_t kv
                    = is_w64 ? static_cast<const ColumnUInt64 &>(*col).getData()[j] : static_cast<const ColumnUInt32 &>(*col).getData()[j];
                uint32_t h = static_cast<uint32_t>(phj_mix64(kv) & mask);
                pids[j] = first ? h : static_cast<uint32_t>((pids[j] ^ h) & mask);
            }
    }

    // ── Histogram ────────────────────────────────────────────────────────────
    std::vector<uint32_t> hist(static_cast<size_t>(P), 0u);
    for (uint32_t p : pids)
        hist[static_cast<size_t>(p)]++;

    // ── Lazy-init accumulators on first use ───────────────────────────────────
    for (int p = 0; p < P; ++p)
    {
        if (!hist[static_cast<size_t>(p)])
            continue;
        PartAccum & a = accum[static_cast<size_t>(p)];
        if (a.cols.empty())
            a.cols = schema.cloneEmptyColumns();
    }

    // ── Column-first scatter — extend existing MutableColumns ─────────────────
    for (size_t ci = 0; ci < ncols; ++ci)
    {
        const IColumn * src_col = src.getByPosition(ci).column.get();
        const ColumnNullable * src_null = dynamic_cast<const ColumnNullable *>(src_col);
        const IColumn * src_nest = src_null ? &src_null->getNestedColumn() : src_col;
        const ColumnUInt8 * src_nm = src_null ? &src_null->getNullMapColumn() : nullptr;
        const ColumnUInt64 * col_u64 = dynamic_cast<const ColumnUInt64 *>(src_nest);
        const ColumnUInt32 * col_u32 = dynamic_cast<const ColumnUInt32 *>(src_nest);

        if (col_u64)
        {
            std::vector<uint64_t *> ptrs(static_cast<size_t>(P), nullptr);
            for (int p = 0; p < P; ++p)
            {
                if (!hist[static_cast<size_t>(p)])
                    continue;
                IColumn * dst = accum[static_cast<size_t>(p)].cols[ci].get();
                const size_t base = accum[static_cast<size_t>(p)].n_rows;
                if (src_null)
                {
                    auto & d = static_cast<ColumnUInt64 &>(static_cast<ColumnNullable &>(*dst).getNestedColumn());
                    d.getData().resize(base + hist[static_cast<size_t>(p)]);
                    ptrs[static_cast<size_t>(p)] = d.getData().data() + base;
                }
                else
                {
                    auto & d = static_cast<ColumnUInt64 &>(*dst);
                    d.getData().resize(base + hist[static_cast<size_t>(p)]);
                    ptrs[static_cast<size_t>(p)] = d.getData().data() + base;
                }
            }
            scatterU64(col_u64->getData().data(), ptrs.data(), pids.data(), rows);
            if (src_null && src_nm)
            {
                std::vector<uint8_t *> nm(static_cast<size_t>(P), nullptr);
                for (int p = 0; p < P; ++p)
                {
                    if (!hist[static_cast<size_t>(p)])
                        continue;
                    const size_t base = accum[static_cast<size_t>(p)].n_rows;
                    auto & d = static_cast<ColumnUInt8 &>(
                        static_cast<ColumnNullable &>(*accum[static_cast<size_t>(p)].cols[ci]).getNullMapColumn());
                    d.getData().resize(base + hist[static_cast<size_t>(p)]);
                    nm[static_cast<size_t>(p)] = reinterpret_cast<uint8_t *>(d.getData().data() + base);
                }
                scatterU8(reinterpret_cast<const uint8_t *>(src_nm->getData().data()), nm.data(), pids.data(), rows);
            }
        }
        else if (col_u32)
        {
            std::vector<uint32_t *> ptrs(static_cast<size_t>(P), nullptr);
            for (int p = 0; p < P; ++p)
            {
                if (!hist[static_cast<size_t>(p)])
                    continue;
                const size_t base = accum[static_cast<size_t>(p)].n_rows;
                IColumn * dst = accum[static_cast<size_t>(p)].cols[ci].get();
                auto & d = src_null ? static_cast<ColumnUInt32 &>(static_cast<ColumnNullable &>(*dst).getNestedColumn())
                                    : static_cast<ColumnUInt32 &>(*dst);
                d.getData().resize(base + hist[static_cast<size_t>(p)]);
                ptrs[static_cast<size_t>(p)] = d.getData().data() + base;
            }
            scatterU32(col_u32->getData().data(), ptrs.data(), pids.data(), rows);
            if (src_null && src_nm)
            {
                std::vector<uint8_t *> nm(static_cast<size_t>(P), nullptr);
                for (int p = 0; p < P; ++p)
                {
                    if (!hist[static_cast<size_t>(p)])
                        continue;
                    const size_t base = accum[static_cast<size_t>(p)].n_rows;
                    auto & d = static_cast<ColumnUInt8 &>(
                        static_cast<ColumnNullable &>(*accum[static_cast<size_t>(p)].cols[ci]).getNullMapColumn());
                    d.getData().resize(base + hist[static_cast<size_t>(p)]);
                    nm[static_cast<size_t>(p)] = reinterpret_cast<uint8_t *>(d.getData().data() + base);
                }
                scatterU8(reinterpret_cast<const uint8_t *>(src_nm->getData().data()), nm.data(), pids.data(), rows);
            }
        }
        else
        {
            // Fallback: row-at-a-time insertFrom for unknown column types.
            for (size_t j = 0; j < rows; ++j)
            {
                const uint32_t p = pids[j];
                if (!accum[static_cast<size_t>(p)].cols[ci])
                    continue; // already guarded by hist check above
                accum[static_cast<size_t>(p)].cols[ci]->insertFrom(*src_col, j);
            }
        }
    }

    // ── Update per-partition row counts ───────────────────────────────────────
    for (int p = 0; p < P; ++p)
        accum[static_cast<size_t>(p)].n_rows += hist[static_cast<size_t>(p)];
}

// Finalise: extract one Block per non-empty partition from the accumulators.
static std::vector<Block> finalizeAccum(const Block & schema, std::vector<PartAccum> & accum, int P)
{
    std::vector<Block> result(static_cast<size_t>(P));
    for (int p = 0; p < P; ++p)
    {
        PartAccum & a = accum[static_cast<size_t>(p)];
        if (a.n_rows == 0)
            continue;
        result[static_cast<size_t>(p)] = schema.cloneWithColumns(std::move(a.cols));
        a.n_rows = 0;
    }
    return result;
}

// ── PHJ cell ──────────────────────────────────────────────────────────────────
PHJPhaseMetrics runPHJCell(
    const ConfigType & cfg,
    const Block & right_sample_block,
    const std::vector<Block> & build_blocks,
    const std::vector<Block> & proto_probe_blocks,
    uint32_t max_threads,
    uint64_t build_rows,
    uint64_t probe_rows) // normalised by caller; not used here
{
    (void)probe_rows;
    const int P = computeAutoPPartitions(cfg, build_rows);
    const int T = static_cast<int>(max_threads);

    PHJPhaseMetrics metrics;
    metrics.P = P;

    // Thread-safe accumulator for CPU ns across T threads.
    struct ThreadAccum
    {
        std::vector<double> cpu_ms_vec;
        std::mutex mu;
        void add(double v)
        {
            std::lock_guard<std::mutex> g(mu);
            cpu_ms_vec.push_back(v);
        }
        double sum()
        {
            double s = 0;
            for (double v : cpu_ms_vec)
                s += v;
            return s;
        }
    };

    // ─── Phase 1: partition build side (streaming accumulator) ──────────────────
    // Each thread accumulates its slice into persistent per-partition MutableColumns
    // (grown across all blocks), then finalises into one Block per partition.
    // Eliminates O(P x ncols x nblocks) = 3.1M column-object allocations.
    // MutableColumns inside PartAccum is non-copyable; use resize() not fill-ctor.
    std::vector<std::vector<PartAccum>> thr_build_accum(static_cast<size_t>(T));
    std::vector<std::vector<Block>> thr_build(static_cast<size_t>(T));
    for (auto & v : thr_build_accum)
        v.resize(static_cast<size_t>(P));
    for (auto & v : thr_build)
        v.resize(static_cast<size_t>(P));

    {
        ThreadAccum acc;
        std::vector<std::thread> ths;
        const size_t n_blk = build_blocks.size();
        for (int t = 0; t < T; ++t)
        {
            const size_t from = static_cast<size_t>(t) * n_blk / static_cast<size_t>(T);
            const size_t to = static_cast<size_t>(t + 1) * n_blk / static_cast<size_t>(T);
            if (from >= to)
                continue;
            ths.emplace_back(
                [&, t, from, to]()
                {
                    uint64_t c0 = cpuNs();
                    auto & a = thr_build_accum[static_cast<size_t>(t)];
                    for (size_t bi = from; bi < to; ++bi)
                        accumulateInto(build_blocks[bi], P, cfg, true, right_sample_block, a);
                    auto finals = finalizeAccum(right_sample_block, a, P);
                    for (int p = 0; p < P; ++p)
                        thr_build[static_cast<size_t>(t)][static_cast<size_t>(p)] = std::move(finals[static_cast<size_t>(p)]);
                    acc.add(static_cast<double>(cpuNs() - c0) / 1e6);
                });
        }
        for (auto & th : ths)
            th.join();
        metrics.part_build_cpu_ms = acc.sum();
    }

    // ─── Phase 2: build P small HashJoins ─────────────────────────────────────
    // T threads, each builds P/T joins by merging T per-thread build sub-blocks.
    std::vector<std::shared_ptr<HashJoin>> part_joins(static_cast<size_t>(P));

    {
        ThreadAccum acc;
        std::vector<std::thread> ths;
        ths.reserve(static_cast<size_t>(T));
        for (int t = 0; t < T; ++t)
        {
            ths.emplace_back(
                [&, t]()
                {
                    const int p_from = t * P / T;
                    const int p_to = (t + 1) * P / T;
                    uint64_t c0 = cpuNs();
                    for (int p = p_from; p < p_to; ++p)
                    {
                        auto join = makePartitionJoin(cfg, right_sample_block);
                        // Each thread produced one Block per partition (not a list).
                        for (int thr = 0; thr < T; ++thr)
                        {
                            const Block & blk = thr_build[static_cast<size_t>(thr)][static_cast<size_t>(p)];
                            if (blk.rows() > 0)
                                join->addBlockToJoin(blk, /*check_limits=*/false);
                        }
                        join->onBuildPhaseFinish();
                        part_joins[static_cast<size_t>(p)] = join;
                    }
                    acc.add(static_cast<double>(cpuNs() - c0) / 1e6);
                });
        }
        for (auto & th : ths)
            th.join();
        metrics.build_ht_cpu_ms = acc.sum();
    }

    // ─── Phase 3: partition probe side (streaming accumulator) ──────────────────
    // Same pattern as Phase 1: one Block per partition per thread at the end.
    std::vector<std::vector<PartAccum>> thr_probe_accum(static_cast<size_t>(T));
    std::vector<std::vector<Block>> thr_probe(static_cast<size_t>(T));
    for (auto & v : thr_probe_accum)
        v.resize(static_cast<size_t>(P));
    for (auto & v : thr_probe)
        v.resize(static_cast<size_t>(P));

    // We need a probe-side schema block for the accumulator initialisation.
    // Use the first non-empty probe block as the schema.
    const Block probe_schema = proto_probe_blocks.empty() ? Block{} : proto_probe_blocks.front();

    {
        ThreadAccum acc;
        std::vector<std::thread> ths;
        const size_t n_blk = proto_probe_blocks.size();
        for (int t = 0; t < T; ++t)
        {
            const size_t from = static_cast<size_t>(t) * n_blk / static_cast<size_t>(T);
            const size_t to = static_cast<size_t>(t + 1) * n_blk / static_cast<size_t>(T);
            if (from >= to)
                continue;
            ths.emplace_back(
                [&, t, from, to]()
                {
                    uint64_t c0 = cpuNs();
                    auto & a = thr_probe_accum[static_cast<size_t>(t)];
                    for (size_t bi = from; bi < to; ++bi)
                        accumulateInto(proto_probe_blocks[bi], P, cfg, false, probe_schema, a);
                    auto finals = finalizeAccum(probe_schema, a, P);
                    for (int p = 0; p < P; ++p)
                        thr_probe[static_cast<size_t>(t)][static_cast<size_t>(p)] = std::move(finals[static_cast<size_t>(p)]);
                    acc.add(static_cast<double>(cpuNs() - c0) / 1e6);
                });
        }
        for (auto & th : ths)
            th.join();
        metrics.part_probe_cpu_ms = acc.sum();
    }

    // ─── Phase 4: probe + generate P small HashJoins ─────────────────────────
    // T threads, each probes P/T partitions using the existing runProbe() path
    // (which fires PROBE_POINT markers → captures probe/generate split).
    std::atomic<uint64_t> total_output_rows{0};
    double probe_cpu_ms_total = 0.0;
    double generate_cpu_ms_total = 0.0;
    std::mutex phase_mu;
    const uint64_t wall_t0 = wallNs();

    {
        std::vector<std::thread> ths;
        ths.reserve(static_cast<size_t>(T));
        for (int t = 0; t < T; ++t)
        {
            ths.emplace_back(
                [&, t]()
                {
                    const int p_from = t * P / T;
                    const int p_to = (t + 1) * P / T;

                    double local_probe_cpu = 0.0, local_gen_cpu = 0.0;
                    uint64_t local_output = 0;

                    for (int p = p_from; p < p_to; ++p)
                    {
                        // Merge T per-thread probe Blocks for partition p.
                        // Each thread now produces exactly ONE Block per partition.
                        std::vector<Block> merged_probe;
                        uint64_t part_probe_rows = 0;
                        for (int thr = 0; thr < T; ++thr)
                        {
                            Block & blk = thr_probe[static_cast<size_t>(thr)][static_cast<size_t>(p)];
                            if (blk.rows() > 0)
                            {
                                part_probe_rows += blk.rows();
                                merged_probe.push_back(std::move(blk));
                            }
                        }

                        if (merged_probe.empty() || part_probe_rows == 0)
                            continue;

                        // Merge all sub-blocks into ONE block per partition.
                        // This amortises ProbeDriver overhead (was ~1526 tiny calls per partition;
                        // now exactly 1 call of ~97.5K rows).
                        Block one_probe;
                        if (merged_probe.size() == 1)
                        {
                            one_probe = std::move(merged_probe[0]);
                        }
                        else
                        {
                            MutableColumns cols = merged_probe[0].cloneEmptyColumns();
                            for (const Block & blk : merged_probe)
                                for (size_t ci = 0; ci < blk.columns(); ++ci)
                                    cols[ci]->insertRangeFrom(*blk.getByPosition(ci).column, 0, blk.rows());
                            one_probe = merged_probe[0].cloneWithColumns(std::move(cols));
                        }

                        // Probe the small HashJoin using ProbeDriver.
                        // la=0: small L2/L3-resident HTs don't benefit from prefetch.
                        DB::generate_phase_prefetch_lookahead = 0;

                        uint64_t part_output_rows = 0;
                        ProbeDriver driver(
                            part_joins[static_cast<size_t>(p)], [&part_output_rows](Block blk) { part_output_rows += blk.rows(); });

                        auto entry = driver.drainBlock(std::move(one_probe), 0);
                        local_probe_cpu += static_cast<double>(entry.phase_probe.cpu_ns) / 1e6;
                        local_gen_cpu += static_cast<double>(entry.phase_generate.cpu_ns) / 1e6;
                        local_output += part_output_rows;
                    }

                    total_output_rows.fetch_add(local_output, std::memory_order_relaxed);
                    {
                        std::lock_guard<std::mutex> g(phase_mu);
                        probe_cpu_ms_total += local_probe_cpu;
                        generate_cpu_ms_total += local_gen_cpu;
                    }
                });
        }
        for (auto & th : ths)
            th.join();
    }

    metrics.total_wall_ms = static_cast<double>(wallNs() - wall_t0) / 1e6;
    metrics.probe_cpu_ms = probe_cpu_ms_total;
    metrics.generate_cpu_ms = generate_cpu_ms_total;
    metrics.output_rows = total_output_rows.load(std::memory_order_relaxed);

    return metrics;
}

} // namespace DB::HashProbeBench
