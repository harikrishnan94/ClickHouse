/// hashprobe-bench/instrumentation/cache_mode.cpp

#include "cache_mode.h"

#include <Interpreters/HashJoin/HashJoin.h>
#include <Interpreters/RowRefs.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#ifdef __linux__
#    include <dirent.h>
#    include <sys/stat.h>
#endif

namespace DB::HashProbeBench
{

// ── LLC size detection ────────────────────────────────────────────────────────

static size_t parseSizeString(const char * s)
{
    // Kernel reports sizes as "32768K", "32M", "1G", or plain bytes.
    char * end = nullptr;
    double val = strtod(s, &end);
    if (!end || end == s)
        return 0;
    switch (*end)
    {
        case 'G':
        case 'g':
            return static_cast<size_t>(val * 1024ULL * 1024 * 1024);
        case 'M':
        case 'm':
            return static_cast<size_t>(val * 1024ULL * 1024);
        case 'K':
        case 'k':
            return static_cast<size_t>(val * 1024ULL);
        default:
            return static_cast<size_t>(val);
    }
}

size_t detectLlcSizeBytes()
{
    constexpr size_t kFallback = 32ULL * 1024 * 1024; // 32 MiB

#ifndef __linux__
    return kFallback;
#else
    // Read all cache levels; keep the largest (== LLC).
    const char * cache_base = "/sys/devices/system/cpu/cpu0/cache";
    DIR * dir = opendir(cache_base);
    if (!dir)
        return kFallback;

    size_t max_size = 0;
    struct dirent * entry;
    while ((entry = readdir(dir)) != nullptr)
    {
        // Only process "index*" directories
        if (strncmp(entry->d_name, "index", 5) != 0)
            continue;

        char path[512];
        snprintf(path, sizeof(path), "%s/%s/size", cache_base, entry->d_name);

        FILE * f = fopen(path, "r");
        if (!f)
            continue;

        char buf[64] = {};
        if (fgets(buf, sizeof(buf), f))
        {
            size_t sz = parseSizeString(buf);
            if (sz > max_size)
                max_size = sz;
        }
        fclose(f);
    }
    closedir(dir);

    return (max_size > 0) ? max_size : kFallback;
#endif
}

// ── LLC warm ──────────────────────────────────────────────────────────────────

void warmLlc(const DB::HashJoin & join)
{
    // Touch the columns stored in the right-table data to bring row data into LLC.
    // The hash map buckets point into this data; reading columns warms the
    // primary storage path used during probing.
    auto data = join.getJoinedData();
    if (!data)
        return;

    volatile uint8_t sink = 0;

    // Walk the ScatteredColumnsList and touch each column's raw storage.
    for (const auto & scattered : data->columns)
    {
        for (const auto & col_ptr : scattered.columns_info.columns)
        {
            if (!col_ptr)
                continue;
            // getRawData() returns a view over the contiguous internal buffer
            // for fixed-width columns (ColumnVector<T>).  For other column
            // types it throws; catch and skip those.
            try
            {
                auto raw = col_ptr->getRawData();
                const auto * ptr = reinterpret_cast<const uint8_t *>(raw.data());
                const size_t sz = raw.size();
                // Read every cache line (64 bytes stride).
                for (size_t off = 0; off < sz; off += 64)
                    sink ^= ptr[off];
            }
            catch (...)
            {
                // Non-contiguous column type — fall back to per-row touch.
                const size_t n = col_ptr->size();
                for (size_t row = 0; row < n; row += 64)
                {
                    try
                    {
                        auto sv = col_ptr->getDataAt(row);
                        sink ^= static_cast<uint8_t>(sv.size());
                    }
                    catch (...)
                    {
                        break;
                    }
                }
            }
        }
    }

    (void)sink; // prevent the reads from being optimised away
}

// ── LLC evict (cold mode) ─────────────────────────────────────────────────────

void evictLlc()
{
    static thread_local std::vector<uint8_t> buf = []()
    {
        const size_t llc = detectLlcSizeBytes();
        return std::vector<uint8_t>(llc * 2);
    }();

    // Two full sequential writes ensure every cache line is evicted.
    for (size_t pass = 0; pass < 2; ++pass)
        for (size_t i = 0; i < buf.size(); i += 64)
            buf[i] = static_cast<uint8_t>(i & 0xFF);
}

} // namespace DB::HashProbeBench
