#pragma once

#include <Columns/IColumn.h>
#include <base/defines.h>
#include <base/types.h>

#include <cstring>

#if defined(__aarch64__) && defined(__ARM_FEATURE_CRC32)
#include <arm_acle.h>
#endif

namespace DB::JoinSlotRouting
{

/** Slot routing of `ConcurrentHashJoin`: one 32-bit route word per row over the prepared join
  * key columns, deliberately independent of the CRC32C-family hashes the slot maps bucket by,
  * so slot selection stays decorrelated from in-table cell placement. A plan of `bits` slots
  * routes a row to slot `word >> (32 - bits)` (MSB-first).
  *
  * The build scatter and the probe dispatch consume ONE shared fold implementation, and the
  * word is a build/probe contract: the fold chain must not change without changing both sides
  * in lockstep, or probe rows visit slots their keys were never inserted into.
  *
  * `key_columns` are the prepared key columns after null-map extraction (nested,
  * non-nullable); a live `ColumnLowCardinality` (kept for the dictionary-aware map types) is
  * routed by its value bytes, so it produces the same words as the plain column of the same
  * values on the other join side.
  */

/// The per-key fold primitives. CRC-32 (ISO polynomial) on AArch64 and a golden-ratio
/// multiply-shift elsewhere - both distinct from the CRC-32C the hash tables bucket by, which
/// is what keeps the route bits independent of `grower.place`.
ALWAYS_INLINE inline UInt32 routeWord(UInt64 key)
{
#if defined(__aarch64__) && defined(__ARM_FEATURE_CRC32)
    return __crc32d(-1U, key);
#else
    return static_cast<UInt32>((key * 0x9E3779B97F4A7C15ULL) >> 32);
#endif
}

ALWAYS_INLINE inline UInt64 mixStep(UInt64 h, UInt64 x)
{
    return (h ^ x) * 0x9E3779B97F4A7C15ULL;
}

ALWAYS_INLINE inline UInt32 finalizeRoute(UInt64 h)
{
    return static_cast<UInt32>(h >> 32);
}

/// Fold `w` bytes at `p` into the accumulator, 8 bytes at a time with a zero-padded tail.
/// The tail dispatches on its size to constant-size copies: a runtime-size copy lowers to a
/// per-row libc `memcpy` call on the runtime-width paths, while constant sizes lower to plain
/// loads (and constant-width callers fold the switch away entirely).
ALWAYS_INLINE inline UInt64 foldBytes(UInt64 h, const char * p, size_t w)
{
    size_t i = 0;
    for (; i + 8 <= w; i += 8)
    {
        UInt64 x = 0;
        memcpy(&x, p + i, sizeof(x));
        h = mixStep(h, x);
    }
    if (i < w)
    {
        UInt64 x = 0;
        switch (w - i) // NOLINT(bugprone-switch-missing-default-case): the tail size is provably in [1, 7]
        {
            case 1: memcpy(&x, p + i, 1); break;
            case 2: memcpy(&x, p + i, 2); break;
            case 3: memcpy(&x, p + i, 3); break;
            case 4: memcpy(&x, p + i, 4); break;
            case 5: memcpy(&x, p + i, 5); break;
            case 6: memcpy(&x, p + i, 6); break;
            case 7: memcpy(&x, p + i, 7); break;
        }
        h = mixStep(h, x);
    }
    return h;
}

/// One route word per row into `words`. The contract-pinning entry point: the slot sink below
/// instantiates the same fold, and tests compare its output against these words.
void computeJoinRouteWords(const ColumnRawPtrs & key_columns, size_t rows, UInt32 * words);

/// The slot sink shared by build scatter and probe dispatch: slot id `word >> (32 - bits)`
/// stored per row directly, no 32-bit word transient. `bits` in [1, 8] -
/// `ConcurrentHashJoin` slot counts are powers of two <= 256.
void computeJoinSlotIds(const ColumnRawPtrs & key_columns, size_t rows, size_t bits, UInt8 * slot_ids);

}
