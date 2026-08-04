#pragma once

#include <bit>
#include <Common/CacheLine.h>
#include <Common/HashTable/FixedHashMap.h>
#include <Common/HashTable/TwoLevelHashTable.h>

/** A `FixedHashMap` that answers the bucket protocol, so it can be built and probed by the same
  * bucket-parallel code as the open-addressing JOIN maps.
  *
  * The cells stay exactly as they are: one flat buffer of `2 ^ size_bits`, addressed by `buf[key]`.
  * Only routing is added, and routing never touches addressing, so offsets, buffer size, iteration
  * order and memory are unchanged from a plain `FixedHashMap` at any bucket count. That is why this
  * map needs no serial counterpart the way the open-addressing maps do: its bucket count costs
  * nothing but the routing arithmetic, so it always uses the default.
  */

/** Routes a key to a bucket for a direct-addressed table.
  *
  * The table's own hash is the identity, which is useless for routing: real key distributions are
  * dense at the low end, so any straight slice of the key - high bits especially - piles everything
  * into one bucket. The range maps make that pathological, since a range is placed in the smallest
  * power-of-two table that fits it, so a 300-key range sits in a 65536-cell table and every key has
  * the same high bits. A multiplicative hash spreads them regardless of distribution.
  *
  * The shift is what keeps the spreading from costing false sharing: keys are routed a cache line
  * at a time, so two threads holding different bucket locks never write the same line. Whole lines
  * are scattered across buckets; cells within a line share one.
  */
template <UInt32 block_shift>
struct FixedRangeBucketHash
{
    /// Golden-ratio multiply; the high half is taken because `getBucketFromHash` reads the top bits.
    template <typename Key>
    size_t ALWAYS_INLINE operator()(Key key) const
    {
        const UInt64 block = static_cast<UInt64>(key) >> block_shift;
        return static_cast<size_t>((block * 0x9E3779B97F4A7C15ULL) >> 32);
    }
};

/// How many cells to route as one unit, as a shift: a whole cache line, so a line never spans two
/// buckets. Rounded down to a power of two, and at least one cell for cells larger than a line.
template <typename Cell>
constexpr UInt32 fixedRangeBlockShift()
{
    constexpr size_t cells_per_line = std::bit_floor(std::max<size_t>(1, DB::CH_CACHE_LINE_SIZE / sizeof(Cell)));
    return static_cast<UInt32>(std::countr_zero(cells_per_line));
}

/// `FixedHashMap` is direct-addressed, so `TwoLevelHashTable` keeps one flat buffer for it and uses
/// the bucket only to route. See `IsDirectAddressedTable` and `FixedRangeStorage`.
template <typename Key, typename Mapped, typename Cell, typename Size, typename Allocator, size_t size_bits>
struct IsDirectAddressedTable<FixedHashMap<Key, Mapped, Cell, Size, Allocator, size_bits>> : std::true_type
{
};

template <typename Key, typename Mapped, size_t size_bits = sizeof(Key) * 8>
using PartitionedFixedHashMap = TwoLevelHashTable<
    Key,
    FixedHashMapCell<Key, Mapped>,
    TrivialHash,
    /// Unused: a fixed table never grows. Named only because the signature requires it.
    TwoLevelHashTableGrower<>,
    HashTableAllocator,
    FixedHashMap<
        Key,
        Mapped,
        FixedHashMapCell<Key, Mapped>,
        FixedHashTableStoredSize<FixedHashMapCell<Key, Mapped>>,
        HashTableAllocator,
        size_bits>,
    DEFAULT_BITS_FOR_BUCKET,
    FixedRangeBucketHash<fixedRangeBlockShift<FixedHashMapCell<Key, Mapped>>()>>;
