#pragma once

#include <bit>
#include <mutex>
#include <type_traits>
#include <vector>
#include <base/defines.h>
#include <Common/HashTable/HashTable.h>

/** Two-level hash table.
  * Represents 256 (or 1ULL << BITS_FOR_BUCKET) small hash tables (buckets of the first level).
  * To determine which one to use, one of the bytes of the hash function is taken.
  *
  * Usually works a little slower than a simple hash table.
  * However, it has advantages in some cases:
  * - if you need to merge two hash tables together, then you can easily parallelize it by buckets;
  * - delay during resizes is amortized, since the small hash tables will be resized separately;
  * - in theory, resizes are cache-local in a larger range of sizes.
  *
  * Dynamic mode (`bits_for_bucket = -1`) uses runtime bucket count with per-bucket address
  * descriptors instead of fixed arrays.
  */

template <size_t initial_size_degree = 8>
struct TwoLevelHashTableGrower : public HashTableGrowerWithPrecalculation<initial_size_degree>
{
    /// Increase the size of the hash table.
    void increaseSize() { this->increaseSizeDegree(this->sizeDegree() >= 15 ? 1 : 2); }
};

constexpr int DEFAULT_BITS_FOR_BUCKET = 8;

/** Is `Impl` a direct-addressed table - one where the cell for a key is found by indexing a buffer
  * with the key itself, rather than by probing from a hash?
  *
  * Such a table cannot give a bucket its own allocation without multiplying memory by the bucket
  * count, because every bucket would still have to span the whole key space. Instead
  * `TwoLevelHashTable` keeps one flat buffer for it and uses the bucket purely to route (see
  * `FixedRangeStorage`). Specialised next to the table it describes.
  */
template <typename Impl>
struct IsDirectAddressedTable : std::false_type
{
};

template <
    typename Key,
    typename Cell,
    typename Hash,
    typename Grower,
    typename Allocator,
    typename ImplTable = HashTable<Key, Cell, Hash, Grower, Allocator>,
    Int32 bits_for_bucket = DEFAULT_BITS_FOR_BUCKET,
    /// When `void` (default), bucket selection reuses the cell-placement hash.
    /// Otherwise this functor selects the bucket independently from `Hash`.
    typename BucketHash = void>
class TwoLevelHashTable : private boost::noncopyable,
                          protected Hash /// empty base optimization
{
protected:
    friend class const_iterator;
    friend class iterator;

    using HashValue = size_t;
    using Self = TwoLevelHashTable;
public:
    using Impl = ImplTable;

    /// Helper: returns true if dynamic bucket count (runtime mode).
    static constexpr bool isRuntimeStorage() { return bits_for_bucket == -1; }
    /// Helper: returns true if fixed bucket count (compile-time mode).
    static constexpr bool isFixedStorage() { return bits_for_bucket >= 0; }
    /// Helper: returns true for a runtime bucket count over a direct-addressed table, where the
    /// buckets route into one shared cell buffer instead of owning their cells.
    static constexpr bool isFixedRangeStorage() { return isRuntimeStorage() && IsDirectAddressedTable<ImplTable>::value; }

    /// Fixed-bucket mode only. Runtime mode (`bits_for_bucket == -1`) uses instance `bucketCount`.
    static constexpr UInt32 numBuckets()
    requires(isFixedStorage())
    {
        return static_cast<UInt32>(1) << static_cast<UInt32>(bits_for_bucket);
    }

private:
    /// Prefix sums of bucket cell-buffer sizes, shared by both storage kinds' `offsetInternal`.
    /// `HashJoin` indexes its per-offset RIGHT/FULL flags by these offsets, so they must reflect
    /// the CURRENT bucket capacities: `compute()` (re)computes them from scratch, and a caller
    /// that inserts further after taking offsets must call it again before trusting new ones.
    class BucketPrefixSums
    {
    public:
        template <typename BucketAt>
        void compute(UInt32 bucket_count, BucketAt && bucket_at)
        {
            prefix.assign(bucket_count, 0);
            size_t run = 0;
            for (UInt32 i = 0; i < bucket_count; ++i)
            {
                prefix[i] = run;
                run += bucket_at(i).getBufferSizeInCells();
            }
            computed = true;
        }

        /// Safe: computes the prefix sums on first use if `compute()` was not already called.
        /// `offsetInternal` is `const` and reached per row from probe threads that share one table,
        /// so the first-use computation is synchronized. `compute()` called directly (through
        /// `computeBucketPrefix`) leaves the flag unarmed, so the first `offset()` after it computes
        /// again; that is idempotent, and it is what keeps the recompute-after-growth contract
        /// working alongside the flag.
        ///
        /// This is for a caller that has no point at which it can say the inserting is over. A caller
        /// that does - `Unified::HashJoin`, whose `freezeMapsForProbing` runs at build finish and
        /// again after any post-build rewrite of the maps - reaches the prefix sums only through
        /// `offsetInternalUnsafe` / `bucketPrefix`, and so never pays the once-flag check per row.
        template <typename BucketAt>
        size_t offset(UInt32 bucket_count, BucketAt && bucket_at, size_t buck, size_t cell_offset)
        {
            std::call_once(compute_once, [&] { compute(bucket_count, bucket_at); });
            return offsetUnsafe(buck, cell_offset);
        }

        /// Unsafe: skips the "is it computed" check `offset()` pays on every call, and never
        /// recomputes. The caller must have called `compute()` itself first (and again after any
        /// bucket growth) - for hot per-row loops that would otherwise pay that check per row.
        size_t offsetUnsafe(size_t buck, size_t cell_offset) const
        {
            chassert(computed);
            return prefix[buck] + cell_offset;
        }

        /// The prefix sums themselves, for a caller that indexes them per bucket rather than
        /// asking for one offset at a time. Same precondition as `offsetUnsafe`.
        const size_t * data() const
        {
            chassert(computed);
            return prefix.data();
        }

    private:
        std::vector<size_t> prefix;
        std::once_flag compute_once;
        bool computed = false;
    };

    class FixedStorage
    {
    private:
        /// `std::conditional_t` names this class even when dynamic(), so the bucket
        /// count must stay well-formed there; the class is not instantiated in that mode.
        static constexpr UInt32 MAX_BUCKET = numBuckets() - 1;

    public:
        FixedStorage() = default;

        explicit FixedStorage(size_t size_hint) { reserve(size_hint); }

        Impl & operator[](size_t bucket) { return buckets[bucket]; }
        const Impl & operator[](size_t bucket) const { return buckets[bucket]; }

        static constexpr UInt32 bucketCount() { return MAX_BUCKET + 1; }
        static constexpr UInt32 maxBucket() { return MAX_BUCKET; }
        static constexpr UInt32 bucketShift() { return 32 - bits_for_bucket; }

        /// NOTE Bad for hash tables with more than 2^32 cells.
        static size_t getBucketFromHash(size_t hash_value) { return (hash_value >> bucketShift()) & MAX_BUCKET; }

        void reserve(size_t num_elements)
        {
            for (auto & bucket : buckets)
                bucket.reserve(num_elements / bucketCount());
        }

        void computeBucketPrefix() const
        {
            prefix_sums.compute(bucketCount(), [this](UInt32 i) -> const Impl & { return buckets[i]; });
        }

        size_t offsetInternal(typename Impl::ConstLookupResult ptr, size_t buck) const
        {
            if (ptr->isZero(buckets[buck]))
                return 0;
            const auto bucket_at = [this](UInt32 i) -> const Impl & { return buckets[i]; };
            return prefix_sums.offset(bucketCount(), bucket_at, buck, static_cast<size_t>(ptr - buckets[buck].buf) + 1);
        }

        size_t offsetInternalUnsafe(typename Impl::ConstLookupResult ptr, size_t buck) const
        {
            if (ptr->isZero(buckets[buck]))
                return 0;
            return prefix_sums.offsetUnsafe(buck, static_cast<size_t>(ptr - buckets[buck].buf) + 1);
        }

        /// The iteration partition. Here a bucket owns its cells, so it coincides with the bucket
        /// partition; a storage that routes into shared cells reports its own partition instead.
        static constexpr UInt32 iterationBuckets() { return bucketCount(); }
        static constexpr UInt32 lastIterationBucket() { return maxBucket(); }

        size_t size() const
        {
            size_t res = 0;
            for (UInt32 i = 0; i < bucketCount(); ++i)
                res += buckets[i].size();
            return res;
        }

        bool empty() const
        {
            for (UInt32 i = 0; i < bucketCount(); ++i)
                if (!buckets[i].empty())
                    return false;
            return true;
        }

        size_t getBufferSizeInBytes() const
        {
            size_t res = 0;
            for (UInt32 i = 0; i < bucketCount(); ++i)
                res += buckets[i].getBufferSizeInBytes();
            return res;
        }

        size_t getBufferSizeInCells() const
        {
            size_t res = 0;
            for (UInt32 i = 0; i < bucketCount(); ++i)
                res += buckets[i].getBufferSizeInCells();
            return res;
        }

        template <typename Func>
        void ALWAYS_INLINE forEachMapped(Func && func)
        {
            for (UInt32 i = 0; i < bucketCount(); ++i)
                buckets[i].forEachMapped(func);
        }

    private:
        Impl buckets[MAX_BUCKET + 1];
        mutable BucketPrefixSums prefix_sums;
    };

    class RuntimeStorage
    {
    public:
        explicit RuntimeStorage(size_t num_buckets_, size_t size_hint = 0)
            : num_buckets(validateBucketCount(num_buckets_))
            , max_bucket(num_buckets - 1)
            , shift(32 - std::countr_zero(num_buckets))
            , buckets(num_buckets)
        {
            if (size_hint)
                reserveBuckets(size_hint);
        }

        Impl & operator[](size_t bucket) { return buckets[bucket]; }
        const Impl & operator[](size_t bucket) const { return buckets[bucket]; }

        UInt32 bucketCount() const { return num_buckets; }
        UInt32 maxBucket() const { return max_bucket; }
        UInt32 bucketShift() const { return shift; }
        UInt32 getBucketFromHash(size_t hash_value) const { return static_cast<UInt32>((hash_value >> shift) & max_bucket); }

        /// The buckets, contiguously, so that a lookup handle can address bucket `i` as `[i]` from
        /// one base pointer it resolved once.
        const Impl * bucketsData() const { return buckets.data(); }

        /// The one bucket that answers every lookup, when there is one - so a lookup handle can drop
        /// the routing entirely rather than route every key to the same place.
        const Impl * soleBucket() const { return num_buckets == 1 ? buckets.data() : nullptr; }

        const size_t * bucketPrefix() const { return prefix_sums.data(); }

        void reserve(size_t num_elements) { reserveBuckets(num_elements); }

        void computeBucketPrefix() const
        {
            prefix_sums.compute(num_buckets, [this](UInt32 i) -> const Impl & { return buckets[i]; });
        }

        size_t offsetInternal(typename Impl::ConstLookupResult ptr, size_t buck) const
        {
            if (ptr->isZero(buckets[buck]))
                return 0;
            const auto bucket_at = [this](UInt32 i) -> const Impl & { return buckets[i]; };
            return prefix_sums.offset(num_buckets, bucket_at, buck, static_cast<size_t>(ptr - buckets[buck].buf) + 1);
        }

        size_t offsetInternalUnsafe(typename Impl::ConstLookupResult ptr, size_t buck) const
        {
            if (ptr->isZero(buckets[buck]))
                return 0;
            return prefix_sums.offsetUnsafe(buck, static_cast<size_t>(ptr - buckets[buck].buf) + 1);
        }

        /// The iteration partition. Here a bucket owns its cells, so it coincides with the bucket
        /// partition; a storage that routes into shared cells reports its own partition instead.
        UInt32 iterationBuckets() const { return num_buckets; }
        UInt32 lastIterationBucket() const { return max_bucket; }

        size_t size() const
        {
            size_t res = 0;
            for (UInt32 i = 0; i < num_buckets; ++i)
                res += buckets[i].size();
            return res;
        }

        bool empty() const
        {
            for (UInt32 i = 0; i < num_buckets; ++i)
                if (!buckets[i].empty())
                    return false;
            return true;
        }

        size_t getBufferSizeInBytes() const
        {
            size_t res = 0;
            for (UInt32 i = 0; i < num_buckets; ++i)
                res += buckets[i].getBufferSizeInBytes();
            return res;
        }

        size_t getBufferSizeInCells() const
        {
            size_t res = 0;
            for (UInt32 i = 0; i < num_buckets; ++i)
                res += buckets[i].getBufferSizeInCells();
            return res;
        }

        template <typename Func>
        void ALWAYS_INLINE forEachMapped(Func && func)
        {
            for (UInt32 i = 0; i < num_buckets; ++i)
                buckets[i].forEachMapped(func);
        }

    private:
        static UInt32 validateBucketCount(size_t num_buckets)
        {
            chassert(num_buckets >= 1 && std::has_single_bit(num_buckets));
            return static_cast<UInt32>(num_buckets);
        }

        void reserveBuckets(size_t num_elements)
        {
            for (auto & bucket : buckets)
                bucket.reserve(num_elements / num_buckets);
        }

        const UInt32 num_buckets;
        const UInt32 max_bucket;
        const UInt32 shift;
        std::vector<Impl> buckets;
        mutable BucketPrefixSums prefix_sums;
    };

    /** Storage for a direct-addressed `Impl`, where the cell for a key is `buf[key]`.
      *
      * There is exactly one table, covering the whole key space, and every bucket IS that table.
      * That is not a degenerate case - it is the correct one. Addressing does not depend on the
      * bucket, so routing `emplace`/`find` through any bucket lands on the same cell, and giving a
      * bucket its own table would multiply memory by the bucket count for no gain.
      *
      * The bucket therefore names a *route*, not a region: it selects which lock a key belongs
      * under, so bucket-parallel builds still get disjointness (distinct keys are distinct cells).
      * Iteration uses a separate partition - one flat pass - because a bucket's cells are scattered
      * across the buffer and iterating per bucket would visit the whole table once per bucket.
      */
    class FixedRangeStorage
    {
    public:
        explicit FixedRangeStorage(size_t num_buckets_, size_t /*size_hint*/ = 0)
            : num_buckets(validateBucketCount(num_buckets_))
            , max_bucket(num_buckets - 1)
            , shift(32 - std::countr_zero(num_buckets))
        {
            /// `min`/`max` are plain members written by every `emplace`, so bucket-parallel inserts
            /// would race on them. One bucket means a serialized build, where the optimization is
            /// both safe and worth keeping.
            if (num_buckets > 1)
                flat.disableMinMaxOptimization();
        }

        Impl & operator[](size_t) { return flat; }
        const Impl & operator[](size_t) const { return flat; }

        UInt32 bucketCount() const { return num_buckets; }
        UInt32 maxBucket() const { return max_bucket; }
        UInt32 bucketShift() const { return shift; }
        UInt32 getBucketFromHash(size_t hash_value) const { return static_cast<UInt32>((hash_value >> shift) & max_bucket); }

        /// Every bucket is the flat table, so it is also the sole one a lookup handle needs, whatever
        /// the bucket count - routing here decides a lock, and a lookup takes none.
        const Impl * bucketsData() const { return &flat; }
        const Impl * soleBucket() const { return &flat; }
        static constexpr const size_t * bucketPrefix() { return nullptr; }

        /// One flat pass: buckets share the cells, so iterating per bucket would repeat the table.
        static constexpr UInt32 iterationBuckets() { return 1; }
        static constexpr UInt32 lastIterationBucket() { return 0; }

        size_t size() const { return flat.size(); }
        bool empty() const { return flat.empty(); }
        size_t getBufferSizeInBytes() const { return flat.getBufferSizeInBytes(); }
        size_t getBufferSizeInCells() const { return flat.getBufferSizeInCells(); }

        template <typename Func>
        void ALWAYS_INLINE forEachMapped(Func && func)
        {
            flat.forEachMapped(func);
        }

        /// Capacity is fixed at construction and the buffer never moves, so there is nothing to
        /// reserve and no prefix sums to compute: an offset is already global, because there is only
        /// ever one buffer to be an offset into.
        void reserve(size_t) { }
        void computeBucketPrefix() const { }

        size_t offsetInternal(typename Impl::ConstLookupResult ptr) const { return flat.offsetInternal(ptr); }
        size_t offsetInternalUnsafe(typename Impl::ConstLookupResult ptr) const { return flat.offsetInternal(ptr); }

    private:
        static UInt32 validateBucketCount(size_t num_buckets_)
        {
            chassert(num_buckets_ >= 1 && std::has_single_bit(num_buckets_));
            return static_cast<UInt32>(num_buckets_);
        }

        Impl flat;
        const UInt32 num_buckets;
        const UInt32 max_bucket;
        const UInt32 shift;
    };

    using Storage = std::conditional_t<
        isFixedRangeStorage(),
        FixedRangeStorage,
        std::conditional_t<isRuntimeStorage(), RuntimeStorage, FixedStorage>>;

public:
    using key_type = typename Impl::key_type;
    using mapped_type = typename Impl::mapped_type;
    using value_type = typename Impl::value_type;
    using cell_type = typename Impl::cell_type;
    using LookupResult = typename Impl::LookupResult;
    using ConstLookupResult = typename Impl::ConstLookupResult;

    Storage impls;

    TwoLevelHashTable()
    requires(isFixedStorage())
    = default;

    explicit TwoLevelHashTable(size_t size_hint)
    requires(isFixedStorage())
        : impls(size_hint)
    {
    }

    explicit TwoLevelHashTable(size_t num_buckets, size_t size_hint = 0)
    requires(isRuntimeStorage())
        : impls(num_buckets, size_hint)
    {
    }

    /// Copy the data from another (normal) hash table. It should have the same hash function.
    template <typename Source>
    explicit TwoLevelHashTable(const Source & src)
    requires(isFixedStorage())
    {
        typename Source::const_iterator it = src.begin();

        /// It is assumed that the zero key (stored separately) is first in iteration order.
        if (it != src.end() && it.getPtr()->isZero(src))
        {
            insert(it->getValue());
            ++it;
        }

        for (; it != src.end(); ++it)
        {
            const Cell * cell = it.getPtr();
            size_t hash_value = cell->getHash(src);
            size_t buck = getBucketFromHash(hash_value);
            impls[buck].insertUniqueNonZero(cell, hash_value);
        }
    }

    size_t hash(const Key & x) const { return Hash::operator()(x); }

    template <Int32 bits_for_bucket_param = bits_for_bucket>
    static size_t getBucketFromHash(size_t hash_value)
    requires(bits_for_bucket_param >= 0)
    {
        return FixedStorage::getBucketFromHash(hash_value);
    }

    UInt32 ALWAYS_INLINE getBucketFromHash(size_t hash_value) const
    requires(isRuntimeStorage())
    {
        return impls.getBucketFromHash(hash_value);
    }

    UInt32 bucketCount() const { return impls.bucketCount(); }
    UInt32 bucketShift() const { return impls.bucketShift(); }

    void reserve(size_t num_elements) { impls.reserve(num_elements); }

    template <typename K>
    size_t ALWAYS_INLINE bucketRoutingHash(const K & key, size_t cell_hash_value) const
    {
        if constexpr (std::is_void_v<BucketHash>)
            return cell_hash_value;
        else
            return BucketHash{}(key);
    }

    /** A handle for looking keys up in this table, held across a run of lookups.
      *
      * Which bucket a hash selects, and where that bucket's cells live, stops changing once the
      * table stops growing - but it is fixed in memory, not in the code, so reaching a cell means
      * loading the bucket array, then that bucket's table, then its buffer. A handle resolves that
      * routing state once, so a lookup loop pays for it once instead of once per key; where there is
      * nothing to route - a single bucket, or buckets sharing one cell buffer - a lookup through the
      * handle is exactly a single-level table's lookup.
      *
      * `has_sole` is fixed for the handle's lifetime and chosen once per probe block: with a sole
      * bucket, `find` / `findWithOffset` / `prefetch` compile as the flat single-level forms (no
      * per-row branch, no prefix add, no routing to carry). Callers must pick the specialisation
      * that matches the table (`withProber` does that); mixing them is a bug.
      *
      * A caller that needs a matched cell's global offset asks for it from the same call
      * (`findWithOffset`) rather than afterwards. That is not a convenience: which bucket a lookup
      * routed to is not recoverable from the cell pointer - recovering it means re-hashing the key,
      * which for a cell that does not save its hash is the whole hash again - so answering
      * afterwards means the handle holds the routing across the call. Holding it means storing it,
      * and a per-row store to a `size_t` field the compiler cannot separate from the prefix array
      * forces the handle's invariants to be reloaded per row as well. Reporting both at once leaves
      * the handle read-only, so all of it stays in registers.
      *
      * A handle belongs to one thread, and is valid only while the table's buckets keep their
      * buffers - that is, for as long as nothing inserts into the table.
      */
    template <bool has_sole>
    class Prober
    {
    public:
        static constexpr bool is_sole = has_sole;

        using key_type = typename Self::key_type;
        using mapped_type = typename Self::mapped_type;
        using value_type = typename Self::value_type;
        using cell_type = typename Self::cell_type;
        using LookupResult = typename Self::ConstLookupResult;
        using ConstLookupResult = typename Self::ConstLookupResult;

        explicit Prober(const Self & table_)
            : table(&table_)
            , state(table_)
        {
        }

        size_t ALWAYS_INLINE hash(const Key & x) const { return table->hash(x); }

        ConstLookupResult ALWAYS_INLINE find(Key x, size_t hash_value) const
        {
            if constexpr (has_sole)
                return state.sole->find(x, hash_value);
            else
                return routedBucket(x, hash_value)->find(x, hash_value);
        }

        ConstLookupResult ALWAYS_INLINE find(Key x) const { return find(x, hash(x)); }

        /// `find`, also reporting the global cell offset of the cell it returns, as
        /// `TwoLevelHashTable` numbers them; zero when nothing was found. Zero is reserved for the
        /// zero-key cell, in every bucket, exactly as a single-level table reserves it - only one
        /// bucket can hold that cell, since the zero key routes like any other. With a sole bucket
        /// the offset is the flat one: the bucket prefix is identically zero, so there is nothing to
        /// add. See the class comment for why this is one call and not two.
        ///
        /// Precondition, inherited from `bucketPrefix()`: `computeBucketPrefix()` has been called
        /// since the last change to any bucket's capacity.
        ConstLookupResult ALWAYS_INLINE findWithOffset(Key x, size_t hash_value, size_t & offset) const
        {
            if constexpr (has_sole)
            {
                const ConstLookupResult ptr = state.sole->find(x, hash_value);
                offset = ptr ? state.sole->offsetInternal(ptr) : 0;
                return ptr;
            }
            else
            {
                const size_t bucket = routedBucketIndex(x, hash_value);
                const Impl * routed = state.buckets + bucket;
                const ConstLookupResult ptr = routed->find(x, hash_value);
                const size_t offset_in_bucket = ptr ? routed->offsetInternal(ptr) : 0;
                offset = offset_in_bucket ? state.prefix[bucket] + offset_in_bucket : 0;
                return ptr;
            }
        }

        ConstLookupResult ALWAYS_INLINE findWithOffset(Key x, size_t & offset) const
        {
            return findWithOffset(x, hash(x), offset);
        }

        /// Same contract as `TwoLevelHashTable::prefetch`, and declared under the same constraint so
        /// that callers testing for the member only see it when the underlying table can prefetch.
        template <typename KeyHolder>
        void ALWAYS_INLINE prefetch(KeyHolder && key_holder) const
        requires requires(const Impl & impl, size_t key_hash) { impl.prefetchByHash(key_hash); }
        {
            const auto & key = keyHolderGetKey(key_holder);
            const auto key_hash = hash(key);
            if constexpr (has_sole)
                state.sole->prefetchByHash(key_hash);
            else
                routedBucket(key, key_hash)->prefetchByHash(key_hash);
            keyHolderDiscardKey(key_holder);
        }

    private:
        template <typename K>
        size_t ALWAYS_INLINE routedBucketIndex(const K & key, size_t hash_value) const
        requires(!has_sole)
        {
            return (table->bucketRoutingHash(key, hash_value) >> state.shift) & state.max_bucket;
        }

        template <typename K>
        const Impl * ALWAYS_INLINE routedBucket(const K & key, size_t hash_value) const
        requires(!has_sole)
        {
            return state.buckets + routedBucketIndex(key, hash_value);
        }

        struct SoleState
        {
            explicit SoleState(const Self & table_)
                : sole(table_.impls.soleBucket())
            {
                chassert(sole);
            }

            const Impl * sole;
        };

        /// Read-only for the handle's lifetime, which is what lets the per-row loop keep all of it in
        /// registers - see the class comment.
        struct RoutedState
        {
            explicit RoutedState(const Self & table_)
                : buckets(table_.impls.bucketsData())
                , prefix(table_.impls.bucketPrefix())
                , shift(table_.impls.bucketShift())
                , max_bucket(table_.impls.maxBucket())
            {
                chassert(!table_.impls.soleBucket());
            }

            const Impl * buckets;
            /// Where each bucket's cells start in the global numbering. Only `findWithOffset` reads
            /// it, so a probe that does not ask for offsets never touches it.
            const size_t * prefix;
            UInt32 shift;
            UInt32 max_bucket;
        };

        /// Only for the hash functions, which are stateless, so this costs no load.
        const Self * table;
        std::conditional_t<has_sole, SoleState, RoutedState> state;
    };

    /// Precondition: `computeBucketPrefix()` has been called since the last change to any bucket's
    /// capacity, as for `offsetInternalUnsafe()`. `has_sole` must match whether this table has a
    /// sole bucket; prefer `withProber` at call sites that decide once per block.
    template <bool has_sole>
    Prober<has_sole> prober() const
    requires(isRuntimeStorage())
    {
        return Prober<has_sole>(*this);
    }

    /// Pick `Prober<true>` or `Prober<false>` once from the table's layout and invoke `f` with it.
    /// The probe block's row loop then sees a specialised `find` / `findWithOffset` / `prefetch` with
    /// no residual sole branch.
    template <typename F>
    decltype(auto) withProber(F && f) const
    requires(isRuntimeStorage())
    {
        if (impls.soleBucket())
            return std::forward<F>(f)(Prober<true>(*this));
        return std::forward<F>(f)(Prober<false>(*this));
    }

protected:
    /// Iteration walks the storage's iteration partition, which is NOT always the bucket partition:
    /// a storage that routes many buckets into one shared cell buffer reports a single iteration
    /// partition, so every populated cell is still visited exactly once.
    typename Impl::iterator beginOfNextNonEmptyBucket(size_t & bucket)
    {
        while (bucket != impls.iterationBuckets() && impls[bucket].empty())
            ++bucket;

        if (bucket != impls.iterationBuckets())
            return impls[bucket].begin();

        --bucket;
        return impls[impls.lastIterationBucket()].end();
    }

    typename Impl::const_iterator beginOfNextNonEmptyBucket(size_t & bucket) const
    {
        while (bucket != impls.iterationBuckets() && impls[bucket].empty())
            ++bucket;

        if (bucket != impls.iterationBuckets())
            return impls[bucket].begin();

        --bucket;
        return impls[impls.lastIterationBucket()].end();
    }

public:
    class iterator /// NOLINT
    {
        Self * container{};
        size_t bucket{};
        typename Impl::iterator current_it{};

        friend class TwoLevelHashTable;

        iterator(Self * container_, size_t bucket_, typename Impl::iterator current_it_)
            : container(container_), bucket(bucket_), current_it(current_it_) {}

    public:
        iterator() = default;

        bool operator== (const iterator & rhs) const { return bucket == rhs.bucket && current_it == rhs.current_it; }
        bool operator!= (const iterator & rhs) const { return !(*this == rhs); }

        iterator & operator++()
        {
            ++current_it;
            if (current_it == container->impls[bucket].end())
            {
                ++bucket;
                current_it = container->beginOfNextNonEmptyBucket(bucket);
            }
            return *this;
        }

        Cell & operator* () const { return *current_it; }
        Cell * operator->() const { return current_it.getPtr(); }
        Cell * getPtr() const { return current_it.getPtr(); }
        size_t getHash() const { return current_it.getHash(); }
        size_t getBucket() const { return bucket; }
    };

    class const_iterator /// NOLINT
    {
        const Self * container{};
        size_t bucket{};
        typename Impl::const_iterator current_it{};

        friend class TwoLevelHashTable;

        const_iterator(const Self * container_, size_t bucket_, typename Impl::const_iterator current_it_)
            : container(container_), bucket(bucket_), current_it(current_it_)
        {
        }

    public:
        const_iterator() = default;
        const_iterator(const iterator & rhs) : container(rhs.container), bucket(rhs.bucket), current_it(rhs.current_it) {} /// NOLINT

        bool operator== (const const_iterator & rhs) const { return bucket == rhs.bucket && current_it == rhs.current_it; }
        bool operator!= (const const_iterator & rhs) const { return !(*this == rhs); }

        const_iterator & operator++()
        {
            ++current_it;
            if (current_it == container->impls[bucket].end())
            {
                ++bucket;
                current_it = container->beginOfNextNonEmptyBucket(bucket);
            }
            return *this;
        }

        const Cell & operator* () const { return *current_it; }
        const Cell * operator->() const { return current_it.getPtr(); }
        const Cell * getPtr() const { return current_it.getPtr(); }
        size_t getHash() const { return current_it.getHash(); }
        size_t getBucket() const { return bucket; }
    };

    const_iterator begin() const
    {
        size_t buck = 0;
        auto impl_it = beginOfNextNonEmptyBucket(buck);
        return { this, buck, impl_it };
    }

    iterator begin()
    {
        size_t buck = 0;
        auto impl_it = beginOfNextNonEmptyBucket(buck);
        return { this, buck, impl_it };
    }

    const_iterator end() const { return {this, impls.lastIterationBucket(), impls[impls.lastIterationBucket()].end()}; }
    iterator end() { return {this, impls.lastIterationBucket(), impls[impls.lastIterationBucket()].end()}; }

    /// Indexes the iteration partition, not the bucket partition - see `beginOfNextNonEmptyBucket`.
    const_iterator iteratorAt(size_t bucket) const
    {
        if (bucket >= impls.iterationBuckets())
            return end();
        auto impl_it = beginOfNextNonEmptyBucket(bucket);
        return { this, bucket, impl_it };
    }

    iterator iteratorAt(size_t bucket)
    {
        if (bucket >= impls.iterationBuckets())
            return end();
        auto impl_it = beginOfNextNonEmptyBucket(bucket);
        return { this, bucket, impl_it };
    }

    std::pair<LookupResult, bool> ALWAYS_INLINE insert(const value_type & x)
    {
        const auto & key = Cell::getKey(x);
        const size_t hash_value = hash(key);
        std::pair<LookupResult, bool> res;
        emplace(key, res.first, res.second, hash_value);
        if (res.second)
            res.first->setMapped(x);
        return res;
    }

    std::pair<LookupResult, bool> ALWAYS_INLINE insert(const Cell & cell)
    {
        const auto hash_value = cell.getHash(*this);
        std::pair<LookupResult, bool> res;
        emplace(cell.getKey(), res.first, res.second, hash_value);
        if (res.second)
            res.first->setMapped(cell.getValue());
        return res;
    }

    /// Constrained so that callers testing for a `prefetch` member (`join_prefetch_supported`) see
    /// it only when the underlying table can actually prefetch. Without the constraint the
    /// declaration alone would advertise support and the call would fail to compile in the body.
    template <typename KeyHolder>
    void ALWAYS_INLINE prefetch(KeyHolder && key_holder) const
    requires requires(const Impl & impl, size_t key_hash) { impl.prefetchByHash(key_hash); }
    {
        const auto & key = keyHolderGetKey(key_holder);
        const auto key_hash = hash(key);
        const auto buck = getBucketFromHash(bucketRoutingHash(key, key_hash));
        impls[buck].prefetchByHash(key_hash);
        keyHolderDiscardKey(key_holder);
    }

    void ALWAYS_INLINE prefetchByHash(size_t key_hash) const
    {
        /// A hash alone cannot identify the bucket when bucket selection does not derive from it.
        if constexpr (!std::is_void_v<BucketHash>)
            return;
        else
            impls[getBucketFromHash(key_hash)].prefetchByHash(key_hash);
    }

    bool ALWAYS_INLINE isEmptyCell(size_t key_hash) const
    {
        if constexpr (!std::is_void_v<BucketHash>)
            return false;
        else
            return impls[getBucketFromHash(key_hash)].isEmptyCell(key_hash);
    }

    template <typename KeyHolder>
    void ALWAYS_INLINE emplace(KeyHolder && key_holder, LookupResult & it, bool & inserted)
    {
        emplace(key_holder, it, inserted, hash(keyHolderGetKey(key_holder)));
    }

    /// Synchronization follows the underlying hash table contract and is the caller's
    /// responsibility. Any external lock must also cover initialization through `it`.
    ///
    /// Only the target bucket is touched, so callers holding one lock per bucket may run this
    /// concurrently for keys that route to different buckets. Nothing shared between buckets is
    /// written here, by design: a bucket-parallel build must not have to synchronize on anything but
    /// its own bucket. Whatever the table derives from all the buckets at once - the prefix sums - is
    /// computed once with `computeBucketPrefix()` when the inserting is over.
    template <typename KeyHolder>
    void ALWAYS_INLINE emplace(KeyHolder && key_holder, LookupResult & it, bool & inserted, size_t hash_value)
    {
        const size_t buck = getBucketFromHash(bucketRoutingHash(keyHolderGetKey(key_holder), hash_value));
        impls[buck].emplace(key_holder, it, inserted, hash_value);
    }

    LookupResult ALWAYS_INLINE find(Key x, size_t hash_value)
    {
        const size_t buck = getBucketFromHash(bucketRoutingHash(x, hash_value));
        return impls[buck].find(x, hash_value);
    }

    ConstLookupResult ALWAYS_INLINE find(Key x, size_t hash_value) const
    {
        return const_cast<std::decay_t<decltype(*this)> *>(this)->find(x, hash_value);
    }

    LookupResult ALWAYS_INLINE find(Key x) { return find(x, hash(x)); }
    ConstLookupResult ALWAYS_INLINE find(Key x) const { return find(x, hash(x)); }

    void write(DB::WriteBuffer & wb) const
    requires(isFixedStorage())
    {
        for (UInt32 i = 0; i < bucketCount(); ++i)
            impls[i].write(wb);
    }

    void writeText(DB::WriteBuffer & wb) const
    requires(isFixedStorage())
    {
        for (UInt32 i = 0; i < bucketCount(); ++i)
        {
            if (i != 0)
                DB::writeChar(',', wb);
            impls[i].writeText(wb);
        }
    }

    void read(DB::ReadBuffer & rb)
    requires(isFixedStorage())
    {
        for (UInt32 i = 0; i < bucketCount(); ++i)
            impls[i].read(rb);
    }

    void readText(DB::ReadBuffer & rb)
    requires(isFixedStorage())
    {
        for (UInt32 i = 0; i < bucketCount(); ++i)
        {
            if (i != 0)
                DB::assertChar(',', rb);
            impls[i].readText(rb);
        }
    }

    /// Aggregate queries answer from the storage rather than looping buckets here: a storage whose
    /// buckets share one cell buffer would otherwise be counted once per bucket.
    size_t size() const { return impls.size(); }
    bool empty() const { return impls.empty(); }
    size_t getBufferSizeInBytes() const { return impls.getBufferSizeInBytes(); }
    size_t getBufferSizeInCells() const { return impls.getBufferSizeInCells(); }

    template <typename Func>
    void ALWAYS_INLINE forEachMapped(Func && func)
    {
        impls.forEachMapped(func);
    }

    bool ALWAYS_INLINE has(const Key & x) const
    {
        const size_t buck = getBucketFromHash(bucketRoutingHash(x, hash(x)));
        return impls[buck].has(x);
    }

    /// (Re)compute the bucket prefix sums `offsetInternal` relies on. Call this once, after the
    /// last insert that may have changed a bucket's capacity, and before anything that reads the
    /// prefix sums without checking whether they are there: `offsetInternalUnsafe`, which skips the
    /// "already computed" check `offsetInternal` pays on every call, and `prober` / `withProber`.
    void computeBucketPrefix() const { impls.computeBucketPrefix(); }

    /// Lazily computes the prefix sums on first use, then reuses them - it does NOT notice later
    /// bucket growth on its own (there is no internal tracking of buffer changes, by design; see
    /// the class-level comment). A caller that inserts more after taking offsets, and needs
    /// correct offsets afterward, must call `computeBucketPrefix()` again itself.
    /// The bucket is recovered from the cell, which only works when a bucket owns its cells: it
    /// costs a re-hash, and a direct-addressed cell has neither a key nor a hash to re-hash. So a
    /// storage that routes into one shared buffer answers from the pointer alone, and this branch
    /// is discarded before `ptr->getHash(*this)` can be instantiated for a cell that has no hash.
    size_t offsetInternal(ConstLookupResult ptr) const
    {
        if constexpr (isFixedRangeStorage())
            return impls.offsetInternal(ptr);
        else
            return impls.offsetInternal(ptr, getBucketFromHash(bucketRoutingHash(ptr->getKey(), ptr->getHash(*this))));
    }

    /// Offset for a cell reached by ITERATION, which already knows which bucket it is in.
    ///
    /// `offsetInternal(ptr)` above has to recover the bucket from the cell, and that costs a re-hash
    /// of the key on every call - dead work for an iterator, which was handed the bucket to begin
    /// with. It also pays the "are the prefix sums computed yet" check per call. A full-table scan
    /// (the RIGHT/FULL non-joined pass) does both per populated cell, so both are worth skipping.
    ///
    /// The bucket passed here is an ITERATION bucket. For the storages whose buckets own their
    /// cells that is the same thing as the bucket partition (`iterationBuckets() == bucketCount()`),
    /// and for the direct-addressed storage there is one iteration partition and the offset comes
    /// from the pointer alone, so the argument is unused.
    ///
    /// Precondition, same as `offsetInternalUnsafe`: `computeBucketPrefix()` has been called since
    /// the last change to any bucket's capacity.
    size_t ALWAYS_INLINE offsetInternalAtBucket(ConstLookupResult ptr, size_t iteration_bucket) const
    {
        if constexpr (isFixedRangeStorage())
            return impls.offsetInternalUnsafe(ptr);
        else
            return impls.offsetInternalUnsafe(ptr, iteration_bucket);
    }

    /// Precondition: `computeBucketPrefix()` has been called since the last change to any
    /// bucket's capacity. See `computeBucketPrefix()`.
    size_t offsetInternalUnsafe(ConstLookupResult ptr) const
    {
        if constexpr (isFixedRangeStorage())
            return impls.offsetInternalUnsafe(ptr);
        else
            return impls.offsetInternalUnsafe(ptr, getBucketFromHash(bucketRoutingHash(ptr->getKey(), ptr->getHash(*this))));
    }
};
