#pragma once

#include <bit>
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

/// Compact per-bucket address material for the `BITS_FOR_BUCKET = -1` (runtime bucket count) mode's
/// hot lookup path (buffer base pointer + mask).
struct TwoLevelHashTableBucketDesc
{
    const void * buf = nullptr;
    size_t mask = 0;
};

constexpr int DEFAULT_BITS_FOR_BUCKET = 8;

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
        template <typename BucketAt>
        size_t offset(UInt32 bucket_count, BucketAt && bucket_at, size_t buck, size_t cell_offset)
        {
            if (!computed)
                compute(bucket_count, bucket_at);
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

    private:
        std::vector<size_t> prefix;
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

        /// No-op: fixed storage has no per-bucket address descriptors to keep in sync.
        void refreshDesc(size_t) { }

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
            , descs(num_buckets)
        {
            if (size_hint)
                reserveBuckets(size_hint);

            refreshAllDescs();
        }

        Impl & operator[](size_t bucket) { return buckets[bucket]; }
        const Impl & operator[](size_t bucket) const { return buckets[bucket]; }

        UInt32 bucketCount() const { return num_buckets; }
        UInt32 maxBucket() const { return max_bucket; }
        UInt32 bucketShift() const { return shift; }
        UInt32 getBucketFromHash(size_t hash_value) const { return static_cast<UInt32>((hash_value >> shift) & max_bucket); }

        void refreshDesc(size_t buck)
        {
            const void * const new_buf = buckets[buck].buf;
            const size_t new_mask = buckets[buck].getBufferSizeInCells() - 1;
            if (descs[buck].buf == new_buf && descs[buck].mask == new_mask)
                return;

            descs[buck].buf = new_buf;
            descs[buck].mask = new_mask;
        }

        const TwoLevelHashTableBucketDesc * bucketDescs() const { return descs.data(); }

        void reserve(size_t num_elements)
        {
            reserveBuckets(num_elements);
            refreshAllDescs();
        }

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

        void refreshAllDescs()
        {
            for (UInt32 i = 0; i < num_buckets; ++i)
                refreshDesc(i);
        }

        const UInt32 num_buckets;
        const UInt32 max_bucket;
        const UInt32 shift;
        std::vector<Impl> buckets;
        std::vector<TwoLevelHashTableBucketDesc> descs;
        mutable BucketPrefixSums prefix_sums;
    };

    using Storage = std::conditional_t<isRuntimeStorage(), RuntimeStorage, FixedStorage>;

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
    const TwoLevelHashTableBucketDesc * bucketDescs() const
    requires(isRuntimeStorage())
    {
        return impls.bucketDescs();
    }

    Impl * singleBucket()
    requires(isRuntimeStorage())
    {
        return bucketCount() == 1 ? &impls[0] : nullptr;
    }
    const Impl * singleBucket() const
    requires(isRuntimeStorage())
    {
        return bucketCount() == 1 ? &impls[0] : nullptr;
    }

    void reserve(size_t num_elements) { impls.reserve(num_elements); }

    template <typename K>
    size_t ALWAYS_INLINE bucketRoutingHash(const K & key, size_t cell_hash_value) const
    {
        if constexpr (std::is_void_v<BucketHash>)
            return cell_hash_value;
        else
            return BucketHash{}(key);
    }

protected:
    typename Impl::iterator beginOfNextNonEmptyBucket(size_t & bucket)
    {
        while (bucket != bucketCount() && impls[bucket].empty())
            ++bucket;

        if (bucket != bucketCount())
            return impls[bucket].begin();

        --bucket;
        return impls[impls.maxBucket()].end();
    }

    typename Impl::const_iterator beginOfNextNonEmptyBucket(size_t & bucket) const
    {
        while (bucket != bucketCount() && impls[bucket].empty())
            ++bucket;

        if (bucket != bucketCount())
            return impls[bucket].begin();

        --bucket;
        return impls[impls.maxBucket()].end();
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

    const_iterator end() const { return {this, impls.maxBucket(), impls[impls.maxBucket()].end()}; }
    iterator end() { return {this, impls.maxBucket(), impls[impls.maxBucket()].end()}; }

    const_iterator iteratorAt(size_t bucket) const
    {
        if (bucket >= bucketCount())
            return end();
        auto impl_it = beginOfNextNonEmptyBucket(bucket);
        return { this, bucket, impl_it };
    }

    iterator iteratorAt(size_t bucket)
    {
        if (bucket >= bucketCount())
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

    template <typename KeyHolder>
    void ALWAYS_INLINE prefetch(KeyHolder && key_holder) const
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
    template <typename KeyHolder>
    void ALWAYS_INLINE emplace(KeyHolder && key_holder, LookupResult & it, bool & inserted, size_t hash_value)
    {
        const size_t buck = getBucketFromHash(bucketRoutingHash(keyHolderGetKey(key_holder), hash_value));
        impls[buck].emplace(key_holder, it, inserted, hash_value);
        impls.refreshDesc(buck);
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

    size_t size() const
    {
        size_t res = 0;
        for (UInt32 i = 0; i < bucketCount(); ++i)
            res += impls[i].size();
        return res;
    }

    bool empty() const
    {
        for (UInt32 i = 0; i < bucketCount(); ++i)
            if (!impls[i].empty())
                return false;
        return true;
    }

    size_t getBufferSizeInBytes() const
    {
        size_t res = 0;
        for (UInt32 i = 0; i < bucketCount(); ++i)
            res += impls[i].getBufferSizeInBytes();
        return res;
    }

    size_t getBufferSizeInCells() const
    {
        size_t res = 0;
        for (UInt32 i = 0; i < bucketCount(); ++i)
            res += impls[i].getBufferSizeInCells();
        return res;
    }

    template <typename Func>
    void ALWAYS_INLINE forEachMapped(Func && func)
    {
        for (UInt32 i = 0; i < bucketCount(); ++i)
            impls[i].forEachMapped(func);
    }

    bool ALWAYS_INLINE has(const Key & x) const
    {
        const size_t buck = getBucketFromHash(bucketRoutingHash(x, hash(x)));
        return impls[buck].has(x);
    }

    /// (Re)compute the bucket prefix sums `offsetInternal` relies on. Call this once, after the
    /// last insert that may have changed a bucket's capacity, before switching a per-row loop to
    /// `offsetInternalUnsafe` - that skips the "already computed" check `offsetInternal` pays on
    /// every call.
    void computeBucketPrefix() const { impls.computeBucketPrefix(); }

    /// Lazily computes the prefix sums on first use, then reuses them - it does NOT notice later
    /// bucket growth on its own (there is no internal tracking of buffer changes, by design; see
    /// the class-level comment). A caller that inserts more after taking offsets, and needs
    /// correct offsets afterward, must call `computeBucketPrefix()` again itself.
    size_t offsetInternal(ConstLookupResult ptr) const
    {
        const size_t buck = getBucketFromHash(bucketRoutingHash(ptr->getKey(), ptr->getHash(*this)));
        return impls.offsetInternal(ptr, buck);
    }

    /// Precondition: `computeBucketPrefix()` has been called since the last change to any
    /// bucket's capacity. See `computeBucketPrefix()`.
    size_t offsetInternalUnsafe(ConstLookupResult ptr) const
    {
        const size_t buck = getBucketFromHash(bucketRoutingHash(ptr->getKey(), ptr->getHash(*this)));
        return impls.offsetInternalUnsafe(ptr, buck);
    }
};
