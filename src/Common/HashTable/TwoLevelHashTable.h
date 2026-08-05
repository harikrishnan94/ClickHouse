#pragma once

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
  * `bits_for_bucket = 0` is one bucket: routing folds to a constant, the single sub-table is stored
  * inline, and every operation compiles to the single-level table's. It is the serial case of the
  * same template rather than a second kind of table.
  */

template <size_t initial_size_degree = 8>
struct TwoLevelHashTableGrower : public HashTableGrowerWithPrecalculation<initial_size_degree>
{
    /// Increase the size of the hash table.
    void increaseSize() { this->increaseSizeDegree(this->sizeDegree() >= 15 ? 1 : 2); }
};

constexpr int DEFAULT_BITS_FOR_BUCKET = 8;

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

    static constexpr bool isFixedRangeStorage() { return IsDirectAddressedTable<ImplTable>::value; }

    static constexpr UInt32 numBuckets() { return static_cast<UInt32>(1) << static_cast<UInt32>(bits_for_bucket); }

private:
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

        /// Lazily computes the prefix sums on first use (`std::call_once`). Prefer
        /// `offsetUnsafe` in hot loops after an explicit `compute()`.
        template <typename BucketAt>
        size_t offset(UInt32 bucket_count, BucketAt && bucket_at, size_t buck, size_t cell_offset)
        {
            std::call_once(compute_once, [&] { compute(bucket_count, bucket_at); });
            return offsetUnsafe(buck, cell_offset);
        }

        size_t offsetUnsafe(size_t buck, size_t cell_offset) const
        {
            chassert(computed);
            return prefix[buck] + cell_offset;
        }

    private:
        std::vector<size_t> prefix;
        std::once_flag compute_once;
        bool computed = false;
    };

    class FixedStorage
    {
    private:
        static constexpr UInt32 MAX_BUCKET = numBuckets() - 1;

    public:
        FixedStorage() = default;

        explicit FixedStorage(size_t size_hint) { reserve(size_hint); }

        Impl & operator[](size_t bucket) { return buckets[bucket]; }
        const Impl & operator[](size_t bucket) const { return buckets[bucket]; }

        static constexpr UInt32 bucketCount() { return MAX_BUCKET + 1; }
        static constexpr UInt32 maxBucket() { return MAX_BUCKET; }
        static constexpr UInt32 bucketShift() { return 32 - bits_for_bucket; }

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
            if constexpr (bucketCount() == 1)
                return static_cast<size_t>(ptr - buckets[0].buf) + 1;
            const auto bucket_at = [this](UInt32 i) -> const Impl & { return buckets[i]; };
            return prefix_sums.offset(bucketCount(), bucket_at, buck, static_cast<size_t>(ptr - buckets[buck].buf) + 1);
        }

        size_t offsetInternalUnsafe(typename Impl::ConstLookupResult ptr, size_t buck) const
        {
            if (ptr->isZero(buckets[buck]))
                return 0;
            if constexpr (bucketCount() == 1)
                return static_cast<size_t>(ptr - buckets[0].buf) + 1;
            return prefix_sums.offsetUnsafe(buck, static_cast<size_t>(ptr - buckets[buck].buf) + 1);
        }

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

    class FixedRangeStorage
    {
    private:
        static constexpr UInt32 MAX_BUCKET = numBuckets() - 1;

    public:
        /// Direct-addressed storage keeps one flat buffer; buckets route locks instead of owning regions.
        FixedRangeStorage()
        {
            /// Do not cache `min`/`max`: parallel inserts race on them.
            flat.disableMinMaxOptimization();
        }

        explicit FixedRangeStorage(size_t /*size_hint*/)
            : FixedRangeStorage()
        {
        }

        Impl & operator[](size_t) { return flat; }
        const Impl & operator[](size_t) const { return flat; }

        static constexpr UInt32 bucketCount() { return MAX_BUCKET + 1; }
        static constexpr UInt32 maxBucket() { return MAX_BUCKET; }
        static constexpr UInt32 bucketShift() { return 32 - bits_for_bucket; }
        static size_t getBucketFromHash(size_t hash_value) { return (hash_value >> bucketShift()) & MAX_BUCKET; }

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

        void reserve(size_t) { }
        void computeBucketPrefix() const { }

        size_t offsetInternal(typename Impl::ConstLookupResult ptr) const { return flat.offsetInternal(ptr); }
        size_t offsetInternalUnsafe(typename Impl::ConstLookupResult ptr) const { return flat.offsetInternal(ptr); }

    private:
        Impl flat;
    };

    using Storage = std::conditional_t<isFixedRangeStorage(), FixedRangeStorage, FixedStorage>;

public:
    using key_type = typename Impl::key_type;
    using mapped_type = typename Impl::mapped_type;
    using value_type = typename Impl::value_type;
    using cell_type = typename Impl::cell_type;
    using LookupResult = typename Impl::LookupResult;
    using ConstLookupResult = typename Impl::ConstLookupResult;

    Storage impls;

    TwoLevelHashTable() = default;

    explicit TwoLevelHashTable(size_t size_hint)
        : impls(size_hint)
    {
    }

    template <typename Source>
    requires(!std::is_arithmetic_v<Source>)
    explicit TwoLevelHashTable(const Source & src)
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

    static size_t ALWAYS_INLINE getBucketFromHash(size_t hash_value) { return Storage::getBucketFromHash(hash_value); }

    static constexpr UInt32 bucketCount() { return Storage::bucketCount(); }
    static constexpr UInt32 bucketShift() { return Storage::bucketShift(); }

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

        /// Only the target bucket is touched; synchronize initialization and call `computeBucketPrefix()` after inserts.
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
    {
        for (UInt32 i = 0; i < bucketCount(); ++i)
            impls[i].write(wb);
    }

    void writeText(DB::WriteBuffer & wb) const
    {
        for (UInt32 i = 0; i < bucketCount(); ++i)
        {
            if (i != 0)
                DB::writeChar(',', wb);
            impls[i].writeText(wb);
        }
    }

    void read(DB::ReadBuffer & rb)
    {
        for (UInt32 i = 0; i < bucketCount(); ++i)
            impls[i].read(rb);
    }

    void readText(DB::ReadBuffer & rb)
    {
        for (UInt32 i = 0; i < bucketCount(); ++i)
        {
            if (i != 0)
                DB::assertChar(',', rb);
            impls[i].readText(rb);
        }
    }

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

    void computeBucketPrefix() const { impls.computeBucketPrefix(); }

    /// Iteration already knows the bucket, so avoid re-hashing; prefix sums must be current.
    size_t offsetInternal(ConstLookupResult ptr) const
    {
        if constexpr (isFixedRangeStorage())
            return impls.offsetInternal(ptr);
        else if constexpr (bucketCount() == 1)
            return impls.offsetInternal(ptr, 0);
        else
            return impls.offsetInternal(ptr, getBucketFromHash(bucketRoutingHash(ptr->getKey(), ptr->getHash(*this))));
    }

    /// Iteration supplies the bucket and avoids re-hashing; call after `computeBucketPrefix()`.
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
        else if constexpr (bucketCount() == 1)
            return impls.offsetInternalUnsafe(ptr, 0);
        else
            return impls.offsetInternalUnsafe(ptr, getBucketFromHash(bucketRoutingHash(ptr->getKey(), ptr->getHash(*this))));
    }
};
