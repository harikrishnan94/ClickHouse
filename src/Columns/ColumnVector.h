#pragma once

#include <Columns/AdoptionHolder.h>
#include <Columns/ColumnFixedSizeHelper.h>
#include <Columns/IColumn.h>
#include <Columns/IColumnImpl.h>
#include <Common/assert_cast.h>
#include <Core/CompareHelper.h>
#include <Core/Field.h>
#include <Core/TypeId.h>
#include <base/TypeName.h>
#include <base/unaligned.h>

#include <bit>
#include <memory>

#include "config.h"

class SipHash;

namespace DB
{

/** A template for columns that use a simple array to store.
 */
template <typename T>
class ColumnVector final : public COWHelper<IColumnHelper<ColumnVector<T>, ColumnFixedSizeHelper>, ColumnVector<T>>
{
    static_assert(!is_decimal<T>);

private:
    using Self = ColumnVector;
    friend class COWHelper<IColumnHelper<Self, ColumnFixedSizeHelper>, Self>;

    struct less;
    struct less_stable;
    struct greater;
    struct greater_stable;
    struct equals;

public:
    using ValueType = T;
    using Container = PaddedPODArray<ValueType>;

private:
    ColumnVector() = default;
    explicit ColumnVector(const size_t n) : data(n) {}
    ColumnVector(const size_t n, const ValueType x) : data(n, x) {}

    /// COW clone(): always produces a heap-owned copy via the iterator-range PODArray ctor,
    /// regardless of whether `src.data` is in adopted mode (the iterator-range ctor allocates
    /// fresh storage and copies element-wise — see PODArray::PODArray(it, it)). The clone's
    /// `adoption_` deliberately default-constructs to null so the cloned column is fully
    /// mutable; the original column retains its producer hold via its own holder. See
    /// adoption-layer spec §Materialization-on-mutation contract and I3.
    ColumnVector(const ColumnVector & src) : data(src.data.begin(), src.data.end()) {}
    ColumnVector(Container::const_iterator begin, Container::const_iterator end) : data(begin, end) { }

    /// Sugar constructor.
    ColumnVector(std::initializer_list<T> il) : data{il} {}

    /// Adopted-mode ctor: wraps an externally-owned buffer of `adopted_n` `T` values starting
    /// at `adopted_data`. `adoption` must be non-null and its two handles must each be
    /// non-null (the public `createAdopted` factory validates and throws; the chassert here
    /// pins the contract in debug builds). The PODArray adopted-mode ctor stores
    /// `&adopted_data` as an opaque non-null owner marker — its value is never dereferenced
    /// (see PODArrayBase `external_owner` docs).
    ColumnVector(T * adopted_data, size_t adopted_n,
                 std::unique_ptr<AdoptionHolder> adoption)
        : data(adopted_data, adopted_n, &adopted_data)
        , adoption_(std::move(adoption))
    {
        chassert(adoption_ != nullptr);
        chassert(adoption_->retain_token != nullptr);
        chassert(adoption_->charge_handle != nullptr);
    }

public:
    bool isNumeric() const override { return is_arithmetic_v<T>; }

    size_t size() const final
    {
        return data.size();
    }

    /// Construct a ColumnVector wrapping producer-owned memory. The buffer at `adopted_data`
    /// must:
    ///   - hold exactly `adopted_n` elements of type T;
    ///   - satisfy ClickHouse's column-storage alignment for T (typically alignof(T));
    ///   - include at least `PaddedPODArray<T>::pad_right` bytes of safely-readable trailing
    ///     padding.
    /// Ownership of `retain_token` and `charge_handle` transfers into the returned column;
    /// both are released exactly once at adopted-state final drop (the last column reference
    /// or COW alias). Both must be non-null; the factory throws LOGICAL_ERROR otherwise.
    ///
    /// Spec authority: adoption-layer spec §Interfaces & contracts (Adopt entry point), I1,
    /// I3, I4; system spec I5, I10; memory-tracker-integration spec I7.
    ///
    /// Return type spelled out (rather than the inherited `MutablePtr` short alias) because
    /// ColumnVector<T> sees two `MutablePtr` aliases through its base chain (COWHelper's
    /// derived-typed one and ColumnFixedSizeHelper -> IColumn's IColumn-typed one); the
    /// unqualified name resolves ambiguously in this class template's scope.
    static typename COWHelper<IColumnHelper<Self, ColumnFixedSizeHelper>, Self>::MutablePtr
    createAdopted(
        T * adopted_data, size_t adopted_n,
        std::shared_ptr<void> retain_token,
        std::shared_ptr<void> charge_handle);

#if !defined(DEBUG_OR_SANITIZER_BUILD)
    void insertFrom(const IColumn & src, size_t n) override
#else
    void doInsertFrom(const IColumn & src, size_t n) override
#endif
    {
        assertOwnedForMutation("ColumnVector::insertFrom");
        data.push_back(assert_cast<const Self &>(src).getData()[n]);
    }

#if !defined(DEBUG_OR_SANITIZER_BUILD)
    void insertManyFrom(const IColumn & src, size_t position, size_t length) override
#else
    void doInsertManyFrom(const IColumn & src, size_t position, size_t length) override
#endif
    {
        assertOwnedForMutation("ColumnVector::insertManyFrom");
        ValueType v = assert_cast<const Self &>(src).getData()[position];
        data.resize_fill(data.size() + length, v);
    }

    void insertMany(const Field & field, size_t length) override
    {
        assertOwnedForMutation("ColumnVector::insertMany");
        data.resize_fill(data.size() + length, static_cast<T>(field.safeGet<T>()));
    }

    void insertData(const char * pos, size_t) override
    {
        assertOwnedForMutation("ColumnVector::insertData");
        data.emplace_back(unalignedLoad<T>(pos));
    }

    void insertDefault() override
    {
        assertOwnedForMutation("ColumnVector::insertDefault");
        data.push_back(T());
    }

    void insertManyDefaults(size_t length) override
    {
        assertOwnedForMutation("ColumnVector::insertManyDefaults");
        data.resize_fill(data.size() + length, T());
    }

    void popBack(size_t n) override
    {
        assertOwnedForMutation("ColumnVector::popBack");
        if (n > size())
            throwCannotPopBack(n, this->getName(), size());

        data.resize_assume_reserved(data.size() - n);
    }

    void deserializeAndInsertFromArena(ReadBuffer & in, const IColumn::SerializationSettings * settings) override;

    void skipSerializedInArena(ReadBuffer & in) const override;

    void updateHashWithValue(size_t n, SipHash & hash) const override;
    void updateHashWithValueRange(size_t begin, size_t end, SipHash & hash) const override;

    void computeHashInto(size_t row_begin, size_t row_end, UInt32 * hash_out, bool initial) const override;

    void updateHashFast(SipHash & hash) const override;

    size_t byteSize() const override
    {
        return data.size() * sizeof(data[0]);
    }

    size_t byteSizeAt(size_t) const override
    {
        return sizeof(data[0]);
    }

    size_t allocatedBytes() const override
    {
        return data.allocated_bytes();
    }

    void protect() override
    {
        data.protect();
    }

    void insertValue(const T value)
    {
        assertOwnedForMutation("ColumnVector::insertValue");
        data.push_back(value);
    }

    template <class U>
    constexpr int compareAtOther(size_t n, size_t m, const ColumnVector<U> & rhs, int nan_direction_hint) const
    {
        return CompareHelper<T, U>::compare(data[n], rhs.data[m], nan_direction_hint);
    }

    /// This method implemented in header because it could be possibly devirtualized.
#if !defined(DEBUG_OR_SANITIZER_BUILD)
    int compareAt(size_t n, size_t m, const IColumn & rhs_, int nan_direction_hint) const final
#else
    int doCompareAt(size_t n, size_t m, const IColumn & rhs_, int nan_direction_hint) const override
#endif
    {
        return CompareHelper<T>::compare(data[n], assert_cast<const Self &>(rhs_).data[m], nan_direction_hint);
    }

    [[nodiscard]] Int64 compareTrackAt(size_t n, size_t m, const IColumn & rhs, int nan_direction_hint) const final
    {
#if defined(DEBUG_OR_SANITIZER_BUILD)
    #define compareAt doCompareAt
#endif
        Int64 res = compareAt(n, m, rhs, nan_direction_hint);

        if (res < 0)
        {
            ++n;
            while (n < size() && (compareAt(n, m, rhs, nan_direction_hint) < 0))
            {
                --res;
                ++n;
            }
        }
        else if (res > 0)
        {
            ++m;
            while (m < assert_cast<const Self &>(rhs).size() && (compareAt(n, m, rhs, nan_direction_hint) > 0))
            {
                ++res;
                ++m;
            }
        }
        return res;
#if defined(DEBUG_OR_SANITIZER_BUILD)
    #undef compareAt
#endif
    }

#if USE_EMBEDDED_COMPILER

    bool isComparatorCompilable() const override;

    llvm::Value * compileComparator(llvm::IRBuilderBase & /*builder*/, llvm::Value * /*lhs*/, llvm::Value * /*rhs*/, llvm::Value * /*nan_direction_hint*/) const override;

#endif

    void compareColumn(const IColumn & rhs, size_t rhs_row_num,
        PaddedPODArray<UInt64> * row_indexes, PaddedPODArray<Int8> & compare_results,
        int direction, int nan_direction_hint) const override;

    void getPermutation(IColumn::PermutationSortDirection direction, IColumn::PermutationSortStability stability,
                    size_t limit, int nan_direction_hint, IColumn::Permutation & res) const override;

    void updatePermutation(IColumn::PermutationSortDirection direction, IColumn::PermutationSortStability stability,
                    size_t limit, int nan_direction_hint, IColumn::Permutation & res, EqualRanges& equal_ranges) const override;

    size_t estimateCardinalityInPermutedRange(const IColumn::Permutation & permutation, const EqualRange & equal_range) const override;

    void reserve(size_t n) override
    {
        assertOwnedForMutation("ColumnVector::reserve");
        data.reserve_exact(n);
    }

    size_t capacity() const override
    {
        return data.capacity();
    }

    void shrinkToFit() override
    {
        assertOwnedForMutation("ColumnVector::shrinkToFit");
        data.shrink_to_fit();
    }

    const char * getFamilyName() const override { return TypeName<T>.data(); }
    TypeIndex getDataType() const override { return TypeToTypeIndex<T>; }

    MutableColumnPtr cloneResized(size_t size) const override;

    Field operator[](size_t n) const override
    {
        chassert(n < data.size()); /// This assert is more strict than the corresponding assert inside PODArray.
        return data[n];
    }


    void get(size_t n, Field & res) const override
    {
        res = (*this)[n];
    }

    void getValueNameImpl(WriteBufferFromOwnString & name_buf, size_t n, const IColumn::Options &) const override;

    UInt64 get64(size_t n) const override;

    Float64 getFloat64(size_t n) const override;
    Float32 getFloat32(size_t n) const override;

    /// Out of range conversion is permitted.
    UInt64 NO_SANITIZE_UNDEFINED getUInt(size_t n) const override
    {
        if constexpr (is_arithmetic_v<T>)
            return UInt64(data[n]);
        else
            throwColumnConvertNotSupported(TypeName<T>, "UInt");
    }

    /// Out of range conversion is permitted.
    Int64 NO_SANITIZE_UNDEFINED getInt(size_t n) const override
    {
        if constexpr (is_arithmetic_v<T>)
            return Int64(data[n]);
        else
            throwColumnConvertNotSupported(TypeName<T>, "Int");
    }

    bool getBool(size_t n) const override
    {
        if constexpr (is_arithmetic_v<T>)
            return bool(data[n]);
        else
            throwColumnConvertNotSupported(TypeName<T>, "bool");
    }

    void insert(const Field & x) override
    {
        assertOwnedForMutation("ColumnVector::insert");
        data.push_back(static_cast<T>(x.safeGet<T>()));
    }

    bool tryInsert(const DB::Field & x) override;

#if !defined(DEBUG_OR_SANITIZER_BUILD)
    void insertRangeFrom(const IColumn & src, size_t start, size_t length) override;
#else
    void doInsertRangeFrom(const IColumn & src, size_t start, size_t length) override;
#endif

    ColumnPtr filter(const IColumn::Filter & filt, ssize_t result_size_hint) const override;

    void filter(const IColumn::Filter & filt) override;

    void expand(const IColumn::Filter & mask, bool inverted) override;

    ColumnPtr permute(const IColumn::Permutation & perm, size_t limit) const override;

    ColumnPtr index(const IColumn & indexes, size_t limit) const override;

    template <typename Type>
    ColumnPtr indexImpl(const PaddedPODArray<Type> & indexes, size_t limit) const;

    ColumnPtr replicate(const IColumn::Offsets & offsets) const override;

    void getExtremes(Field & min, Field & max, size_t start, size_t end) const override;

    bool canBeInsideNullable() const override { return true; }
    bool isFixedAndContiguous() const override { return true; }
    size_t sizeOfValueIfFixed() const override { return sizeof(T); }
    std::span<char> insertRawUninitialized(size_t count) override;

    std::string_view getRawData() const override
    {
        return {reinterpret_cast<const char*>(data.data()), byteSize()};
    }

    std::string_view getDataAt(size_t n) const override
    {
        return std::string_view(reinterpret_cast<const char *>(&data[n]), sizeof(data[n]));
    }

    bool isDefaultAt(size_t n) const override
    {
        if constexpr (is_floating_point<T>)
        {
            /// For floating-point types, use bit_cast to compare raw bit patterns instead of
            /// arithmetic equality. IEEE 754 defines -0.0 == +0.0, so the arithmetic check
            /// would incorrectly treat -0.0 as the default value, losing the sign on
            /// deserialization. Comparing bits directly distinguishes the two: +0.0 is
            /// all-zero bits, while -0.0 has its sign bit set.
            ///
            /// std::conditional_t selects an unsigned integer type of the same size as T,
            /// satisfying the requirement of std::bit_cast that both types have equal size.
            /// Unsigned integers are chosen because their value equals their bit pattern,
            /// making the comparison to 0 unambiguous.
            using Bits = std::conditional_t<sizeof(T) == 2, UInt16,
                         std::conditional_t<sizeof(T) == 4, UInt32, UInt64>>;
            return std::bit_cast<Bits>(data[n]) == 0;
        }
        else
        {
            return data[n] == T{};
        }
    }

    bool structureEquals(const IColumn & rhs) const override
    {
        return typeid(rhs) == typeid(ColumnVector<T>);
    }

    ColumnPtr createWithOffsets(const IColumn::Offsets & offsets, const ColumnConst & column_with_default_value, size_t total_rows, size_t shift) const override;

    void updateAt(const IColumn & src, size_t dst_pos, size_t src_pos) override;

    ColumnPtr compress(bool force_compression) const override;

    /// Replace elements that match the filter with zeroes. If inverted replaces not matched elements.
    void applyZeroMap(const IColumn::Filter & filt, bool inverted = false);

    /** More efficient methods of manipulation - to manipulate with data directly. */
    /// Guarded against direct mutation of adopted (producer-owned) memory: per VC1 and
    /// adoption-layer spec I3, callers that try to mutate via this non-const accessor are
    /// considered misuse and must instead route through `IColumn::mutate()` to COW-materialize
    /// a heap-owned copy first. The const overload below (the AC1 hot path for sum() batch
    /// aggregation) deliberately stays unguarded.
    Container & getData()
    {
        assertOwnedForMutation("ColumnVector::getData()");
        return data;
    }

    const Container & getData() const
    {
        return data;
    }

    const T & getElement(size_t n) const
    {
        return data[n];
    }

    /// Guarded for the same reason as the non-const getData(): the returned reference
    /// is a writable handle into the (potentially producer-owned) value buffer. Per the
    /// adoption-layer §Materialization-on-mutation contract any path that yields a
    /// writable reference to an adopted column's storage is misuse and must instead go
    /// through IColumn::mutate(). The const overload above is the read hot path.
    T & getElement(size_t n)
    {
        assertOwnedForMutation("ColumnVector::getElement");
        return data[n];
    }

protected:
    Container data;

private:
    /// Heap-allocated adoption holder. Non-null iff this column wraps externally-owned
    /// (producer) memory; null on default-constructed and on COW-cloned columns. The
    /// holder owns the retain_token (pins producer SHM region for the column's lifetime)
    /// and charge_handle (releases adopted bytes back to MemoryTracker on destruction);
    /// both shared_ptrs are released exactly once when the column is destroyed, satisfying
    /// system spec I5 (Retain correctness) and memory-tracker-integration spec I7
    /// (charge/release pairing). See adoption-layer spec §Retain and charge handle
    /// semantics and AdoptionHolder.h.
    ///
    /// Single 8-byte member rather than two ~16-byte std::shared_ptr<void> inline members:
    /// this matters because ColumnVector<T> is the most-used column type in ClickHouse and
    /// every owned (non-adopted) instance pays the layout cost.
    ///
    /// Trailing underscore disambiguates from the constructor's homonymous parameter; the
    /// project-wide convention permits this with NOLINT (see ChargeHandle.h for precedent).
    std::unique_ptr<AdoptionHolder> adoption_; // NOLINT(readability-identifier-naming)

    /// Defense-in-depth guard called at the top of every public direct mutator (any
    /// method that writes through `data` outside the COW `mutate()` entry point). Adopted
    /// columns wrap producer-owned storage; mutation through them would silently corrupt
    /// the producer's memory and violate adoption-layer spec I3 + §Materialization-on-
    /// mutation contract. Out-of-line via the existing `throwAdoptedColumnAccessor` helper
    /// in IColumn.cpp so this header does not need to include Exception.h.
    void assertOwnedForMutation(const char * method) const
    {
        if (adoption_)
            throwAdoptedColumnAccessor(method);
    }
};

template <class TCol>
concept is_col_vector = std::is_same_v<TCol, ColumnVector<typename TCol::ValueType>>;

/// Prevent implicit template instantiation of ColumnVector for common types

extern template class ColumnVector<UInt8>;
extern template class ColumnVector<UInt16>;
extern template class ColumnVector<UInt32>;
extern template class ColumnVector<UInt64>;
extern template class ColumnVector<UInt128>;
extern template class ColumnVector<UInt256>;
extern template class ColumnVector<Int8>;
extern template class ColumnVector<Int16>;
extern template class ColumnVector<Int32>;
extern template class ColumnVector<Int64>;
extern template class ColumnVector<Int128>;
extern template class ColumnVector<Int256>;
extern template class ColumnVector<BFloat16>;
extern template class ColumnVector<Float32>;
extern template class ColumnVector<Float64>;
extern template class ColumnVector<UUID>;
extern template class ColumnVector<IPv4>;
extern template class ColumnVector<IPv6>;

}
