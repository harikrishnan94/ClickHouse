#pragma once

#include <base/TypeName.h>
#include <Core/Field.h>
#include <Core/TypeId.h>
#include <Common/typeid_cast.h>
#include <Columns/AdoptionHolder.h>
#include <Columns/ColumnFixedSizeHelper.h>
#include <Columns/IColumn.h>

#include <memory>

namespace DB
{

/// A ColumnVector for Decimals
template <is_decimal T>
class ColumnDecimal final : public COWHelper<IColumnHelper<ColumnDecimal<T>, ColumnFixedSizeHelper>, ColumnDecimal<T>>
{
private:
    using Self = ColumnDecimal;
    friend class COWHelper<IColumnHelper<Self, ColumnFixedSizeHelper>, Self>;

public:
    using ValueType = T;
    using NativeT = typename T::NativeType;
    using Container = PaddedPODArray<T>;

private:
    ColumnDecimal(const size_t n, UInt32 scale_)
    :   data(n),
        scale(scale_)
    {}

    /// COW clone(): the iterator-range PODArray ctor always allocates fresh heap
    /// storage and copies element-wise, regardless of whether `src.data` is in
    /// adopted mode. The clone's `adoption_` deliberately default-constructs to
    /// null so the cloned column is fully mutable; the original retains its
    /// producer hold via its own holder. Mirrors `ColumnVector`'s COW clone (see
    /// adoption-layer spec §Materialization-on-mutation contract and I3).
    ColumnDecimal(const ColumnDecimal & src)
    :   data(src.data.begin(), src.data.end()),
        scale(src.scale)
    {}

    /// Adopted-mode ctor: wraps an externally-owned (producer SHM) buffer of
    /// `adopted_n` `T` values at `adopted_data`, carrying the decimal `scale_`
    /// derived from the SQL/handshake DataType (the scale is not on the wire).
    /// `adoption` must be non-null with both handles non-null; the public
    /// `createAdopted` factory validates and throws, the chasserts pin the
    /// contract in debug builds. The PODArray adopted-mode ctor stores
    /// `&adopted_data` as an opaque non-null owner marker (never dereferenced).
    ColumnDecimal(T * adopted_data, size_t adopted_n, UInt32 scale_,
                  std::unique_ptr<AdoptionHolder> adoption)
    :   data(adopted_data, adopted_n, &adopted_data),
        scale(scale_),
        adoption_(std::move(adoption))
    {
        chassert(adoption_ != nullptr);
        chassert(adoption_->retain_token != nullptr);
        chassert(adoption_->charge_handle != nullptr);
    }

public:
    /// Construct a ColumnDecimal wrapping producer-owned memory. The buffer at
    /// `adopted_data` must hold exactly `adopted_n` elements of type `T`, satisfy
    /// `T`'s natural alignment (`alignof(T)`; 16 for `Decimal128`), and include at
    /// least `PaddedPODArray<T>::pad_right` bytes of safely-readable trailing
    /// padding. `scale` is the decimal scale from the (cross-validated) DataType.
    /// Ownership of `retain_token` and `charge_handle` transfers into the returned
    /// column and both release exactly once at adopted-state final drop. Both must
    /// be non-null; the factory throws BAD_ARGUMENTS otherwise. Only the
    /// adopted decimal set is supported (`Decimal32/64/128`, `DateTime64`); any
    /// other `T` (e.g. `Decimal256`, `Time64`) throws BAD_ARGUMENTS so an
    /// unintended instantiation fails loudly rather than wrapping unvalidated
    /// memory. Spec authority: adoption-layer spec §Interfaces & contracts, I1,
    /// I3, I4; system spec I5, I10; memory-tracker-integration spec I7.
    static typename COWHelper<IColumnHelper<Self, ColumnFixedSizeHelper>, Self>::MutablePtr
    createAdopted(
        T * adopted_data, size_t adopted_n, UInt32 scale,
        std::shared_ptr<void> retain_token,
        std::shared_ptr<void> charge_handle);

    const char * getFamilyName() const final;
    TypeIndex getDataType() const final;

    bool isNumeric() const final { return false; }
    bool canBeInsideNullable() const final { return true; }
    bool isFixedAndContiguous() const final { return true; }
    size_t sizeOfValueIfFixed() const final { return sizeof(T); }
    std::span<char> insertRawUninitialized(size_t count) final;

    size_t size() const final { return data.size(); }
    size_t byteSize() const final { return data.size() * sizeof(data[0]); }
    size_t byteSizeAt(size_t) const final { return sizeof(data[0]); }
    size_t allocatedBytes() const final { return data.allocated_bytes(); }
    void protect() final { data.protect(); }
    void reserve(size_t n) final { assertOwnedForMutation("ColumnDecimal::reserve"); data.reserve_exact(n); }
    size_t capacity() const final { return data.capacity(); }
    void shrinkToFit() final { assertOwnedForMutation("ColumnDecimal::shrinkToFit"); data.shrink_to_fit(); }

#if !defined(DEBUG_OR_SANITIZER_BUILD)
    void insertFrom(const IColumn & src, size_t n) final { assertOwnedForMutation("ColumnDecimal::insertFrom"); data.push_back(static_cast<const Self &>(src).getData()[n]); }
#else
    void doInsertFrom(const IColumn & src, size_t n) final { assertOwnedForMutation("ColumnDecimal::insertFrom"); data.push_back(static_cast<const Self &>(src).getData()[n]); }
#endif

#if !defined(DEBUG_OR_SANITIZER_BUILD)
    void insertManyFrom(const IColumn & src, size_t position, size_t length) final
#else
    void doInsertManyFrom(const IColumn & src, size_t position, size_t length) final
#endif
    {
        assertOwnedForMutation("ColumnDecimal::insertManyFrom");
        ValueType v = assert_cast<const Self &>(src).getData()[position];
        data.resize_fill(data.size() + length, v);
    }

    void insertData(const char * src, size_t /*length*/) final;
    void insertDefault() final { assertOwnedForMutation("ColumnDecimal::insertDefault"); data.push_back(T()); }
    void insertManyDefaults(size_t length) final { assertOwnedForMutation("ColumnDecimal::insertManyDefaults"); data.resize_fill(data.size() + length); }
    void insert(const Field & x) final { assertOwnedForMutation("ColumnDecimal::insert"); data.push_back(x.safeGet<T>()); }
    bool tryInsert(const Field & x) final;
#if !defined(DEBUG_OR_SANITIZER_BUILD)
    void insertRangeFrom(const IColumn & src, size_t start, size_t length) final;
#else
    void doInsertRangeFrom(const IColumn & src, size_t start, size_t length) final;
#endif

    void popBack(size_t n) final
    {
        assertOwnedForMutation("ColumnDecimal::popBack");
        if (n > size())
            throwCannotPopBack(n, this->getName(), size());

        data.resize_assume_reserved(data.size() - n);
    }

    std::string_view getRawData() const final
    {
        return {reinterpret_cast<const char*>(data.data()), byteSize()};
    }

    std::string_view getDataAt(size_t n) const final
    {
        return {reinterpret_cast<const char *>(&data[n]), sizeof(data[n])};
    }

    Float64 getFloat64(size_t n) const final;

    void deserializeAndInsertFromArena(ReadBuffer & in, const IColumn::SerializationSettings * settings) final;
    void skipSerializedInArena(ReadBuffer & in) const final;
    void updateHashWithValue(size_t n, SipHash & hash) const final;
    void updateHashWithValueRange(size_t begin, size_t end, SipHash & hash) const final;
    void computeHashInto(size_t row_begin, size_t row_end, UInt32 * hash_out, bool initial) const final;
    void updateHashFast(SipHash & hash) const final;
#if !defined(DEBUG_OR_SANITIZER_BUILD)
    int compareAt(size_t n, size_t m, const IColumn & rhs_, int nan_direction_hint) const final;
#else
    int doCompareAt(size_t n, size_t m, const IColumn & rhs_, int nan_direction_hint) const final;
#endif
    [[nodiscard]] Int64 compareTrackAt(size_t n, size_t m, const IColumn & rhs, int nan_direction_hint) const final;
    void getPermutation(IColumn::PermutationSortDirection direction, IColumn::PermutationSortStability stability,
                        size_t limit, int nan_direction_hint, IColumn::Permutation & res) const final;
    void updatePermutation(IColumn::PermutationSortDirection direction, IColumn::PermutationSortStability stability,
                        size_t limit, int, IColumn::Permutation & res, EqualRanges& equal_ranges) const final;
    size_t estimateCardinalityInPermutedRange(const IColumn::Permutation & permutation, const EqualRange & equal_range) const final;


    MutableColumnPtr cloneResized(size_t size) const final;

    Field operator[](size_t n) const final { return DecimalField<ValueType>(data[n], scale); }
    void get(size_t n, Field & res) const final { res = (*this)[n]; }
    void getValueNameImpl(WriteBufferFromOwnString & name_buf, size_t n, const IColumn::Options &options) const final;
    bool getBool(size_t n) const final { return bool(data[n].value); }
    Int64 getInt(size_t n) const final { return Int64(data[n].value); }
    UInt64 get64(size_t n) const final;
    bool isDefaultAt(size_t n) const final { return data[n].value == 0; }

    ColumnPtr filter(const IColumn::Filter & filt, ssize_t result_size_hint) const final;
    void filter(const IColumn::Filter & filt) final;
    void expand(const IColumn::Filter & mask, bool inverted) final;

    ColumnPtr permute(const IColumn::Permutation & perm, size_t limit) const final;
    ColumnPtr index(const IColumn & indexes, size_t limit) const final;

    template <typename Type>
    ColumnPtr indexImpl(const PaddedPODArray<Type> & indexes, size_t limit) const;

    ColumnPtr replicate(const IColumn::Offsets & offsets) const final;
    void getExtremes(Field & min, Field & max, size_t start, size_t end) const final;

    bool structureEquals(const IColumn & rhs) const final
    {
        if (auto rhs_concrete = typeid_cast<const ColumnDecimal<T> *>(&rhs))
            return scale == rhs_concrete->scale;
        return false;
    }

    void updateAt(const IColumn & src, size_t dst_pos, size_t src_pos) final;

    ColumnPtr compress(bool force_compression) const final;

    void insertValue(const T value) { assertOwnedForMutation("ColumnDecimal::insertValue"); data.push_back(value); }
    /// Guarded against direct mutation of adopted (producer-owned) memory: per the
    /// adoption-layer §Materialization-on-mutation contract (I3), callers wanting to
    /// mutate must route through `IColumn::mutate()` to COW-materialize first. The
    /// const overload (the aggregation read hot path) stays unguarded.
    Container & getData() { assertOwnedForMutation("ColumnDecimal::getData()"); return data; }
    const Container & getData() const { return data; }
    const T & getElement(size_t n) const { return data[n]; }
    T & getElement(size_t n) { assertOwnedForMutation("ColumnDecimal::getElement"); return data[n]; }

    UInt32 getScale() const { return scale; }

protected:
    Container data;
    UInt32 scale;

private:
    /// Heap-allocated adoption holder. Non-null iff this column wraps externally-owned
    /// (producer SHM) memory; null on default-constructed and COW-cloned columns. Owns the
    /// retain_token (pins the producer SHM region for the column's lifetime) and the
    /// charge_handle (releases adopted bytes back to the MemoryTracker on destruction); both
    /// release exactly once at column destruction, satisfying system spec I5 and
    /// memory-tracker-integration spec I7. Mirrors `ColumnVector`'s `adoption_`. See
    /// AdoptionHolder.h.
    std::unique_ptr<AdoptionHolder> adoption_; // NOLINT(readability-identifier-naming)

    /// Defense-in-depth guard called at the top of every public direct mutator (any method
    /// that writes through `data` outside the COW `mutate()` entry). Adopted columns wrap
    /// producer-owned storage; mutation through them would silently corrupt producer memory
    /// and violate adoption-layer spec I3. Out-of-line via `throwAdoptedColumnAccessor` in
    /// IColumn.cpp so this header does not need Exception.h.
    void assertOwnedForMutation(const char * method) const
    {
        if (adoption_)
            throwAdoptedColumnAccessor(method);
    }
};

template <class TCol>
concept is_col_over_big_decimal = std::is_same_v<TCol, ColumnDecimal<typename TCol::ValueType>>
    && is_decimal<typename TCol::ValueType> && is_over_big_int<typename TCol::NativeT>;

template <class TCol>
concept is_col_int_decimal = std::is_same_v<TCol, ColumnDecimal<typename TCol::ValueType>>
    && is_decimal<typename TCol::ValueType> && std::is_integral_v<typename TCol::NativeT>;

template <class> class ColumnVector;
template <class T> struct ColumnVectorOrDecimalT { using Col = ColumnVector<T>; };
template <is_decimal T> struct ColumnVectorOrDecimalT<T> { using Col = ColumnDecimal<T>; };
template <class T> using ColumnVectorOrDecimal = typename ColumnVectorOrDecimalT<T>::Col;

template <is_decimal T>
template <typename Type>
ColumnPtr ColumnDecimal<T>::indexImpl(const PaddedPODArray<Type> & indexes, size_t limit) const
{
    chassert(limit <= indexes.size());

    auto res = this->create(limit, scale);
    typename Self::Container & res_data = res->getData();
    for (size_t i = 0; i < limit; ++i)
        res_data[i] = data[indexes[i]];

    return res;
}


/// Prevent implicit template instantiation of ColumnDecimal for common decimal types

extern template class ColumnDecimal<Decimal32>;
extern template class ColumnDecimal<Decimal64>;
extern template class ColumnDecimal<Decimal128>;
extern template class ColumnDecimal<Decimal256>;
extern template class ColumnDecimal<DateTime64>;
extern template class ColumnDecimal<Time64>;


}
