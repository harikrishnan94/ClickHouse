#pragma once

#include <cstring>
#include <memory>
#include <utility>

#include <IO/WriteHelpers.h>
#include <Columns/AdoptionHolder.h>
#include <Columns/IColumn.h>
#include <Columns/IColumnImpl.h>
#include <Common/PODArray.h>
#include <Common/memcpySmall.h>
#include <base/memcmpSmall.h>
#include <Common/assert_cast.h>
#include <Core/Field.h>

#include <base/defines.h>


class Collator;
class SipHash;

namespace DB
{

class Arena;

/// Column for String values.
class ColumnString final : public COWHelper<IColumnHelper<ColumnString>, ColumnString>
{
public:
    using Char = UInt8;
    using Chars = PaddedPODArray<UInt8>;

    static constexpr size_t min_size_to_compress = 4096;

private:
    friend class COWHelper<IColumnHelper<ColumnString>, ColumnString>;

    /// Maps i'th position to offset to i+1'th element. Last offset maps to the end of all chars (is the size of all chars).
    Offsets offsets;

    /// Bytes of strings, placed contiguously. Note that strings are not zero-terminated and could contain zero bytes in the middle.
    Chars chars;

    /// Heap-allocated adoption holder. Non-null iff this column wraps producer-owned (SHM)
    /// memory via createAdopted(). Owns the retain_token (pins producer SHM region for the
    /// column's lifetime) and charge_handle (releases adopted bytes back to MemoryTracker
    /// on destruction); both shared_ptrs are released exactly once at this column's
    /// destruction (or on the COW clone path: the clone leaves this null, so producer
    /// memory release is gated only on outstanding references to the originally adopted
    /// instance). See adoption-layer spec §Retain and charge handle semantics, system
    /// spec I5, and AdoptionHolder.h.
    ///
    /// Single 8-byte member rather than two ~16-byte std::shared_ptr<void> inline members:
    /// non-adopted ColumnString instances (the common case) pay only 8 bytes of layout
    /// overhead for this feature.
    ///
    /// Trailing underscore disambiguates from constructor parameters; matches the
    /// convention established in T2.1's ColumnVector and T1.4's ChargeHandle.
    std::unique_ptr<AdoptionHolder> adoption_; // NOLINT(readability-identifier-naming)

    /// Defense-in-depth guard called at the top of every public direct mutator (any method
    /// that writes through `chars` or `offsets` outside the COW `mutate()` entry point).
    /// Adopted columns wrap producer-owned storage; mutation through them would silently
    /// corrupt producer memory and violate adoption-layer spec I3 + §Materialization-on-
    /// mutation contract. Inline here so the guard compiles down to a single null-check.
    void assertOwnedForMutation(const char * method) const
    {
        if (adoption_)
            throwAdoptedAccessorWrite(method);
    }

    size_t ALWAYS_INLINE offsetAt(ssize_t i) const { return offsets[i - 1]; }

    /// Size of i-th element
    size_t ALWAYS_INLINE sizeAt(ssize_t i) const
    {
        chassert(offsets[i] >= offsets[i - 1]);
        return offsets[i] - offsets[i - 1];
    }

    struct ComparatorBase;

    using ComparatorAscendingUnstable = ComparatorAscendingUnstableImpl<ComparatorBase>;
    using ComparatorAscendingStable = ComparatorAscendingStableImpl<ComparatorBase>;
    using ComparatorDescendingUnstable = ComparatorDescendingUnstableImpl<ComparatorBase>;
    using ComparatorDescendingStable = ComparatorDescendingStableImpl<ComparatorBase>;
    using ComparatorEqual = ComparatorEqualImpl<ComparatorBase>;

    struct ComparatorCollationBase;

    using ComparatorCollationAscendingUnstable = ComparatorAscendingUnstableImpl<ComparatorCollationBase>;
    using ComparatorCollationAscendingStable = ComparatorAscendingStableImpl<ComparatorCollationBase>;
    using ComparatorCollationDescendingUnstable = ComparatorDescendingUnstableImpl<ComparatorCollationBase>;
    using ComparatorCollationDescendingStable = ComparatorDescendingStableImpl<ComparatorCollationBase>;
    using ComparatorCollationEqual = ComparatorEqualImpl<ComparatorCollationBase>;

    ColumnString() = default;
    ColumnString(const ColumnString & src);

    /// Adopted-mode constructor. Wraps externally-owned (typically SHM-producer) chars and
    /// offsets buffers without copying. Both PaddedPODArrays are constructed in adopted
    /// mode (mutators throw, dealloc is a no-op); the producer-side lifetime is pinned by
    /// `adoption`'s retain_token / charge_handle which RAII-release on this column's
    /// destruction. The opaque owner markers are the addresses of the parameter pointers
    /// themselves; PaddedPODArray treats them as void * sentinels and never dereferences
    /// them.
    ColumnString(UInt8 * adopted_chars, size_t adopted_chars_size,
                 UInt64 * adopted_offsets, size_t adopted_rows,
                 std::unique_ptr<AdoptionHolder> adoption)
        : offsets(adopted_offsets, adopted_rows, &adopted_offsets)
        , chars(adopted_chars, adopted_chars_size, &adopted_chars)
        , adoption_(std::move(adoption))
    {
        chassert(adoption_ != nullptr);
        chassert(adoption_->retain_token != nullptr);
        chassert(adoption_->charge_handle != nullptr);
    }

    /// Throw helpers (kept out-of-line so this header does not need Exception.h).
    [[noreturn]] static void throwAdoptedFactoryRequiresHandles();
    [[noreturn]] static void throwAdoptedAccessorWrite(const char * which);

public:
    const char * getFamilyName() const override { return "String"; }
    TypeIndex getDataType() const override { return TypeIndex::String; }

    size_t size() const override
    {
        return offsets.size();
    }

    size_t byteSize() const override
    {
        return chars.size() + offsets.size() * sizeof(offsets[0]);
    }

    size_t byteSizeAt(size_t n) const override
    {
        chassert(n < size());
        return sizeAt(n) + sizeof(offsets[0]);
    }

    size_t allocatedBytes() const override
    {
        return chars.allocated_bytes() + offsets.allocated_bytes();
    }

    void protect() override;

    MutableColumnPtr cloneResized(size_t to_size) const override;

    Field operator[](size_t n) const override
    {
        chassert(n < size());
        return Field(&chars[offsetAt(n)], sizeAt(n));
    }

    void get(size_t n, Field & res) const override
    {
        chassert(n < size());
        res = std::string_view{reinterpret_cast<const char *>(&chars[offsetAt(n)]), sizeAt(n)};
    }

    void getValueNameImpl(WriteBufferFromOwnString & name_buf, size_t n, const Options & options) const override
    {
        if (options.notFull(name_buf))
            writeQuoted(std::string_view{reinterpret_cast<const char *>(&chars[offsetAt(n)]), sizeAt(n)}, name_buf);
    }

    std::string_view getDataAt(size_t n) const override
    {
        chassert(n < size());
        return std::string_view(reinterpret_cast<const char *>(&chars[offsetAt(n)]), sizeAt(n));
    }

    bool isDefaultAt(size_t n) const override
    {
        chassert(n < size());
        return sizeAt(n) == 0;
    }

    void insert(const Field & x) override
    {
        assertOwnedForMutation("insert");
        const String & s = x.safeGet<String>();
        const size_t old_size = chars.size();
        const size_t size_to_append = s.size();
        const size_t new_size = old_size + size_to_append;

        chars.resize(new_size);
        memcpy(chars.data() + old_size, s.data(), size_to_append);
        offsets.push_back(new_size);
    }

    bool tryInsert(const Field & x) override
    {
        assertOwnedForMutation("tryInsert");
        if (x.getType() != Field::Types::Which::String)
            return false;

        insert(x);
        return true;
    }

#if !defined(DEBUG_OR_SANITIZER_BUILD)
    void insertFrom(const IColumn & src_, size_t n) override
#else
    void doInsertFrom(const IColumn & src_, size_t n) override
#endif
    {
        assertOwnedForMutation("insertFrom");
        const ColumnString & src = assert_cast<const ColumnString &>(src_);
        const size_t size_to_append = src.sizeAt(n);

        if (size_to_append == 0)
        {
            /// shortcut for empty string
            offsets.push_back(chars.size());
        }
        else
        {
            const size_t old_size = chars.size();
            const size_t offset = src.offsetAt(n);
            const size_t new_size = old_size + size_to_append;

            chars.resize(new_size);
            memcpySmallAllowReadWriteOverflow15(chars.data() + old_size, &src.chars[offset], size_to_append);
            offsets.push_back(new_size);
        }
    }

#if !defined(DEBUG_OR_SANITIZER_BUILD)
    void insertManyFrom(const IColumn & src, size_t position, size_t length) override;
#else
    void doInsertManyFrom(const IColumn & src, size_t position, size_t length) override;
#endif

    void insertData(const char * pos, size_t length) override
    {
        assertOwnedForMutation("insertData");
        const size_t old_size = chars.size();
        const size_t new_size = old_size + length;

        chars.resize(new_size);
        if (length)
            memcpy(chars.data() + old_size, pos, length);
        offsets.push_back(new_size);
    }

    void popBack(size_t n) override
    {
        assertOwnedForMutation("popBack");
        if (n > size())
            throwCannotPopBack(n, getName(), size());

        size_t nested_n = offsets.back() - offsetAt(offsets.size() - n);
        chars.resize(chars.size() - nested_n);
        offsets.resize_assume_reserved(offsets.size() - n);
    }

    ColumnCheckpointPtr getCheckpoint() const override;
    void updateCheckpoint(ColumnCheckpoint & checkpoint) const override;
    void rollback(const ColumnCheckpoint & checkpoint) override;

    void collectSerializedValueSizes(PaddedPODArray<UInt64> & sizes, const UInt8 * is_null, const IColumn::SerializationSettings * settings) const override;

    std::optional<size_t> getSerializedValueSize(size_t n, const IColumn::SerializationSettings * settings) const override;

    std::string_view serializeValueIntoArena(size_t n, Arena & arena, char const *& begin, const IColumn::SerializationSettings * settings) const override;
    ALWAYS_INLINE char * serializeValueIntoMemory(size_t n, char * memory, const IColumn::SerializationSettings * settings) const override;

    void batchSerializeValueIntoMemory(VectorWithMemoryTracking<char *> & memories, const IColumn::SerializationSettings * settings) const override;

    void deserializeAndInsertFromArena(ReadBuffer & in, const IColumn::SerializationSettings * settings) override;

    void skipSerializedInArena(ReadBuffer & in) const override;

    void updateHashWithValue(size_t n, SipHash & hash) const override;
    void updateHashWithValueRange(size_t begin, size_t end, SipHash & hash) const override;

    void computeHashInto(size_t row_begin, size_t row_end, UInt32 * hash_out, bool initial) const override;

    void updateHashFast(SipHash & hash) const override;

#if !defined(DEBUG_OR_SANITIZER_BUILD)
    void insertRangeFrom(const IColumn & src, size_t start, size_t length) override;
#else
    void doInsertRangeFrom(const IColumn & src, size_t start, size_t length) override;
#endif

    ColumnPtr filter(const Filter & filt, ssize_t result_size_hint) const override;

    void filter(const Filter & filt) override;

    void expand(const Filter & mask, bool inverted) override;

    ColumnPtr permute(const Permutation & perm, size_t limit) const override;

    ColumnPtr index(const IColumn & indexes, size_t limit) const override;

    template <typename Type>
    ColumnPtr indexImpl(const PaddedPODArray<Type> & indexes, size_t limit) const;

    void insertDefault() override
    {
        assertOwnedForMutation("insertDefault");
        auto last = offsets.back();
        offsets.push_back(last);
    }

    void insertManyDefaults(size_t length) override
    {
        assertOwnedForMutation("insertManyDefaults");
        auto last = offsets.back();
        for (size_t i = 0; i < length; ++i)
            offsets.push_back(last);
    }

#if !defined(DEBUG_OR_SANITIZER_BUILD)
    int compareAt(size_t n, size_t m, const IColumn & rhs_, int /*nan_direction_hint*/) const override
#else
    int doCompareAt(size_t n, size_t m, const IColumn & rhs_, int /*nan_direction_hint*/) const override
#endif
    {
        const ColumnString & rhs = assert_cast<const ColumnString &>(rhs_);
        return memcmpSmallAllowOverflow15(chars.data() + offsetAt(n), sizeAt(n), rhs.chars.data() + rhs.offsetAt(m), rhs.sizeAt(m));
    }

#if USE_EMBEDDED_COMPILER
    bool isComparatorCompilable() const override;
    llvm::Value * compileComparator(llvm::IRBuilderBase & b, llvm::Value * lhs, llvm::Value * rhs, llvm::Value * /*nan_direction_hint*/) const override;
#endif

    /// Variant of compareAt for string comparison with respect of collation.
    int compareAtWithCollation(size_t n, size_t m, const IColumn & rhs_, int, const Collator & collator) const override;

    void getPermutation(IColumn::PermutationSortDirection direction, IColumn::PermutationSortStability stability,
                    size_t limit, int nan_direction_hint, Permutation & res) const override;

    void updatePermutation(IColumn::PermutationSortDirection direction, IColumn::PermutationSortStability stability,
                    size_t limit, int, Permutation & res, EqualRanges & equal_ranges) const override;

    /// Sorting with respect of collation.
    void getPermutationWithCollation(const Collator & collator, IColumn::PermutationSortDirection direction, IColumn::PermutationSortStability stability,
                    size_t limit, int, Permutation & res) const override;

    void updatePermutationWithCollation(const Collator & collator, IColumn::PermutationSortDirection direction, IColumn::PermutationSortStability stability,
                    size_t limit, int, Permutation & res, EqualRanges & equal_ranges) const override;

    size_t estimateCardinalityInPermutedRange(const Permutation & permutation, const EqualRange & equal_range) const override;

    ColumnPtr replicate(const Offsets & replicate_offsets) const override;

    ColumnPtr compress(bool force_compression) const override;

    void reserve(size_t n) override;
    size_t capacity() const override;
    void prepareForSquashing(const VectorWithMemoryTracking<ColumnPtr> & source_columns, size_t factor) override;
    void shrinkToFit() override;

    void getExtremes(Field & min, Field & max, size_t start, size_t end) const override;

    bool canBeInsideNullable() const override { return true; }

    bool structureEquals(const IColumn & rhs) const override
    {
        return typeid(rhs) == typeid(ColumnString);
    }

    /// Construct a ColumnString wrapping producer-owned chars and offsets buffers without
    /// copying. Returned column carries a retain_token (pins producer SHM region for the
    /// column's lifetime) and a charge_handle (returns adopted bytes to MemoryTracker on
    /// destruction); both are RAII-released exactly once when the last reference to the
    /// adopted state drops (the COW clone path materialises a fully owned copy and does
    /// not propagate either handle — see [adoption-layer spec §Retain and charge handle
    /// semantics](file:///home/hari/auto_click/specs/adoption-layer.md#interfaces--contracts)
    /// and system spec I5).
    ///
    /// Requirements on inputs (the caller — the adoption layer T3.1 — must enforce these
    /// via per-column descriptor validation before calling; createAdopted does not
    /// re-validate, the contract puts the burden on the producer ABI documentation per
    /// [shm-block-stream spec §Per-type buffer layout](file:///home/hari/auto_click/specs/shm-block-stream.md#per-type-buffer-layout)):
    ///   - adopted_chars[0 .. adopted_chars_size + PaddedPODArray<UInt8>::pad_right - 1]
    ///     is safely readable (column-storage trailing safe-read padding).
    ///   - adopted_offsets[0 .. adopted_rows - 1] is safely readable, plus
    ///     PaddedPODArray<UInt64>::pad_right bytes of trailing safe-read padding.
    ///   - The byte 8 bytes BEFORE adopted_offsets[0] (i.e. adopted_offsets[-1] read as
    ///     UInt64) MUST equal 0. ColumnString::offsetAt(0) is implemented as offsets[-1]
    ///     and is exercised by every read of row 0. The producer ABI satisfies this as
    ///     part of the safe-read padding contract — it places the offsets buffer 8 bytes
    ///     into its allocated region and zero-initialises the preceding UInt64 slot. This
    ///     mirrors PaddedPODArray<UInt64>'s own pad_left zero-init invariant. The SHM
    ///     consumer-side enforcement of this invariant lives in `AdoptionLayer.cpp` —
    ///     `validateStringDescriptor` rejects descriptors whose `offsets_offset < 8` and
    ///     whose pre-offsets-buffer sentinel byte is non-zero with
    ///     `SHM_BUFFER_LAYOUT_INVALID`; createAdopted therefore does not re-validate.
    ///   - Content-level monotonicity (precondition 21) and terminal offset value
    ///     (precondition 22) are NOT validated here. They are content-level, lazy, and
    ///     surfaced via validateAdoptedOffsets() (called by the adoption layer or by the
    ///     consumer-side read path that would otherwise observe the violation). Authority:
    ///     adoption-layer §Constraints, I4.
    ///   - adopted_chars and adopted_offsets are aligned to alignof(UInt8)==1 and
    ///     alignof(UInt64)==8 respectively.
    ///   - Both retain_token and charge_handle are non-null; LOGICAL_ERROR otherwise.
    static MutablePtr createAdopted(
        UInt8 * adopted_chars, size_t adopted_chars_size,
        UInt64 * adopted_offsets, size_t adopted_rows,
        std::shared_ptr<void> retain_token,
        std::shared_ptr<void> charge_handle)
    {
        if (!retain_token || !charge_handle)
            throwAdoptedFactoryRequiresHandles();
        auto holder = std::make_unique<AdoptionHolder>(std::move(retain_token), std::move(charge_handle));
        return create(adopted_chars, adopted_chars_size,
                      adopted_offsets, adopted_rows,
                      std::move(holder));
    }

    /// Non-const accessors throw on adopted columns: writing through getChars().data()[i]
    /// or getOffsets().data()[i] would silently corrupt producer memory (see VC1 finding
    /// and adoption-layer spec I3). Callers that need to mutate MUST first call
    /// IColumn::mutate(), which COW-clones into a fresh owned ColumnString. The const
    /// overloads are AC1 read-path hot lanes (cityHash64(), length()) and MUST remain
    /// unguarded for performance.
    Chars & getChars()
    {
        assertOwnedForMutation("getChars");
        return chars;
    }
    const Chars & getChars() const { return chars; }

    Offsets & getOffsets()
    {
        assertOwnedForMutation("getOffsets");
        return offsets;
    }
    const Offsets & getOffsets() const { return offsets; }

    // Throws an exception if offsets/chars are messed up
    void validate() const;

    /// Lazy content-level validation for adopted columns: precondition 21 (offsets are
    /// monotonically non-decreasing) and precondition 22 (terminal offset equals chars
    /// buffer size). Throws SHM_BUFFER_LAYOUT_INVALID on violation. Const and idempotent;
    /// safe to call repeatedly. Treats offsets[-1] as the implicit 0 sentinel (the adopted
    /// factory's contract requires the producer ABI to satisfy this). The plain validate()
    /// above only checks precondition 22 with LOGICAL_ERROR — it is kept for callers that
    /// expect that contract; this method is additive for the SHM-adoption surface.
    /// Authority: adoption-layer §Constraints, I4.
    void validateAdoptedOffsets() const;

    bool isCollationSupported() const override { return true; }

    /// Constructs a ColumnUInt64 representing the `.size` subcolumn, derived from the string offsets.
    ColumnPtr createSizeSubcolumn() const;
};


}
