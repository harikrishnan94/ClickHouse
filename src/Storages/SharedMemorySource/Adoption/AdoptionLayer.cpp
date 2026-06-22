#include <Storages/SharedMemorySource/Adoption/AdoptionLayer.h>
#include <Storages/SharedMemorySource/Wire/WireTypeMapping.h>

#include <Columns/ColumnString.h>
#include <Columns/ColumnVector.h>
#include <Columns/ColumnsNumber.h>
#include <Columns/IColumn.h>
#include <Core/TypeId.h>
#include <DataTypes/DataTypeString.h>
#include <DataTypes/DataTypesNumber.h>
#include <Common/Exception.h>

#include <cstring>
#include <memory>
#include <utility>


namespace DB
{

namespace ErrorCodes
{
    extern const int SHM_SCHEMA_MISMATCH;
    extern const int SHM_BUFFER_LAYOUT_INVALID;
    extern const int SHM_BLOCK_FRAMING_INVALID;
}

namespace
{

/// Bounds-check helper. Detects offset+size overflow AND end-past-region.
bool fitsInRegion(uint64_t off, uint64_t size, uint64_t padding, size_t region_size) noexcept
{
    const uint64_t sum1 = off + size;
    if (sum1 < off)
        return false;
    const uint64_t sum2 = sum1 + padding;
    if (sum2 < sum1)
        return false;
    return sum2 <= region_size;
}

/// Validate the single value buffer of a fixed-width column (UInt64 and every
/// numeric/date wire type). `elem_size` is the element byte width returned by
/// `wireFixedWidthSize`; the natural alignment of such a buffer equals its
/// element width, so the same value drives both the alignment and the byte math.
void validateFixedWidthDescriptor(
    const SharedMemoryWire::ColumnDescriptor & d, UInt64 row_count, size_t region_size,
    size_t column_index, size_t elem_size)
{
    /// Precondition 13: declared value-buffer offset satisfies the element's natural alignment.
    if ((d.value_offset % elem_size) != 0)
        throw Exception(ErrorCodes::SHM_BUFFER_LAYOUT_INVALID,
            "column {} (fixed-width {}B): value_offset={} is not {}-aligned (precondition 13)",
            column_index, elem_size, d.value_offset, elem_size);

    /// Precondition 26 (block-framing): value_count must equal row_count.
    if (d.value_count != row_count)
        throw Exception(ErrorCodes::SHM_BLOCK_FRAMING_INVALID,
            "column {} (fixed-width {}B): value_count={} != row_count={} (precondition 26)",
            column_index, elem_size, d.value_count, row_count);

    /// Precondition 14: value_count * elem_size + value_padding fits at value_offset.
    const uint64_t bytes = d.value_count * elem_size;
    if (d.value_count != 0 && bytes / elem_size != d.value_count)
        throw Exception(ErrorCodes::SHM_BUFFER_LAYOUT_INVALID,
            "column {} (fixed-width {}B): value_count={} * {} overflows (precondition 14)",
            column_index, elem_size, d.value_count, elem_size);
    if (!fitsInRegion(d.value_offset, bytes, d.value_padding, region_size))
        throw Exception(ErrorCodes::SHM_BUFFER_LAYOUT_INVALID,
            "column {} (fixed-width {}B): value buffer [offset={}, bytes={}, padding={}] "
            "does not fit in data region of size {} (precondition 14)",
            column_index, elem_size, d.value_offset, bytes, d.value_padding, region_size);

    /// Precondition 15: value_padding >= PADDING_FOR_SIMD.
    if (d.value_padding < SharedMemoryWire::PADDING_FOR_SIMD)
        throw Exception(ErrorCodes::SHM_BUFFER_LAYOUT_INVALID,
            "column {} (fixed-width {}B): value_padding={} < PADDING_FOR_SIMD={} (precondition 15)",
            column_index, elem_size, d.value_padding, SharedMemoryWire::PADDING_FOR_SIMD);
}

/// Adopt one fixed-width column as a `ColumnVector<T>` (zero copy). `expected_tag`
/// is the wire tag the handshake-validated schema implies for this column; a
/// per-block descriptor declaring a different tag is a post-handshake descriptor
/// inconsistency (buffer-layout-invalid, not schema-mismatch — see the rationale
/// in the original UInt64 branch).
template <typename T>
ColumnPtr adoptFixedWidthColumn(
    const SharedMemoryWire::ColumnDescriptor & desc,
    SharedMemoryWire::WireColumnType expected_tag,
    UInt64 row_count,
    const char * data_region_base,
    size_t data_region_size,
    size_t column_index,
    const std::string & column_name,
    const RetainToken & retain_token,
    const std::shared_ptr<void> & charge_token)
{
    if (desc.type != static_cast<uint32_t>(expected_tag))
        throw Exception(ErrorCodes::SHM_BUFFER_LAYOUT_INVALID,
            "column {} ('{}'): per-block descriptor declares wire-tag {} but schema "
            "(validated at handshake) expects {} — handshake-time schema equality is "
            "precondition 6 (SHM_SCHEMA_MISMATCH); a per-block descriptor inconsistency "
            "after the handshake passed is precondition-13/16-style buffer-layout-invalid",
            column_index, column_name, desc.type, static_cast<uint32_t>(expected_tag));

    validateFixedWidthDescriptor(desc, row_count, data_region_size, column_index, sizeof(T));

    auto * data_ptr = reinterpret_cast<T *>(
        const_cast<char *>(data_region_base) + desc.value_offset);
    return ColumnVector<T>::createAdopted(data_ptr, desc.value_count, retain_token, charge_token);
}

void validateStringDescriptor(
    const SharedMemoryWire::ColumnDescriptor & d,
    UInt64 row_count,
    size_t region_size,
    const char * data_region_base,
    size_t column_index)
{
    /// Precondition 16: chars alignment — UInt8 alignment is trivially satisfied.

    /// Finding 6 / offsets[-1] zero sentinel — bounds half:
    ///
    /// ColumnString::offsetAt(0) is implemented as `offsets[-1]` (a one-element back-step
    /// into the offsets buffer's pad_left region). The producer MUST leave 8 zero-valued
    /// bytes immediately before `offsets[0]` to satisfy this read; therefore
    /// `offsets_offset` MUST be `>= sizeof(uint64_t)`. The wire ABI doc records this
    /// requirement (see shm-block-stream-abi-v1.md §`ColumnString` / offsets[-1] zero
    /// sentinel). Checking the bounds BEFORE the alignment check ensures that
    /// `offsets_offset` values in `[1, 7]` surface as this more specific error rather
    /// than the generic "not 8-aligned" precondition 17 error.
    if (d.offsets_offset < sizeof(uint64_t))
        throw Exception(ErrorCodes::SHM_BUFFER_LAYOUT_INVALID,
            "column {} (String): offsets_offset={} must be >= {} bytes to leave room for "
            "the offsets[-1] zero sentinel required by the column-storage contract "
            "(ColumnString::offsetAt(0) reads offsets[-1]; see ABI doc §ColumnString)",
            column_index, d.offsets_offset, sizeof(uint64_t));

    /// Precondition 17: offsets_offset satisfies UInt64 alignment.
    if ((d.offsets_offset % alignof(uint64_t)) != 0)
        throw Exception(ErrorCodes::SHM_BUFFER_LAYOUT_INVALID,
            "column {} (String): offsets_offset={} is not {}-aligned (precondition 17)",
            column_index, d.offsets_offset, alignof(uint64_t));

    /// Precondition 26 (block-framing): offsets_count must equal row_count.
    if (d.offsets_count != row_count)
        throw Exception(ErrorCodes::SHM_BLOCK_FRAMING_INVALID,
            "column {} (String): offsets_count={} != row_count={} (precondition 26)",
            column_index, d.offsets_count, row_count);

    /// Precondition 18: chars region fits.
    if (!fitsInRegion(d.value_offset, d.value_count, d.value_padding, region_size))
        throw Exception(ErrorCodes::SHM_BUFFER_LAYOUT_INVALID,
            "column {} (String): chars [offset={}, size={}, padding={}] "
            "does not fit in data region of size {} (precondition 18)",
            column_index, d.value_offset, d.value_count, d.value_padding, region_size);

    /// Precondition 19: offsets region fits.
    const uint64_t offs_bytes = d.offsets_count * sizeof(uint64_t);
    if (d.offsets_count != 0 && offs_bytes / sizeof(uint64_t) != d.offsets_count)
        throw Exception(ErrorCodes::SHM_BUFFER_LAYOUT_INVALID,
            "column {} (String): offsets_count={} * sizeof(UInt64) overflows (precondition 19)",
            column_index, d.offsets_count);
    if (!fitsInRegion(d.offsets_offset, offs_bytes, d.offsets_padding, region_size))
        throw Exception(ErrorCodes::SHM_BUFFER_LAYOUT_INVALID,
            "column {} (String): offsets [offset={}, bytes={}, padding={}] "
            "does not fit in data region of size {} (precondition 19)",
            column_index, d.offsets_offset, offs_bytes, d.offsets_padding, region_size);

    /// Precondition 20: both paddings >= PADDING_FOR_SIMD.
    if (d.value_padding < SharedMemoryWire::PADDING_FOR_SIMD)
        throw Exception(ErrorCodes::SHM_BUFFER_LAYOUT_INVALID,
            "column {} (String): chars padding={} < PADDING_FOR_SIMD={} (precondition 20)",
            column_index, d.value_padding, SharedMemoryWire::PADDING_FOR_SIMD);
    if (d.offsets_padding < SharedMemoryWire::PADDING_FOR_SIMD)
        throw Exception(ErrorCodes::SHM_BUFFER_LAYOUT_INVALID,
            "column {} (String): offsets padding={} < PADDING_FOR_SIMD={} (precondition 20)",
            column_index, d.offsets_padding, SharedMemoryWire::PADDING_FOR_SIMD);

    /// Finding 6 / offsets[-1] zero sentinel — value half:
    ///
    /// Bounds were checked at the top of the function (offsets_offset >= 8) and the
    /// alignment + region-fit checks just above ensure the read at
    /// `data_region_base + offsets_offset - 8` is in-region and 8-aligned. We memcpy
    /// into a stack uint64_t both to be defensive about alignment-strict targets and to
    /// keep TSan/UBSan from flagging a reinterpret_cast through a const char *.
    uint64_t sentinel = 0;
    std::memcpy(
        &sentinel,
        data_region_base + d.offsets_offset - sizeof(uint64_t),
        sizeof(uint64_t));
    if (sentinel != 0)
        throw Exception(ErrorCodes::SHM_BUFFER_LAYOUT_INVALID,
            "column {} (String): offsets[-1] sentinel at offset {} must be zero "
            "(got {}); see column-storage contract and shm-block-stream-abi-v1 "
            "§ColumnString / offsets[-1] zero sentinel",
            column_index, d.offsets_offset - sizeof(uint64_t), sentinel);
}

}

Columns adopt(
    const std::vector<SharedMemoryWire::ColumnDescriptor> & descriptors,
    const std::vector<std::pair<std::string, DataTypePtr>> & schema,
    const char * data_region_base,
    size_t data_region_size,
    UInt64 row_count,
    RetainToken retain_token,
    ChargeHandle charge_handle)
{
    /// `retain_token` is a shared_ptr<void> taken by value: its local copy in this scope is
    /// the on-throw rollback handle (drops to refcount 0 on stack unwind). `charge_handle`
    /// is move-only and taken by value: its destructor runs on stack unwind for the same
    /// reason. The columns_local accumulator below stays empty until each per-column ctor
    /// succeeds; on any throw, columns_local goes out of scope and any partially-built
    /// adopted column releases its own share. This is the `system.md` I10 / `adoption-layer.md`
    /// §Retain and charge handle semantics rollback contract.
    if (descriptors.size() != schema.size())
        throw Exception(ErrorCodes::SHM_BLOCK_FRAMING_INVALID,
            "adopt: descriptors.size()={} != schema.size()={} (precondition 12)",
            descriptors.size(), schema.size());

    /// Wrap the move-only ChargeHandle in a shared_ptr so it can ride alongside the retain
    /// token in every column. This is the phase-1 implementation choice noted in the brief
    /// and the spec (`adoption-layer.md` §Retain and charge handle semantics): the
    /// "LAST adopted column owns the unique releaser" is rephrased as "the ChargeHandle's
    /// destructor runs when the last shared_ptr<ChargeHandle> reference drops" — which is
    /// the same final-drop ordering, modulo the std::shared_ptr abstraction.
    auto charge_shared = std::make_shared<ChargeHandle>(std::move(charge_handle));
    /// Type-erase for the column factories (they take std::shared_ptr<void>); the deleter
    /// inside `charge_shared`'s control block keeps the correct ChargeHandle destructor in
    /// place under the std::shared_ptr<void> alias, so when the last reference drops the
    /// ChargeHandle destructor runs (not a raw `delete void*` UB).
    std::shared_ptr<void> charge_token = charge_shared;

    Columns columns_local;
    columns_local.reserve(descriptors.size());

    for (size_t i = 0; i < descriptors.size(); ++i)
    {
        const auto & desc = descriptors[i];
        const auto & type = schema[i].second;

        /// Type dispatch — precondition 6's adoption-side membership/equality catch. The
        /// SQL-side membership gate lives in T3.4 (TableFunctionShm); the handshake-side
        /// equality cross-validation lives in T3.2a (PollableShmSource::ensureAttached).
        /// Here we re-check the runtime type against the descriptor's declared wire type as
        /// a last line of defence — both must agree, otherwise the producer is publishing
        /// against a schema different from the one negotiated at handshake.
        const auto type_id = type->getTypeId();

        if (type_id == TypeIndex::String)
        {
            /// A per-block descriptor wire-tag inconsistency once past the handshake-time
            /// schema equality gate (precondition 6) is a buffer-layout/descriptor issue,
            /// not a schema-mismatch.
            if (desc.type != static_cast<uint32_t>(SharedMemoryWire::WireColumnType::String))
                throw Exception(ErrorCodes::SHM_BUFFER_LAYOUT_INVALID,
                    "column {} ('{}'): per-block descriptor declares wire-tag {} but schema "
                    "(validated at handshake) expects {} for String — handshake-time schema "
                    "equality is precondition 6 (SHM_SCHEMA_MISMATCH); per-block descriptor "
                    "inconsistency after the handshake passed is precondition-13/16-style "
                    "buffer-layout-invalid",
                    i, schema[i].first, desc.type,
                    static_cast<uint32_t>(SharedMemoryWire::WireColumnType::String));

            validateStringDescriptor(desc, row_count, data_region_size, data_region_base, i);

            auto * chars_ptr = reinterpret_cast<UInt8 *>(
                const_cast<char *>(data_region_base) + desc.value_offset);
            auto * offsets_ptr = reinterpret_cast<UInt64 *>(
                const_cast<char *>(data_region_base) + desc.offsets_offset);
            auto col = ColumnString::createAdopted(
                chars_ptr, desc.value_count,
                offsets_ptr, desc.offsets_count,
                retain_token, charge_token);
            columns_local.emplace_back(std::move(col));
        }
        else if (const auto wire_tag = SharedMemoryWire::tryWireColumnTypeForTypeIndex(type_id))
        {
            /// Fixed-width adoption. `wire_tag` is the tag the handshake-validated schema
            /// implies; `adoptFixedWidthColumn` re-checks the per-block descriptor tag against
            /// it and validates the single value buffer. The switch only selects the column
            /// element type `T`; the validation/creation logic lives in the helper.
            const auto tag = *wire_tag;
            ColumnPtr col;
            switch (type_id)
            {
                case TypeIndex::UInt8:    col = adoptFixedWidthColumn<UInt8>   (desc, tag, row_count, data_region_base, data_region_size, i, schema[i].first, retain_token, charge_token); break;
                case TypeIndex::UInt16:   col = adoptFixedWidthColumn<UInt16>  (desc, tag, row_count, data_region_base, data_region_size, i, schema[i].first, retain_token, charge_token); break;
                case TypeIndex::UInt32:   col = adoptFixedWidthColumn<UInt32>  (desc, tag, row_count, data_region_base, data_region_size, i, schema[i].first, retain_token, charge_token); break;
                case TypeIndex::UInt64:   col = adoptFixedWidthColumn<UInt64>  (desc, tag, row_count, data_region_base, data_region_size, i, schema[i].first, retain_token, charge_token); break;
                case TypeIndex::Int8:     col = adoptFixedWidthColumn<Int8>    (desc, tag, row_count, data_region_base, data_region_size, i, schema[i].first, retain_token, charge_token); break;
                case TypeIndex::Int16:    col = adoptFixedWidthColumn<Int16>   (desc, tag, row_count, data_region_base, data_region_size, i, schema[i].first, retain_token, charge_token); break;
                case TypeIndex::Int32:    col = adoptFixedWidthColumn<Int32>   (desc, tag, row_count, data_region_base, data_region_size, i, schema[i].first, retain_token, charge_token); break;
                case TypeIndex::Int64:    col = adoptFixedWidthColumn<Int64>   (desc, tag, row_count, data_region_base, data_region_size, i, schema[i].first, retain_token, charge_token); break;
                case TypeIndex::Float32:  col = adoptFixedWidthColumn<Float32> (desc, tag, row_count, data_region_base, data_region_size, i, schema[i].first, retain_token, charge_token); break;
                case TypeIndex::Float64:  col = adoptFixedWidthColumn<Float64> (desc, tag, row_count, data_region_base, data_region_size, i, schema[i].first, retain_token, charge_token); break;
                /// Date/DateTime/Date32 adopt through their fixed-width storage column.
                case TypeIndex::Date:     col = adoptFixedWidthColumn<UInt16>  (desc, tag, row_count, data_region_base, data_region_size, i, schema[i].first, retain_token, charge_token); break;
                case TypeIndex::DateTime: col = adoptFixedWidthColumn<UInt32>  (desc, tag, row_count, data_region_base, data_region_size, i, schema[i].first, retain_token, charge_token); break;
                case TypeIndex::Date32:   col = adoptFixedWidthColumn<Int32>   (desc, tag, row_count, data_region_base, data_region_size, i, schema[i].first, retain_token, charge_token); break;
                default:
                    /// Unreachable: `tryWireColumnTypeForTypeIndex` returned a fixed-width tag
                    /// only for the cases above (String is handled in the branch before this).
                    throw Exception(ErrorCodes::SHM_SCHEMA_MISMATCH,
                        "column {} ('{}'): type '{}' has a wire tag but no adoption arm (orchestration bug)",
                        i, schema[i].first, type->getName());
            }
            columns_local.emplace_back(std::move(col));
        }
        else
        {
            /// Late catch for precondition 6 on the adoption side. Anything that escapes the
            /// SQL-side gate AND the handshake cross-validation is a programming error in
            /// the orchestration; we still raise the typed exception per `adoption-layer.md`
            /// §Unsupported types.
            throw Exception(ErrorCodes::SHM_SCHEMA_MISMATCH,
                "column {} ('{}'): unsupported type '{}' (adoption-layer supports {})",
                i, schema[i].first, type->getName(), SharedMemoryWire::supportedShmTypeList());
        }
    }

    return columns_local;
}

}
