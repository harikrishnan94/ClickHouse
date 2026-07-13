#include <Columns/ColumnsScatter.h>

#include <Columns/ColumnConst.h>
#include <Common/Arena.h>
#include <Common/Exception.h>
#include <Common/assert_cast.h>

#include <Core/TypeId.h>

#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <utility>

namespace DB
{

namespace ErrorCodes
{
extern const int LOGICAL_ERROR;
extern const int NOT_IMPLEMENTED;
}

}

namespace DB::ColumnsScatter
{

namespace
{

/// ------------------------------------------------------------------------------------------------
/// Kernel bodies — ported from the radix scatter kernels of RadixHashJoin.cpp (themselves a
/// width-generic port of the benchmark scatter in src/Common/benchmarks/hash_join_bench.cpp),
/// generalized over the partition-id type. Structure, constants, and invariants are unchanged;
/// see the ScatterScratch invariant in the header.
/// ------------------------------------------------------------------------------------------------

using NtLine = char __attribute__((vector_size(LINE_BYTES)));

/// Compile-time width variant for the hot single-key path (the loop unrolls fully).
template <size_t width>
ALWAYS_INLINE UInt32 routeWordFixed(const char * p)
{
    if constexpr (width == sizeof(UInt64))
    {
        UInt64 key{};
        __builtin_memcpy_inline(&key, p, sizeof(key));
        return routeWord(key);
    }
    else
    {
        return finalizeRoute(foldBytes(0, p, width));
    }
}

/// The routing source per row. The single-column key kernel computes the partition from the key (and
/// optionally emits it as a pid); the payload kernels reload the emitted pid.
template <size_t width, typename Pid>
struct RouteFromKey
{
    const char * keys;
    UInt32 shift;
    UInt32 mask;
    Pid * pids; /// null when there are no columns to consume the ids

    ALWAYS_INLINE UInt32 partition(size_t i) const
    {
        const UInt32 p = (routeWordFixed<width>(keys + i * width) >> shift) & mask;
        if (pids)
            pids[i] = static_cast<Pid>(p);
        return p;
    }
};

template <typename Pid>
struct RouteFromKeyGeneric
{
    const char * keys;
    size_t width;
    UInt32 shift;
    UInt32 mask;
    Pid * pids;

    ALWAYS_INLINE UInt32 partition(size_t i) const
    {
        const UInt32 p = (routeWordBytes(keys + i * width, width) >> shift) & mask;
        if (pids)
            pids[i] = static_cast<Pid>(p);
        return p;
    }
};

template <typename Pid>
struct RouteFromPids
{
    const Pid * pids;
    ALWAYS_INLINE UInt32 partition(size_t i) const { return pids[i]; }
};

template <size_t width, typename Route>
void scatterDirect(Route route, const char * data, size_t n, char ** cursors)
{
    for (size_t i = 0; i < n; ++i)
    {
        const UInt32 p = route.partition(i);
        char * dst = cursors[p];
        __builtin_memcpy_inline(dst, data + i * width, width);
        cursors[p] = dst + width;
    }
}

template <typename Route>
void scatterDirectGeneric(Route route, const char * data, size_t n, size_t w, char ** cursors)
{
    for (size_t i = 0; i < n; ++i)
    {
        const UInt32 p = route.partition(i);
        char * dst = cursors[p];
        memcpy(dst, data + i * w, w);
        cursors[p] = dst + w;
    }
}

template <size_t width, typename Route>
void scatterSwwc(Route route, const char * data, size_t n, ScatterScratch & scratch)
{
    /// Hoisted like `staging`: the char*/vector NT store defeats TBAA hoisting, so without this the
    /// compiler reloads scratch.cursors/fill.data() every row.
    char * const staging = scratch.staging;
    char ** const cursors = scratch.cursors.data();
    UInt32 * const fill = scratch.fill.data();

    for (size_t i = 0; i < n; ++i)
    {
        const UInt32 p = route.partition(i);
        char * line = staging + static_cast<size_t>(p) * LINE_BYTES;
        UInt32 f = fill[p];
        __builtin_memcpy_inline(line + f, data + i * width, width);
        f += width;
        if (f == LINE_BYTES)
        {
            char * cur = cursors[p];
            const UInt32 m = static_cast<UInt32>(reinterpret_cast<uintptr_t>(cur) & (LINE_BYTES - 1));
            if (m) /// first flush of a misaligned stream: emit the partial head line with regular stores
            {
                __builtin_memcpy(cur, line + m, LINE_BYTES - m);
                cursors[p] = cur + (LINE_BYTES - m);
            }
            else
            {
                __builtin_nontemporal_store(*reinterpret_cast<const NtLine *>(line), reinterpret_cast<NtLine *>(cur));
                cursors[p] = cur + LINE_BYTES;
            }
            f = 0;
        }
        fill[p] = f;
    }
}

template <size_t width, typename Route>
ALWAYS_INLINE void scatterOne(Route route, const char * data, size_t n, bool use_swwc, ScatterScratch & scratch)
{
    if (use_swwc)
        scatterSwwc<width>(route, data, n, scratch);
    else
        scatterDirect<width>(route, data, n, scratch.cursors.data());
}

template <typename Pid>
void scatterKeyChunkImpl(
    size_t kw, const char * keys, size_t n, UInt32 shift, UInt32 mask, Pid * pids, bool use_swwc, ScatterScratch & scratch)
{
    switch (kw)
    {
        case 4: scatterOne<4>(RouteFromKey<4, Pid>{keys, shift, mask, pids}, keys, n, use_swwc, scratch); break;
        case 8: scatterOne<8>(RouteFromKey<8, Pid>{keys, shift, mask, pids}, keys, n, use_swwc, scratch); break;
        case 16: scatterOne<16>(RouteFromKey<16, Pid>{keys, shift, mask, pids}, keys, n, use_swwc, scratch); break;
        default:
            scatterDirectGeneric(RouteFromKeyGeneric<Pid>{keys, kw, shift, mask, pids}, keys, n, kw, scratch.cursors.data());
            break;
    }
}

template <typename Pid>
void scatterPidChunkImpl(size_t w, const Pid * pids, const char * data, size_t n, bool use_swwc, ScatterScratch & scratch)
{
    RouteFromPids<Pid> route{pids};
    switch (w)
    {
        case 1: scatterOne<1>(route, data, n, use_swwc, scratch); break;
        case 2: scatterOne<2>(route, data, n, use_swwc, scratch); break;
        case 4: scatterOne<4>(route, data, n, use_swwc, scratch); break;
        case 8: scatterOne<8>(route, data, n, use_swwc, scratch); break;
        case 16: scatterOne<16>(route, data, n, use_swwc, scratch); break;
        default: scatterDirectGeneric(route, data, n, w, scratch.cursors.data()); break;
    }
}

/// Histogram one chunk's rows from a single key column. At low fanout `lanes` (4 * fanout, caller
/// owned, persistent across chunks) breaks the load-increment-store dependency chain.
/// hist and lanes are each written on one branch (a clang-tidy false positive flags them const-able).
template <size_t width, typename Counter>
void histogramKeyT(
    const char * keys,
    size_t n,
    UInt32 shift,
    UInt32 mask,
    Counter * hist,
    Counter * lanes,
    size_t fanout) /// NOLINT(readability-non-const-parameter)
{
    if (!lanes)
    {
        for (size_t i = 0; i < n; ++i)
            ++hist[(routeWordFixed<width>(keys + i * width) >> shift) & mask];
        return;
    }
    size_t i = 0;
    for (; i + 4 <= n; i += 4)
    {
        ++lanes[0 * fanout + ((routeWordFixed<width>(keys + (i + 0) * width) >> shift) & mask)];
        ++lanes[1 * fanout + ((routeWordFixed<width>(keys + (i + 1) * width) >> shift) & mask)];
        ++lanes[2 * fanout + ((routeWordFixed<width>(keys + (i + 2) * width) >> shift) & mask)];
        ++lanes[3 * fanout + ((routeWordFixed<width>(keys + (i + 3) * width) >> shift) & mask)];
    }
    for (; i < n; ++i)
        ++lanes[(i & 3) * fanout + ((routeWordFixed<width>(keys + i * width) >> shift) & mask)];
}

template <typename Counter>
void histogramKeyGeneric(
    const char * keys, size_t width, size_t n, UInt32 shift, UInt32 mask, Counter * hist, Counter * lanes, size_t fanout)
{
    if (!lanes)
    {
        for (size_t i = 0; i < n; ++i)
            ++hist[(routeWordBytes(keys + i * width, width) >> shift) & mask];
        return;
    }
    for (size_t i = 0; i < n; ++i)
        ++lanes[(i & 3) * fanout + ((routeWordBytes(keys + i * width, width) >> shift) & mask)];
}

template <typename Counter>
void histogramKeyChunkImpl(
    size_t kw, const char * keys, size_t n, UInt32 shift, UInt32 mask, Counter * hist, Counter * lanes, size_t fanout)
{
    switch (kw)
    {
        case 4: histogramKeyT<4>(keys, n, shift, mask, hist, lanes, fanout); break;
        case 8: histogramKeyT<8>(keys, n, shift, mask, hist, lanes, fanout); break;
        case 16: histogramKeyT<16>(keys, n, shift, mask, hist, lanes, fanout); break;
        default: histogramKeyGeneric(keys, kw, n, shift, mask, hist, lanes, fanout); break;
    }
}

/// Histogram one chunk's rows from precomputed route words (composite-key mode).
template <typename Counter>
void histogramRouteChunkImpl(const UInt32 * routes, size_t n, UInt32 shift, UInt32 mask, Counter * hist, Counter * lanes, size_t fanout)
{
    if (!lanes)
    {
        for (size_t i = 0; i < n; ++i)
            ++hist[(routes[i] >> shift) & mask];
        return;
    }
    size_t i = 0;
    for (; i + 4 <= n; i += 4)
    {
        ++lanes[0 * fanout + ((routes[i + 0] >> shift) & mask)];
        ++lanes[1 * fanout + ((routes[i + 1] >> shift) & mask)];
        ++lanes[2 * fanout + ((routes[i + 2] >> shift) & mask)];
        ++lanes[3 * fanout + ((routes[i + 3] >> shift) & mask)];
    }
    for (; i < n; ++i)
        ++lanes[(i & 3) * fanout + ((routes[i] >> shift) & mask)];
}

/// Histogram one chunk's rows from precomputed pids (no shift/mask: pids are final).
template <typename Pid, typename Counter>
void histogramPidChunkImpl(const Pid * pids, size_t n, Counter * hist, Counter * lanes, size_t fanout)
{
    if (!lanes)
    {
        for (size_t i = 0; i < n; ++i)
            ++hist[pids[i]];
        return;
    }
    size_t i = 0;
    for (; i + 4 <= n; i += 4)
    {
        ++lanes[0 * fanout + pids[i + 0]];
        ++lanes[1 * fanout + pids[i + 1]];
        ++lanes[2 * fanout + pids[i + 2]];
        ++lanes[3 * fanout + pids[i + 3]];
    }
    for (; i < n; ++i)
        ++lanes[(i & 3) * fanout + pids[i]];
}

template <typename Counter>
void reduceHistogramLanesImpl(Counter * hist, const Counter * lanes, size_t fanout)
{
    for (size_t p = 0; p < fanout; ++p)
        hist[p] += lanes[0 * fanout + p] + lanes[1 * fanout + p] + lanes[2 * fanout + p] + lanes[3 * fanout + p];
}

/// ------------------------------------------------------------------------------------------------
/// Introspection state
/// ------------------------------------------------------------------------------------------------

thread_local DispatchTrace * dispatch_trace = nullptr;

ALWAYS_INLINE void traceDispatch(TypeIndex type, ScatterKernelId kernel)
{
    if (dispatch_trace) [[unlikely]]
        dispatch_trace->entries.push_back({type, kernel});
}

/// ------------------------------------------------------------------------------------------------
/// Layer-1 typed kernels
/// ------------------------------------------------------------------------------------------------

template <typename Pid>
using SourcePids = std::span<const std::span<const Pid>>;

template <typename Pid>
using ScatterKernel = MutableColumns (*)(std::span<const IColumn * const>, SourcePids<Pid>, std::span<const UInt32>);

/// Raw-byte kernel for every fixed-and-contiguous type with insertRawUninitialized support
/// (ColumnVector, ColumnDecimal, ColumnFixedString). One runtime-width body; the per-chunk width
/// switch inside scatterPidChunkImpl selects the compile-time kernel.
template <typename Pid>
MutableColumns scatterFixedWidth(std::span<const IColumn * const> sources, SourcePids<Pid> pids, std::span<const UInt32> rows_per_shard)
{
    const IColumn & sample = *sources[0];
    const size_t width = sample.sizeOfValueIfFixed();
    const size_t num_shards = rows_per_shard.size();
    const bool use_swwc = num_shards >= SWWC_MIN_FANOUT && widthSupportsSwwc(width);

    /// `getDataType` equality (checked at the entry) cannot distinguish FixedString widths; a
    /// mismatch here would stride source bytes wrongly and corrupt every shard silently.
    for (size_t b = 1; b < sources.size(); ++b)
        if (sources[b]->sizeOfValueIfFixed() != width)
            throw Exception(
                ErrorCodes::LOGICAL_ERROR,
                "Source column {} has value width {} but source 0 has width {}",
                b,
                sources[b]->sizeOfValueIfFixed(),
                width);

    MutableColumns result(num_shards);
    ScatterScratch scratch;
    scratch.init(num_shards, use_swwc);
    for (size_t s = 0; s < num_shards; ++s)
    {
        auto [column, raw] = allocateUninitializedFixed(sample, rows_per_shard[s]);
        scratch.seed(s, raw.data());
        result[s] = std::move(column);
    }

    for (size_t b = 0; b < sources.size(); ++b)
        scatterPidChunkImpl(width, pids[b].data(), sources[b]->getRawData().data(), sources[b]->size(), use_swwc, scratch);

    scratch.drain();
    return result;
}

/// Fallback: delegate to legacy IColumn::scatter per source, merging with insertRangeFrom. Preserves
/// each type's legacy scatter semantics (LowCardinality keeps its physical type and shares one
/// dictionary; ColumnAggregateFunction results view the source arena). Sources must be normalized.
template <typename Pid>
MutableColumns scatterFallback(std::span<const IColumn * const> sources, SourcePids<Pid> pids, std::span<const UInt32> rows_per_shard)
{
    const size_t num_shards = rows_per_shard.size();

    IColumn::Selector selector;
    if (sources.size() == 1)
    {
        selector.resize_exact(pids[0].size());
        for (size_t j = 0; j < pids[0].size(); ++j)
            selector[j] = pids[0][j];
        auto parts = sources[0]->scatter(num_shards, selector);
        MutableColumns result(num_shards);
        for (size_t s = 0; s < num_shards; ++s)
            result[s] = std::move(parts[s]);
        return result;
    }

    MutableColumns result(num_shards);
    for (size_t s = 0; s < num_shards; ++s)
    {
        result[s] = sources[0]->cloneEmpty();
        if (rows_per_shard[s])
            result[s]->reserve(rows_per_shard[s]);
    }
    for (size_t b = 0; b < sources.size(); ++b)
    {
        selector.resize_exact(pids[b].size());
        for (size_t j = 0; j < pids[b].size(); ++j)
            selector[j] = pids[b][j];
        auto parts = sources[b]->scatter(num_shards, selector);
        for (size_t s = 0; s < num_shards; ++s)
            if (parts[s]->size())
                result[s]->insertRangeFrom(*parts[s], 0, parts[s]->size());
    }
    return result;
}

/// ------------------------------------------------------------------------------------------------
/// Dispatch table: TypeIndex -> kernel. Sized by the underlying type so indexing is unconditionally
/// in bounds; unregistered types (and LowCardinality, deliberately) take the fallback.
/// ------------------------------------------------------------------------------------------------

constexpr size_t SCATTER_TABLE_SIZE = static_cast<size_t>(std::numeric_limits<std::underlying_type_t<TypeIndex>>::max()) + 1;

constexpr std::array<TypeIndex, 25> FIXED_WIDTH_TYPES = {
    TypeIndex::UInt8,     TypeIndex::UInt16,   TypeIndex::UInt32,   TypeIndex::UInt64,     TypeIndex::UInt128, TypeIndex::UInt256,
    TypeIndex::Int8,      TypeIndex::Int16,    TypeIndex::Int32,    TypeIndex::Int64,      TypeIndex::Int128,  TypeIndex::Int256,
    TypeIndex::BFloat16,  TypeIndex::Float32,  TypeIndex::Float64,  TypeIndex::UUID,       TypeIndex::IPv4,    TypeIndex::IPv6,
    TypeIndex::Decimal32, TypeIndex::Decimal64, TypeIndex::Decimal128, TypeIndex::Decimal256, TypeIndex::DateTime64, TypeIndex::Time64,
    TypeIndex::FixedString};

/// The kernel-id table is the single source of registration truth; the function-pointer table is
/// DERIVED from it, so the recorded trace equals the executed kernel by construction (a new type
/// family in U2+ is a one-place edit: register the id here, map it in kernelForId).
constexpr std::array<ScatterKernelId, SCATTER_TABLE_SIZE> buildKernelIdTable()
{
    std::array<ScatterKernelId, SCATTER_TABLE_SIZE> table{};
    table.fill(ScatterKernelId::Fallback);
    for (TypeIndex type : FIXED_WIDTH_TYPES)
        table[static_cast<size_t>(type)] = ScatterKernelId::FixedWidth;
    return table;
}

constexpr auto KERNEL_ID_TABLE = buildKernelIdTable();

template <typename Pid>
constexpr ScatterKernel<Pid> kernelForId(ScatterKernelId id)
{
    switch (id)
    {
        case ScatterKernelId::FixedWidth:
            return &scatterFixedWidth<Pid>;
        case ScatterKernelId::ConstCompact: /// not a dispatch-table kernel: handled before dispatch
        case ScatterKernelId::Fallback:
            return &scatterFallback<Pid>;
    }
    UNREACHABLE();
}

template <typename Pid>
constexpr std::array<ScatterKernel<Pid>, SCATTER_TABLE_SIZE> buildScatterTable()
{
    std::array<ScatterKernel<Pid>, SCATTER_TABLE_SIZE> table{};
    for (size_t i = 0; i < SCATTER_TABLE_SIZE; ++i)
        table[i] = kernelForId<Pid>(KERNEL_ID_TABLE[i]);
    return table;
}

/// ------------------------------------------------------------------------------------------------
/// Wrapper normalization: strip ColumnConst/ColumnReplicated/ColumnSparse at every nesting level,
/// preserving ColumnLowCardinality as a leaf (its legacy scatter is type-preserving and O(indexes)).
/// Mirrors `IColumn::convertToFullIfNeeded` minus the LowCardinality conversion.
/// ------------------------------------------------------------------------------------------------

bool hasAnySubcolumn(const IColumn & column)
{
    bool found = false;
    column.forEachSubcolumn([&](const auto &) { found = true; });
    return found;
}

/// Conservative: any column WITH subcolumns takes the recursive path, because a composite may hide
/// a `ColumnConst`/`ColumnSparse`/`ColumnReplicated` at a nesting level a top-level probe cannot
/// see. The probe is deliberately generic (any subcolumn) rather than a per-type list — a
/// hand-maintained type switch is exactly the thing that silently misses a newly added composite.
/// Clean leaf batches (the hot case) skip normalization entirely.
bool mayNeedNormalization(const IColumn & column)
{
    return column.isConst() || column.isSparse() || column.isReplicated() || hasAnySubcolumn(column);
}

ColumnPtr normalizeRepresentation(const ColumnPtr & column)
{
    ColumnPtr converted
        = column->convertToFullColumnIfConst()->convertToFullColumnIfReplicated()->convertToFullColumnIfSparse();

    /// LowCardinality is a preserved leaf: its own kernel/fallback keeps the physical type; its
    /// dictionary must not be rewritten here.
    if (converted->getDataType() == TypeIndex::LowCardinality)
        return converted;

    Columns new_subcolumns;
    bool any_changed = false;
    converted->forEachSubcolumn(
        [&](const IColumn::WrappedPtr & subcolumn)
        {
            auto normalized = normalizeRepresentation(subcolumn);
            any_changed |= (normalized.get() != subcolumn.get());
            new_subcolumns.push_back(std::move(normalized));
        });

    if (!any_changed)
        return converted;

    auto mutable_column = IColumn::mutate(std::move(converted));
    size_t i = 0;
    mutable_column->forEachMutableSubcolumn([&](IColumn::WrappedPtr & subcolumn) { subcolumn = std::move(new_subcolumns[i++]); });
    return std::move(mutable_column);
}

/// ------------------------------------------------------------------------------------------------
/// Layer-1 entry
/// ------------------------------------------------------------------------------------------------

/// All-const batches with byte-identical values stay compact: `cloneResized` per shard, O(1)
/// memory. Equality must be byte-exact (`serializeValueIntoArena`), never `compareAt`: a physical
/// split must preserve exact bytes (+0.0 vs -0.0, NaN payloads). Returns empty when the
/// optimization does not apply (distinct values, or non-serializable nested values with more than
/// one source).
MutableColumns tryScatterAllConst(std::span<const IColumn * const> sources, std::span<const UInt32> rows_per_shard)
{
    const auto & first = assert_cast<const ColumnConst &>(*sources[0]);
    if (sources.size() > 1)
    {
        Arena arena;
        const char * ref_begin = nullptr;
        std::string_view ref;
        try
        {
            ref = first.getDataColumn().serializeValueIntoArena(0, arena, ref_begin, nullptr);
            for (size_t b = 1; b < sources.size(); ++b)
            {
                const auto & other = assert_cast<const ColumnConst &>(*sources[b]);
                const char * begin = nullptr;
                std::string_view serialized = other.getDataColumn().serializeValueIntoArena(0, arena, begin, nullptr);
                if (serialized != ref)
                    return {};
            }
        }
        catch (const Exception & e)
        {
            /// Values that cannot be serialized (e.g. ColumnFunction) cannot be compared byte-exactly:
            /// skip the optimization rather than fail the scatter.
            if (e.code() == ErrorCodes::NOT_IMPLEMENTED)
                return {};
            throw;
        }
    }

    MutableColumns result(rows_per_shard.size());
    for (size_t s = 0; s < rows_per_shard.size(); ++s)
        result[s] = first.cloneResized(rows_per_shard[s]);
    return result;
}

template <typename Pid>
void countRowsPerShardImpl(SourcePids<Pid> pids_per_source, std::span<UInt32> rows_per_shard)
{
    const size_t num_shards = rows_per_shard.size();
    const bool interleave = num_shards <= HIST_INTERLEAVE_MAX_FANOUT;
    PaddedPODArray<UInt32> lanes;
    if (interleave)
    {
        lanes.resize(4 * num_shards);
        memset(lanes.data(), 0, 4 * num_shards * sizeof(UInt32));
    }
    for (const auto & pids : pids_per_source)
        histogramPidChunkImpl(pids.data(), pids.size(), rows_per_shard.data(), interleave ? lanes.data() : nullptr, num_shards);
    if (interleave)
        reduceHistogramLanesImpl(rows_per_shard.data(), lanes.data(), num_shards);
}

template <typename Pid>
MutableColumns scatterImpl(
    std::span<const IColumn * const> sources, SourcePids<Pid> pids_per_source, size_t num_shards, std::span<const UInt32> rows_per_shard)
{
    if (sources.empty())
        throw Exception(ErrorCodes::LOGICAL_ERROR, "Cannot scatter an empty batch of source columns");
    if (sources.size() != pids_per_source.size())
        throw Exception(
            ErrorCodes::LOGICAL_ERROR,
            "Number of source columns ({}) does not match number of pid spans ({})",
            sources.size(),
            pids_per_source.size());
    if (num_shards == 0)
        throw Exception(ErrorCodes::LOGICAL_ERROR, "Cannot scatter into zero shards");
    if (!rows_per_shard.empty() && rows_per_shard.size() != num_shards)
        throw Exception(
            ErrorCodes::LOGICAL_ERROR,
            "Size of rows_per_shard ({}) does not match num_shards ({})",
            rows_per_shard.size(),
            num_shards);

    size_t total_rows = 0;
    for (size_t b = 0; b < sources.size(); ++b)
    {
        if (sources[b]->size() != pids_per_source[b].size())
            throw Exception(
                ErrorCodes::LOGICAL_ERROR,
                "Source column {} has {} rows but {} partition ids",
                b,
                sources[b]->size(),
                pids_per_source[b].size());
        /// One virtual call per chunk; violating the same-concrete-type contract would otherwise
        /// silently produce wrong results in the raw-byte kernels.
        if (sources[b]->getDataType() != sources[0]->getDataType())
            throw Exception(
                ErrorCodes::LOGICAL_ERROR,
                "Source column {} has type {} but source 0 has type {}",
                b,
                sources[b]->getName(),
                sources[0]->getName());
        total_rows += pids_per_source[b].size();
#ifdef DEBUG_OR_SANITIZER_BUILD
        for (const Pid pid : pids_per_source[b])
            chassert(static_cast<size_t>(pid) < num_shards);
#endif
    }

#ifdef DEBUG_OR_SANITIZER_BUILD
    /// Caller-provided rows_per_shard drives exact-sized destination allocation that the kernels
    /// then treat as write cursors: an undercount is a heap overflow in release builds. Verify the
    /// contents here by recounting (after the pid-range chasserts above, so the recount itself
    /// cannot index out of bounds).
    if (!rows_per_shard.empty())
    {
        PaddedPODArray<UInt32> recounted;
        recounted.resize(num_shards);
        memset(recounted.data(), 0, num_shards * sizeof(UInt32));
        countRowsPerShardImpl<Pid>(pids_per_source, {recounted.data(), num_shards});
        for (size_t s = 0; s < num_shards; ++s)
            chassert(recounted[s] == rows_per_shard[s]);
    }
#endif

    /// The fast kernels and the compact-const path size destinations from UInt32 counts.
    const bool fits_32 = total_rows <= std::numeric_limits<UInt32>::max();

    PaddedPODArray<UInt32> counted;
    auto ensure_counts = [&]() -> std::span<const UInt32>
    {
        if (!rows_per_shard.empty())
            return rows_per_shard;
        if (counted.empty())
        {
            counted.resize(num_shards);
            memset(counted.data(), 0, num_shards * sizeof(UInt32));
            countRowsPerShardImpl<Pid>(pids_per_source, {counted.data(), num_shards});
        }
        return {counted.data(), num_shards};
    };

    if (fits_32)
    {
        bool all_const = true;
        for (const IColumn * source : sources)
            all_const &= source->isConst();
        if (all_const)
        {
            auto compact = tryScatterAllConst(sources, ensure_counts());
            if (!compact.empty())
            {
                traceDispatch(sources[0]->getDataType(), ScatterKernelId::ConstCompact);
                return compact;
            }
        }
    }

    /// Normalize transparent wrappers once, at the boundary; kernels and the fallback assume
    /// wrapper-free input at every nesting level.
    Columns normalized_holder;
    ColumnRawPtrs normalized_sources;
    bool any_needs_normalization = false;
    for (const IColumn * source : sources)
        any_needs_normalization |= mayNeedNormalization(*source);
    if (any_needs_normalization)
    {
        normalized_holder.reserve(sources.size());
        normalized_sources.reserve(sources.size());
        for (const IColumn * source : sources)
        {
            normalized_holder.push_back(normalizeRepresentation(source->getPtr()));
            normalized_sources.push_back(normalized_holder.back().get());
        }
        sources = std::span<const IColumn * const>(normalized_sources.data(), normalized_sources.size());
    }

    const TypeIndex type = sources[0]->getDataType();

    if (!fits_32)
    {
        /// size_t-safe path: the legacy scatter sizes each destination itself from 64-bit counts;
        /// skip the UInt32 counting entirely (zero counts only suppress the reserve).
        traceDispatch(type, ScatterKernelId::Fallback);
        PaddedPODArray<UInt32> zero_counts;
        zero_counts.resize_fill(num_shards, 0);
        return scatterFallback<Pid>(sources, pids_per_source, std::span<const UInt32>(zero_counts.data(), num_shards));
    }

    static constexpr auto table = buildScatterTable<Pid>();
    const ScatterKernelId kernel_id = KERNEL_ID_TABLE[static_cast<size_t>(type)];
    traceDispatch(type, kernel_id);
    return table[static_cast<size_t>(type)](sources, pids_per_source, ensure_counts());
}

}

/// ------------------------------------------------------------------------------------------------
/// Layer-0 exported wrappers
/// ------------------------------------------------------------------------------------------------

void scatterKeyChunk(
    size_t key_width, const char * keys, size_t n, UInt32 shift, UInt32 mask, UInt16 * pids_out, bool use_swwc, ScatterScratch & scratch)
{
    scatterKeyChunkImpl(key_width, keys, n, shift, mask, pids_out, use_swwc, scratch);
}

void scatterKeyChunk(
    size_t key_width, const char * keys, size_t n, UInt32 shift, UInt32 mask, UInt32 * pids_out, bool use_swwc, ScatterScratch & scratch)
{
    scatterKeyChunkImpl(key_width, keys, n, shift, mask, pids_out, use_swwc, scratch);
}

void scatterPidChunk(size_t width, const UInt16 * pids, const char * data, size_t n, bool use_swwc, ScatterScratch & scratch)
{
    scatterPidChunkImpl(width, pids, data, n, use_swwc, scratch);
}

void scatterPidChunk(size_t width, const UInt32 * pids, const char * data, size_t n, bool use_swwc, ScatterScratch & scratch)
{
    scatterPidChunkImpl(width, pids, data, n, use_swwc, scratch);
}

void histogramKeyChunk(size_t key_width, const char * keys, size_t n, UInt32 shift, UInt32 mask, UInt32 * hist, UInt32 * lanes, size_t fanout)
{
    histogramKeyChunkImpl(key_width, keys, n, shift, mask, hist, lanes, fanout);
}

void histogramKeyChunk(size_t key_width, const char * keys, size_t n, UInt32 shift, UInt32 mask, UInt64 * hist, UInt64 * lanes, size_t fanout)
{
    histogramKeyChunkImpl(key_width, keys, n, shift, mask, hist, lanes, fanout);
}

void histogramRouteChunk(const UInt32 * routes, size_t n, UInt32 shift, UInt32 mask, UInt32 * hist, UInt32 * lanes, size_t fanout)
{
    histogramRouteChunkImpl(routes, n, shift, mask, hist, lanes, fanout);
}

void histogramRouteChunk(const UInt32 * routes, size_t n, UInt32 shift, UInt32 mask, UInt64 * hist, UInt64 * lanes, size_t fanout)
{
    histogramRouteChunkImpl(routes, n, shift, mask, hist, lanes, fanout);
}

void histogramPidChunk(const UInt16 * pids, size_t n, UInt32 * hist, UInt32 * lanes, size_t fanout)
{
    histogramPidChunkImpl(pids, n, hist, lanes, fanout);
}

void histogramPidChunk(const UInt16 * pids, size_t n, UInt64 * hist, UInt64 * lanes, size_t fanout)
{
    histogramPidChunkImpl(pids, n, hist, lanes, fanout);
}

void histogramPidChunk(const UInt32 * pids, size_t n, UInt32 * hist, UInt32 * lanes, size_t fanout)
{
    histogramPidChunkImpl(pids, n, hist, lanes, fanout);
}

void histogramPidChunk(const UInt32 * pids, size_t n, UInt64 * hist, UInt64 * lanes, size_t fanout)
{
    histogramPidChunkImpl(pids, n, hist, lanes, fanout);
}

void reduceHistogramLanes(UInt32 * hist, const UInt32 * lanes, size_t fanout)
{
    reduceHistogramLanesImpl(hist, lanes, fanout);
}

void reduceHistogramLanes(UInt64 * hist, const UInt64 * lanes, size_t fanout)
{
    reduceHistogramLanesImpl(hist, lanes, fanout);
}

std::pair<MutableColumnPtr, std::span<char>> allocateUninitializedFixed(const IColumn & sample, size_t rows)
{
    auto column = sample.cloneEmpty();
    auto raw = column->insertRawUninitialized(rows);
    chassert(raw.size() == rows * sample.sizeOfValueIfFixed());
    return {std::move(column), raw};
}

/// ------------------------------------------------------------------------------------------------
/// Introspection
/// ------------------------------------------------------------------------------------------------

const char * toString(ScatterKernelId id)
{
    switch (id)
    {
        case ScatterKernelId::FixedWidth: return "FixedWidth";
        case ScatterKernelId::ConstCompact: return "ConstCompact";
        case ScatterKernelId::Fallback: return "Fallback";
    }
    UNREACHABLE();
}

ScatterKernelId plannedKernel(const IColumn & column)
{
    return KERNEL_ID_TABLE[static_cast<size_t>(column.getDataType())];
}

DispatchTrace * exchangeDispatchTrace(DispatchTrace * trace)
{
    return std::exchange(dispatch_trace, trace);
}

/// ------------------------------------------------------------------------------------------------
/// Layer 1
/// ------------------------------------------------------------------------------------------------

void countRowsPerShard(std::span<const std::span<const UInt16>> pids_per_source, std::span<UInt32> rows_per_shard)
{
    countRowsPerShardImpl(pids_per_source, rows_per_shard);
}

void countRowsPerShard(std::span<const std::span<const UInt32>> pids_per_source, std::span<UInt32> rows_per_shard)
{
    countRowsPerShardImpl(pids_per_source, rows_per_shard);
}

MutableColumns scatter(
    std::span<const IColumn * const> source_columns,
    std::span<const std::span<const UInt16>> pids_per_source,
    size_t num_shards,
    std::span<const UInt32> rows_per_shard)
{
    return scatterImpl(source_columns, pids_per_source, num_shards, rows_per_shard);
}

MutableColumns scatter(
    std::span<const IColumn * const> source_columns,
    std::span<const std::span<const UInt32>> pids_per_source,
    size_t num_shards,
    std::span<const UInt32> rows_per_shard)
{
    return scatterImpl(source_columns, pids_per_source, num_shards, rows_per_shard);
}

}
