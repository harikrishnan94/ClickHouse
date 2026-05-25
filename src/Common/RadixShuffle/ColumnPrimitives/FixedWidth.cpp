#include <Common/RadixShuffle/ColumnPrimitives/FixedWidth.h>

#include <Columns/ColumnDecimal.h>
#include <Columns/ColumnFixedString.h>
#include <Columns/ColumnVector.h>
#include <Common/RadixShuffle/HashCombiner.h>
#include <Common/RadixShuffle/HashKernels.h>
#include <Common/TargetSpecific.h>
#include <Common/assert_cast.h>

#if defined(__x86_64__)
#    include <immintrin.h>
#endif

#include <cstdlib>
#include <cstring>
#include <new>

namespace DB::RadixShuffle
{

namespace
{

/// Partition threshold below which scatter uses a stack-allocated typed
/// write-pointer array (fast path, stays L1-resident).  For partitions
/// above this threshold scatter works directly through the char* pointers
/// in ScatterState::fixed_ptrs (no stack VLA, slightly slower per row).
constexpr size_t SCATTER_STACK_PTRS = 1024;


/// Refresh the fixed-chunk write pointer for partition p.
/// elem_size is in bytes (sizeof(T) for fixed-width, n for FixedString).
[[gnu::always_inline]] inline void refreshFixedPtr(void *& ptr, size_t p, size_t slot_idx, const PartReservation * dst, size_t elem_size)
{
    if (dst[p].fixed != nullptr)
    {
        const size_t slot_off = dst[p].fixed->slot_byte_offsets[slot_idx];
        ptr = static_cast<char *>(dst[p].fixed->data) + slot_off + dst[p].begin_row * elem_size;
    }
    else
    {
        ptr = nullptr;
    }
}


/// Refresh all stale or uninitialised fixed-chunk write pointers.
/// On the first call (state.initialized == false) every partition is
/// refreshed with an O(P) loop.  On subsequent calls only the stale
/// partitions (flagged in stale_fixed_bitset) are touched.
template <size_t ElemSize>
[[gnu::always_inline]] inline void
refreshFixedPtrs(ScatterState & state, size_t slot_idx, size_t partitions, const PartReservation * dst, const uint64_t * stale_fixed_bitset)
{
    if (!state.initialized)
    {
        for (size_t p = 0; p < partitions; ++p)
            refreshFixedPtr(state.fixed_ptrs[p], p, slot_idx, dst, ElemSize);
        state.initialized = true;
        return;
    }

    // Selective refresh: walk only set bits in the stale bitset.
    const size_t words = (partitions + 63) / 64;
    for (size_t word = 0; word < words; ++word)
    {
        uint64_t bits = stale_fixed_bitset[word];
        while (bits)
        {
            const size_t bit = static_cast<size_t>(__builtin_ctzll(bits));
            const size_t p = word * 64 + bit;
            if (p < partitions)
                refreshFixedPtr(state.fixed_ptrs[p], p, slot_idx, dst, ElemSize);
            bits &= bits - 1;
        }
    }
}


template <typename T>
[[gnu::hot]] void scatterFixed(
    const ColumnPrimitives & self,
    const PartSchema & /*schema*/,
    const IColumn & src_,
    const uint16_t * pids,
    size_t n,
    size_t partitions,
    const PartReservation * dst,
    ScatterState & state,
    const uint64_t * stale_fixed_bitset)
{
    const auto & col = assert_cast<const ColumnVector<T> &>(src_);
    const T * src = col.getData().data();

    const size_t slot_idx = self.fixed_slot_indices[0];

    refreshFixedPtrs<sizeof(T)>(state, slot_idx, partitions, dst, stale_fixed_bitset);

    if (partitions <= SCATTER_STACK_PTRS)
    {
        // Fast path: stack-allocated typed pointer array — stays L1-resident.
        T * ptrs[SCATTER_STACK_PTRS];
        for (size_t p = 0; p < partitions; ++p)
            ptrs[p] = static_cast<T *>(state.fixed_ptrs[p]);
        for (size_t j = 0; j < n; ++j)
            *ptrs[pids[j]]++ = src[j];
        for (size_t p = 0; p < partitions; ++p)
            state.fixed_ptrs[p] = ptrs[p];
    }
    else
    {
        // Large-P fallback: work directly through char* to avoid a stack VLA.
        for (size_t j = 0; j < n; ++j)
        {
            const uint16_t p = pids[j];
            std::memcpy(state.fixed_ptrs[p], &src[j], sizeof(T));
            state.fixed_ptrs[p] = static_cast<char *>(state.fixed_ptrs[p]) + sizeof(T);
        }
    }
}


[[gnu::hot]] void scatterFixedString(
    const ColumnPrimitives & self,
    const PartSchema & /*schema*/,
    const IColumn & src_,
    const uint16_t * pids,
    size_t n_rows,
    size_t partitions,
    const PartReservation * dst,
    ScatterState & state,
    const uint64_t * stale_fixed_bitset)
{
    const auto & col = assert_cast<const ColumnFixedString &>(src_);
    const size_t n = col.getN();
    const auto * src = reinterpret_cast<const unsigned char *>(col.getChars().data());

    const size_t slot_idx = self.fixed_slot_indices[0];

    // Element size is dynamic (n), so we cannot use the compile-time template.
    if (!state.initialized)
    {
        for (size_t p = 0; p < partitions; ++p)
            refreshFixedPtr(state.fixed_ptrs[p], p, slot_idx, dst, n);
        state.initialized = true;
    }
    else
    {
        const size_t words = (partitions + 63) / 64;
        for (size_t word = 0; word < words; ++word)
        {
            uint64_t bits = stale_fixed_bitset[word];
            while (bits)
            {
                const size_t bit = static_cast<size_t>(__builtin_ctzll(bits));
                const size_t p = word * 64 + bit;
                if (p < partitions)
                    refreshFixedPtr(state.fixed_ptrs[p], p, slot_idx, dst, n);
                bits &= bits - 1;
            }
        }
    }

    if (partitions <= SCATTER_STACK_PTRS)
    {
        unsigned char * ptrs[SCATTER_STACK_PTRS];
        for (size_t p = 0; p < partitions; ++p)
            ptrs[p] = static_cast<unsigned char *>(state.fixed_ptrs[p]);
        for (size_t j = 0; j < n_rows; ++j)
        {
            unsigned char * out = ptrs[pids[j]];
            std::memcpy(out, src + j * n, n);
            ptrs[pids[j]] = out + n;
        }
        for (size_t p = 0; p < partitions; ++p)
            state.fixed_ptrs[p] = ptrs[p];
    }
    else
    {
        for (size_t j = 0; j < n_rows; ++j)
        {
            const uint16_t p = pids[j];
            std::memcpy(state.fixed_ptrs[p], src + j * n, n);
            state.fixed_ptrs[p] = static_cast<char *>(state.fixed_ptrs[p]) + n;
        }
    }
}


template <typename T>
[[gnu::hot]] void scatterDecimal(
    const ColumnPrimitives & self,
    const PartSchema & /*schema*/,
    const IColumn & src_,
    const uint16_t * pids,
    size_t n,
    size_t partitions,
    const PartReservation * dst,
    ScatterState & state,
    const uint64_t * stale_fixed_bitset)
{
    const auto & col = assert_cast<const ColumnDecimal<T> &>(src_);
    const T * src = col.getData().data();

    const size_t slot_idx = self.fixed_slot_indices[0];

    refreshFixedPtrs<sizeof(T)>(state, slot_idx, partitions, dst, stale_fixed_bitset);

    if (partitions <= SCATTER_STACK_PTRS)
    {
        T * ptrs[SCATTER_STACK_PTRS];
        for (size_t p = 0; p < partitions; ++p)
            ptrs[p] = static_cast<T *>(state.fixed_ptrs[p]);
        for (size_t j = 0; j < n; ++j)
            *ptrs[pids[j]]++ = src[j];
        for (size_t p = 0; p < partitions; ++p)
            state.fixed_ptrs[p] = ptrs[p];
    }
    else
    {
        for (size_t j = 0; j < n; ++j)
        {
            const uint16_t p = pids[j];
            std::memcpy(state.fixed_ptrs[p], &src[j], sizeof(T));
            state.fixed_ptrs[p] = static_cast<char *>(state.fixed_ptrs[p]) + sizeof(T);
        }
    }
}


template <typename T>
ResumePosition reconstructFixed(
    const ColumnPrimitives & self,
    const PartSchema & /*schema*/,
    const PartReservationView * views,
    size_t n_views,
    ResumePosition start,
    IColumn & target)
{
    auto & col = assert_cast<ColumnVector<T> &>(target);
    auto & data = col.getData();
    const size_t cap = data.capacity();
    size_t cur = data.size();

    const size_t slot_idx = self.fixed_slot_indices[0];

    size_t vi = start.view_index;
    size_t in_view = start.rows_consumed_in_view;
    while (vi < n_views)
    {
        const PartReservationView & v = views[vi];
        const size_t view_rows = v.row_end - v.row_begin;
        const size_t available = view_rows - in_view;
        const size_t room = cap - cur;
        if (room == 0)
            break;
        const size_t take = std::min(available, room);

        const size_t slot_off = v.fixed->slot_byte_offsets[slot_idx];
        const T * src = reinterpret_cast<const T *>(static_cast<const char *>(v.fixed->data) + slot_off) + v.row_begin + in_view;

        data.resize_assume_reserved(cur + take);
        std::memcpy(data.data() + cur, src, take * sizeof(T));
        cur += take;

        in_view += take;
        if (in_view == view_rows)
        {
            ++vi;
            in_view = 0;
        }
        else
        {
            break;
        }
    }
    return ResumePosition{vi, in_view};
}


template <typename T>
ResumePosition reconstructDecimal(
    const ColumnPrimitives & self,
    const PartSchema & /*schema*/,
    const PartReservationView * views,
    size_t n_views,
    ResumePosition start,
    IColumn & target)
{
    auto & col = assert_cast<ColumnDecimal<T> &>(target);
    auto & data = col.getData();
    const size_t cap = data.capacity();
    size_t cur = data.size();

    const size_t slot_idx = self.fixed_slot_indices[0];

    size_t vi = start.view_index;
    size_t in_view = start.rows_consumed_in_view;
    while (vi < n_views)
    {
        const PartReservationView & v = views[vi];
        const size_t view_rows = v.row_end - v.row_begin;
        const size_t available = view_rows - in_view;
        const size_t room = cap - cur;
        if (room == 0)
            break;
        const size_t take = std::min(available, room);

        const size_t slot_off = v.fixed->slot_byte_offsets[slot_idx];
        const T * src = reinterpret_cast<const T *>(static_cast<const char *>(v.fixed->data) + slot_off) + v.row_begin + in_view;

        data.resize_assume_reserved(cur + take);
        std::memcpy(data.data() + cur, src, take * sizeof(T));
        cur += take;

        in_view += take;
        if (in_view == view_rows)
        {
            ++vi;
            in_view = 0;
        }
        else
        {
            break;
        }
    }
    return ResumePosition{vi, in_view};
}


ResumePosition reconstructFixedString(
    const ColumnPrimitives & self,
    const PartSchema & /*schema*/,
    const PartReservationView * views,
    size_t n_views,
    ResumePosition start,
    IColumn & target)
{
    auto & col = assert_cast<ColumnFixedString &>(target);
    auto & chars = col.getChars();
    const size_t n = col.getN();
    const size_t cap_rows = chars.capacity() / n;
    size_t cur_rows = chars.size() / n;

    const size_t slot_idx = self.fixed_slot_indices[0];

    size_t vi = start.view_index;
    size_t in_view = start.rows_consumed_in_view;
    while (vi < n_views)
    {
        const PartReservationView & v = views[vi];
        const size_t view_rows = v.row_end - v.row_begin;
        const size_t available = view_rows - in_view;
        const size_t room = cap_rows - cur_rows;
        if (room == 0)
            break;
        const size_t take = std::min(available, room);

        const size_t slot_off = v.fixed->slot_byte_offsets[slot_idx];
        const auto * src = static_cast<const unsigned char *>(v.fixed->data) + slot_off + (v.row_begin + in_view) * n;

        chars.resize_assume_reserved((cur_rows + take) * n);
        auto * dst_ptr = reinterpret_cast<unsigned char *>(chars.data()) + cur_rows * n;
        std::memcpy(dst_ptr, src, take * n);
        cur_rows += take;

        in_view += take;
        if (in_view == view_rows)
        {
            ++vi;
            in_view = 0;
        }
        else
        {
            break;
        }
    }
    return ResumePosition{vi, in_view};
}


template <typename T>
void hashFixed(
    const ColumnPrimitives & /*self*/,
    const PartSchema & /*schema*/,
    const IColumn & src_,
    size_t offset,
    size_t n,
    bool initial,
    uint32_t * out)
{
    const T * data = assert_cast<const ColumnVector<T> &>(src_).getData().data() + offset;
    if (initial)
        hashBatch32Direct(data, static_cast<int>(n), out);
    else
        hashBatch32Combine(data, static_cast<int>(n), out);
}


template <typename T>
void hashDecimal(
    const ColumnPrimitives & /*self*/,
    const PartSchema & /*schema*/,
    const IColumn & src_,
    size_t offset,
    size_t n,
    bool initial,
    uint32_t * out)
{
    const auto & data = assert_cast<const ColumnDecimal<T> &>(src_).getData();
    for (size_t i = 0; i < n; ++i)
    {
        const uint32_t h = hashOne32(data[offset + i].value);
        out[i] = initial ? h : hashCombine(out[i], h);
    }
}


void hashFixedString(
    const ColumnPrimitives & /*self*/,
    const PartSchema & /*schema*/,
    const IColumn & src_,
    size_t offset,
    size_t n_rows,
    bool initial,
    uint32_t * out)
{
    const auto & col = assert_cast<const ColumnFixedString &>(src_);
    const size_t n = col.getN();
    const auto * data = reinterpret_cast<const unsigned char *>(col.getChars().data()) + offset * n;
    for (size_t i = 0; i < n_rows; ++i)
    {
        const uint32_t h = hashBytes32(data + i * n, n);
        out[i] = initial ? h : hashCombine(out[i], h);
    }
}

// ── NT-flush helpers for SWWC ─────────────────────────────────────────────────
// Each flush writes exactly 64 bytes (one cache line) regardless of T's size.
// kSlotsPerFlush<T> = 64 / sizeof(T): number of T-elements per flush.
//
//   sizeof=8  (uint64) → 8 slots    sizeof=4 (uint32) → 16 slots
//   sizeof=2  (uint16) → 32 slots   sizeof=1 (uint8)  → 64 slots
//   sizeof=16 (UUID)   → 4 slots    sizeof=32 (u256)  → 2 slots
//
// flushStagedNT/Scalar       — typed T*&  (NumericScatterColumn compat)
// flushStagedNTInPlace/…     — void*&     (raw scatter path, in-place update)

template <typename T>
static constexpr size_t kSlotsPerFlush = 64 / sizeof(T);

#if USE_MULTITARGET_CODE

DECLARE_X86_64_V4_SPECIFIC_CODE(

    template <typename T> inline void flushStagedNT(const T * staging_p, T *& out_p) // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)
    {
        _mm512_stream_si512(reinterpret_cast<__m512i *>(out_p), _mm512_load_si512(reinterpret_cast<const __m512i *>(staging_p)));
        out_p += kSlotsPerFlush<T>;
    }

    template <typename T>
    inline void flushStagedNTInPlace(const T * staging_p, void *& out_void) // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)
    {
        T * out_p = static_cast<T *>(out_void);
        _mm512_stream_si512(reinterpret_cast<__m512i *>(out_p), _mm512_load_si512(reinterpret_cast<const __m512i *>(staging_p)));
        out_void = out_p + kSlotsPerFlush<T>;
    }

    ) // DECLARE_X86_64_V4_SPECIFIC_CODE


DECLARE_X86_64_V3_SPECIFIC_CODE(

    template <typename T> inline void flushStagedNT(const T * staging_p, T *& out_p) // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)
    {
        const auto * s = reinterpret_cast<const __m256i *>(staging_p);
        auto * d = reinterpret_cast<__m256i *>(out_p);
        _mm256_stream_si256(d, _mm256_load_si256(s));
        _mm256_stream_si256(d + 1, _mm256_load_si256(s + 1));
        out_p += kSlotsPerFlush<T>;
    }

    template <typename T>
    inline void flushStagedNTInPlace(const T * staging_p, void *& out_void) // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)
    {
        T * out_p = static_cast<T *>(out_void);
        const auto * s = reinterpret_cast<const __m256i *>(staging_p);
        auto * d = reinterpret_cast<__m256i *>(out_p);
        _mm256_stream_si256(d, _mm256_load_si256(s));
        _mm256_stream_si256(d + 1, _mm256_load_si256(s + 1));
        out_void = out_p + kSlotsPerFlush<T>;
    }

    ) // DECLARE_X86_64_V3_SPECIFIC_CODE

#endif // USE_MULTITARGET_CODE

template <typename T>
[[gnu::always_inline]] inline void flushStagedScalar(const T * staging_p, T *& out_p)
{
    constexpr size_t S = kSlotsPerFlush<T>;
    for (size_t i = 0; i < S; ++i)
        out_p[i] = staging_p[i];
    out_p += S;
}

template <typename T>
[[gnu::always_inline]] inline void flushStagedScalarInPlace(const T * staging_p, void *& out_void)
{
    T * out_p = static_cast<T *>(out_void);
    constexpr size_t S = kSlotsPerFlush<T>;
    for (size_t i = 0; i < S; ++i)
        out_p[i] = staging_p[i];
    out_void = out_p + S;
}


// ── Direct pids computation (hash + mask in one SIMD pass) ───────────────────

/// Computes pids[j] = hash(src[offset+j]) & mask in one SIMD pass.
/// Delegates to hashBatch32 which has the same MULTITARGET ISA dispatch.
template <typename T>
void computePidsFixed(const ColumnPrimitives & /*self*/, const IColumn & src_, size_t offset, int n, uint32_t mask, uint32_t * pids)
{
    const T * data = assert_cast<const ColumnVector<T> &>(src_).getData().data() + offset;
    hashBatch32(data, n, mask, pids);
}


// ── Raw-output-pointer scatter (OutBlock model) ───────────────────────────────
// self + partitions are absent from all four signatures (see ColumnPrimitives.h).
// Dropping two parameters reduces the x86-64 register pressure so pids and
// positions stay in integer parameter registers (rdx, rcx) rather than being
// spilled to the stack, eliminating the 2 extra memory reads per row that were
// the primary regression vs. the baseline IScatterColumn path.

/// Direct scatter: reads IColumn[offset..offset+n), writes via raw_write_ptrs[p].
///
/// raw_write_ptrs is a persistent per-partition write-pointer array, refreshed
/// only when on_grow_raw allocates a new OutBlock for a partition.  Between
/// refreshes the pointers are advanced here, exactly as `UInt64Column::out_[]`
/// is advanced by `scatter_direct` in the reference implementation.
///
/// The array is allocated as T*[] and viewed through void** in ScatterState
/// (type erasure).  `reinterpret_cast<T**>` restores the original type so the
/// hot loop is verbatim `*wp[pids[j]]++ = src[j]` — no per-row cast, no
/// copy-in/copy-out, matching the reference pattern exactly.
template <typename T>
[[gnu::hot]] void scatterRawFixed(const IColumn & src_, size_t offset, const uint32_t * pids, int n, ScatterState & state)
{
    const T * src = assert_cast<const ColumnVector<T> &>(src_).getData().data() + offset;
    T ** wp = reinterpret_cast<T **>(state.raw_write_ptrs); // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)
    for (int j = 0; j < n; ++j)
        *wp[pids[j]]++ = src[j];
}


/// SWWC scatter.
///
/// Register allocation with 6 parameters (no self, no partitions):
///   rdi=src → compute r12=src_data (then src on stack, 1 load/row — same as baseline)
///   rsi=offset
///   rdx=pids → rbx (callee-saved) — NO STACK SPILL
///   rcx=positions → rbp (callee-saved) — NO STACK SPILL
///   r8=n → r14 (callee-saved)
///   r9=state → r13=raw_write_ptrs, r12=staging (loaded once, no vector.data())
///   r15 = j (loop counter)
///
/// Equivalent to the baseline `NumericScatterColumn::scatter_staged` where
/// `this->out_` (T**) → `raw_write_ptrs` (void**) and `this->staging_` (T*) →
/// the `staging` pointer loaded from `state.swwc_staging`.
template <typename T>
[[gnu::hot]] void
scatterRawSwwcFixed(const IColumn & src_, size_t offset, const uint32_t * pids, const uint32_t * positions, int n, ScatterState & state)
{
    // staging is pre-allocated by on_grow_raw — no lazy-init guard here.
    // kSlotsPerFlush<T> = 64/sizeof(T): flush exactly one 64-byte cache line.
    constexpr size_t S = kSlotsPerFlush<T>;
    constexpr uint32_t flush_trigger = static_cast<uint32_t>(S) - 1;

    const T * src = assert_cast<const ColumnVector<T> &>(src_).getData().data() + offset;

    // Access staging and raw_write_ptrs via state in the loop body so the
    // compiler keeps `state` in one callee-saved register — same pattern as the
    // baseline's `this` giving access to `staging_` and `out_[]` via offsets.
    for (int j = 0; j < n; ++j)
    {
        const uint32_t p = pids[j];
        // positions[j] is a raw row counter (uint8_t, wraps at 256).
        // Mask to this column's slot range so each column type flushes at the
        // correct granularity regardless of what other columns have in the batch.
        const uint32_t slot = positions[j] & flush_trigger;
        reinterpret_cast<T *>(state.swwc_staging)[p * S + slot] = src[j];
        if (slot == flush_trigger)
        {
#if USE_MULTITARGET_CODE
            if (isArchSupported(TargetArch::x86_64_v4))
                TargetSpecific::x86_64_v4::flushStagedNTInPlace(reinterpret_cast<T *>(state.swwc_staging) + p * S, state.raw_write_ptrs[p]);
            else if (isArchSupported(TargetArch::x86_64_v3))
                TargetSpecific::x86_64_v3::flushStagedNTInPlace(reinterpret_cast<T *>(state.swwc_staging) + p * S, state.raw_write_ptrs[p]);
            else
                flushStagedScalarInPlace(reinterpret_cast<T *>(state.swwc_staging) + p * S, state.raw_write_ptrs[p]);
#else
            flushStagedScalarInPlace(reinterpret_cast<T *>(state.swwc_staging) + p * S, state.raw_write_ptrs[p]);
#endif
        }
    }
}


/// Drain: copies residual staged values for partition `p` to its output pointer.
/// `cnt` is the raw per-partition row counter from the operator; the actual
/// residual is `cnt & (kSlotsPerFlush<T> - 1)` — the elements staged since
/// the last full flush for this column type.
template <typename T>
void drainRawFixed(const ColumnPrimitives & /*self*/, size_t p, uint32_t cnt, ScatterState & state)
{
    if (state.swwc_staging == nullptr)
        return;
    constexpr size_t S = kSlotsPerFlush<T>;
    const uint32_t residual = cnt & static_cast<uint32_t>(S - 1);
    if (residual == 0)
        return;
    const T * s = reinterpret_cast<const T *>(state.swwc_staging) + p * S;
    T ** wp = reinterpret_cast<T **>(state.raw_write_ptrs); // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)
    T * dst = wp[p];
    for (uint32_t i = 0; i < residual; ++i)
        dst[i] = s[i];
    wp[p] = dst + residual;
}


/// Update the write pointer for partition `p` when a new output block is
/// allocated.  On the first call, also pre-allocates `raw_write_ptrs` and
/// `swwc_staging` so that `scatterRawSwwcFixed` needs no lazy-init guard.
///
/// Staging layout: P partitions × kSlotsPerFlush<T> × sizeof(T) = P × 64 bytes
/// (always one 64-byte cache line per partition, independent of T).
template <typename T>
void onGrowRawFixed(const ColumnPrimitives & /*self*/, size_t p, void * col_base, size_t /*capacity*/, ScatterState & state)
{
    if (state.raw_write_ptrs == nullptr)
    {
        const size_t P = state.fixed_ptrs.size();
        state.raw_write_ptrs = static_cast<void **>(std::calloc(P, sizeof(void *)));
        if (!state.raw_write_ptrs)
            throw std::bad_alloc{};
        // One 64-byte cache line per partition, 64-byte aligned for NT stores.
        if (posix_memalign(reinterpret_cast<void **>(&state.swwc_staging), 64, P * 64) != 0)
            throw std::bad_alloc{};
        std::memset(state.swwc_staging, 0, P * 64);
    }
    state.raw_write_ptrs[p] = col_base;
}


// ── Decimal raw-scatter ───────────────────────────────────────────────────────
// Decimal<T> wraps an integer NativeType with identical memory layout.
// We reinterpret the column data as NativeType* and delegate to the same
// algorithms used for ColumnVector<NativeType>.

template <typename T>
void computePidsDecimal(const ColumnPrimitives & /*self*/, const IColumn & src_, size_t offset, int n, uint32_t mask, uint32_t * pids)
{
    using NativeT = typename T::NativeType;
    const NativeT * data = reinterpret_cast<const NativeT *>( // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)
                               assert_cast<const ColumnDecimal<T> &>(src_).getData().data())
        + offset;
    hashBatch32(data, n, mask, pids);
}

template <typename T>
[[gnu::hot]] void scatterRawDecimal(const IColumn & src_, size_t offset, const uint32_t * pids, int n, ScatterState & state)
{
    using NativeT = typename T::NativeType;
    const NativeT * src = reinterpret_cast<const NativeT *>( // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)
                              assert_cast<const ColumnDecimal<T> &>(src_).getData().data())
        + offset;
    NativeT ** wp = reinterpret_cast<NativeT **>(state.raw_write_ptrs); // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)
    for (int j = 0; j < n; ++j)
        *wp[pids[j]]++ = src[j];
}

template <typename T>
[[gnu::hot]] void
scatterRawSwwcDecimal(const IColumn & src_, size_t offset, const uint32_t * pids, const uint32_t * positions, int n, ScatterState & state)
{
    using NativeT = typename T::NativeType;
    constexpr size_t S = kSlotsPerFlush<NativeT>;
    constexpr uint32_t flush_trigger = static_cast<uint32_t>(S) - 1;
    const NativeT * src = reinterpret_cast<const NativeT *>( // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)
                              assert_cast<const ColumnDecimal<T> &>(src_).getData().data())
        + offset;
    for (int j = 0; j < n; ++j)
    {
        const uint32_t p = pids[j];
        // Mask positions[j] to this column's slot range (see scatterRawSwwcFixed comment).
        const uint32_t slot = positions[j] & flush_trigger;
        reinterpret_cast<NativeT *>(state.swwc_staging)[p * S + slot] = src[j]; // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)
        if (slot == flush_trigger)
        {
#if USE_MULTITARGET_CODE
            if (isArchSupported(TargetArch::x86_64_v4))
                TargetSpecific::x86_64_v4::flushStagedNTInPlace(
                    reinterpret_cast<NativeT *>(state.swwc_staging) + p * S,
                    state.raw_write_ptrs[p]); // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)
            else if (isArchSupported(TargetArch::x86_64_v3))
                TargetSpecific::x86_64_v3::flushStagedNTInPlace(
                    reinterpret_cast<NativeT *>(state.swwc_staging) + p * S,
                    state.raw_write_ptrs[p]); // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)
            else
                flushStagedScalarInPlace(
                    reinterpret_cast<NativeT *>(state.swwc_staging) + p * S,
                    state.raw_write_ptrs[p]); // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)
#else
            flushStagedScalarInPlace(
                reinterpret_cast<NativeT *>(state.swwc_staging) + p * S,
                state.raw_write_ptrs[p]); // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)
#endif
        }
    }
}


// ── FixedString raw-scatter ───────────────────────────────────────────────────
// Element size is a runtime value (col.getN()), so SWWC is not supported
// (staging slot count cannot be a compile-time constant).  RadixPartitionOperator
// falls back to direct scatter when scatter_raw_swwc is nullptr.

void computePidsFixedString(const ColumnPrimitives & /*self*/, const IColumn & src_, size_t offset, int n, uint32_t mask, uint32_t * pids)
{
    const auto & col = assert_cast<const ColumnFixedString &>(src_);
    const size_t elem = col.getN();
    const auto * data = reinterpret_cast<const unsigned char *>(col.getChars().data()) + offset * elem;
    for (int j = 0; j < n; ++j)
        pids[j] = hashBytes32(data + j * elem, elem) & mask;
}

[[gnu::hot]] void scatterRawFixedString(const IColumn & src_, size_t offset, const uint32_t * pids, int n, ScatterState & state)
{
    const auto & col = assert_cast<const ColumnFixedString &>(src_);
    const size_t elem = col.getN();
    const auto * src = reinterpret_cast<const unsigned char *>(col.getChars().data()) + offset * elem;
    unsigned char ** wp = reinterpret_cast<unsigned char **>(state.raw_write_ptrs); // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)
    for (int j = 0; j < n; ++j)
    {
        std::memcpy(wp[pids[j]], src + j * elem, elem);
        wp[pids[j]] += elem;
    }
}


} // namespace


template <typename T>
ColumnPrimitives makeFixedWidth()
{
    ColumnPrimitives cp;
    cp.scatter = &scatterFixed<T>;
    cp.reconstruct = &reconstructFixed<T>;
    cp.hash = &hashFixed<T>;
    cp.compute_pids = &computePidsFixed<T>;
    cp.scatter_raw = &scatterRawFixed<T>;
    cp.scatter_raw_swwc = &scatterRawSwwcFixed<T>;
    cp.drain_raw = &drainRawFixed<T>;
    cp.on_grow_raw = &onGrowRawFixed<T>;
    cp.raw_elem_size = sizeof(T);
    return cp;
}

template ColumnPrimitives makeFixedWidth<UInt8>();
template ColumnPrimitives makeFixedWidth<UInt16>();
template ColumnPrimitives makeFixedWidth<UInt32>();
template ColumnPrimitives makeFixedWidth<UInt64>();
template ColumnPrimitives makeFixedWidth<UInt128>();
template ColumnPrimitives makeFixedWidth<UInt256>();
template ColumnPrimitives makeFixedWidth<Int8>();
template ColumnPrimitives makeFixedWidth<Int16>();
template ColumnPrimitives makeFixedWidth<Int32>();
template ColumnPrimitives makeFixedWidth<Int64>();
template ColumnPrimitives makeFixedWidth<Int128>();
template ColumnPrimitives makeFixedWidth<Int256>();
template ColumnPrimitives makeFixedWidth<BFloat16>();
template ColumnPrimitives makeFixedWidth<Float32>();
template ColumnPrimitives makeFixedWidth<Float64>();
template ColumnPrimitives makeFixedWidth<UUID>();
template ColumnPrimitives makeFixedWidth<IPv4>();
template ColumnPrimitives makeFixedWidth<IPv6>();


template <typename T>
ColumnPrimitives makeDecimal()
{
    using NativeT = typename T::NativeType;
    ColumnPrimitives cp;
    cp.scatter = &scatterDecimal<T>;
    cp.reconstruct = &reconstructDecimal<T>;
    cp.hash = &hashDecimal<T>;
    cp.compute_pids = &computePidsDecimal<T>;
    cp.scatter_raw = &scatterRawDecimal<T>;
    cp.scatter_raw_swwc = &scatterRawSwwcDecimal<T>;
    cp.drain_raw = &drainRawFixed<NativeT>;
    cp.on_grow_raw = &onGrowRawFixed<NativeT>;
    cp.raw_elem_size = sizeof(NativeT);
    return cp;
}

template ColumnPrimitives makeDecimal<Decimal32>();
template ColumnPrimitives makeDecimal<Decimal64>();
template ColumnPrimitives makeDecimal<Decimal128>();
template ColumnPrimitives makeDecimal<Decimal256>();
template ColumnPrimitives makeDecimal<DateTime64>();
template ColumnPrimitives makeDecimal<Time64>();


ColumnPrimitives makeFixedString(size_t n)
{
    ColumnPrimitives cp;
    cp.scatter = &scatterFixedString;
    cp.reconstruct = &reconstructFixedString;
    cp.hash = &hashFixedString;
    cp.aux = n;
    cp.compute_pids = &computePidsFixedString;
    cp.scatter_raw = &scatterRawFixedString;
    // scatter_raw_swwc and drain_raw are null — RadixPartitionOperator
    // falls back to direct scatter when scatter_raw_swwc is null.
    cp.on_grow_raw = &onGrowRawFixed<unsigned char>;
    cp.raw_elem_size = n;
    return cp;
}

}
