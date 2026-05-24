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
// flushStagedNT / flushStagedScalar          — typed T*&  (NumericScatterColumn)
// flushStagedNTInPlace / flushStagedScalarInPlace — void*& (raw scatter path)
//
// The InPlace variants pass the reference directly into fixed_ptrs[p] so the
// pointer updates in-place without a separate load-cast-store sequence.

#if USE_MULTITARGET_CODE

DECLARE_X86_64_V4_SPECIFIC_CODE(

    template <typename T> inline void flushStagedNT(const T * staging_p, T *& out_p) // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)
    {
        if constexpr (sizeof(T) == 8)
            _mm512_stream_si512(reinterpret_cast<__m512i *>(out_p), _mm512_load_si512(reinterpret_cast<const __m512i *>(staging_p)));
        else
            for (int i = 0; i < 8; ++i)
                out_p[i] = staging_p[i];
        out_p += 8;
    }

    template <typename T>
    inline void flushStagedNTInPlace(const T * staging_p, void *& out_void) // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)
    {
        T * out_p = static_cast<T *>(out_void);
        if constexpr (sizeof(T) == 8)
            _mm512_stream_si512(reinterpret_cast<__m512i *>(out_p), _mm512_load_si512(reinterpret_cast<const __m512i *>(staging_p)));
        else
            for (int i = 0; i < 8; ++i)
                out_p[i] = staging_p[i];
        out_void = out_p + 8;
    }

    ) // DECLARE_X86_64_V4_SPECIFIC_CODE

#endif // USE_MULTITARGET_CODE

template <typename T>
[[gnu::always_inline]] inline void flushStagedScalar(const T * staging_p, T *& out_p)
{
    for (int i = 0; i < 8; ++i)
        out_p[i] = staging_p[i];
    out_p += 8;
}

template <typename T>
[[gnu::always_inline]] inline void flushStagedScalarInPlace(const T * staging_p, void *& out_void)
{
    T * out_p = static_cast<T *>(out_void);
    for (int i = 0; i < 8; ++i)
        out_p[i] = staging_p[i];
    out_void = out_p + 8;
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
template <typename T>
[[gnu::hot]] void scatterRawFixed(const IColumn & src_, size_t offset, const uint32_t * pids, int n, ScatterState & state)
{
    const T * src = assert_cast<const ColumnVector<T> &>(src_).getData().data() + offset;
    // raw_write_ptrs is a plain void** — loaded once into a callee-saved register,
    // no vector.data() double-indirection in the hot loop.
    void ** wp = state.raw_write_ptrs;
    const size_t partitions = state.fixed_ptrs.size();

    if (partitions <= SCATTER_STACK_PTRS)
    {
        T * ptrs[SCATTER_STACK_PTRS];
        for (size_t p = 0; p < partitions; ++p)
            ptrs[p] = static_cast<T *>(wp[p]);
        for (int j = 0; j < n; ++j)
            *ptrs[pids[j]]++ = src[j];
        for (size_t p = 0; p < partitions; ++p)
            wp[p] = ptrs[p]; // T* → void* implicit
    }
    else
    {
        for (int j = 0; j < n; ++j)
        {
            const uint32_t p = pids[j];
            T * out = static_cast<T *>(wp[p]);
            *out++ = src[j];
            wp[p] = out; // T* → void* implicit
        }
    }
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
    const T * src = assert_cast<const ColumnVector<T> &>(src_).getData().data() + offset;

    if (!state.swwc_staging_initialized)
    {
        const size_t partitions = state.fixed_ptrs.size();
        if (posix_memalign(reinterpret_cast<void **>(&state.swwc_staging), 64, partitions * 8 * sizeof(T)) != 0)
            throw std::bad_alloc{};
        std::memset(state.swwc_staging, 0, partitions * 8 * sizeof(T));
        state.swwc_staging_initialized = true;
    }

    // Access staging and raw_write_ptrs via state in the loop body — don't preload
    // them into named variables before the loop.  The compiler then keeps `state`
    // (= r9 → one callee-saved register) as the base pointer for both arrays,
    // exactly mirroring the baseline's `this` (r12) that gave access to both
    // `staging_` and `out_[]` via member offsets.  This frees two callee-saved
    // registers (previously consumed by a pre-loaded staging pointer and a
    // pre-loaded wp pointer) so that pids and positions can stay in registers
    // instead of being spilled to the stack.
    for (int j = 0; j < n; ++j)
    {
        const uint32_t p = pids[j];
        const uint32_t slot = positions[j];
        reinterpret_cast<T *>(state.swwc_staging)[static_cast<size_t>(p) * 8 + slot] = src[j];
        if (slot == 7)
        {
            // state.raw_write_ptrs[p] is void*& — updated in-place by flush.
#if USE_MULTITARGET_CODE
            if (isArchSupported(TargetArch::x86_64_v4))
                TargetSpecific::x86_64_v4::flushStagedNTInPlace(
                    reinterpret_cast<T *>(state.swwc_staging) + static_cast<size_t>(p) * 8, state.raw_write_ptrs[p]);
            else
                flushStagedScalarInPlace(reinterpret_cast<T *>(state.swwc_staging) + static_cast<size_t>(p) * 8, state.raw_write_ptrs[p]);
#else
            flushStagedScalarInPlace(reinterpret_cast<T *>(state.swwc_staging) + static_cast<size_t>(p) * 8, state.raw_write_ptrs[p]);
#endif
        }
    }
}


/// Drain: copies `cnt` staged values for partition `p` to its output pointer.
template <typename T>
void drainRawFixed(size_t p, uint32_t cnt, ScatterState & state)
{
    if (!state.swwc_staging_initialized || cnt == 0)
        return;
    const T * s = reinterpret_cast<const T *>(state.swwc_staging) + p * 8;
    T * dst = static_cast<T *>(state.raw_write_ptrs[p]);
    for (uint32_t i = 0; i < cnt; ++i)
        dst[i] = s[i];
    state.raw_write_ptrs[p] = dst + cnt; // T* → void* implicit
}


/// Update the write pointer for partition `p` when a new output block is
/// allocated.  Initializes raw_write_ptrs lazily on the first call.
template <typename T>
void onGrowRawFixed(size_t p, void * col_base, ScatterState & state)
{
    if (state.raw_write_ptrs == nullptr)
    {
        // Lazily allocate; calloc so unused slots are null-safe.
        state.raw_write_ptrs = static_cast<void **>(std::calloc(state.fixed_ptrs.size(), sizeof(void *)));
        if (!state.raw_write_ptrs)
            throw std::bad_alloc{};
    }
    state.raw_write_ptrs[p] = col_base;
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
    ColumnPrimitives cp;
    cp.scatter = &scatterDecimal<T>;
    cp.reconstruct = &reconstructDecimal<T>;
    cp.hash = &hashDecimal<T>;
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
    return cp;
}

}
