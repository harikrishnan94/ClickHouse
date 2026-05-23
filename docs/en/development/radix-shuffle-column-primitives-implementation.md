---
description: 'Implementation notes for the RadixShuffle column primitives (concrete C++ types, function signatures, hash kernels, allocator internals, migration checklist)'
sidebar_label: 'RadixShuffle Implementation'
sidebar_position: 81
slug: /development/radix-shuffle-column-primitives-implementation
title: 'RadixShuffle Column Primitives — Implementation Notes'
doc_type: 'reference'
---

# RadixShuffle Column Primitives — Implementation Notes {#radix-shuffle-implementation-notes}

This document is the companion to the [RadixShuffle Column Primitives spec](/development/radix-shuffle-column-primitives). It covers the concrete C++ shapes that realise the spec: struct definitions, function typedefs, hash kernel internals, SIMD multi-versioning strategy, allocator internals, and a file-by-file migration checklist from the current code to the target design.

## File layout {#file-layout}

All files live under `src/Common/RadixShuffle/`.

**Current (as-implemented) file structure:**

```
src/Common/RadixShuffle/
  PartSchema.h             ← SlotRole, FixedSlot, PartSchema  (new)
  PartitionTypes.h         ← FixedChunk, DataChunk, PartReservation,
                              PartReserveGrant, PartReservationView,
                              ResumePosition
  Allocator.h
  Allocator.cpp
  ColumnPrimitives.h       ← updated ScatterFn / ReconstructFn / HashFn typedefs
  ColumnPrimitivesDispatch.h
  ColumnPrimitivesDispatch.cpp
  HashCombiner.h           ← uint32_t hashCombine  (fixed width)
  HashKernels.h            ← fmix32 + SIMD-multi-versioned hash bodies  (new)
  ColumnPrimitives/
    FixedWidth.h / .cpp
    String.h / .cpp
    Nullable.h / .cpp
```

## C++ types {#cpp-types}

### Schema types (`PartSchema.h`) {#schema-types}

```cpp
enum class SlotRole { Values, Offsets, NullMap, FixedStringChars };

struct FixedSlot
{
    size_t col_idx;
    SlotRole role;
    size_t element_size;
    size_t alignment;
};

struct PartSchema
{
    std::vector<FixedSlot> fixed_slots;
    std::vector<size_t> slot_byte_offset;  // byte offset at row 0 in fixed chunk
    size_t fixed_bytes_per_row;            // stride including inter-slot alignment padding
    bool has_varlen_portion;
};
```

`PartSchema` is built once at allocator construction and is immutable thereafter. `slot_byte_offset[s]` is computed by the schema builder by walking `fixed_slots` in order and aligning each slot's start to `fixed_slots[s].alignment`. `fixed_bytes_per_row` is the sum of all slot element sizes plus padding.

### Chunk and reservation types (`PartitionTypes.h`) {#chunk-types}

```cpp
struct FixedChunk
{
    void * data = nullptr;
    size_t row_capacity = 0;

    /// Per-slot byte offsets within `data` at row 0.  Column-major layout:
    /// addr(slot_s, row_r) = slot_byte_offsets[s] + r * fixed_slots[s].element_size.
    /// Computed at chunk allocation time for this chunk's row_capacity;
    /// points into the handle's arena.  Length = PartSchema::fixed_slots.size().
    const size_t * slot_byte_offsets = nullptr;
};

struct DataChunk
{
    unsigned char * bytes = nullptr;
    size_t byte_capacity = 0;
};

struct PartReservation
{
    FixedChunk * fixed = nullptr;
    size_t begin_row = 0;
    size_t reserved_rows = 0;
    DataChunk * data = nullptr;   // null for fixed-only partitions
    size_t begin_byte = 0;
    size_t reserved_bytes = 0;
};

struct PartReserveGrant
{
    size_t granted_rows = 0;
    size_t granted_varlen_bytes = 0;
    PartReservation slice;
    bool fully_satisfied = false;
};

/// Per-partition view tuple used by reconstruct.
struct PartReservationView
{
    const FixedChunk * fixed = nullptr;
    size_t row_begin = 0;
    size_t row_end = 0;
    const DataChunk * data = nullptr;  // null for fixed-only primitives
    size_t byte_begin = 0;
    size_t byte_end = 0;
};

/// Resume cursor for the reconstruct pump pattern.
struct ResumePosition
{
    size_t view_index = 0;
    size_t rows_consumed_in_view = 0;
};
```

### Column primitives type (`ColumnPrimitives.h`) {#column-primitives-type}

```cpp
struct ColumnPrimitives
{
    ScatterFn scatter = nullptr;
    ReconstructFn reconstruct = nullptr;
    HashFn hash = nullptr;

    /// Indices into PartSchema::fixed_slots that this primitive reads/writes.
    std::vector<size_t> fixed_slot_indices;

    /// True if this primitive writes to the data chunk.
    bool writes_varlen = false;

    /// For Nullable(X): owning pointer to the nested column primitive.
    std::shared_ptr<const ColumnPrimitives> nested;

    /// Auxiliary scalar (e.g., FixedString row width n).
    size_t aux = 0;
};
```

## Function signatures {#function-signatures}

### Primitive function pointer typedefs {#primitive-typedefs}

```cpp
/// Scatter primitive: routes n source rows into per-partition destinations.
/// pids[j] ∈ [0, partitions) for all j.
using ScatterFn = void (*)(
    const ColumnPrimitives & self,
    const PartSchema & schema,
    const IColumn & src,
    const uint16_t * pids,
    size_t n,
    size_t partitions,
    PartReservation * dst);

/// Reconstruct primitive: appends rows from views into target up to capacity.
/// Returns the position of the first unconsumed row across the view list.
using ReconstructFn = ResumePosition (*)(
    const ColumnPrimitives & self,
    const PartSchema & schema,
    const PartReservationView * views,
    size_t n_views,
    ResumePosition start,
    IColumn & target);

/// Hash primitive: updates out[i] = hashCombine(out[i], h(src[i])) for i ∈ [0, n).
using HashFn = void (*)(
    const ColumnPrimitives & self,
    const PartSchema & schema,
    const IColumn & src,
    size_t n,
    uint32_t * out);
```

### Dispatcher and schema builder {#dispatcher-signature}

`resolveColumnPrimitives` resolves the function-pointer triple for a single type and returns a `ColumnPrimitives` with **empty** `fixed_slot_indices`.  It is an internal building block; callers should not use it directly.

`buildSchemaAndPrimitives` is the canonical entry point.  It resolves all primitives and populates their `fixed_slot_indices` in a single pass using an internal `PartSchemaBuilder`:

```cpp
/// Internal: resolves the scatter/reconstruct/hash function pointers only.
/// fixed_slot_indices is left empty.  Use buildSchemaAndPrimitives instead.
[[nodiscard]] ColumnPrimitives resolveColumnPrimitives(const IDataType & type);

/// Canonical entry point.  Resolves all primitives and builds the PartSchema
/// in one pass.  fixed_slot_indices is populated and consistent with
/// schema.fixed_slots on return.
struct SchemaAndPrimitives
{
    PartSchema schema;
    std::vector<ColumnPrimitives> primitives;
};
[[nodiscard]] SchemaAndPrimitives buildSchemaAndPrimitives(
    const std::vector<DataTypePtr> & types);
```

Internally `buildSchemaAndPrimitives` calls `resolveColumnPrimitives` for each type, then `assignSlotIndices` walks the primitive tree (handling `Nullable` recursion) and appends slots to a `PartSchemaBuilder`.  The `NullMap` slot is always inserted **before** the nested primitive's slots so that `Nullable::scatter` can address it via `fixed_slot_indices[0]`.

### Handle reserve {#handle-reserve-signature}

```cpp
/// Per-batch reservation across all P partitions (hot path).
/// rows[p] and varlen_bytes[p] are the request for partition p.
/// grants[p] receives the allocation result for partition p.
/// stale_fixed_bitset is ceil(P / 64) uint64_t words; bit p is set iff
/// partition p's FixedChunk was newly allocated during this call.
void Handle::reserve(
    const size_t * rows,
    const size_t * varlen_bytes,
    PartReserveGrant * grants,
    uint64_t * stale_fixed_bitset);
```

The bitset is provided and zeroed by the caller; `reserve` sets bits but never clears them, so the caller can OR results from multiple calls in a single batch pass if needed.

## Hash kernel internals {#hash-kernel-internals}

### `hashCombine` (32-bit) {#hash-combine}

`HashCombiner.h` provides the uniform hash combiner used by every primitive:

```cpp
[[gnu::always_inline]] inline uint32_t hashCombine(uint32_t prior, uint32_t h) noexcept
{
    return prior ^ (h + 0x9e3779b9U + (prior << 6) + (prior >> 2));
}
```

`0x9e3779b9` is the 32-bit fractional part of phi (golden ratio). The `<<6 / >>2` shift pair is the canonical 32-bit `boost::hash_combine` mixer. This replaces the current `HashCombiner.h` which incorrectly uses the 64-bit constant `0x9e3779b97f4a7c15ULL` with `<<12 / >>4` shifts and returns `uint64_t`.

### `fmix32` per-row finalizer {#fmix32}

`HashKernels.h` provides the MurmurHash3 32-bit finalizer used by fixed-width and string per-row mixers:

```cpp
[[gnu::always_inline]] inline uint32_t fmix32(uint32_t x) noexcept
{
    x ^= x >> 16;
    x *= 0x85ebca6bU;
    x ^= x >> 13;
    x *= 0xc2b2ae35U;
    x ^= x >> 16;
    return x;
}
```

This is the 32-bit analogue of the existing `intHash64Local` (64-bit MurmurHash3 `fmix64`) used in `ColumnPrimitives/FixedWidth.cpp`. For types wider than 4 bytes, bytes are folded through the finalizer in 4-byte chunks, accumulating into a `uint32_t` state.

### SIMD multi-versioning {#simd-multi-versioning}

**Not in v1.** Hash kernels are scalar in v1; the multi-versioning scaffold is tracked as a `TODO` comment at the top of `[src/Common/RadixShuffle/HashKernels.h](src/Common/RadixShuffle/HashKernels.h)`. The design below describes the intended future implementation.

Hash kernel bodies should be emitted in multiple SIMD variants using the `MULTITARGET_FUNCTION_X86_V3` macro from `src/Common/TargetSpecific.h`. The runtime dispatch pattern follows the existing ClickHouse convention seen in `[src/Common/iota.cpp](src/Common/iota.cpp)`:

```cpp
MULTITARGET_FUNCTION_X86_V3(
    MULTITARGET_FUNCTION_HEADER(static void NO_INLINE),
    hashColumnImpl, MULTITARGET_FUNCTION_BODY((/* args */) { /* body */ })
)

void hashColumn(/* args */)
{
#if USE_MULTITARGET_CODE
    if (isArchSupported(TargetArch::x86_64_v3))
        return hashColumnImpl_x86_64_v3(/* args */);
#endif
    return hashColumnImpl(/* args */);
}
```

This produces a baseline scalar variant and an `x86_64_v3` (AVX2 + AVX-512) variant, with runtime selection via `isArchSupported`. The macro expansion is compile-time; no virtual dispatch is introduced.

## Allocator internals {#allocator-internals}

### Per-handle arena {#allocator-arena}

Each `Handle` owns a private bump-allocated arena. Arena pages are `DEFAULT_ARENA_PAGE_BYTES` (64 KiB) by default, allocated lazily on first use from the system allocator, and freed only at parent allocator destruction. This preserves the monotonic-no-per-chunk-dealloc invariant: chunks returned to callers remain valid for the allocator's entire lifetime.

The arena backs both `FixedChunk` headers and their data payloads, and `DataChunk` headers and their byte payloads, in one bump-allocated stream.

### Per-handle per-partition writable tails {#allocator-tails}

Each handle maintains a flat array of per-partition state structs (indexed by partition index), each holding:

- `FixedChunk * fixed_tail` — the current appendable fixed chunk for that partition.
- `size_t fixed_next_row` — next free row index within `fixed_tail`.
- `size_t fixed_remaining_rows` — rows left in the current `fixed_tail`.
- `DataChunk * data_tail` — the current appendable data chunk (null for fixed-only schemas).
- `size_t data_next_byte` — next free byte index within `data_tail`.
- `size_t data_remaining_bytes` — bytes left in the current `data_tail`.
- `size_t reserved_rows` / `reserved_bytes` — cumulative totals for growth-factor computation.

When `rows[p]` rows do not fit in `fixed_tail`'s remaining capacity:

1. A new `FixedChunk` is allocated from the handle's arena, sized to at least `max(MIN_CHUNK_FLOOR_ROWS, rows[p])` rows (rounded up to a multiple of the request to eliminate per-chunk tail waste).
2. Per-chunk `slot_byte_offsets` are computed (column-major layout, one offset per slot) and stored in the arena alongside the chunk header.  Bit `p` is set in `stale_fixed_bitset`.
3. `fixed_tail` pointers and counters are updated.

When `varlen_bytes[p]` bytes do not fit in `data_tail`:

1. A new `DataChunk` is allocated, sized to `max(MIN_CHUNK_FLOOR_BYTES_DATA, varlen_bytes[p])` bytes.
2. No bit is set in the stale bitset — data pointers are always re-read from `PartReservation.data` per batch.
3. `data_tail` and `data_next_byte` are updated.

### Sharded atomic counters {#allocator-counters}

Each handle maintains per-handle atomic counters for bookkeeping:

```cpp
alignas(64) std::atomic<uint64_t> local_reserved_bytes{0};
alignas(64) std::atomic<uint64_t> local_allocated_bytes{0};
alignas(64) std::atomic<uint64_t> local_chunks{0};
alignas(64) std::atomic<uint64_t> local_active_partitions{0};
```

Each counter occupies its own cache line to avoid false sharing. `Allocator::totalReservedBytes()` and `totalAllocatedBytes()` sum across all handles lazily, under the cold-path handle-pool mutex. Tests that read running totals during live execution observe an eventually-consistent view (`memory_order_relaxed` is sufficient — no ordering with other allocator state is required).

### Handle-pool mutex {#allocator-mutex}

The parent `Allocator` guards the handle pool with a single `std::mutex`, held only on the cold-path `acquire` / `release` / totals-read paths. No lock is held during `reserve` — handle-level state is private to the owning thread, so the reservation path is entirely contention-free.

## Migration checklist {#migration-checklist}

The following file-by-file actions migrate the current code to the target design. No file outside `src/Common/RadixShuffle/` needs to change for the v1 primitives milestone.

### `PartitionTypes.h` {#migrate-partition-types}

- Remove `struct Chunk` (the current monolithic chunk bundling `primary`, `offsets`, `null_map`).
- Remove `struct ColumnDesc`, `struct ReservationRequest`, `struct Reservation`.
- Add `struct FixedChunk`, `struct DataChunk`, `struct PartReservation`, `struct PartReserveGrant`, `struct PartReservationView`.
- Keep `struct ResumePosition` (unchanged).
- `struct ChunkRangeView` is superseded by `PartReservationView`; remove it.

### New `PartSchema.h` {#migrate-part-schema}

Create this header with `enum class SlotRole`, `struct FixedSlot`, and `struct PartSchema`. Provide a free function `PartSchema buildPartSchema(const std::vector<ColumnPrimitives> & primitives)` that walks the column-primitive list and constructs the schema by assigning slot indices and computing `slot_byte_offset[]` and `fixed_bytes_per_row`.

### `Allocator.h` {#migrate-allocator-h}

- Constructor: replace `std::vector<ColumnDesc> column_descs` with `PartSchema schema`.
- `Handle::reserve`: replace `void reserve(size_t col_idx, const ReservationRequest *, Reservation *)` with `void reserve(const size_t * rows, const size_t * varlen_bytes, PartReserveGrant * grants, uint64_t * stale_fixed_bitset)`.
- Remove `columnDesc(size_t)` and `numColumns()` accessors that expose `ColumnDesc`.
- Update `Handle::PerChain` (internal) from a per-(column, partition) flat vector to a per-partition array of `(fixed_tail, fixed_next_row, data_tail, data_next_byte)` structs.

### `Allocator.cpp` {#migrate-allocator-cpp}

- Rework `Handle::ensureChunk` into two helpers: `ensureFixed(part_idx, rows)` and `ensureData(part_idx, bytes)`, each setting the stale bit on the output bitset only for the fixed case.
- Update `Handle::reserve` body accordingly.
- Remove all `col_idx`-indexed logic; replace with `part_idx`-indexed logic.
- Update the waste-bound bookkeeping: `local_active_partitions` tracks partitions with at least one chunk, not `(column, partition)` pairs.

### `ColumnPrimitives.h` {#migrate-column-primitives-h}

- Replace `using ScatterFn = void (*)(... const uint32_t * pids ..., Reservation * dst)` with the new signature using `uint16_t * pids` and `PartReservation * dst`, plus the `PartSchema &` argument.
- Replace `using HashFn = void (*)(... uint64_t * out)` with `uint32_t * out`.
- Update `ReconstructFn` to take `const PartReservationView *` and `const PartSchema &`.
- In `struct ColumnPrimitives`: remove `ColumnDesc column_desc`; add `std::vector<size_t> fixed_slot_indices` and `bool writes_varlen`.

### `HashCombiner.h` {#migrate-hash-combiner}

- Change `uint64_t hashCombine(uint64_t prior, uint64_t h)` to `uint32_t hashCombine(uint32_t prior, uint32_t h)`.
- Replace the constant `0x9e3779b97f4a7c15ULL` with `0x9e3779b9U`.
- Replace the mixer shifts `(prior << 12) + (prior >> 4)` with `(prior << 6) + (prior >> 2)`.

### New `HashKernels.h` {#migrate-hash-kernels}

Create this header with `fmix32` and the `MULTITARGET_FUNCTION_X86_V3`-wrapped hash kernel bodies for fixed-width, string, and nullable column types.

### `ColumnPrimitives/FixedWidth.cpp` {#migrate-fixed-width}

- Change the `pids` type from `const uint32_t *` to `const uint16_t *` in `scatterFixed` and `scatterFixedString`.
- Change `dst` from `Reservation *` to `PartReservation *`; update write-pointer derivation to use the planar address formula: `static_cast<T *>(dst[p].fixed->data) + slot_byte_offset / sizeof(T) + dst[p].begin_row`.
- Change per-row hash output from `uint64_t *` to `uint32_t *`; replace `intHash64Local` with `fmix32` from `HashKernels.h`; call `hashCombine` with `uint32_t` operands.
- Remove `MAX_PARTITIONS = 1024`; replace with `static_assert(P ≤ 65536)` (enforced by the `uint16_t` pid type).

### `ColumnPrimitives/String.cpp` {#migrate-string}

- Same `pids` and `dst` type changes as above.
- The `off_ptrs` array now indexes into the `Offsets` slot of `dst[p].fixed->data` using the planar formula.
- The `char_ptrs` array and `abs_byte` tracking move to `dst[p].data->bytes + dst[p].begin_byte`.
- Change hash output to `uint32_t *`; replace `intHash64Local` + 64-bit `hashCombine` with `fmix32` + 32-bit `hashCombine`.

### `ColumnPrimitives/Nullable.cpp` {#migrate-nullable}

- Same `pids` and `dst` type changes.
- `null_ptrs` now indexes into the `NullMap` slot of `dst[p].fixed->data` via the planar formula.
- Hash: update to `uint32_t` throughout; the null byte is hashed with `fmix32` before calling `hashCombine`.

### `ColumnPrimitivesDispatch.cpp` {#migrate-dispatcher}

**Already implemented.**  The file is split into two concerns:

- `resolveLeaf` / `resolveColumnPrimitives`: resolve the function-pointer triple only.  `fixed_slot_indices` is left empty (internal building block).
- `PartSchemaBuilder` + `assignSlotIndices` + `buildSchemaAndPrimitives`: walk the type tree, insert slots into the schema, and populate each primitive's `fixed_slot_indices` and `writes_varlen`.  `NullMap` is always the first slot for `Nullable` columns.

Callers should use `buildSchemaAndPrimitives`; `resolveColumnPrimitives` is kept public for testing and introspection.
