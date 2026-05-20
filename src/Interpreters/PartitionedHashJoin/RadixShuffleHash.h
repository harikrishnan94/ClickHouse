#pragma once

#include <cstddef>
#include <cstdint>

namespace DB
{

/// Phase 1 of the batched radix partition algorithm:
///   for each row j in [0, n): pids[j] = hash(key_data[j]) & mask
///
/// hashOneKeyIntoIds is the per-column primitive:
///   data     — raw column data starting at the first row of this batch
///   elem_sz  — bytes per element (1,2,4,8)
///   n        — number of rows in this batch
///   mask     — P - 1 (P must be a power of two)
///   pids     — output array, length >= n
///   first    — if true, initialise pids; if false, XOR-combine
///
/// Runtime-dispatches to the best available SIMD tier:
///   x86_64-v4 (AVX-512DQ): 8 keys per ZMM, VPMULLQ
///   x86_64-v3 (AVX2):      4 keys per YMM, algebraic-identity 64-bit mul
///   Default (scalar):      one row at a time
void hashOneKeyIntoIds(const void * data, size_t elem_sz, size_t n, uint64_t mask, uint16_t * pids, bool first);

}
