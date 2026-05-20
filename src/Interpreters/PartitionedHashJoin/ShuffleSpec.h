#pragma once

#include <base/types.h>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace DB
{

/// Describes one flat scatter slot (inner data OR null map of a nullable column).
struct ShuffleColDesc
{
    size_t block_pos = 0; /// column index in the incoming Block
    size_t elem_bytes = 0; /// sizeof each element (inner type size, or 1 for null map)
    bool is_nullable = false; /// true → this is the INNER DATA slot of a Nullable column
    bool is_nullmap = false; /// true → this is the NULL MAP slot (uint8, same block_pos as preceding inner slot)
};

/// Immutable per-IJoin configuration built once at PartitionedHashJoin ctor time.
struct ShuffleSpec
{
    size_t P = 0; /// partition count, power-of-two, 64 ≤ P ≤ 1024
    size_t batch_size = 0; /// max(1024, min(32768, P*16))
    bool use_swwc = false; /// true when should_use_swwc(total_cols, P)

    /// Key columns (driving hash). Multi-key XOR-combine semantics.
    std::vector<ShuffleColDesc> key_cols;

    /// Kept payload columns (scattered verbatim, no hash contribution).
    std::vector<ShuffleColDesc> payload_cols;

    /// All columns in scatter order: key_cols first, then payload_cols.
    /// scatter_cols[i].elem_bytes drives the scatter element size.
    std::vector<ShuffleColDesc> scatter_cols;

    /// Element sizes for growPartitionOutput: length == scatter_cols.size().
    std::vector<size_t> col_elem_bytes;

    [[nodiscard]] size_t totalCols() const { return scatter_cols.size(); }

    /// Auto-select direct vs SWWC scatter.
    ///
    /// Crossover thresholds (re-measured post-SWWC inline-flush correctness fix).
    /// The earlier numbers came from the reference benchmark whose buggy
    /// "fill-all-then-flush-all" SWWC silently dropped writes for partitions
    /// receiving > 8 rows per batch, inflating SWWC's apparent throughput
    /// (especially for K=8 at small P). After the fix:
    ///
    ///   K = 1               : SWWC wins at P ≥ 1024 (was P ≥ 512)
    ///   K ∈ [2, 7]          : SWWC wins at P ≥ 64
    ///   K ≥ 8, T ≥ 32       : SWWC wins for any P ≥ 4
    ///   K ≥ 8, T ≤ 16       : SWWC wins at P ≥ 64
    ///
    /// Note: PartitionedHashJoin clamps P to [64, 1024], so for K ≥ 2 the
    /// "P ≥ 64" branches always evaluate true in practice — but the explicit
    /// thresholds preserve the right behaviour if the clamp ever loosens.
    static bool shouldUseSWWC(size_t K, size_t P_val, size_t max_threads)
    {
        if (K == 1)
            return P_val >= 1024;
        if (K <= 7)
            return P_val >= 64;
        /// K ≥ 8: high-K joins win with SWWC at high thread counts irrespective
        /// of P (NT stores cut cross-thread cache-line contention); at low
        /// thread counts the win only materialises once P is large enough that
        /// direct scatter's working set falls out of L2.
        return max_threads >= 32 || P_val >= 64;
    }
};

}
