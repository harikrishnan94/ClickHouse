#pragma once

#include <Columns/IColumn.h>
#include <Interpreters/HashJoin/ScatteredBlock.h>
#include <Interpreters/RowRefs.h>
#include <Common/Arena.h>
#include <Common/PODArray.h>
#include <base/types.h>

#include <type_traits>
#include <vector>

namespace DB
{
namespace Unified
{

/** The probe of the single-map `joinRightColumns`, as two independent choices.
  *
  * A LOOKUP DRIVER walks a range of rows and hands each row's lookup result to a SINK. The
  * driver is the swappable part - the ordinary sequential `findKey` below, or later an
  * out-of-order ring that overlaps the dependent loads of many rows' hash-table walks. The
  * sink decides what the join does with each result.
  *
  * There is one loop, not two, because the sink is what makes the probe fused or two-phase:
  *   - a sink that emits the match straight away IS the fused probe, and nothing runs after it;
  *   - a sink that records the outcome hands the batch to a second pass, which lets the
  *     batch's lookups overlap each other's memory latency instead of each one waiting for the
  *     previous row's output to be emitted.
  * Whether that second pass earns its scratch traffic depends on the join kind, so the choice
  * lives at the call site instead of in a second copy of the loop.
  */

/// Rows looked up before their matches are processed. Measured flat from 256 to 65536 over a
/// 3-cell x 12-length scan and indistinguishable between 1024 and 4096 over the full 144-cell
/// matrix, so this is not a tuning knob. Cost is monotone only below ~16, where the phase
/// switch cannot be amortised.
inline constexpr size_t PROBE_BATCH_ROWS = 8192;

template <typename Selector>
ALWAYS_INLINE size_t selectorIndexAt(const Selector & selector, size_t k)
{
    if constexpr (std::is_same_v<std::decay_t<Selector>, ScatteredBlock::Selector>)
        return selector[k];
    else if constexpr (std::is_same_v<std::decay_t<Selector>, ScatteredBlock::Indexes>)
        return selector.getData()[k];
    else
        return selector.first + k;
}

/** The mapped values that fit the outcome word, so a second pass can rebuild them on the stack
  * and never dereference the hash-table cell again. Both are exactly 8 bytes (there are
  * `static_assert`s on that in `RowRefs.h`) and an occupied cell's value is never zero: a
  * `RowRef` always carries `INLINE_FLAG` in bit 63, and a cell's `RowRefList` always holds at
  * least one row. That is what lets zero be the "no match" sentinel.
  */
template <typename Mapped>
inline constexpr bool probe_mapped_fits_word
    = std::is_same_v<std::remove_const_t<Mapped>, RowRef> || std::is_same_v<std::remove_const_t<Mapped>, RowRefList>;

template <typename Mapped>
requires probe_mapped_fits_word<Mapped>
ALWAYS_INLINE UInt64 mappedWordOf(const Mapped & mapped)
{
    if constexpr (std::is_same_v<std::remove_const_t<Mapped>, RowRefList>)
        return mapped.word;
    else
        return mapped.encode();
}

template <typename Mapped>
requires probe_mapped_fits_word<Mapped>
ALWAYS_INLINE Mapped mappedFromWord(UInt64 word)
{
    if constexpr (std::is_same_v<Mapped, RowRefList>)
        return RowRefList::fromWord(word);
    else
        return RowRef::fromWord(word);
}

/** What a recording sink hands to the second pass, indexed by the row's position in the batch.
  *
  * `found[j]`:
  *   0         - the row matched nothing. A miss and a row excluded by the skip bytes (a NULL
  *               key or an ON-section condition) are recorded the same way, because the emit
  *               treats them the same way - so no sink and no second pass carries skip logic.
  *   otherwise - the matched cell's mapped value COPIED BY VALUE when it fits a word, else the
  *               bits of a pointer to the mapped value inside the cell (ASOF's `AsofRowRefs`).
  *               Copying it out in the same visit that reads the cell is what keeps the second
  *               pass off the hash table: a batch later the cell's line has left the cache, and
  *               re-reading it through a recorded pointer would be a second random miss per
  *               matched row.
  *
  * `offset[j]` is the matched cell's `JoinUsedFlags` offset. It is filled - and only sized -
  * for the join kinds that keep used flags, since nothing else reads it.
  */
struct ProbeOutcomes
{
    /// The sink writes `found[0 .. count)`. This is a raw pointer rather than the array itself
    /// because for some joins the outcomes ARE the output: a lazy ALL join that adds missing
    /// rows appends exactly one word per probe row to `LazyOutput::row_refs`, and that word is
    /// the one the sink already has - a match's cell word, or zero, which is what `addDefault`
    /// appends for a miss. Pointing the sink straight at that array saves writing every word
    /// twice. `scratch` is the storage for the joins where that does not hold.
    UInt64 * found = nullptr;
    PODArray<UInt64> offset;
    PODArray<UInt64> scratch;

    void useScratch(size_t rows, bool need_flags)
    {
        scratch.resize(rows);
        found = scratch.data();
        if (need_flags)
            offset.resize(rows);
    }

    /// `external` must have room for `rows`, and must not be reallocated while the sink and
    /// the second pass run over it.
    void useExternal(UInt64 * external, size_t rows, bool need_flags)
    {
        found = external;
        if (need_flags)
            offset.resize(rows);
    }
};

/** The ordinary lookup driver: one `findKey` per row, in row order.
  *
  * `prefetch_at` is the probe's look-ahead software prefetch, called with the ABSOLUTE row so
  * its look-ahead reaches past the end of a batch; a driver that drives its own prefetching
  * ignores it.
  *
  * Every row of `[begin, begin + count)` reaches the sink exactly once, in row order. Both
  * properties are part of the contract: a recording sink's second pass reads the whole range
  * and cannot tell an unwritten entry from a miss, and an emitting sink must produce output
  * rows in row order. A driver that completes lookups out of order must still deliver them to
  * the sink in order, or only be used with sinks that write by position.
  *
  * `fast_path` (skip_data == nullptr) is folded per P4: ONE `run` instantiation dispatches to
  * two NO_INLINE inner loops, so the single-map probe is no longer instantiated x2 for
  * skip vs no-skip. The loops stay outlined: ALWAYS_INLINE here doubled the hot `run` body
  * (both skip and no-skip loops in one I-cache footprint) and regressed String keys ~3%+.
  */
struct SequentialLookup
{
    template <typename KeyGetter, typename Map, typename Selector, typename PrefetchAt, typename Sink>
    static void run(
        KeyGetter & key_getter,
        const Map & map,
        const Selector & selector,
        const UInt8 * skip_data,
        Arena & pool,
        size_t begin,
        size_t count,
        PrefetchAt && prefetch_at,
        Sink && sink)
    {
        if (skip_data == nullptr)
            runImpl</*with_skip=*/false>(
                key_getter, map, selector, skip_data, pool, begin, count, prefetch_at, sink);
        else
            runImpl</*with_skip=*/true>(
                key_getter, map, selector, skip_data, pool, begin, count, prefetch_at, sink);
    }

    template <bool with_skip, typename KeyGetter, typename Map, typename Selector, typename PrefetchAt, typename Sink>
    NO_INLINE static void runImpl(
        KeyGetter & key_getter,
        const Map & map,
        const Selector & selector,
        const UInt8 * skip_data [[maybe_unused]],
        Arena & pool,
        size_t begin,
        size_t count,
        PrefetchAt && prefetch_at,
        Sink && sink)
    {
        for (size_t j = 0; j < count; ++j)
        {
            prefetch_at(begin + j);

            const size_t ind = selectorIndexAt(selector, begin + j);

            if constexpr (with_skip)
            {
                if (skip_data[ind])
                {
                    sink.miss(j, begin + j);
                    continue;
                }
            }

            sink.result(j, begin + j, ind, key_getter.findKey(map, ind, pool));
        }
    }
};

/** Multi-clause (OR-of-equi-joins) lookup driver under C15.
  *
  * The driver, not the sink, owns clause iteration:
  *   - per row, for each clause in order, consume that clause's skip bytes; a skipped clause
  *     is not a missed row (the row may still match a later clause);
  *   - call `sink.result(..., find_result, known_rows, is_last_disjunct)` once per MATCHING
  *     clause, in clause order; stop after the first match when `stop_after_first_match`;
  *   - call `sink.miss(j, row)` exactly once if no clause matched;
  *   - own `KnownRowsHolder` per row (passed into the sink) and call `sink.finish(row)` once
  *     per consumed row so C16's `push_back` of `offsets_to_replicate` stays at the row
  *     boundary.
  *
  * C13: loop stops when `current_offset >= max_joined_rows` and returns rows consumed (`i`).
  * `fast_path` is folded the same way as `SequentialLookup` (P4): two NO_INLINE inner loops
  * inside one `run` instantiation, keyed on whether any clause has skip bytes.
  */
struct SequentialMultiLookup
{
    template <
        bool stop_after_first_match,
        typename KnownRowsHolder,
        typename KeyGetter,
        typename Map,
        typename Selector,
        typename PrefetchAt,
        typename Sink>
    static size_t run(
        std::vector<KeyGetter> & key_getter_vector,
        const std::vector<const Map *> & mapv,
        const Selector & selector,
        const std::vector<const UInt8 *> & skip_datas,
        Arena & pool,
        size_t rows,
        size_t max_joined_rows,
        IColumn::Offset & current_offset,
        PrefetchAt && prefetch_at,
        Sink && sink)
    {
        if (skip_datas.empty())
            return runImpl<stop_after_first_match, /*with_skip=*/false, KnownRowsHolder>(
                key_getter_vector,
                mapv,
                selector,
                skip_datas,
                pool,
                rows,
                max_joined_rows,
                current_offset,
                prefetch_at,
                sink);
        return runImpl<stop_after_first_match, /*with_skip=*/true, KnownRowsHolder>(
            key_getter_vector,
            mapv,
            selector,
            skip_datas,
            pool,
            rows,
            max_joined_rows,
            current_offset,
            prefetch_at,
            sink);
    }

    template <
        bool stop_after_first_match,
        bool with_skip,
        typename KnownRowsHolder,
        typename KeyGetter,
        typename Map,
        typename Selector,
        typename PrefetchAt,
        typename Sink>
    NO_INLINE static size_t runImpl(
        std::vector<KeyGetter> & key_getter_vector,
        const std::vector<const Map *> & mapv,
        const Selector & selector,
        const std::vector<const UInt8 *> & skip_datas [[maybe_unused]],
        Arena & pool,
        size_t rows,
        size_t max_joined_rows,
        IColumn::Offset & current_offset,
        PrefetchAt && prefetch_at,
        Sink && sink)
    {
        /// Clause count follows the key getters (and join_on_keys at the call site), not
        /// mapv: dispatch still routes `join_on_keys.empty()` into this overload, where
        /// `key_getter_vector` is empty while `mapv` may still be non-empty.
        chassert(key_getter_vector.size() == mapv.size() || key_getter_vector.empty());
        const size_t num_clauses = key_getter_vector.size();
        size_t i = 0;
        for (; i < rows && current_offset < max_joined_rows; ++i)
        {
            prefetch_at(i);

            const size_t ind = selectorIndexAt(selector, i);

            bool right_row_found = false;
            KnownRowsHolder known_rows;
            for (size_t onexpr_idx = 0; onexpr_idx < num_clauses; ++onexpr_idx)
            {
                if constexpr (with_skip)
                {
                    if (skip_datas[onexpr_idx] && skip_datas[onexpr_idx][ind])
                        continue;
                }

                auto find_result = key_getter_vector[onexpr_idx].findKey(*mapv[onexpr_idx], ind, pool);
                if (find_result.isFound())
                {
                    right_row_found = true;
                    const bool is_last_disjunct = onexpr_idx + 1 == num_clauses;
                    sink.result(i, i, ind, find_result, known_rows, is_last_disjunct);

                    if constexpr (stop_after_first_match)
                        break;
                }
            }

            if (!right_row_found)
                sink.miss(i, i);

            sink.finish(i);
        }
        return i;
    }
};

/** Sink: record the outcome and emit nothing, so a second pass can do the whole emit later.
  *
  * The hit/miss test here was also tried as a conditional move rather than a branch, on the
  * theory that mispredicting it inside the lookup discards the speculation that overlaps the
  * next rows' misses. It removed exactly one branch from the generated loop and changed
  * nothing measurable - within +-1% over a 25-cell cardinality x match-rate grid, at every
  * match rate including the 50% worst case - so the branch stays and the variant is gone.
  */
template <bool need_flags>
struct RecordOutcomeSink
{
    ProbeOutcomes & outcomes;

    ALWAYS_INLINE void miss(size_t j, size_t /* row */) { outcomes.found[j] = 0; }

    template <typename FindResult>
    ALWAYS_INLINE void result(size_t j, size_t row, size_t /* ind */, const FindResult & find_result)
    {
        using Mapped = std::remove_reference_t<decltype(std::declval<FindResult &>().getMapped())>;

        if (!find_result.isFound())
        {
            miss(j, row);
            return;
        }

        auto & mapped = find_result.getMapped();
        if constexpr (probe_mapped_fits_word<Mapped>)
            outcomes.found[j] = mappedWordOf(mapped);
        else
            outcomes.found[j] = reinterpret_cast<UInt64>(&mapped);

        if constexpr (need_flags)
            outcomes.offset[j] = find_result.getOffset();
    }
};

}

}
