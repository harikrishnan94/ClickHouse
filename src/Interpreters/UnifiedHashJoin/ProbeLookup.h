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

/** Single-map probe lookup: walk a row range, record each `findKey` outcome, then let a
  * second pass emit. Recording (rather than emitting inside the loop) lets later rows'
  * hash-table walks overlap the memory latency of earlier ones; whether that pays for the
  * scratch traffic is a call-site choice (`lookupBatch` + consume), not a second copy of
  * this loop.
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

/** What the recording pass hands to the emit pass, indexed by the row's position in the batch.
  *
  * `found[j]`:
  *   0         - the row matched nothing. A miss and a row excluded by the skip bytes (a NULL
  *               key or an ON-section condition) are recorded the same way, because the emit
  *               treats them the same way - so no second pass carries skip logic.
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
    /// The lookup writes `found[0 .. count)`. This is a raw pointer rather than the array itself
    /// because for some joins the outcomes ARE the output: a lazy ALL join that adds missing
    /// rows appends exactly one word per probe row to `LazyOutput::row_refs`, and that word is
    /// the one the lookup already has - a match's cell word, or zero, which is what `addDefault`
    /// appends for a miss. Pointing the lookup straight at that array saves writing every word
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

    /// `external` must have room for `rows`, and must not be reallocated while the lookup and
    /// the second pass run over it.
    void useExternal(UInt64 * external, size_t rows, bool need_flags)
    {
        found = external;
        if (need_flags)
            offset.resize(rows);
    }
};

/** Record the outcome and emit nothing, so a second pass can do the whole emit later.
  * Branch preferred over cmov here; the difference was not measurable.
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

/** One `findKey` per row, in row order. Writes every position of `[0, count)` so a miss and an
  * unwritten entry are never confused, and delivers results in row order for the emit pass.
  *
  * `prefetch_at` is the probe's look-ahead software prefetch, called with the ABSOLUTE row so
  * its look-ahead reaches past the end of a batch.
  *
  * `fast_path` (skip_data == nullptr) folds into two NO_INLINE inner loops so the single-map
  * probe is not instantiated twice for skip vs no-skip. The loops stay outlined: ALWAYS_INLINE
  * here doubled the hot `run` body and regressed String keys ~3%+.
  */
struct SequentialLookup
{
    template <bool need_flags, typename KeyGetter, typename Map, typename Selector, typename PrefetchAt>
    static void run(
        KeyGetter & key_getter,
        const Map & map,
        const Selector & selector,
        const UInt8 * skip_data,
        Arena & pool,
        size_t begin,
        size_t count,
        PrefetchAt && prefetch_at,
        ProbeOutcomes & outcomes)
    {
        RecordOutcomeSink<need_flags> sink{outcomes};
        if (skip_data == nullptr)
            runImpl</*with_skip=*/false, need_flags>(
                key_getter, map, selector, skip_data, pool, begin, count, prefetch_at, sink);
        else
            runImpl</*with_skip=*/true, need_flags>(
                key_getter, map, selector, skip_data, pool, begin, count, prefetch_at, sink);
    }

    template <bool with_skip, bool need_flags, typename KeyGetter, typename Map, typename Selector, typename PrefetchAt>
    NO_INLINE static void runImpl(
        KeyGetter & key_getter,
        const Map & map,
        const Selector & selector,
        const UInt8 * skip_data [[maybe_unused]],
        Arena & pool,
        size_t begin,
        size_t count,
        PrefetchAt && prefetch_at,
        RecordOutcomeSink<need_flags> & sink)
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

}

}
