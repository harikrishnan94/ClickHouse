#pragma once

#include <Interpreters/HashJoin/HashJoin.h>

#include <concepts>

namespace DB
{

struct JoinOnKeyColumns;

}

/// What a lookup method of the hash join probe implements, and the arguments of a key search. A new lookup method includes
/// this header, not `FindKeys.h`.
namespace DB::JoinProbe
{

/// The rows of a block that one key search visits: the range `[first, first + size)`, or, with `rows` set, the
/// increasing rows `rows[0..size)`.
struct ProbeRows
{
    size_t first = 0;
    const UInt32 * rows = nullptr;
    size_t size = 0;
};

/// What the key search found for one left row in one ON disjunct: 0 if the key is not there, is NULL, or the ON section
/// masks the row out. Otherwise the `RowRef` word (`MapsOne`), the `RowRefList` word (`MapsAll`), the address of the key's
/// `SortedLookupVectorBase` (`MapsAsof`), or 1 (`MapsSet`).
using MatchWord = UInt64;

/// What `findKeys` writes besides the match words, and which rows it visits.
enum class FindKeysMode : uint8_t
{
    Words,
    /// Also the cell offset of each key, for a join with a used flag per key.
    WordsAndKeyOffsets,
    /// Stops after the row at which the right rows of the found keys reach `max_output_rows`.
    WordsUpToRowLimit,
    WordsAndKeyOffsetsUpToRowLimit,
    /// Visits the rows of the list `rows.rows`, not a range.
    WordsOfRowList,
};

/// One key search: what it reads and where it writes.
struct FindKeysArgs
{
    const HashJoinTable & table;
    const JoinOnKeyColumns & keys;
    ProbeRows rows;
    bool enable_prefetch = false;
    /// Block row `ind` writes `words[ind - base]`, and `key_offsets[ind - base]` if `key_offsets` is set.
    size_t base = 0;
    MatchWord * words = nullptr;
    size_t * key_offsets = nullptr;
    size_t max_output_rows = 0;
};

/// Returns the rows it visited: all of `rows`, unless the row limit stopped it.
using FindKeysFn = size_t (*)(const FindKeysArgs &);

/// A lookup method visits every row of `args.rows`, finds its key, and calls `on_row(i, ind, find_result)` for the position `i`
/// and the block row `ind`; `on_row` may return false to stop after that row. It returns the rows it visited. A row with a NULL
/// key or one the ON-section mask filters out gets a result that is not found; `fast_path` assumes there is none.
/// `findKeys` also passes `keep_key_getter = true` (a fifth template argument, false by default), because it searches once per
/// batch: the method may then keep a getter that is costly to build for the later batches, as `keyGetterOf` does.
/// `retires_rows_in_order`: `on_row` comes in increasing `i`. A fused function and the row limit of `findKeys` need it, because
/// they count the output in row order.
template <typename Lookup>
concept LookupMethod = requires(const FindKeysArgs & args)
{
    { Lookup::retires_rows_in_order } -> std::convertible_to<bool>;
    {
        Lookup::template run<HashJoin::Type::key64, HashJoin::MapsOne, true, false>(args, [](size_t, size_t, const auto &) { })
    } -> std::same_as<size_t>;
};

}
