#pragma once

#include <Interpreters/HashJoin/HashJoinTable.h>
#include <Interpreters/JoinUtils.h>

#include <atomic>
#include <concepts>

/// What an emplace method of the hash join build implements, and the arguments of one insert. A new emplace method includes
/// this header, not `InsertKeys.h`.
namespace DB::JoinBuild
{

/// The rows of one insert: the range `[first, first + size)`, or, with `rows` set, the increasing rows `rows[0..size)`.
struct BuildRows
{
    size_t first = 0;
    const UInt64 * rows = nullptr;
    size_t size = 0;
};

/// One insert into one map: what it reads besides the map and the key getter, and where it writes. No other insert into the
/// map runs meanwhile.
struct InsertKeysArgs
{
    BuildRows rows;
    ConstNullMapPtr null_map = nullptr;
    const JoinCommon::JoinMask & join_mask;
    Arena & pool;
    bool enable_prefetch = false;
    /// The insert adds its count of new keys here, also when it throws: the keys added before the throw stay in the map.
    std::atomic<size_t> & num_keys;
    /// For a key getter that builds many keys at once (`PreparedJoinKeys`): the skip bytes of the block, and, if set, the
    /// keys that the slot scatter of the parallel layout built for every row of the block that it routed (`buildAllKeys`).
    const UInt8 * skip = nullptr;
    const PaddedPODArray<UInt128> * prebuilt_keys = nullptr;

    /// What storing a row reads. The `ASOF` fields are set only for `MapsAsof`.
    UInt32 block_no = 0;
    bool any_take_last_row = false;
    const IColumn * asof_column = nullptr;
    TypeIndex asof_type{};
    ASOFJoinInequality asof_inequality{};
};

/// An emplace method emplaces into `map` the key of every row of `args.rows`, except the rows with a NULL key and the rows
/// the ON-section mask filters out. It calls `store(ind, emplace_result)` for the block row `ind` right after each emplace,
/// before the next one: an emplace can move cells, so an emplace result is valid only until the next emplace. The store
/// calls `isInserted` of the emplace result and, except for a set map, `getMapped`. `row_list` tells whether
/// `args.rows.rows` is set. Returns whether it skipped a NULL key.
///
/// `run` must be `ALWAYS_INLINE`: the store captures the locals of `insertKeys` by reference, and they stay in registers only
/// when `run` is inlined into it.
template <typename Emplace>
concept EmplaceMethod = requires(
    JoinMapOf<HashJoin::Type::key64, HashJoin::MapsOne> & map,
    typename KeyGetterForType<HashJoin::Type::key64, JoinMapOf<HashJoin::Type::key64, HashJoin::MapsOne>>::Type & key_getter,
    const InsertKeysArgs & args)
{
    { Emplace::template run<HashJoin::MapsOne, /*row_list=*/false>(map, key_getter, args, [](size_t, auto &) { }) } -> std::same_as<bool>;
};

}
