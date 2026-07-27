#pragma once
#include <Interpreters/HashJoin/HashJoin.h>
#include <Interpreters/HashJoin/HashJoinMethods.h>

namespace DB
{

/** The order-preserving routed probe of `ConcurrentHashJoin` (`parallel_hash`). The left block is
  * NOT scattered: the whole block runs through one `joinBlockImpl` whose per-row lookup follows
  * the row's route word into that slot's hash map (`slot = slot_ids[ind]`), while the emit stays
  * in left-row order through the standard `AddedColumns`/`HashJoinResult` machinery — built from
  * slot 0, whose shared `StoredColumnsIndex` resolves every slot's stored blocks. Used flags stay
  * per-slot with slot-local offsets, so `NotJoinedHash`/`getNonJoinedBlocks` are untouched.
  *
  * The per-row loop is split into find and emit phases behind a `precomputed` seam: above the
  * AMAC engagement threshold the find half runs as the out-of-order AMAC find pass (see
  * `AmacProbe.h`) and the emit half consumes its per-row results in order; below it the plain
  * in-order routed loop runs — both emit in left-row order, so the probe preserves the left
  * block order by construction.
  *
  * `MapsTemplate` is `HashJoin::MapsOne`/`MapsAll`/`MapsAsof`, exactly as in `HashJoinMethods`.
  */
template <JoinKind KIND, JoinStrictness STRICTNESS, typename MapsTemplate>
class RoutedHashJoinMethods
{
public:
    /// `slot_joins` are the per-slot `HashJoin`s (all sharing one `StoredColumnsIndex`);
    /// `block` wraps the ORIGINAL left block (zero-copy); `slot_ids` carries one route word per
    /// source-block row (null when there is a single slot).
    static JoinResultPtr joinBlockImpl(
        const std::vector<const HashJoin *> & slot_joins,
        ScatteredBlock block,
        const Block & block_with_columns_to_add,
        const UInt64 * slot_ids);

private:
    template <typename AddedColumns>
    static size_t switchJoinRightColumns(
        const std::vector<const HashJoin *> & slot_joins,
        AddedColumns & added_columns,
        const ScatteredBlock::Selector & selector,
        const UInt64 * slot_ids);

    /// The pre-loop dispatch layer: the additional-filter path (mixed ON conditions) and the
    /// selector-shape split.
    template <typename KeyGetter, typename Map, typename AddedColumns>
    static size_t joinRightColumnsRouted(
        const std::vector<const HashJoin *> & slot_joins,
        KeyGetter && key_getter,
        const std::vector<const Map *> & maps_by_slot,
        const std::vector<JoinStuff::JoinUsedFlags *> & flags_by_slot,
        AddedColumns & added_columns,
        const ScatteredBlock::Selector & selector,
        const UInt64 * slot_ids);

    /// The routed probe loop (single join clause; `parallel_hash` supports no disjuncts).
    template <typename KeyGetter, typename Map, typename AddedColumns, typename Selector>
    static size_t joinRightColumns(
        const std::vector<const HashJoin *> & slot_joins,
        KeyGetter & key_getter,
        const std::vector<const Map *> & maps_by_slot,
        const std::vector<JoinStuff::JoinUsedFlags *> & flags_by_slot,
        AddedColumns & added_columns,
        const Selector & selector,
        const UInt64 * slot_ids);
};

/// Instantiated ahead in the RoutedHashJoin*.cpp files (one per strictness), mirroring the
/// `HashJoinMethods` list above.
extern template class RoutedHashJoinMethods<JoinKind::Left, JoinStrictness::RightAny, HashJoin::MapsOne>;
extern template class RoutedHashJoinMethods<JoinKind::Left, JoinStrictness::RightAny, HashJoin::MapsAll>;
extern template class RoutedHashJoinMethods<JoinKind::Left, JoinStrictness::Any, HashJoin::MapsOne>;
extern template class RoutedHashJoinMethods<JoinKind::Left, JoinStrictness::Any, HashJoin::MapsAll>;
extern template class RoutedHashJoinMethods<JoinKind::Left, JoinStrictness::All, HashJoin::MapsAll>;
extern template class RoutedHashJoinMethods<JoinKind::Left, JoinStrictness::Semi, HashJoin::MapsOne>;
extern template class RoutedHashJoinMethods<JoinKind::Left, JoinStrictness::Semi, HashJoin::MapsAll>;
extern template class RoutedHashJoinMethods<JoinKind::Left, JoinStrictness::Anti, HashJoin::MapsOne>;
extern template class RoutedHashJoinMethods<JoinKind::Left, JoinStrictness::Anti, HashJoin::MapsAll>;
extern template class RoutedHashJoinMethods<JoinKind::Left, JoinStrictness::Asof, HashJoin::MapsAsof>;

extern template class RoutedHashJoinMethods<JoinKind::Right, JoinStrictness::RightAny, HashJoin::MapsAll>;
extern template class RoutedHashJoinMethods<JoinKind::Right, JoinStrictness::Any, HashJoin::MapsAll>;
extern template class RoutedHashJoinMethods<JoinKind::Right, JoinStrictness::All, HashJoin::MapsAll>;
extern template class RoutedHashJoinMethods<JoinKind::Right, JoinStrictness::Semi, HashJoin::MapsAll>;
extern template class RoutedHashJoinMethods<JoinKind::Right, JoinStrictness::Anti, HashJoin::MapsAll>;
extern template class RoutedHashJoinMethods<JoinKind::Right, JoinStrictness::Asof, HashJoin::MapsAsof>;

extern template class RoutedHashJoinMethods<JoinKind::Inner, JoinStrictness::RightAny, HashJoin::MapsOne>;
extern template class RoutedHashJoinMethods<JoinKind::Inner, JoinStrictness::RightAny, HashJoin::MapsAll>;
extern template class RoutedHashJoinMethods<JoinKind::Inner, JoinStrictness::Any, HashJoin::MapsOne>;
extern template class RoutedHashJoinMethods<JoinKind::Inner, JoinStrictness::Any, HashJoin::MapsAll>;
extern template class RoutedHashJoinMethods<JoinKind::Inner, JoinStrictness::All, HashJoin::MapsAll>;
extern template class RoutedHashJoinMethods<JoinKind::Inner, JoinStrictness::Semi, HashJoin::MapsOne>;
extern template class RoutedHashJoinMethods<JoinKind::Inner, JoinStrictness::Anti, HashJoin::MapsOne>;
extern template class RoutedHashJoinMethods<JoinKind::Inner, JoinStrictness::Asof, HashJoin::MapsAsof>;

extern template class RoutedHashJoinMethods<JoinKind::Full, JoinStrictness::RightAny, HashJoin::MapsAll>;
extern template class RoutedHashJoinMethods<JoinKind::Full, JoinStrictness::Any, HashJoin::MapsAll>;
extern template class RoutedHashJoinMethods<JoinKind::Full, JoinStrictness::All, HashJoin::MapsAll>;
extern template class RoutedHashJoinMethods<JoinKind::Full, JoinStrictness::Semi, HashJoin::MapsOne>;
extern template class RoutedHashJoinMethods<JoinKind::Full, JoinStrictness::Anti, HashJoin::MapsOne>;
extern template class RoutedHashJoinMethods<JoinKind::Full, JoinStrictness::Asof, HashJoin::MapsAsof>;
}
