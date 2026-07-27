#include <Interpreters/HashJoin/HashJoinRoutedMethodsImpl.h>

namespace DB
{
template class RoutedHashJoinMethods<JoinKind::Left, JoinStrictness::Asof, HashJoin::MapsAsof>;
template class RoutedHashJoinMethods<JoinKind::Right, JoinStrictness::Asof, HashJoin::MapsAsof>;
template class RoutedHashJoinMethods<JoinKind::Inner, JoinStrictness::Asof, HashJoin::MapsAsof>;
template class RoutedHashJoinMethods<JoinKind::Full, JoinStrictness::Asof, HashJoin::MapsAsof>;
}
