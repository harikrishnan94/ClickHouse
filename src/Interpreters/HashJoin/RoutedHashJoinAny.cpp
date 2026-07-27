#include <Interpreters/HashJoin/HashJoinRoutedMethodsImpl.h>

namespace DB
{
template class RoutedHashJoinMethods<JoinKind::Left, JoinStrictness::Any, HashJoin::MapsOne>;
template class RoutedHashJoinMethods<JoinKind::Left, JoinStrictness::Any, HashJoin::MapsAll>;
template class RoutedHashJoinMethods<JoinKind::Right, JoinStrictness::Any, HashJoin::MapsAll>;
template class RoutedHashJoinMethods<JoinKind::Inner, JoinStrictness::Any, HashJoin::MapsOne>;
template class RoutedHashJoinMethods<JoinKind::Inner, JoinStrictness::Any, HashJoin::MapsAll>;
template class RoutedHashJoinMethods<JoinKind::Full, JoinStrictness::Any, HashJoin::MapsAll>;
}
