#include <Interpreters/HashJoin/HashJoinRoutedMethodsImpl.h>

namespace DB
{
template class RoutedHashJoinMethods<JoinKind::Left, JoinStrictness::Semi, HashJoin::MapsOne>;
template class RoutedHashJoinMethods<JoinKind::Left, JoinStrictness::Semi, HashJoin::MapsAll>;
template class RoutedHashJoinMethods<JoinKind::Right, JoinStrictness::Semi, HashJoin::MapsAll>;
template class RoutedHashJoinMethods<JoinKind::Inner, JoinStrictness::Semi, HashJoin::MapsOne>;
template class RoutedHashJoinMethods<JoinKind::Full, JoinStrictness::Semi, HashJoin::MapsOne>;
}
