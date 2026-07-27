#include <Interpreters/HashJoin/HashJoinRoutedMethodsImpl.h>

namespace DB
{
template class RoutedHashJoinMethods<JoinKind::Left, JoinStrictness::RightAny, HashJoin::MapsOne>;
template class RoutedHashJoinMethods<JoinKind::Left, JoinStrictness::RightAny, HashJoin::MapsAll>;
template class RoutedHashJoinMethods<JoinKind::Right, JoinStrictness::RightAny, HashJoin::MapsAll>;
template class RoutedHashJoinMethods<JoinKind::Inner, JoinStrictness::RightAny, HashJoin::MapsOne>;
template class RoutedHashJoinMethods<JoinKind::Inner, JoinStrictness::RightAny, HashJoin::MapsAll>;
template class RoutedHashJoinMethods<JoinKind::Full, JoinStrictness::RightAny, HashJoin::MapsAll>;
}
