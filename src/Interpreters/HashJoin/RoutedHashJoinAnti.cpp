#include <Interpreters/HashJoin/HashJoinRoutedMethodsImpl.h>

namespace DB
{
template class RoutedHashJoinMethods<JoinKind::Left, JoinStrictness::Anti, HashJoin::MapsOne>;
template class RoutedHashJoinMethods<JoinKind::Left, JoinStrictness::Anti, HashJoin::MapsAll>;
template class RoutedHashJoinMethods<JoinKind::Right, JoinStrictness::Anti, HashJoin::MapsAll>;
template class RoutedHashJoinMethods<JoinKind::Inner, JoinStrictness::Anti, HashJoin::MapsOne>;
template class RoutedHashJoinMethods<JoinKind::Full, JoinStrictness::Anti, HashJoin::MapsOne>;
}
