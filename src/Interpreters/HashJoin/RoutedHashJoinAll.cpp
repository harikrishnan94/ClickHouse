#include <Interpreters/HashJoin/HashJoinRoutedMethodsImpl.h>

namespace DB
{
template class RoutedHashJoinMethods<JoinKind::Left, JoinStrictness::All, HashJoin::MapsAll>;
template class RoutedHashJoinMethods<JoinKind::Right, JoinStrictness::All, HashJoin::MapsAll>;
template class RoutedHashJoinMethods<JoinKind::Inner, JoinStrictness::All, HashJoin::MapsAll>;
template class RoutedHashJoinMethods<JoinKind::Full, JoinStrictness::All, HashJoin::MapsAll>;
}
