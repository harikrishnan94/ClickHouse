#include <Interpreters/UnifiedHashJoin/HashJoinMethodsImpl.h>

namespace DB
{
namespace Unified
{
template class HashJoinMethods<JoinKind::Right, JoinStrictness::RightAny, HashJoin::MapsAll>;
}

}
