#include <Interpreters/UnifiedHashJoin/HashJoinMethodsImpl.h>

namespace DB
{
namespace Unified
{
template class HashJoinMethods<JoinKind::Left, JoinStrictness::Any, HashJoin::MapsAll>;
}

}
