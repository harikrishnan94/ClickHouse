#include <Interpreters/UnifiedHashJoin/HashJoinMethodsImpl.h>

namespace DB
{
namespace Unified
{
template class HashJoinMethods<JoinKind::Full, JoinStrictness::Asof, HashJoin::MapsAsof>;
}

}
