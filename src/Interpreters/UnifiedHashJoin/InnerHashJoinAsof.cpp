
#include <Interpreters/UnifiedHashJoin/HashJoinMethodsImpl.h>

namespace DB
{
namespace Unified
{
template class HashJoinMethods<JoinKind::Inner, JoinStrictness::Asof, HashJoin::MapsAsof>;
}

}
