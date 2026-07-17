#include <Interpreters/PartitionedHashJoin/PartitionedHashJoinProbeImpl.h>

namespace DB
{

/// Explicit instantiations of the routed probe for RIGHT kinds (see `probeDispatch`).
template JoinResultPtr PartitionedHashJoin::probeImpl<JoinKind::Right, JoinStrictness::All, HashJoin::MapsAll>(Block);
template JoinResultPtr PartitionedHashJoin::probeImpl<JoinKind::Right, JoinStrictness::RightAny, HashJoin::MapsAll>(Block);
template JoinResultPtr PartitionedHashJoin::probeImpl<JoinKind::Right, JoinStrictness::Any, HashJoin::MapsAll>(Block);
template JoinResultPtr PartitionedHashJoin::probeImpl<JoinKind::Right, JoinStrictness::Semi, HashJoin::MapsAll>(Block);
template JoinResultPtr PartitionedHashJoin::probeImpl<JoinKind::Right, JoinStrictness::Anti, HashJoin::MapsAll>(Block);

}
