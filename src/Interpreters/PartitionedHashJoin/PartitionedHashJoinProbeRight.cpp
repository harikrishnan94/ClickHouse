#include <Interpreters/PartitionedHashJoin/PartitionedHashJoinProbeImpl.h>

namespace DB
{

/// Explicit instantiations of the routed probe for RIGHT kinds (see `probeDispatch`).
template JoinResultPtr PartitionedHashJoin::probeImpl<JoinKind::Right, JoinStrictness::All, HashJoin::MapsAll>(Block, size_t);
template JoinResultPtr PartitionedHashJoin::probeImpl<JoinKind::Right, JoinStrictness::RightAny, HashJoin::MapsAll>(Block, size_t);
template JoinResultPtr PartitionedHashJoin::probeImpl<JoinKind::Right, JoinStrictness::Any, HashJoin::MapsAll>(Block, size_t);
template JoinResultPtr PartitionedHashJoin::probeImpl<JoinKind::Right, JoinStrictness::Semi, HashJoin::MapsAll>(Block, size_t);
template JoinResultPtr PartitionedHashJoin::probeImpl<JoinKind::Right, JoinStrictness::Anti, HashJoin::MapsAll>(Block, size_t);

}
