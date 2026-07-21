#include <Interpreters/PartitionedHashJoin/PartitionedHashJoinProbeImpl.h>

namespace DB
{

/// Explicit instantiations of the routed probe for FULL kinds (see `probeDispatch`).
template JoinResultPtr PartitionedHashJoin::probeImpl<JoinKind::Full, JoinStrictness::All, HashJoin::MapsAll>(Block, size_t);
template JoinResultPtr PartitionedHashJoin::probeImpl<JoinKind::Full, JoinStrictness::RightAny, HashJoin::MapsAll>(Block, size_t);

}
