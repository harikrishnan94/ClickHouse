#include <Interpreters/PartitionedHashJoin/PartitionedHashJoinProbeImpl.h>

namespace DB
{

/// Explicit instantiations of the routed probe for INNER kinds (see `probeDispatch`).
template JoinResultPtr PartitionedHashJoin::probeImpl<JoinKind::Inner, JoinStrictness::All, HashJoin::MapsAll>(Block, size_t);
template JoinResultPtr PartitionedHashJoin::probeImpl<JoinKind::Inner, JoinStrictness::RightAny, HashJoin::MapsOne>(Block, size_t);
template JoinResultPtr PartitionedHashJoin::probeImpl<JoinKind::Inner, JoinStrictness::RightAny, HashJoin::MapsAll>(Block, size_t);
template JoinResultPtr PartitionedHashJoin::probeImpl<JoinKind::Inner, JoinStrictness::Any, HashJoin::MapsOne>(Block, size_t);
template JoinResultPtr PartitionedHashJoin::probeImpl<JoinKind::Inner, JoinStrictness::Asof, HashJoin::MapsAsof>(Block, size_t);

}
