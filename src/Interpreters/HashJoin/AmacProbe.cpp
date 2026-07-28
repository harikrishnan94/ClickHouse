#include <Interpreters/HashJoin/AmacProbeImpl.h>

namespace DB
{

#define M(TYPE) AMAC_FIND_PASS_INSTANTIATIONS(, TYPE)
APPLY_FOR_AMAC_BUILD_JOIN_VARIANTS(M)
#undef M

}
