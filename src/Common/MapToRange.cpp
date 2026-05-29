#include <Common/MapToRange.h>
#include <Common/TargetSpecific.h>

namespace DB
{

MULTITARGET_FUNCTION_X86_V4_V3(
    MULTITARGET_FUNCTION_HEADER(static void NO_INLINE),
    mapToRangeImpl,
    MULTITARGET_FUNCTION_BODY(
        (const UInt32 * hashes, size_t n, UInt32 range_size, UInt64 * result) /// NOLINT(bugprone-macro-repeated-side-effects)
        {
            for (size_t i = 0; i < n; ++i)
                result[i] = (static_cast<UInt64>(hashes[i]) * range_size) >> 32;
        }))

void mapToRange(const UInt32 * hashes, size_t n, UInt32 range_size, UInt64 * result)
{
#if USE_MULTITARGET_CODE
    if (isArchSupported(TargetArch::x86_64_v4))
    {
        mapToRangeImpl_x86_64_v4(hashes, n, range_size, result);
        return;
    }
    if (isArchSupported(TargetArch::x86_64_v3))
    {
        mapToRangeImpl_x86_64_v3(hashes, n, range_size, result);
        return;
    }
#endif
    mapToRangeImpl(hashes, n, range_size, result);
}

} // namespace DB
