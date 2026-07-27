#include <Interpreters/HashJoin/AmacMode.h>

#include <cstdlib>
#include <cstring>

namespace DB
{

namespace
{

AmacMode modeFromEnv()
{
    const char * value = getenv("CLICKHOUSE_JOIN_AMAC"); /// NOLINT(concurrency-mt-unsafe)
    if (!value)
        return AmacMode::Auto;
    if (strcmp(value, "0") == 0 || strcmp(value, "off") == 0)
        return AmacMode::Off;
    if (strcmp(value, "force") == 0)
        return AmacMode::Force;
    return AmacMode::Auto;
}

/// The environment is read once, on the first engagement check; `setAmacModeForTests` then
/// simply overwrites the cached value (tests set it before any concurrent build runs).
AmacMode & amacModeStorage()
{
    static AmacMode mode = modeFromEnv();
    return mode;
}

}

AmacMode joinAmacMode()
{
    return amacModeStorage();
}

void setAmacModeForTests(AmacMode mode)
{
    amacModeStorage() = mode;
}

}
