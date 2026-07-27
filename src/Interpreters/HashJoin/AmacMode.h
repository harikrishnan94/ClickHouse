#pragma once

namespace DB
{

/** Process-level engagement mode of the AMAC build-insert ring (see `AmacRing.h`).
  * By the requester's decision this is a diagnostic hook, NOT a user-facing setting: the ring is
  * meant to be an implementation detail of `parallel_hash` with the `Auto` predicate as the only
  * long-term policy, and the hook exists so that A/B harnesses and tests can pin either path
  * without a server restart carrying user-visible surface.
  */
enum class AmacMode
{
    Off, /// never engage; every insert takes the sequential loop
    Auto, /// engage when the map and the section pass the size thresholds (the default)
    Force /// engage whenever the map type supports it, ignoring the size thresholds
};

/// Reads the `CLICKHOUSE_JOIN_AMAC` environment variable ONCE per process:
/// "0"/"off" -> `Off`, "force" -> `Force`, unset or anything else (incl. "1"/"auto") -> `Auto`.
AmacMode joinAmacMode();

/// Overrides the mode for unit tests (takes precedence over the environment).
void setAmacModeForTests(AmacMode mode);

}
