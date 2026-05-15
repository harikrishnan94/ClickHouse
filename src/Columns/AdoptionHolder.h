#pragma once

#include <memory>
#include <utility>

namespace DB
{

/// Per-column heap-allocated holder for the two shared_ptr handles that pin SHM
/// adoption state (the producer-side retain_token and the MemoryTracker
/// charge_handle). Non-adopted columns hold a null AdoptionHolder pointer
/// (8 bytes per column instance) instead of two inline std::shared_ptr<void>
/// members (~32 bytes per instance on common 64-bit ABIs). Adopted columns
/// allocate one of these on construction; it's freed exactly once on column
/// destruction, which is also when both handles release.
///
/// Authority for the two-handle contract:
///   - adoption-layer spec §Retain and charge handle semantics
///   - system spec I5 (Retain correctness), I7 (charge/release pairing)
struct AdoptionHolder
{
    std::shared_ptr<void> retain_token;
    std::shared_ptr<void> charge_handle;

    AdoptionHolder(std::shared_ptr<void> rt, std::shared_ptr<void> ch)
        : retain_token(std::move(rt)), charge_handle(std::move(ch))
    {
    }
};

}
