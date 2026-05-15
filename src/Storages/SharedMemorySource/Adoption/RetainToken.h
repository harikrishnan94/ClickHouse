#pragma once

#include <functional>
#include <memory>
#include <utility>

namespace DB
{

/// A RetainToken pins a producer-side SHM region for as long as any ClickHouse object
/// references its bytes. The pollable source (T3.2) creates one at retain-acquisition time
/// (producer-slot refcount 0 -> 1) and the adoption layer (T3.1) threads it through every
/// adopted IColumn (or derived handle) via std::shared_ptr<void> aliasing. When the last
/// alias-copy drops, the release callback fires exactly once and is responsible for
/// decrementing the producer-side retain refcount in the SHM control plane.
///
/// Spec authority: system spec §Cross-component invariants I5 (Retain correctness);
/// adoption-layer spec §Retain and charge handle semantics.
using RetainToken = std::shared_ptr<void>;

/// Construct a RetainToken backed by `release_callback`. The callback runs exactly once when
/// the last RetainToken alias-copy drops.
///
/// The callback must not throw: it executes inside the shared_ptr control block's deleter,
/// and std::shared_ptr's contract terminates the process if an exception escapes deletion.
/// (We use std::function<void()> rather than std::function<void() noexcept> because the
/// standard library does not provide a partial specialization for noexcept-qualified
/// function types; the noexcept obligation is contractual, enforced by the std::terminate
/// fallback above.)
///
/// Passing an empty std::function is benign - the resulting token's deleter is a no-op.
inline RetainToken makeRetainToken(std::function<void()> release_callback)
{
    struct Holder
    {
        std::function<void()> cb;
        explicit Holder(std::function<void()> c) noexcept : cb(std::move(c)) {}
        ~Holder()
        {
            if (cb)
                cb();
        }
    };
    return std::shared_ptr<void>(new Holder(std::move(release_callback)));
}

}
