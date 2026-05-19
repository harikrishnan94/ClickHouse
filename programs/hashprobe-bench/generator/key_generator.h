#pragma once

/// hashprobe-bench/generator/key_generator.h
///
/// KeyGenerator: deterministic build- and probe-side key stream generator.
///
/// Build side:   `build_rows` rows drawn from a universe of `build_distinct_keys`
///               distinct key tuples.  For ALL strictness at least one key repeats.
/// Probe side:   `probe_rows` rows; exactly round(match_rate x probe_rows) rows come
///               from [1, build_distinct_keys] and the rest come from
///               [build_distinct_keys+1, maxKeyValue()].
///               All real key values are >= 1; zero is reserved for NULL encoding only.
///
/// Null fraction is applied per-cell after the key value is assigned.
/// A.2 invariants are enforced at construction time (not runtime).

#include <hashprobe_bench/config.h>
#include <hashprobe_bench/types.h>

#include <cstdint>
#include <memory>
#include <vector>

namespace DB::HashProbeBench
{

class KeyGenerator
{
public:
    struct Params
    {
        KeyShape         shape;
        StrictnessConfig strictness;
        uint64_t         build_rows;
        uint64_t         build_distinct_keys; ///< A.2 invariants checked in ctor
        uint64_t         probe_rows;
        double           match_rate;          ///< [0.0, 1.0]
        double           null_fraction;       ///< [0.0, 1.0] per-cell null probability
        uint64_t         seed;
    };

    /// One key row (build or probe).
    struct KeyRow
    {
        std::vector<uint64_t> key_values;          ///< shape.n values; 0 for NULL cells
        std::vector<uint8_t>  null_mask;           ///< shape.n bytes; 1=NULL
        uint64_t              payload;             ///< Row index; unique per stream
        bool                  from_build_universe; ///< true iff key in [1, build_distinct_keys]
    };

    class Iterator
    {
    public:
        virtual ~Iterator() = default;
        /// Fills out, returns true while rows remain; false when exhausted.
        virtual bool next(KeyRow & out) = 0;
    };

    /// Constructor enforces A.2 invariants; throws std::invalid_argument on violation.
    explicit KeyGenerator(Params p);

    /// Fresh build-side iterator, deterministic from seed.
    std::unique_ptr<Iterator> buildIterator() const;

    /// Fresh probe-side iterator, deterministic from seed.
    std::unique_ptr<Iterator> probeIterator() const;

    const Params &   params()               const { return p_; }
    uint64_t         getBuildRows()          const { return p_.build_rows; }
    uint64_t         getBuildDistinctKeys()  const { return p_.build_distinct_keys; }
    double           getBuildRowToKeyRatio() const;

    /// Max representable key value: 2^31-1 (W32) or 2^63-1 (W64).
    uint64_t maxKeyValue() const;

private:
    Params p_;
};

} // namespace DB::HashProbeBench
