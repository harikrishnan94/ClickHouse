#pragma once

#include <atomic>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include <base/types.h>

#include <Core/Block.h>
#include <Interpreters/IJoin.h>
#include <Common/ThreadPool.h>

namespace DB
{
class TableJoin;
}

namespace DB::JoinBench
{

/// Prevents the compiler from optimizing benchmark kernels away.
extern std::atomic<UInt64> g_sink;

/// ClickHouse thread pool (threads carry ThreadStatus, so per-thread memory tracking is cheap).
/// Threads stay warm between runs (max_free_threads == max_threads), so per-iteration timing
/// does not include thread creation.
using SimpleThreadPool = ThreadPoolImpl<ThreadFromGlobalPoolImpl</*propagate_opentelemetry_context*/ false, /*global_trace_collector_allowed*/ false>>;

class WorkerPool
{
public:
    explicit WorkerPool(size_t num_threads_);

    /// Runs task(thread_index) on all threads. Returns elapsed wall seconds.
    double run(const std::function<void(size_t)> & task);

    size_t size() const { return num_threads; }

private:
    size_t num_threads;
    SimpleThreadPool pool;
};

struct JoinStats
{
    double build_sec = 0;
    double probe_sec = 0;
    size_t matches = 0;

    double total() const { return build_sec + probe_sec; }
};

/// One join algorithm under test. Implementations use the driver's worker pool internally;
/// the driver times the two phases.
class IJoinBench
{
public:
    virtual ~IJoinBench() = default;
    virtual std::string name() const = 0;

    /// Consume the build (right) side.
    virtual void build(const std::vector<Block> & blocks) = 0;

    /// Join the probe (left) side, materializing real output Blocks (dropped after counting).
    /// Returns the number of output rows.
    virtual size_t probe(const std::vector<Block> & blocks) = 0;

    /// Optional sub-phase timing details for reporting.
    virtual std::string phaseBreakdown() const { return {}; }
};

/// A partition holds a list of scattered column chunks.
struct Chunk
{
    Columns columns;
    size_t rows = 0;
};
using ChunkList = std::vector<Chunk>;

/// Splits log2(p_star) partition bits into passes of at most log2(f_max) bits each.
std::vector<size_t> computePassBits(size_t p_star, size_t f_max);

/// Multi-pass radix scatter of one side by the hash of its first (key) column: this is the
/// partitioning code the radix join runs, also used to measure the scatter bandwidth term.
std::vector<ChunkList> scatterSide(WorkerPool & pool, const std::vector<Block> & blocks, const std::vector<size_t> & pass_bits);

/// Shared setup of the join metadata: INNER ALL join on the first column of each side.
std::shared_ptr<TableJoin> makeTableJoin(const Block & left_header, const Block & right_header);

/// Materializes all output blocks of one join result, returns the number of output rows.
size_t drainJoinResult(JoinResultPtr result);

/// The driver: times the two phases of a join implementation through the common interface.
JoinStats driveJoin(IJoinBench & join, const std::vector<Block> & build_blocks, const std::vector<Block> & probe_blocks);

}
