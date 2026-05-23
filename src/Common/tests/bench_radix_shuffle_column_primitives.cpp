#include <Columns/ColumnDecimal.h>
#include <Columns/ColumnFixedString.h>
#include <Columns/ColumnNullable.h>
#include <Columns/ColumnString.h>
#include <Columns/ColumnVector.h>
#include <Columns/ColumnsNumber.h>
#include <Common/RadixShuffle/Allocator.h>
#include <Common/RadixShuffle/ColumnPrimitives.h>
#include <Common/RadixShuffle/ColumnPrimitives/FixedWidth.h>
#include <Common/RadixShuffle/ColumnPrimitives/Nullable.h>
#include <Common/RadixShuffle/ColumnPrimitives/String.h>
#include <Common/RadixShuffle/ColumnPrimitivesDispatch.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include <fmt/format.h>


namespace
{

using namespace DB;
namespace rs = DB::RadixShuffle;


/// Workload configuration knob.
struct Workload
{
    size_t batch_size;
    size_t partitions;
    size_t columns;
    size_t threads;
    size_t total_rows;
};


/// Reported numbers per configuration.
struct Result
{
    std::string column_type;
    size_t batch_size = 0;
    size_t partitions = 0;
    size_t columns = 0;
    size_t threads = 0;
    size_t total_rows = 0;
    double median_ns_per_row = 0.0;
    /// Source-side scatter bandwidth in GB/s, aggregated across all threads.
    /// Computed as (total source bytes scattered) / wall_seconds / 1e9,
    /// where "total source bytes" is the sum over (threads × columns ×
    /// batches_per_thread) of the source column's `byteSize()`.
    double total_bandwidth_gbs = 0.0;
};


/// Column-type knob for the sweep.
struct ColumnSpec
{
    std::string name;
    /// Factory: build one column with `n` rows from `seed`.
    std::function<MutableColumnPtr(size_t n, uint64_t seed)> make_column;
    /// ColumnPrimitives triple for this column type.
    std::function<rs::ColumnPrimitives()> make_column_primitives;
    /// Per-row "byte cost" in the source column (variable-length only).
    /// Returns 0 for fixed-width columns.
    std::function<size_t(const IColumn &, size_t row)> row_bytes;
};


/// Compute the per-(column × partition) byte cost from the source column,
/// for variable-length columns. Returns the total bytes for each partition.
std::vector<size_t> bytesPerPartition(const IColumn & col, const ColumnSpec & spec, const uint32_t * pids, size_t n, size_t P)
{
    std::vector<size_t> out(P, 0);
    if (!spec.row_bytes)
        return out;
    for (size_t i = 0; i < n; ++i)
        out[pids[i]] += spec.row_bytes(col, i);
    return out;
}


/// Quickly build a histogram (pid counts) for one batch.
void buildHistogram(const uint32_t * pids, size_t n, size_t P, std::vector<size_t> & out)
{
    out.assign(P, 0);
    for (size_t i = 0; i < n; ++i)
        ++out[pids[i]];
}


/// Per-thread state used by the benchmark. The RNG is default-constructed
/// here and explicitly seeded in `init` per-thread; the predictability of
/// the default constructor is intentional (reproducible workloads).
/// NOLINTNEXTLINE(bugprone-random-generator-seed,cert-msc32-c,cert-msc51-cpp)
struct ThreadState
{
    std::mt19937_64 rng;
    std::vector<uint32_t> pids;
    std::vector<rs::ReservationRequest> requests;
    std::vector<rs::Reservation> dst;
    std::vector<size_t> histogram;

    void init(size_t batch_size, size_t partitions, uint64_t seed)
    {
        rng.seed(seed);
        pids.resize(batch_size);
        requests.resize(partitions);
        dst.resize(partitions);
        histogram.resize(partitions);
    }

    /// Fill `pids` uniformly over [0, P).
    void newPids(size_t P)
    {
        std::uniform_int_distribution<uint32_t> dist(0, static_cast<uint32_t>(P - 1));
        for (auto & p : pids)
            p = dist(rng);
    }
};


/// Run scatter for one column, one batch, on the calling thread. The
/// per-thread state's `requests`, `dst`, and `histogram` are reused.
double scatterOneBatch(
    rs::Handle * handle,
    size_t col_idx,
    const rs::ColumnPrimitives & primitives,
    const IColumn & src,
    ThreadState & ts,
    size_t P,
    const std::vector<size_t> & bytes_per_partition_for_this_batch)
{
    /// Build histogram.
    buildHistogram(ts.pids.data(), ts.pids.size(), P, ts.histogram);

    for (size_t p = 0; p < P; ++p)
    {
        ts.requests[p].rows = ts.histogram[p];
        ts.requests[p].bytes = bytes_per_partition_for_this_batch.empty() ? 0 : bytes_per_partition_for_this_batch[p];
    }
    handle->reserve(col_idx, ts.requests.data(), ts.dst.data());

    const auto t0 = std::chrono::steady_clock::now();
    primitives.scatter(primitives, src, ts.pids.data(), ts.pids.size(), P, ts.dst.data());
    const auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::nano>(t1 - t0).count();
}


/// Run one workload configuration (batch_size × P × columns × threads ×
/// total_rows) for one column type. Returns the per-row median ns/row for
/// scatter, plus the total threads × rows-per-second bandwidth.
Result runConfig(const ColumnSpec & spec, const Workload & wl)
{
    const size_t batches_per_thread = std::max<size_t>(1, wl.total_rows / wl.threads / wl.batch_size);
    const size_t rows_per_thread = batches_per_thread * wl.batch_size;

    /// Pre-build per-thread source columns (one per column, reused across
    /// batches). Each thread gets its own to avoid contention reading the
    /// same column.
    std::vector<std::vector<MutableColumnPtr>> per_thread_sources(wl.threads);
    for (size_t t = 0; t < wl.threads; ++t)
    {
        per_thread_sources[t].reserve(wl.columns);
        for (size_t c = 0; c < wl.columns; ++c)
            per_thread_sources[t].push_back(spec.make_column(wl.batch_size, t * 1000 + c + 1));
    }

    /// Build column descs (same for every column — they are all the same type).
    rs::ColumnPrimitives proto = spec.make_column_primitives();
    std::vector<rs::ColumnDesc> descs(wl.columns, proto.column_desc);
    rs::Allocator alloc(descs, wl.partitions, rows_per_thread * wl.threads);

    /// Per-thread results: median scatter ns/row across batches.
    std::vector<double> per_thread_median_ns_per_row(wl.threads, 0.0);
    std::vector<double> per_thread_total_ns(wl.threads, 0.0);

    /// Sync barrier for the start of the parallel section.
    std::atomic<size_t> ready{0};
    std::atomic<bool> go{false};

    auto thread_fn = [&](size_t tid)
    {
        ThreadState ts;
        ts.init(wl.batch_size, wl.partitions, /*seed=*/tid * 7919 + 1);

        rs::Handle * h = alloc.acquire();

        std::vector<rs::ColumnPrimitives> primitives(wl.columns, proto);

        /// Wait for the launch signal.
        ready.fetch_add(1, std::memory_order_release);
        while (!go.load(std::memory_order_acquire))
            std::this_thread::yield();

        std::vector<double> samples;
        samples.reserve(batches_per_thread);
        double total_ns = 0.0;
        for (size_t b = 0; b < batches_per_thread; ++b)
        {
            ts.newPids(wl.partitions);
            for (size_t c = 0; c < wl.columns; ++c)
            {
                const IColumn & src = *per_thread_sources[tid][c];
                std::vector<size_t> bpp;
                if (spec.row_bytes)
                    bpp = bytesPerPartition(src, spec, ts.pids.data(), wl.batch_size, wl.partitions);
                const double ns = scatterOneBatch(h, c, primitives[c], src, ts, wl.partitions, bpp);
                samples.push_back(ns);
                total_ns += ns;
            }
        }

        alloc.release(h);

        std::sort(samples.begin(), samples.end());
        double median_ns_per_call = 0.0;
        if (!samples.empty())
        {
            const size_t mid = samples.size() / 2;
            median_ns_per_call = samples[mid];
        }
        per_thread_median_ns_per_row[tid] = median_ns_per_call / static_cast<double>(wl.batch_size);
        per_thread_total_ns[tid] = total_ns;
    };

    std::vector<std::thread> threads;
    threads.reserve(wl.threads);
    for (size_t t = 0; t < wl.threads; ++t)
        threads.emplace_back(thread_fn, t);

    /// Wait until all threads are at the barrier.
    while (ready.load(std::memory_order_acquire) < wl.threads)
        std::this_thread::yield();

    /// Launch.
    const auto t_start = std::chrono::steady_clock::now();
    go.store(true, std::memory_order_release);

    for (auto & th : threads)
        th.join();
    const auto t_end = std::chrono::steady_clock::now();
    const double wall_seconds = std::chrono::duration<double>(t_end - t_start).count();

    /// Aggregate.
    Result r;
    r.column_type = spec.name;
    r.batch_size = wl.batch_size;
    r.partitions = wl.partitions;
    r.columns = wl.columns;
    r.threads = wl.threads;
    r.total_rows = rows_per_thread * wl.threads;

    /// Median across threads of the per-thread median ns/row.
    std::vector<double> tmp = per_thread_median_ns_per_row;
    std::sort(tmp.begin(), tmp.end());
    r.median_ns_per_row = tmp[tmp.size() / 2];

    /// Aggregate source-side scatter bandwidth in GB/s. Each thread reads
    /// every per-thread source column `batches_per_thread` times; the
    /// per-batch source byte cost is the source column's `byteSize()`.
    double total_source_bytes = 0.0;
    for (size_t t = 0; t < wl.threads; ++t)
        for (size_t c = 0; c < wl.columns; ++c)
            total_source_bytes += static_cast<double>(per_thread_sources[t][c]->byteSize()) * static_cast<double>(batches_per_thread);
    r.total_bandwidth_gbs = total_source_bytes / wall_seconds / 1.0e9;
    return r;
}


/// Helper: build a ColumnVector<T> of size `n` with deterministic random
/// bytes derived from `seed`. Works for any trivially-copyable T.
template <typename T>
MutableColumnPtr makeVectorCol(size_t n, uint64_t seed)
{
    auto col = ColumnVector<T>::create();
    auto & data = col->getData();
    data.resize(n);
    std::mt19937_64 rng(seed); // NOLINT(cert-msc32-c,cert-msc51-cpp)
    auto * bytes = reinterpret_cast<unsigned char *>(data.data());
    for (size_t i = 0; i < n * sizeof(T); i += sizeof(uint64_t))
    {
        const uint64_t w = rng();
        const size_t k = std::min(sizeof(uint64_t), n * sizeof(T) - i);
        std::memcpy(bytes + i, &w, k);
    }
    return col;
}


/// Helper: build a ColumnDecimal<T> of size `n` with deterministic random
/// values derived from `seed`.
template <typename T>
MutableColumnPtr makeDecimalCol(size_t n, uint64_t seed)
{
    auto col = ColumnDecimal<T>::create(0, 4);
    auto & data = col->getData();
    data.resize(n);
    std::mt19937_64 rng(seed); // NOLINT(cert-msc32-c,cert-msc51-cpp)
    auto * bytes = reinterpret_cast<unsigned char *>(data.data());
    for (size_t i = 0; i < n * sizeof(T); i += sizeof(uint64_t))
    {
        const uint64_t w = rng();
        const size_t k = std::min(sizeof(uint64_t), n * sizeof(T) - i);
        std::memcpy(bytes + i, &w, k);
    }
    return col;
}


/// Helper: build a ColumnFixedString(W) of size `n`.
MutableColumnPtr makeFixedStringCol(size_t n, size_t width, uint64_t seed)
{
    auto col = ColumnFixedString::create(width);
    auto & chars = col->getChars();
    chars.resize(n * width);
    std::mt19937_64 rng(seed); // NOLINT(cert-msc32-c,cert-msc51-cpp)
    auto * bytes = reinterpret_cast<unsigned char *>(chars.data());
    for (size_t i = 0; i < n * width; i += sizeof(uint64_t))
    {
        const uint64_t w = rng();
        const size_t k = std::min(sizeof(uint64_t), n * width - i);
        std::memcpy(bytes + i, &w, k);
    }
    return col;
}


/// Helper: build a ColumnString with strings of varying length.
MutableColumnPtr makeStringCol(size_t n, uint64_t seed)
{
    auto col = ColumnString::create();
    std::mt19937_64 rng(seed); // NOLINT(cert-msc32-c,cert-msc51-cpp)
    std::uniform_int_distribution<size_t> dist(8, 24);
    std::string buf;
    for (size_t i = 0; i < n; ++i)
    {
        const size_t len = dist(rng);
        buf.resize(len);
        for (size_t k = 0; k < len; ++k)
            buf[k] = static_cast<char>('a' + (rng() & 0x0f));
        col->insertData(buf.data(), buf.size());
    }
    return col;
}


/// Helper: wrap any nested column in a ColumnNullable with a random null map.
MutableColumnPtr wrapNullable(MutableColumnPtr nested, uint64_t seed)
{
    const size_t n = nested->size();
    auto null_col = ColumnUInt8::create();
    auto & nm = null_col->getData();
    nm.resize(n);
    std::mt19937_64 rng(seed); // NOLINT(cert-msc32-c,cert-msc51-cpp)
    for (auto & b : nm)
        b = (rng() & 0x1) ? 1u : 0u;
    return ColumnNullable::create(std::move(nested), std::move(null_col));
}


/// Per-row byte cost for ColumnString (used to size the allocator's
/// per-partition byte requests).
size_t stringRowBytes(const IColumn & col, size_t row)
{
    const auto & sc = static_cast<const ColumnString &>(col);
    const auto & offs = sc.getOffsets();
    const UInt64 prev = (row == 0) ? 0 : offs[row - 1];
    return offs[row] - prev;
}


/// Per-row byte cost for Nullable(String): defers to the nested string's
/// per-row byte cost (the null map is fixed-width).
size_t nullableStringRowBytes(const IColumn & col, size_t row)
{
    const auto & nc = static_cast<const ColumnNullable &>(col);
    return stringRowBytes(nc.getNestedColumn(), row);
}


/// Build the column-spec table for the sweep. Coverage spans every scope-D
/// leaf type (every `ColumnVector<T>`, every `ColumnDecimal<T>`,
/// `ColumnFixedString` at several N, `ColumnString`) plus a representative
/// sample of `ColumnNullable(X)` wrappers (one per category).
std::vector<ColumnSpec> buildSpecs()
{
    std::vector<ColumnSpec> out;

    /// ColumnVector<T> — every numeric T from scope D.
    out.push_back({"ColumnVector<UInt8>", &makeVectorCol<UInt8>, []() { return rs::makeFixedWidth<UInt8>(); }, {}});
    out.push_back({"ColumnVector<UInt16>", &makeVectorCol<UInt16>, []() { return rs::makeFixedWidth<UInt16>(); }, {}});
    out.push_back({"ColumnVector<UInt32>", &makeVectorCol<UInt32>, []() { return rs::makeFixedWidth<UInt32>(); }, {}});
    out.push_back({"ColumnVector<UInt64>", &makeVectorCol<UInt64>, []() { return rs::makeFixedWidth<UInt64>(); }, {}});
    out.push_back({"ColumnVector<UInt128>", &makeVectorCol<UInt128>, []() { return rs::makeFixedWidth<UInt128>(); }, {}});
    out.push_back({"ColumnVector<UInt256>", &makeVectorCol<UInt256>, []() { return rs::makeFixedWidth<UInt256>(); }, {}});
    out.push_back({"ColumnVector<Int8>", &makeVectorCol<Int8>, []() { return rs::makeFixedWidth<Int8>(); }, {}});
    out.push_back({"ColumnVector<Int16>", &makeVectorCol<Int16>, []() { return rs::makeFixedWidth<Int16>(); }, {}});
    out.push_back({"ColumnVector<Int32>", &makeVectorCol<Int32>, []() { return rs::makeFixedWidth<Int32>(); }, {}});
    out.push_back({"ColumnVector<Int64>", &makeVectorCol<Int64>, []() { return rs::makeFixedWidth<Int64>(); }, {}});
    out.push_back({"ColumnVector<Int128>", &makeVectorCol<Int128>, []() { return rs::makeFixedWidth<Int128>(); }, {}});
    out.push_back({"ColumnVector<Int256>", &makeVectorCol<Int256>, []() { return rs::makeFixedWidth<Int256>(); }, {}});
    out.push_back({"ColumnVector<BFloat16>", &makeVectorCol<BFloat16>, []() { return rs::makeFixedWidth<BFloat16>(); }, {}});
    out.push_back({"ColumnVector<Float32>", &makeVectorCol<Float32>, []() { return rs::makeFixedWidth<Float32>(); }, {}});
    out.push_back({"ColumnVector<Float64>", &makeVectorCol<Float64>, []() { return rs::makeFixedWidth<Float64>(); }, {}});
    out.push_back({"ColumnVector<UUID>", &makeVectorCol<UUID>, []() { return rs::makeFixedWidth<UUID>(); }, {}});
    out.push_back({"ColumnVector<IPv4>", &makeVectorCol<IPv4>, []() { return rs::makeFixedWidth<IPv4>(); }, {}});
    out.push_back({"ColumnVector<IPv6>", &makeVectorCol<IPv6>, []() { return rs::makeFixedWidth<IPv6>(); }, {}});

    /// ColumnDecimal<T> — every Decimal width plus DateTime64 / Time64.
    out.push_back({"ColumnDecimal<Decimal32>", &makeDecimalCol<Decimal32>, []() { return rs::makeDecimal<Decimal32>(); }, {}});
    out.push_back({"ColumnDecimal<Decimal64>", &makeDecimalCol<Decimal64>, []() { return rs::makeDecimal<Decimal64>(); }, {}});
    out.push_back({"ColumnDecimal<Decimal128>", &makeDecimalCol<Decimal128>, []() { return rs::makeDecimal<Decimal128>(); }, {}});
    out.push_back({"ColumnDecimal<Decimal256>", &makeDecimalCol<Decimal256>, []() { return rs::makeDecimal<Decimal256>(); }, {}});
    out.push_back({"ColumnDecimal<DateTime64>", &makeDecimalCol<DateTime64>, []() { return rs::makeDecimal<DateTime64>(); }, {}});
    out.push_back({"ColumnDecimal<Time64>", &makeDecimalCol<Time64>, []() { return rs::makeDecimal<Time64>(); }, {}});

    /// ColumnFixedString — a representative sweep of widths.
    for (size_t w : {size_t{1}, size_t{4}, size_t{16}, size_t{32}, size_t{64}, size_t{128}})
    {
        out.push_back({
            "ColumnFixedString(" + std::to_string(w) + ")",
            [w](size_t n, uint64_t seed) { return makeFixedStringCol(n, w, seed); },
            [w]() { return rs::makeFixedString(w); },
            {},
        });
    }

    /// ColumnString.
    out.push_back({
        "ColumnString",
        &makeStringCol,
        []() { return rs::makeString(); },
        &stringRowBytes,
    });

    /// ColumnNullable(X) — a representative sample. One per category: small
    /// fixed-width, medium fixed-width, wide fixed-width, decimal,
    /// FixedString, and String. The composite-type contract is identical
    /// across nested types, so this sample is sufficient to characterize
    /// the wrapper's per-row overhead.
    out.push_back({
        "Nullable(UInt32)",
        [](size_t n, uint64_t seed) { return wrapNullable(makeVectorCol<UInt32>(n, seed), seed ^ 0xa5a5ULL); },
        []() { return rs::makeNullable(rs::makeFixedWidth<UInt32>()); },
        {},
    });
    out.push_back({
        "Nullable(Int64)",
        [](size_t n, uint64_t seed) { return wrapNullable(makeVectorCol<Int64>(n, seed), seed ^ 0xa5a5ULL); },
        []() { return rs::makeNullable(rs::makeFixedWidth<Int64>()); },
        {},
    });
    out.push_back({
        "Nullable(UInt128)",
        [](size_t n, uint64_t seed) { return wrapNullable(makeVectorCol<UInt128>(n, seed), seed ^ 0xa5a5ULL); },
        []() { return rs::makeNullable(rs::makeFixedWidth<UInt128>()); },
        {},
    });
    out.push_back({
        "Nullable(Decimal64)",
        [](size_t n, uint64_t seed) { return wrapNullable(makeDecimalCol<Decimal64>(n, seed), seed ^ 0xa5a5ULL); },
        []() { return rs::makeNullable(rs::makeDecimal<Decimal64>()); },
        {},
    });
    out.push_back({
        "Nullable(FixedString(16))",
        [](size_t n, uint64_t seed) { return wrapNullable(makeFixedStringCol(n, 16, seed), seed ^ 0xa5a5ULL); },
        []() { return rs::makeNullable(rs::makeFixedString(16)); },
        {},
    });
    out.push_back({
        "Nullable(String)",
        [](size_t n, uint64_t seed) { return wrapNullable(makeStringCol(n, seed), seed ^ 0xa5a5ULL); },
        []() { return rs::makeNullable(rs::makeString()); },
        &nullableStringRowBytes,
    });

    return out;
}


/// Parse `--key value` and `--key=value` pairs from argv. Returns empty
/// optional if the flag is missing.
std::optional<std::string> getArg(int argc, char ** argv, const std::string & key)
{
    for (int i = 1; i < argc; ++i)
    {
        const std::string a = argv[i];
        if (a == "--" + key)
        {
            if (i + 1 < argc)
                return std::string(argv[i + 1]);
            return std::string();
        }
        if (a.starts_with("--" + key + "="))
            return a.substr(key.size() + 3);
    }
    return std::nullopt;
}


bool hasFlag(int argc, char ** argv, const std::string & key)
{
    for (int i = 1; i < argc; ++i)
        if (argv[i] == "--" + key)
            return true;
    return false;
}


void printResultsStdout(const std::vector<Result> & rows)
{
    std::cout << fmt::format(
        "{:<22} {:>8} {:>4} {:>3} {:>3} {:>12} {:>16} {:>14}\n", "type", "batch", "P", "K", "T", "rows", "ns/row(median)", "GB/s(total)");
    for (const auto & r : rows)
    {
        std::cout << fmt::format(
            "{:<22} {:>8} {:>4} {:>3} {:>3} {:>12} {:>16.3f} {:>14.3f}\n",
            r.column_type,
            r.batch_size,
            r.partitions,
            r.columns,
            r.threads,
            r.total_rows,
            r.median_ns_per_row,
            r.total_bandwidth_gbs);
    }
}


void printResultsCsv(const std::vector<Result> & rows, std::ostream & os)
{
    os << "type,batch,P,K,T,total_rows,ns_per_row_median,total_bandwidth_gbs\n";
    for (const auto & r : rows)
    {
        os << r.column_type << ',' << r.batch_size << ',' << r.partitions << ',' << r.columns << ',' << r.threads << ',' << r.total_rows
           << ',' << r.median_ns_per_row << ',' << r.total_bandwidth_gbs << '\n';
    }
}


std::vector<size_t> parseList(const std::string & s)
{
    std::vector<size_t> out;
    size_t pos = 0;
    while (pos < s.size())
    {
        size_t end = s.find(',', pos);
        if (end == std::string::npos)
            end = s.size();
        if (end > pos)
            out.push_back(std::stoull(s.substr(pos, end - pos)));
        pos = end + 1;
    }
    return out;
}

}


int main(int argc, char ** argv)
{
    if (hasFlag(argc, argv, "help") || hasFlag(argc, argv, "h"))
    {
        std::cout << "bench_radix_shuffle_column_primitives — measure RadixShuffle column-primitive per-row cost\n"
                  << "\nOptions:\n"
                  << "  --total-rows N        Total rows per (column-type, configuration). Default 1'000'000.\n"
                  << "  --batches  L          Batch-size sweep, comma-separated. Default 1024,4096,16384.\n"
                  << "  --partitions L        Partition-count sweep. Default 4,8,16,32,64,128,256.\n"
                  << "  --columns L           Column-count sweep. Default 1,2,4,8.\n"
                  << "  --threads L           Thread-count sweep. Default 1,4,8,16,32,48.\n"
                  << "  --types L             Column-type sweep (comma-separated names). Default: all scope-D types.\n"
                  << "  --csv FILE            Write CSV output to FILE.\n"
                  << "  --quick               Smoke-test mode: collapse the sweep to a small subset (one batch,\n"
                  << "                        one P, one K, two T values, all types) so the bench finishes fast.\n"
                  << "\nDefault sweep matches spec §4.1 (Batch ∈ {1024,4096,16384}, P ∈ {4,8,16,32,64,128,256}, K ∈ {1,2,4,8}, T ∈ "
                     "{1,4,8,16,32,48}).\n";
        return 0;
    }

    const bool quick = hasFlag(argc, argv, "quick");

    const size_t default_total_rows = quick ? 200000 : 1000000;
    const size_t total_rows = std::stoull(getArg(argc, argv, "total-rows").value_or(std::to_string(default_total_rows)));

    const std::vector<size_t> batches = parseList(getArg(argc, argv, "batches").value_or(quick ? "4096" : "1024,4096,16384"));
    // NOLINTBEGIN(readability-identifier-naming) -- P, K, T match the spec's notation.
    const std::vector<size_t> P_list = parseList(getArg(argc, argv, "partitions").value_or(quick ? "16" : "4,8,16,32,64,128,256"));
    const std::vector<size_t> K_list = parseList(getArg(argc, argv, "columns").value_or(quick ? "1" : "1,2,4,8"));
    const std::vector<size_t> T_list = parseList(getArg(argc, argv, "threads").value_or(quick ? "1,8" : "1,4,8,16,32,48"));
    // NOLINTEND(readability-identifier-naming)

    auto specs = buildSpecs();
    std::optional<std::string> types_arg = getArg(argc, argv, "types");
    if (types_arg)
    {
        std::vector<ColumnSpec> filtered;
        const std::string & csv = *types_arg;
        size_t pos = 0;
        while (pos < csv.size())
        {
            size_t end = csv.find(',', pos);
            if (end == std::string::npos)
                end = csv.size();
            std::string name = csv.substr(pos, end - pos);
            for (auto & s : specs)
                if (s.name == name)
                    filtered.push_back(s);
            pos = end + 1;
        }
        specs = std::move(filtered);
    }

    std::vector<Result> results;
    for (const auto & spec : specs)
    {
        // NOLINTBEGIN(readability-identifier-naming) -- P, K, T match the spec's notation.
        for (size_t batch : batches)
            for (size_t P : P_list)
                for (size_t K : K_list)
                    for (size_t T : T_list)
                    {
                        Workload wl;
                        wl.batch_size = batch;
                        wl.partitions = P;
                        wl.columns = K;
                        wl.threads = T;
                        wl.total_rows = total_rows;
                        results.push_back(runConfig(spec, wl));
                    }
        // NOLINTEND(readability-identifier-naming)
    }

    printResultsStdout(results);

    auto csv_path = getArg(argc, argv, "csv");
    if (csv_path)
    {
        std::ofstream f(*csv_path);
        if (!f.is_open())
        {
            std::cerr << fmt::format("Failed to open CSV path: {}\n", *csv_path);
            return 1;
        }
        printResultsCsv(results, f);
    }

    return 0;
}
