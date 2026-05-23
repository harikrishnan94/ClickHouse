#include <Columns/ColumnNullable.h>
#include <Columns/ColumnString.h>
#include <Columns/ColumnVector.h>
#include <Columns/ColumnsNumber.h>
#include <DataTypes/DataTypeNullable.h>
#include <DataTypes/DataTypeString.h>
#include <DataTypes/DataTypesNumber.h>
#include <Common/RadixShuffle/ColumnPrimitivesDispatch.h>
#include <Common/RadixShuffle/RadixPartitioner.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
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


// ───────────────────────── column builders ─────────────────────────


MutableColumnPtr makeUInt32Col(size_t n, uint64_t seed)
{
    std::mt19937_64 rng(seed);
    auto col = ColumnVector<UInt32>::create();
    col->reserve(n);
    for (size_t i = 0; i < n; ++i)
        col->insertValue(static_cast<UInt32>(rng()));
    return col;
}

MutableColumnPtr makeUInt64Col(size_t n, uint64_t seed)
{
    std::mt19937_64 rng(seed);
    auto col = ColumnVector<UInt64>::create();
    col->reserve(n);
    for (size_t i = 0; i < n; ++i)
        col->insertValue(rng());
    return col;
}

MutableColumnPtr makeStringCol(size_t n, uint64_t seed)
{
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<size_t> len_dist(4, 32);
    auto col = ColumnString::create();
    std::string buf;
    for (size_t i = 0; i < n; ++i)
    {
        const size_t len = len_dist(rng);
        buf.resize(len);
        for (auto & c : buf)
            c = static_cast<char>((rng() % 95) + 32);
        col->insertData(buf.data(), buf.size());
    }
    return col;
}

MutableColumnPtr makeNullableUInt32Col(size_t n, uint64_t seed)
{
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<int> null_dist(0, 4);
    auto nested = ColumnVector<UInt32>::create();
    auto null_map = ColumnUInt8::create();
    for (size_t i = 0; i < n; ++i)
    {
        null_map->insertValue(null_dist(rng) == 0 ? 1 : 0);
        nested->insertValue(static_cast<UInt32>(rng()));
    }
    return ColumnNullable::create(std::move(nested), std::move(null_map));
}


// ───────────────────────── column spec ─────────────────────────


struct ColumnSpec
{
    std::string name;
    std::function<MutableColumnPtr(size_t, uint64_t)> make_column;
    DataTypePtr dtype;
};


std::vector<ColumnSpec> buildColumnSpecs()
{
    return {
        {"UInt32", makeUInt32Col, std::make_shared<DataTypeUInt32>()},
        {"UInt64", makeUInt64Col, std::make_shared<DataTypeUInt64>()},
        {"String", makeStringCol, std::make_shared<DataTypeString>()},
        {"Nullable(UInt32)", makeNullableUInt32Col, std::make_shared<DataTypeNullable>(std::make_shared<DataTypeUInt32>())},
    };
}


// ───────────────────────── workload ─────────────────────────


struct Workload
{
    size_t batch_size;
    size_t partitions;
    size_t columns;
    size_t threads;
    size_t total_rows;
};


struct Result
{
    std::string column_type;
    size_t batch_size = 0;
    size_t partitions = 0;
    size_t columns = 0;
    size_t threads = 0;
    size_t total_rows = 0;
    double median_ns_per_row = 0.0;
    double total_bandwidth_gbs = 0.0;
};


// ───────────────────────── benchmark kernel ─────────────────────────


uint64_t runThread(
    const rs::PartSchema & schema,
    const std::vector<rs::ColumnPrimitives> & primitives,
    const std::vector<MutableColumnPtr> & columns,
    size_t P,
    size_t batches,
    size_t batch_size,
    std::vector<double> & batch_times_ns)
{
    rs::RadixPartitionerOptions opts;
    opts.batch_size_override = batch_size;
    rs::RadixPartitioner part(schema, primitives, P, {0}, opts);

    uint64_t total_source_bytes = 0;
    for (size_t b = 0; b < batches; ++b)
    {
        DB::Columns cols(columns.size());
        for (size_t k = 0; k < columns.size(); ++k)
            cols[k] = columns[k]->getPtr();

        const auto t0 = std::chrono::steady_clock::now();
        part.process(cols);
        const auto t1 = std::chrono::steady_clock::now();

        batch_times_ns.push_back(static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count()));

        for (size_t k = 0; k < columns.size(); ++k)
            total_source_bytes += columns[k]->byteSize();
    }
    part.finish();
    return total_source_bytes;
}


Result runBenchmark(const ColumnSpec & spec, const Workload & wl)
{
    const size_t B = wl.batch_size;
    const size_t P = wl.partitions;
    const size_t K = wl.columns;
    const size_t T = wl.threads;
    const size_t N = wl.total_rows;

    // Build schema + primitives from K copies of the same column type.
    std::vector<DataTypePtr> types(K, spec.dtype);
    auto sp = rs::buildSchemaAndPrimitives(types);

    const size_t batches_per_thread = std::max<size_t>(1, N / (T * B));

    // Build source columns per thread.
    std::vector<std::vector<MutableColumnPtr>> cols_per_thread(T);
    for (size_t t = 0; t < T; ++t)
    {
        cols_per_thread[t].resize(K);
        for (size_t k = 0; k < K; ++k)
            cols_per_thread[t][k] = spec.make_column(B, t * 100 + k);
    }

    std::vector<std::vector<double>> times_per_thread(T);
    std::vector<std::atomic<uint64_t>> src_bytes_per_thread(T);
    for (auto & s : src_bytes_per_thread)
        s.store(0);

    const auto wall_t0 = std::chrono::steady_clock::now();
    std::vector<std::thread> threads;
    threads.reserve(T);
    for (size_t t = 0; t < T; ++t)
    {
        threads.emplace_back(
            [&, t]()
            {
                times_per_thread[t].reserve(batches_per_thread);
                const uint64_t sb = runThread(sp.schema, sp.primitives, cols_per_thread[t], P, batches_per_thread, B, times_per_thread[t]);
                src_bytes_per_thread[t].store(sb);
            });
    }
    for (auto & thr : threads)
        thr.join();

    const auto wall_t1 = std::chrono::steady_clock::now();
    const double wall_sec = std::chrono::duration<double>(wall_t1 - wall_t0).count();

    std::vector<double> all_times;
    for (const auto & tv : times_per_thread)
        all_times.insert(all_times.end(), tv.begin(), tv.end());
    std::sort(all_times.begin(), all_times.end());

    const double median_ns = all_times.empty() ? 0.0 : all_times[all_times.size() / 2];
    const double median_ns_per_row = (B > 0 && K > 0) ? median_ns / static_cast<double>(B * K) : 0.0;

    uint64_t total_src_bytes = 0;
    for (const auto & s : src_bytes_per_thread)
        total_src_bytes += s.load();
    const double bandwidth_gbs = (wall_sec > 0) ? static_cast<double>(total_src_bytes) / wall_sec / 1e9 : 0.0;

    return Result{spec.name, B, P, K, T, N, median_ns_per_row, bandwidth_gbs};
}


// ───────────────────────── CLI ─────────────────────────


struct CLIConfig
{
    size_t total_rows = 8 * 1024 * 1024;
    std::vector<size_t> batch_sizes = {1024, 4096, 16384};
    std::vector<size_t> partitions_list = {4, 8, 16, 32, 64, 128, 256};
    std::vector<size_t> columns_list = {1, 2, 4, 8};
    std::vector<size_t> threads_list = {1, 4, 8, 16, 32, 48};
    std::string column_type;
    bool csv = false;
    std::string csv_path = "bench_radix_partition.csv";
    bool full_sweep = false;
};


CLIConfig parseCLI(int argc, char ** argv)
{
    CLIConfig cfg;
    for (int i = 1; i < argc; ++i)
    {
        const std::string arg = argv[i];
        if (arg == "--csv" && i + 1 < argc)
        {
            cfg.csv = true;
            cfg.csv_path = argv[++i];
        }
        else if (arg == "--csv")
        {
            cfg.csv = true;
        }
        else if (arg == "--total-rows" && i + 1 < argc)
        {
            cfg.total_rows = std::stoull(argv[++i]);
        }
        else if (arg == "--batch-size" && i + 1 < argc)
        {
            cfg.batch_sizes = {std::stoull(argv[++i])};
        }
        else if (arg == "--partitions" && i + 1 < argc)
        {
            cfg.partitions_list = {std::stoull(argv[++i])};
        }
        else if (arg == "--columns" && i + 1 < argc)
        {
            cfg.columns_list = {std::stoull(argv[++i])};
        }
        else if (arg == "--threads" && i + 1 < argc)
        {
            cfg.threads_list = {std::stoull(argv[++i])};
        }
        else if (arg == "--column-type" && i + 1 < argc)
        {
            cfg.column_type = argv[++i];
        }
        else if (arg == "--full-sweep")
        {
            cfg.full_sweep = true;
        }
    }
    return cfg;
}


void printHeader()
{
    fmt::print(
        "{:<18} {:>6} {:>5} {:>4} {:>4} {:>12} {:>12} {:>14}\n",
        "column_type",
        "batch",
        "P",
        "K",
        "T",
        "total_rows",
        "ns/row(med)",
        "bandwidth_gbs");
    fmt::print("{}\n", std::string(80, '-'));
}

void printResult(const Result & r)
{
    fmt::print(
        "{:<18} {:>6} {:>5} {:>4} {:>4} {:>12} {:>12.2f} {:>14.2f}\n",
        r.column_type,
        r.batch_size,
        r.partitions,
        r.columns,
        r.threads,
        r.total_rows,
        r.median_ns_per_row,
        r.total_bandwidth_gbs);
}

} // namespace


int main(int argc, char ** argv)
{
    const CLIConfig cfg = parseCLI(argc, argv);
    const std::vector<ColumnSpec> specs = buildColumnSpecs();
    const size_t hw_threads = std::thread::hardware_concurrency();

    printHeader();

    for (const auto & spec : specs)
    {
        if (!cfg.column_type.empty() && spec.name != cfg.column_type)
            continue;

        for (const size_t batch : cfg.batch_sizes)
        {
            for (const size_t P : cfg.partitions_list)
            {
                for (const size_t K : cfg.columns_list)
                {
                    for (const size_t T : cfg.threads_list)
                    {
                        if (T > hw_threads && !cfg.full_sweep)
                            continue;
                        const Workload wl{batch, P, K, T, cfg.total_rows};
                        printResult(runBenchmark(spec, wl));
                    }
                }
            }
        }
    }

    return 0;
}
