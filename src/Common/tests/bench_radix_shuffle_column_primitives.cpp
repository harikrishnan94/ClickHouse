#include <Columns/ColumnDecimal.h>
#include <Columns/ColumnFixedString.h>
#include <Columns/ColumnNullable.h>
#include <Columns/ColumnString.h>
#include <Columns/ColumnVector.h>
#include <Columns/ColumnsNumber.h>
#include <Common/RadixShuffle/Allocator.h>
#include <Common/RadixShuffle/ColumnPrimitives.h>
#include <Common/RadixShuffle/ColumnPrimitivesDispatch.h>
#include <DataTypes/DataTypeNullable.h>
#include <DataTypes/DataTypeString.h>
#include <DataTypes/DataTypesNumber.h>

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


struct ColumnSpec
{
    std::string name;
    std::function<MutableColumnPtr(size_t n, uint64_t seed)> make_column;
    std::function<rs::SchemaAndPrimitives(size_t k)> make_schema_and_primitives;
    std::function<size_t(const IColumn &, size_t row)> row_bytes; // 0 for fixed
};


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


// ───────────────────────── per-partition histogram ─────────────────────────


void buildHistogram(const uint16_t * pids, size_t n, size_t P, std::vector<size_t> & out)
{
    out.assign(P, 0);
    for (size_t i = 0; i < n; ++i)
        ++out[pids[i]];
}



// ───────────────────────── one-thread scatter kernel ─────────────────────────


/// Run one thread's worth of scatter batches.  Returns the number of source
/// bytes scattered (for bandwidth computation).
uint64_t runThread(
    rs::Handle * handle,
    const rs::PartSchema & schema,
    const std::vector<rs::ColumnPrimitives> & primitives,
    const std::vector<MutableColumnPtr> & columns, // K columns, each batch_size rows
    const std::vector<uint16_t> & pids,
    size_t P,
    size_t batches,
    std::vector<double> & batch_times_ns)
{
    const size_t K = columns.size();
    const size_t batch_size = pids.size();

    std::vector<size_t> hist(P);
    std::vector<size_t> varlen(P, 0);
    std::vector<rs::PartReserveGrant> grants(P);
    std::vector<uint64_t> stale((P + 63) / 64, 0);

    uint64_t total_source_bytes = 0;

    for (size_t b = 0; b < batches; ++b)
    {
        buildHistogram(pids.data(), batch_size, P, hist);

        /// Compute varlen bytes if any column is String.
        std::fill(varlen.begin(), varlen.end(), 0);
        if (schema.has_varlen_portion)
        {
            for (size_t k = 0; k < K; ++k)
            {
                if (const auto * cs = typeid_cast<const ColumnString *>(columns[k].get()))
                    for (size_t i = 0; i < batch_size; ++i)
                    {
                        const UInt64 end = cs->getOffsets()[i];
                        const UInt64 prev = (i == 0) ? 0 : cs->getOffsets()[i - 1];
                        varlen[pids[i]] += end - prev;
                    }
                else if (const auto * cn = typeid_cast<const ColumnNullable *>(columns[k].get()))
                {
                    if (const auto * cs2 = typeid_cast<const ColumnString *>(&cn->getNestedColumn()))
                        for (size_t i = 0; i < batch_size; ++i)
                        {
                            const UInt64 end = cs2->getOffsets()[i];
                            const UInt64 prev = (i == 0) ? 0 : cs2->getOffsets()[i - 1];
                            varlen[pids[i]] += end - prev;
                        }
                }
            }
        }

        const auto t0 = std::chrono::steady_clock::now();

        std::fill(stale.begin(), stale.end(), 0);
        handle->reserve(hist.data(), varlen.data(), grants.data(), stale.data());

        std::vector<rs::PartReservation> dst(P);
        for (size_t p = 0; p < P; ++p)
            dst[p] = grants[p].slice;

        for (size_t k = 0; k < K; ++k)
            primitives[k].scatter(primitives[k], schema, *columns[k], pids.data(), batch_size, P, dst.data());

        const auto t1 = std::chrono::steady_clock::now();
        batch_times_ns.push_back(
            static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count()));

        for (size_t k = 0; k < K; ++k)
            total_source_bytes += columns[k]->byteSize();
    }

    return total_source_bytes;
}


// ───────────────────────── benchmark runner ─────────────────────────


Result runBenchmark(const ColumnSpec & spec, const Workload & wl)
{
    const size_t N = wl.total_rows;
    const size_t B = wl.batch_size;
    const size_t P = wl.partitions;
    const size_t K = wl.columns;
    const size_t T = wl.threads;

    auto sp = spec.make_schema_and_primitives(K);
    const rs::PartSchema & schema = sp.schema;
    std::vector<rs::ColumnPrimitives> & primitives = sp.primitives;

    // Build pid arrays once per thread (different seeds)
    std::vector<std::vector<uint16_t>> pids_per_thread(T);
    for (size_t t = 0; t < T; ++t)
    {
        pids_per_thread[t].resize(B);
        std::mt19937_64 rng(t * 1000 + 42);
        std::uniform_int_distribution<uint16_t> dist(0, static_cast<uint16_t>(P - 1));
        for (auto & p : pids_per_thread[t])
            p = dist(rng);
    }

    // Build source columns per thread (K columns of B rows each)
    std::vector<std::vector<MutableColumnPtr>> cols_per_thread(T);
    for (size_t t = 0; t < T; ++t)
    {
        cols_per_thread[t].resize(K);
        for (size_t k = 0; k < K; ++k)
            cols_per_thread[t][k] = spec.make_column(B, t * 100 + k);
    }

    const size_t batches_per_thread = std::max<size_t>(1, N / (T * B));

    std::vector<std::vector<double>> times_per_thread(T);
    std::vector<std::atomic<uint64_t>> source_bytes_per_thread(T);
    for (auto & sb : source_bytes_per_thread)
        sb.store(0);

    rs::Allocator alloc(schema, P, N);

    const auto wall_t0 = std::chrono::steady_clock::now();

    std::vector<std::thread> threads;
    threads.reserve(T);
    for (size_t t = 0; t < T; ++t)
    {
        threads.emplace_back([&, t]()
        {
            rs::Handle * h = alloc.acquire();
            times_per_thread[t].reserve(batches_per_thread);
            const uint64_t sb = runThread(
                h, schema, primitives,
                cols_per_thread[t], pids_per_thread[t], P,
                batches_per_thread, times_per_thread[t]);
            source_bytes_per_thread[t].store(sb);
            alloc.release(h);
        });
    }
    for (auto & thr : threads)
        thr.join();

    const auto wall_t1 = std::chrono::steady_clock::now();
    const double wall_sec = std::chrono::duration<double>(wall_t1 - wall_t0).count();

    // Aggregate per-batch times across all threads
    std::vector<double> all_times;
    for (const auto & tv : times_per_thread)
        all_times.insert(all_times.end(), tv.begin(), tv.end());

    std::sort(all_times.begin(), all_times.end());
    double median_ns = all_times.empty() ? 0.0 : all_times[all_times.size() / 2];
    double median_ns_per_row = (B > 0 && K > 0) ? median_ns / (static_cast<double>(B * K)) : 0.0;

    uint64_t total_source_bytes = 0;
    for (const auto & sb : source_bytes_per_thread)
        total_source_bytes += sb.load();
    const double bandwidth_gbs = (wall_sec > 0) ? static_cast<double>(total_source_bytes) / wall_sec / 1e9 : 0.0;

    return Result{
        spec.name,
        B, P, K, T,
        N,
        median_ns_per_row,
        bandwidth_gbs};
}


// ───────────────────────── column spec registry ─────────────────────────


std::vector<ColumnSpec> buildColumnSpecs()
{
    return {
        {
            "UInt32",
            makeUInt32Col,
            [](size_t k) {
                std::vector<DataTypePtr> types(k, std::make_shared<DataTypeUInt32>());
                return rs::buildSchemaAndPrimitives(types);
            },
            nullptr},
        {
            "UInt64",
            makeUInt64Col,
            [](size_t k) {
                std::vector<DataTypePtr> types(k, std::make_shared<DataTypeUInt64>());
                return rs::buildSchemaAndPrimitives(types);
            },
            nullptr},
        {
            "String",
            makeStringCol,
            [](size_t k) {
                std::vector<DataTypePtr> types(k, std::make_shared<DataTypeString>());
                return rs::buildSchemaAndPrimitives(types);
            },
            [](const IColumn & col, size_t row) -> size_t {
                const auto & cs = assert_cast<const ColumnString &>(col);
                const auto & offs = cs.getOffsets();
                return offs[row] - (row == 0 ? 0 : offs[row - 1]);
            }},
        {
            "Nullable(UInt32)",
            makeNullableUInt32Col,
            [](size_t k) {
                std::vector<DataTypePtr> types(
                    k, std::make_shared<DataTypeNullable>(std::make_shared<DataTypeUInt32>()));
                return rs::buildSchemaAndPrimitives(types);
            },
            nullptr},
    };
}


// ───────────────────────── workload sweep definition ─────────────────────────


// ───────────────────────── CLI parsing ─────────────────────────


struct CLIConfig
{
    size_t total_rows = 8 * 1024 * 1024;
    std::vector<size_t> batch_sizes = {1024, 4096, 16384};
    std::vector<size_t> partitions_list = {4, 8, 16, 32, 64, 128, 256};
    std::vector<size_t> columns_list = {1, 2, 4, 8};
    std::vector<size_t> threads_list = {1, 4, 8, 16, 32, 48};
    std::string column_type; // empty = all
    bool csv = false;
    std::string csv_path = "bench_radix_shuffle.csv";
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


// ───────────────────────── output ─────────────────────────


void printHeader()
{
    fmt::print(
        "{:<18} {:>6} {:>5} {:>4} {:>4} {:>12} {:>12} {:>14}\n",
        "column_type", "batch", "P", "K", "T",
        "total_rows", "ns/row(med)", "bandwidth_gbs");
    fmt::print("{}\n", std::string(80, '-'));
}

void printResult(const Result & r)
{
    fmt::print(
        "{:<18} {:>6} {:>5} {:>4} {:>4} {:>12} {:>12.2f} {:>14.2f}\n",
        r.column_type, r.batch_size, r.partitions, r.columns, r.threads,
        r.total_rows, r.median_ns_per_row, r.total_bandwidth_gbs);
}

void writeCsvHeader(std::ostream & out)
{
    out << "column_type,batch_size,partitions,columns,threads,total_rows,"
        << "median_ns_per_row,total_bandwidth_gbs\n";
}

void writeCsvRow(std::ostream & out, const Result & r)
{
    out << r.column_type << "," << r.batch_size << "," << r.partitions << ","
        << r.columns << "," << r.threads << "," << r.total_rows << ","
        << r.median_ns_per_row << "," << r.total_bandwidth_gbs << "\n";
}

} // namespace


int main(int argc, char ** argv)
{
    const CLIConfig cfg = parseCLI(argc, argv);
    const std::vector<ColumnSpec> specs = buildColumnSpecs();

    std::ofstream csv_out;
    if (cfg.csv)
    {
        csv_out.open(cfg.csv_path);
        writeCsvHeader(csv_out);
    }

    printHeader();

    const auto hw_threads = std::thread::hardware_concurrency();

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
                        const Result r = runBenchmark(spec, wl);
                        printResult(r);
                        if (cfg.csv)
                            writeCsvRow(csv_out, r);
                    }
                }
            }
        }
    }

    return 0;
}
