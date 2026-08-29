#include "ch_table.h"
#include "platform.h"
#include "umbra.h"
#include "zipf.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <numeric>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

enum class Design
{
    Unchained,
    Chained,
    ChMutex,
    ChSpin,
};

inline const char * design_name(Design d)
{
    switch (d)
    {
        case Design::Unchained: return "unchained";
        case Design::Chained: return "chained";
        case Design::ChMutex: return "ch-mutex";
        case Design::ChSpin: return "ch-spin";
    }
    return "?";
}

struct Config
{
    uint64_t rows = 32'000'000;
    uint64_t distinct = 8'000'000;
    std::vector<double> skews{0.0, 0.5, 1.0, 1.25, 1.5};
    std::vector<size_t> threads;
    size_t umbra_parts = 256;
    size_t ch_parts = 256;
    size_t block_size = 65536;
    size_t reps = 5;
    size_t warmup = 1;
    uint64_t seed = 1;
    bool pin = true;
    bool validate = true;
    std::vector<Design> designs{
        Design::Unchained,
        Design::Chained,
        Design::ChMutex,
        Design::ChSpin,
    };
    std::string csv_path;
    bool quick = false;
};

static std::vector<std::string> split_csv(const std::string & s)
{
    std::vector<std::string> out;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, ','))
        if (!item.empty())
            out.push_back(item);
    return out;
}

static std::vector<double> parse_doubles(const std::string & s)
{
    std::vector<double> v;
    for (const auto & p : split_csv(s))
        v.push_back(std::stod(p));
    return v;
}

static std::vector<size_t> parse_sizes(const std::string & s)
{
    std::vector<size_t> v;
    for (const auto & p : split_csv(s))
        v.push_back(static_cast<size_t>(std::stoull(p)));
    return v;
}

static std::vector<Design> parse_designs(const std::string & s)
{
    std::vector<Design> v;
    for (const auto & p : split_csv(s))
    {
        if (p == "unchained")
            v.push_back(Design::Unchained);
        else if (p == "chained")
            v.push_back(Design::Chained);
        else if (p == "ch-mutex")
            v.push_back(Design::ChMutex);
        else if (p == "ch-spin")
            v.push_back(Design::ChSpin);
        else if (p == "all")
        {
            v = {
                Design::Unchained,
                Design::Chained,
                Design::ChMutex,
                Design::ChSpin,
            };
        }
        else
            throw std::runtime_error("unknown design: " + p);
    }
    return v;
}

static std::vector<size_t> default_threads()
{
    const size_t n = ncpus();
    const size_t cand[] = {1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 192, 256};
    std::vector<size_t> t;
    for (size_t c : cand)
        if (c <= n)
            t.push_back(c);
    if (t.empty() || t.back() != n)
        t.push_back(n);
    return t;
}

static void usage()
{
    std::cerr << "join_ht_build_bench — parallel hash-table build microbenchmark\n"
              << "  --rows N              total 64-bit keys (default 32000000)\n"
              << "  --distinct N          Zipf domain size, keys in 1..N (default 8000000)\n"
              << "  --s LIST              Zipf s, comma-separated; 0 = uniform (default 0,0.5,1,1.25,1.5)\n"
              << "  --threads LIST        thread counts (default 1,2,4,... up to ncpu)\n"
              << "  --umbra-parts N       unchained collection partitions, power of two (default 256)\n"
              << "  --ch-parts N          ClickHouse sub-tables, power of two (default 256)\n"
              << "  --block-size N        ClickHouse input block rows (default 65536)\n"
              << "  --reps N              timed repetitions (default 5)\n"
              << "  --warmup N            discarded runs per cell (default 1)\n"
              << "  --seed N              generator seed (default 1)\n"
              << "  --pin 0|1             pin thread i to CPU i (default 1)\n"
              << "  --validate 0|1        histogram check after each build (default 1)\n"
              << "  --designs LIST        unchained,chained,ch-mutex,ch-spin,all\n"
              << "  --csv PATH            write one CSV row per timed run\n"
              << "  --quick               tiny sweep for a smoke test\n";
}

static Config parse_args(int argc, char ** argv)
{
    Config c;
    c.threads = default_threads();
    for (int i = 1; i < argc; ++i)
    {
        const std::string a = argv[i];
        auto need = [&](const char * name) -> std::string
        {
            if (i + 1 >= argc)
                throw std::runtime_error(std::string("missing value for ") + name);
            return argv[++i];
        };
        if (a == "-h" || a == "--help")
        {
            usage();
            std::exit(0);
        }
        else if (a == "--rows")
            c.rows = std::stoull(need("--rows"));
        else if (a == "--distinct")
            c.distinct = std::stoull(need("--distinct"));
        else if (a == "--s")
            c.skews = parse_doubles(need("--s"));
        else if (a == "--threads")
            c.threads = parse_sizes(need("--threads"));
        else if (a == "--umbra-parts")
            c.umbra_parts = std::stoul(need("--umbra-parts"));
        else if (a == "--ch-parts")
            c.ch_parts = std::stoul(need("--ch-parts"));
        else if (a == "--block-size")
            c.block_size = std::stoul(need("--block-size"));
        else if (a == "--reps")
            c.reps = std::stoul(need("--reps"));
        else if (a == "--warmup")
            c.warmup = std::stoul(need("--warmup"));
        else if (a == "--seed")
            c.seed = std::stoull(need("--seed"));
        else if (a == "--pin")
            c.pin = std::stoi(need("--pin")) != 0;
        else if (a == "--validate")
            c.validate = std::stoi(need("--validate")) != 0;
        else if (a == "--designs")
            c.designs = parse_designs(need("--designs"));
        else if (a == "--csv")
            c.csv_path = need("--csv");
        else if (a == "--quick")
            c.quick = true;
        else
            throw std::runtime_error("unknown argument: " + a);
    }
    if (c.quick)
    {
        c.rows = 1'000'000;
        c.distinct = 200'000;
        c.skews = {0.0, 1.0};
        c.threads = {1, std::min<size_t>(4, ncpus()), ncpus()};
        std::sort(c.threads.begin(), c.threads.end());
        c.threads.erase(std::unique(c.threads.begin(), c.threads.end()), c.threads.end());
        c.reps = 2;
        c.warmup = 0;
    }
    if (c.distinct == 0 || c.rows == 0)
        throw std::runtime_error("rows and distinct must be > 0");
    if (c.distinct > (1ull << 32))
        throw std::runtime_error("distinct too large for the histogram");
    return c;
}

static std::vector<uint64_t> generate_keys(const Config & c, double s, std::vector<uint32_t> & hist)
{
    hist.assign(c.distinct + 1, 0);
    std::vector<uint64_t> keys(c.rows);
    Zipf zipf(c.distinct, s);
    const size_t gen_threads = std::min(c.threads.empty() ? ncpus() : *std::max_element(c.threads.begin(), c.threads.end()), ncpus());
    std::vector<std::vector<uint32_t>> local(gen_threads, std::vector<uint32_t>(c.distinct + 1, 0));

    parallel_for(
        gen_threads,
        false,
        [&](size_t tid)
        {
            uint64_t state = c.seed ^ (static_cast<uint64_t>(std::llround(s * 1000.0)) << 32) ^ (tid * 0x9E3779B97F4A7C15ull);
            const uint64_t begin = c.rows * tid / gen_threads;
            const uint64_t end = c.rows * (tid + 1) / gen_threads;
            auto & h = local[tid];
            for (uint64_t i = begin; i < end; ++i)
            {
                const uint64_t k = zipf.sample(splitmix64(state));
                keys[i] = k;
                h[k] += 1;
            }
        });

    for (size_t t = 0; t < gen_threads; ++t)
        for (uint64_t k = 1; k <= c.distinct; ++k)
            hist[k] += local[t][k];
    return keys;
}

static double occupancy_imbalance(const uint64_t * keys, uint64_t n, size_t parts, bool umbra)
{
    if (parts == 0)
        return 0;
    std::vector<uint64_t> c(parts, 0);
    const uint32_t mask = static_cast<uint32_t>(parts - 1);
    const uint32_t logp = ceil_log2_u64(parts);
    for (uint64_t i = 0; i < n; ++i)
    {
        if (umbra)
            c[umbra_hash64(keys[i]) >> (64u - logp)] += 1;
        else
            c[ch_bucket(ch_hash64(keys[i]), mask)] += 1;
    }
    uint64_t mx = 0;
    for (uint64_t x : c)
        mx = std::max(mx, x);
    const double mean = static_cast<double>(n) / static_cast<double>(parts);
    return mean == 0 ? 0 : static_cast<double>(mx) / mean;
}

struct RunRow
{
    Design design{};
    double s = 0;
    size_t threads = 0;
    StageNs stages{};
    bool ok = true;
    std::string err;
};

static RunRow run_one(Design d, const Config & cfg, double s, size_t threads, const uint64_t * keys, const uint32_t * hist)
{
    RunRow row;
    row.design = d;
    row.s = s;
    row.threads = threads;
    try
    {
        switch (d)
        {
            case Design::Unchained: {
                UnchainedTable table;
                row.stages = build_unchained(table, keys, cfg.rows, threads, cfg.umbra_parts, cfg.pin);
                if (cfg.validate)
                    validate_unchained(table, hist, cfg.distinct);
                break;
            }
            case Design::Chained: {
                ChainedTable table;
                row.stages = build_chained(table, keys, cfg.rows, threads, cfg.umbra_parts, cfg.pin);
                if (cfg.validate)
                    validate_chained(table, hist, cfg.distinct);
                break;
            }
            case Design::ChMutex: {
                ChTable<MutexLatch> table;
                row.stages = build_ch(table, keys, cfg.rows, threads, cfg.ch_parts, cfg.block_size, cfg.distinct, cfg.pin);
                if (cfg.validate)
                    validate_ch(table, hist, cfg.distinct);
                break;
            }
            case Design::ChSpin: {
                ChTable<SpinLatch> table;
                row.stages = build_ch(table, keys, cfg.rows, threads, cfg.ch_parts, cfg.block_size, cfg.distinct, cfg.pin);
                if (cfg.validate)
                    validate_ch(table, hist, cfg.distinct);
                break;
            }
        }
    }
    catch (const std::exception & e)
    {
        row.ok = false;
        row.err = e.what();
    }
    return row;
}

static uint64_t median_u64(std::vector<uint64_t> v)
{
    if (v.empty())
        return 0;
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

struct CellStats
{
    Design design{};
    double s = 0;
    size_t threads = 0;
    uint64_t med_ns = 0;
    uint64_t min_ns = 0;
    uint64_t max_ns = 0;
    uint64_t med_collect = 0;
    uint64_t med_count = 0;
    uint64_t med_insert = 0;
    uint64_t med_latch = 0;
    bool ok = true;
    std::string err;
};

static CellStats summarize(const std::vector<RunRow> & runs)
{
    CellStats st;
    st.design = runs[0].design;
    st.s = runs[0].s;
    st.threads = runs[0].threads;
    std::vector<uint64_t> tot;
    std::vector<uint64_t> col;
    std::vector<uint64_t> cnt;
    std::vector<uint64_t> ins;
    std::vector<uint64_t> lat;
    for (const auto & r : runs)
    {
        if (!r.ok)
        {
            st.ok = false;
            st.err = r.err;
            return st;
        }
        tot.push_back(r.stages.total());
        col.push_back(r.stages.collect);
        cnt.push_back(r.stages.count);
        ins.push_back(r.stages.insert);
        lat.push_back(r.stages.latch);
    }
    st.med_ns = median_u64(tot);
    st.min_ns = *std::min_element(tot.begin(), tot.end());
    st.max_ns = *std::max_element(tot.begin(), tot.end());
    st.med_collect = median_u64(col);
    st.med_count = median_u64(cnt);
    st.med_insert = median_u64(ins);
    st.med_latch = median_u64(lat);
    return st;
}

static double mrps(uint64_t rows, uint64_t ns)
{
    if (ns == 0)
        return 0;
    return static_cast<double>(rows) * 1e3 / static_cast<double>(ns); /// million rows / s  (rows / (ns/1e9) / 1e6)
}

static void print_header(const Config & cfg)
{
    std::cout << "# join_ht_build_bench\n"
              << "# host: " << host_summary() << "\n"
              << "# rows=" << cfg.rows << " distinct=" << cfg.distinct << " block=" << cfg.block_size << " umbra_parts=" << cfg.umbra_parts
              << " ch_parts=" << cfg.ch_parts << " reps=" << cfg.reps << " warmup=" << cfg.warmup << " seed=" << cfg.seed
              << " pin=" << cfg.pin << "\n"
              << "# timing: unchained collect=bump-partition, count=part-prefix, insert=count+prefix+copy per partition\n"
              << "#         chained insert=slab materialize + atomic xchg/or (no partition)\n"
              << "#         ch-* collect=max-thread scatter, insert=max-thread latch+emplace, wall=parallel region\n"
              << "#         mmap/prefault/reserve are outside the timed region\n"
              << "# latch_pct = 100 * sum_thread(latch_ns) / (threads * wall_ns); CPU share of acquire vs wall\n"
              << "#\n";
    std::cout << std::left << "design           s    threads   med_ms   min_ms   max_ms    Mrps  "
              << "collect_ms  count_ms  insert_ms  latch_pct  imb   ok\n";
}

static void print_cell(const Config & cfg, const CellStats & st, double imb)
{
    const double med_ms = st.med_ns / 1e6;
    const double min_ms = st.min_ns / 1e6;
    const double max_ms = st.max_ns / 1e6;
    const double spread = st.med_ns ? (static_cast<double>(st.max_ns - st.min_ns) / static_cast<double>(st.med_ns)) : 0;
    (void)spread;
    double latch_pct = 0;
    if (st.threads && st.med_ns)
        latch_pct = 100.0 * (static_cast<double>(st.med_latch) / static_cast<double>(st.threads)) / static_cast<double>(st.med_ns);

    char buf[512];
    std::snprintf(
        buf,
        sizeof(buf),
        "%-14s %5.2f %8zu %8.2f %8.2f %8.2f %7.2f  %10.2f %9.2f %10.2f  %8.2f %5.2f  %s\n",
        design_name(st.design),
        st.s,
        st.threads,
        med_ms,
        min_ms,
        max_ms,
        mrps(cfg.rows, st.med_ns),
        st.med_collect / 1e6,
        st.med_count / 1e6,
        st.med_insert / 1e6,
        latch_pct,
        imb,
        st.ok ? "ok" : st.err.c_str());
    std::cout << buf;
}

static void print_fig12_note(const Config & cfg, const std::vector<CellStats> & cells)
{
    std::cout << "\n# s=0 scaling vs Birler et al. Figure 12 (shape, not numbers): "
                 "unchained should flatten earlier than chained-atomic.\n";
    auto find = [&](Design d, size_t t) -> const CellStats *
    {
        for (const auto & c : cells)
            if (c.ok && c.design == d && c.s == 0.0 && c.threads == t)
                return &c;
        return nullptr;
    };
    const CellStats * u1 = find(Design::Unchained, 1);
    const CellStats * c1 = find(Design::Chained, 1);
    if (!u1 || !c1)
        return;

    std::cout << "# speedup vs 1 thread at s=0:\n";
    std::cout << "#   threads  unchained  chained\n";
    for (size_t t : cfg.threads)
    {
        const CellStats * u = find(Design::Unchained, t);
        const CellStats * c = find(Design::Chained, t);
        if (!u || !c)
            continue;
        const double su = static_cast<double>(u1->med_ns) / static_cast<double>(u->med_ns);
        const double sc = static_cast<double>(c1->med_ns) / static_cast<double>(c->med_ns);
        char buf[128];
        std::snprintf(buf, sizeof(buf), "#   %7zu  %9.2f  %7.2f\n", t, su, sc);
        std::cout << buf;
    }

    /// Knee: first thread count where next doubling gains < 20% for unchained vs chained.
    auto knee = [&](Design d)
    {
        size_t last = cfg.threads.front();
        const CellStats * prev = find(d, last);
        for (size_t i = 1; i < cfg.threads.size(); ++i)
        {
            const CellStats * cur = find(d, cfg.threads[i]);
            if (!prev || !cur)
                continue;
            const double gain = static_cast<double>(prev->med_ns) / static_cast<double>(cur->med_ns);
            if (gain < 1.20)
                return last;
            last = cfg.threads[i];
            prev = cur;
        }
        return last;
    };
    const size_t ku = knee(Design::Unchained);
    const size_t kc = knee(Design::Chained);
    std::cout << "# first thread count where a further step gains < 20%: unchained=" << ku << " chained=" << kc
              << (ku < kc ? "  (ordering matches the paper's figure)\n" : "  (ordering does NOT match the paper; inspect the table)\n");
}

int main(int argc, char ** argv)
{
    try
    {
        Config cfg = parse_args(argc, argv);
        (void)bloom_tags();
        (void)cycles_per_sec();

        std::optional<std::ofstream> csv;
        if (!cfg.csv_path.empty())
        {
            csv.emplace(cfg.csv_path);
            *csv << "design,s,threads,rep,wall_ns,collect_ns,count_ns,insert_ns,latch_ns,ok\n";
        }

        print_header(cfg);
        std::vector<CellStats> cells;

        for (double s : cfg.skews)
        {
            std::vector<uint32_t> hist;
            const std::vector<uint64_t> keys = generate_keys(cfg, s, hist);
            const double imb_u = occupancy_imbalance(keys.data(), cfg.rows, cfg.umbra_parts, true);
            const double imb_c = occupancy_imbalance(keys.data(), cfg.rows, cfg.ch_parts, false);

            for (size_t threads : cfg.threads)
            {
                for (Design d : cfg.designs)
                {
                    for (size_t w = 0; w < cfg.warmup; ++w)
                        (void)run_one(d, cfg, s, threads, keys.data(), hist.data());

                    std::vector<RunRow> runs;
                    runs.reserve(cfg.reps);
                    for (size_t r = 0; r < cfg.reps; ++r)
                    {
                        RunRow row = run_one(d, cfg, s, threads, keys.data(), hist.data());
                        if (csv)
                            *csv << design_name(row.design) << ',' << s << ',' << threads << ',' << r << ',' << row.stages.total() << ','
                                 << row.stages.collect << ',' << row.stages.count << ',' << row.stages.insert << ',' << row.stages.latch
                                 << ',' << (row.ok ? 1 : 0) << '\n';
                        runs.push_back(std::move(row));
                    }
                    CellStats st = summarize(runs);
                    const double imb = (d == Design::Unchained || d == Design::Chained) ? imb_u : imb_c;
                    print_cell(cfg, st, imb);
                    std::cout.flush();
                    cells.push_back(st);
                }
            }
        }

        print_fig12_note(cfg, cells);
        return 0;
    }
    catch (const std::exception & e)
    {
        std::cerr << "error: " << e.what() << '\n';
        return 1;
    }
}
