/// hashprobe-bench/instrumentation/per_block_log.cpp

#include "per_block_log.h"

#include <cerrno>
#include <cstring>
#include <print>
#include <stdexcept>
#include <string>

namespace DB::HashProbeBench
{

// ── CSV header strings ────────────────────────────────────────────────────────

static const char * kProbeBlockHeader = "probe_block_idx,"
                                        "probe_block_rows,"
                                        "joinblock_probe_wall_ns,"
                                        "joinblock_probe_cpu_ns,"
                                        "result_emit_wall_ns,"
                                        "result_emit_cpu_ns,"
                                        "output_block_count,"
                                        "hw_cycles,"
                                        "hw_instructions,"
                                        "hw_ipc,"
                                        "hw_llc_load,"
                                        "hw_llc_miss_pct,"
                                        "hw_branches,"
                                        "hw_br_miss_pct,"
                                        "hw_dtlb_load,"
                                        "hw_dtlb_miss_pct,"
                                        "caller_tid,"
                                        "joinblock_start_ns,"
                                        "last_next_end_ns,"
                                        "probe_wall_ns,probe_cpu_ns,probe_cycles,probe_instructions,probe_ipc,probe_llc_load,probe_llc_miss_pct,probe_branches,probe_br_miss_pct,probe_dtlb_load,probe_dtlb_miss_pct,probe_hw_available,"
                                        "generate_wall_ns,generate_cpu_ns,generate_cycles,generate_instructions,generate_ipc,generate_llc_load,generate_llc_miss_pct,generate_branches,generate_br_miss_pct,generate_dtlb_load,generate_dtlb_miss_pct,generate_hw_available\n";

static const char * kOutputBlockHeader = "probe_block_idx,"
                                         "output_block_idx,"
                                         "output_block_rows,"
                                         "next_wall_ns,"
                                         "next_cpu_ns,"
                                         "is_last\n";

static double pct(uint64_t numerator, uint64_t denominator)
{
    return denominator > 0 ? static_cast<double>(numerator) * 100.0 / static_cast<double>(denominator) : 0.0;
}

static void printPhaseMetrics(FILE * file, const PhaseMetrics & p)
{
    const double ipc = p.hw_cycles > 0 ? static_cast<double>(p.hw_instructions) / static_cast<double>(p.hw_cycles) : 0.0;
    std::print(
        file,
        ",{:.17g},{:.17g},{},{},{:.17g},{},{:.4g},{},{:.4g},{},{:.4g},{}",
        p.wall_ns,
        p.cpu_ns,
        p.hw_cycles,
        p.hw_instructions,
        ipc,
        p.hw_llc_load,
        pct(p.hw_llc_miss, p.hw_llc_load),
        p.hw_branches,
        pct(p.hw_branch_miss, p.hw_branches),
        p.hw_dtlb_load,
        pct(p.hw_dtlb_miss, p.hw_dtlb_load),
        p.hw_available ? 1 : 0);
}

// ── PerBlockLog ───────────────────────────────────────────────────────────────

PerBlockLog::PerBlockLog(std::string output_dir)
    : output_dir_(std::move(output_dir))
{
}

void PerBlockLog::open()
{
    if (isOpen())
        return;

    std::string probe_path = output_dir_ + "/probe_block_log.csv";
    std::string output_path = output_dir_ + "/output_block_log.csv";

    probe_file_ = fopen(probe_path.c_str(), "w");
    if (!probe_file_)
        throw std::runtime_error("PerBlockLog: cannot open " + probe_path + ": " + strerror(errno));

    output_file_ = fopen(output_path.c_str(), "w");
    if (!output_file_)
    {
        fclose(probe_file_);
        probe_file_ = nullptr;
        throw std::runtime_error("PerBlockLog: cannot open " + output_path + ": " + strerror(errno));
    }

    fputs(kProbeBlockHeader, probe_file_);
    fputs(kOutputBlockHeader, output_file_);
}

void PerBlockLog::writeProbeBlock(const ProbeBlockEntry & e)
{
    if (!probe_file_)
        open();

    const double llc_miss_pct = (e.hw_llc_load > 0) ? static_cast<double>(e.hw_llc_miss) * 100.0 / static_cast<double>(e.hw_llc_load) : 0.0;
    const double br_miss_pct
        = (e.hw_branches > 0) ? static_cast<double>(e.hw_branch_miss) * 100.0 / static_cast<double>(e.hw_branches) : 0.0;
    const double dtlb_miss_pct
        = (e.hw_dtlb_load > 0) ? static_cast<double>(e.hw_dtlb_miss) * 100.0 / static_cast<double>(e.hw_dtlb_load) : 0.0;
    std::print(
        probe_file_,
        "{},{},{:.17g},{:.17g},{:.17g},{:.17g},{},{},{},{:.17g},{},{:.4g},{},{:.4g},{},{:.4g},{},{:.17g},{:.17g}",
        e.probe_block_idx,
        e.probe_block_rows,
        e.joinblock_probe_wall_ns,
        e.joinblock_probe_cpu_ns,
        e.result_emit_wall_ns,
        e.result_emit_cpu_ns,
        e.output_block_count,
        e.hw_cycles,
        e.hw_instructions,
        e.hw_ipc,
        e.hw_llc_load,
        llc_miss_pct,
        e.hw_branches,
        br_miss_pct,
        e.hw_dtlb_load,
        dtlb_miss_pct,
        e.caller_tid,
        e.joinblock_start_ns,
        e.last_next_end_ns);
    printPhaseMetrics(probe_file_, e.phase_probe);
    printPhaseMetrics(probe_file_, e.phase_generate);
    std::println(probe_file_, "");
}

void PerBlockLog::writeOutputBlock(const OutputBlockEntry & e)
{
    if (!output_file_)
        open();

    std::println(
        output_file_,
        "{},{},{},{:.17g},{:.17g},{}",
        e.probe_block_idx,
        e.output_block_idx,
        e.output_block_rows,
        e.next_wall_ns,
        e.next_cpu_ns,
        e.is_last ? 1 : 0);
}

void PerBlockLog::close()
{
    if (probe_file_)
    {
        fflush(probe_file_);
        fclose(probe_file_);
        probe_file_ = nullptr;
    }
    if (output_file_)
    {
        fflush(output_file_);
        fclose(output_file_);
        output_file_ = nullptr;
    }
}

} // namespace DB::HashProbeBench
