#pragma once

/// hashprobe-bench/instrumentation/per_block_log.h
///
/// Structured CSV writer for per-probe-block and per-output-block log entries (H2).
///
/// Writes two CSV files under <output_dir>/:
///   probe_block_log.csv   — one row per ProbeBlockEntry
///   output_block_log.csv  — one row per OutputBlockEntry
///
/// Both files include a header row.  Values are written with full double precision
/// (%.17g) so nanosecond timestamps round-trip losslessly.
///
/// Usage:
///   PerBlockLog log("/tmp/hashprobe-out");
///   log.open();                           // creates/truncates the CSV files
///   log.writeProbeBlock(entry);           // one call per probe block
///   log.writeOutputBlock(entry);          // one call per output block
///   log.close();                          // flushes and closes

#include <hashprobe_bench/artifact.h>

#include <cstdio>
#include <string>

namespace DB::HashProbeBench
{

class PerBlockLog
{
public:
    /// output_dir must already exist.  Files are opened on the first write
    /// if open() is not called explicitly.
    explicit PerBlockLog(std::string output_dir);
    ~PerBlockLog() { close(); }

    PerBlockLog(const PerBlockLog &) = delete;
    PerBlockLog & operator=(const PerBlockLog &) = delete;

    /// Open (or create) the two CSV files, writing header rows.
    /// Safe to call multiple times; no-op if already open.
    void open();

    /// Append one ProbeBlockEntry row to probe_block_log.csv.
    void writeProbeBlock(const ProbeBlockEntry & e);

    /// Append one OutputBlockEntry row to output_block_log.csv.
    void writeOutputBlock(const OutputBlockEntry & e);

    /// Flush and close both files.
    void close();

    bool isOpen() const { return probe_file_ != nullptr && output_file_ != nullptr; }

private:
    std::string output_dir_;
    FILE * probe_file_  = nullptr;
    FILE * output_file_ = nullptr;
};

} // namespace DB::HashProbeBench
