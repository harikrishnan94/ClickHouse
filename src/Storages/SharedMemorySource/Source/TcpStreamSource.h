#pragma once

#if defined(OS_LINUX)

#include <DataTypes/IDataType.h>
#include <Processors/Chunk.h>
#include <Processors/ISource.h>
#include <Storages/SharedMemorySource/Tracker/AdoptedByteCharger.h>
#include <Common/Stopwatch.h>
#include <base/types.h>

#include <atomic>
#include <cstdint>
#include <vector>


namespace DB
{

/// Hot-Cold Phase 1 — TCP transport consumer (decision D-HC-0101/0104). The producer-per-stream
/// worker listens on its own TCP port (D-HC-0102); this source connects, reads the handshake
/// (schema cross-validation, identical to the SHM handshake), then reads BLOCK frames
/// (`Wire/TcpFrame.h`). Each block's payload is the SAME frame-relative data-region bytes the SHM
/// data plane carries, so the existing `adopt()` reconstructs columns identically — the recv buffer
/// becomes the `data_region_base` and a RetainToken frees it on last-alias drop (the TCP analog of
/// the SHM slot release). TCP delivery is inherently one copy (kernel recv), so this is adopt over
/// an owned recv buffer — the natural generalisation of Phase-0 copy mode. Counters: the ShmCopied*
/// family (TCP is a copy transport; `ShmCopiedBlocks` is the offload-oracle signal).
///
/// Blocking-recv leaf source: `generate()` blocks on the socket (with SO_RCVTIMEO slices so it can
/// observe cancellation and the stall budget); an empty Chunk signals EOS. `onCancel()` shuts the
/// socket down to unblock a pending recv.
class TcpStreamSource final : public ISource
{
public:
    TcpStreamSource(
        SharedHeader header,
        String host_,
        UInt16 port_,
        std::vector<DataTypePtr> full_column_types_,
        std::vector<String> full_column_names_,
        std::vector<String> requested_column_names_,
        UInt64 stall_timeout_ms_);

    ~TcpStreamSource() override;

    TcpStreamSource(const TcpStreamSource &) = delete;
    TcpStreamSource & operator=(const TcpStreamSource &) = delete;

    String getName() const override { return "TcpStreamSource"; }

protected:
    Chunk generate() override;
    void onCancel() noexcept override;

private:
    /// Connect (retrying ECONNREFUSED while the producer comes up), read + cross-validate the
    /// handshake, build the projection map. Lazy on first generate().
    void ensureConnected();
    /// Read one BLOCK frame; returns an empty Chunk on the EOS frame.
    Chunk recvBlock();
    /// Read exactly `n` bytes; honours cancellation + the stall budget across SO_RCVTIMEO slices;
    /// peer close mid-stream throws producer-death/framing-invalid.
    void recvAll(void * dst, size_t n);

    String host;
    UInt16 port;
    std::vector<DataTypePtr> full_column_types;
    std::vector<String> full_column_names;
    std::vector<String> requested_column_names;
    std::vector<size_t> projection_indices;
    UInt64 stall_timeout_ms;

    int sock_fd = -1;
    bool connected = false;
    bool eos_observed = false;
    std::atomic<bool> cancelled{false};

    AdoptedByteCharger charger;
    Stopwatch stall_timer;
};

}

#endif
