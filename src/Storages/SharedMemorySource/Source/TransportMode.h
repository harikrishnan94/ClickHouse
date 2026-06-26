#pragma once

#if defined(OS_LINUX)

#include <base/types.h>

#include <cstdint>
#include <string_view>


namespace DB
{

/// Consumer-side transport for the experimental `streamed_table()` source.
///
/// Selected PER QUERY via the optional 3rd literal argument to `streamed_table(...)`
/// (Hot-Cold decision D-HC-0001) — no ClickHouse server GUC, no rebuild to switch:
///   ""  | "shm" | "shm:adopt" -> Adopt    zero-copy adoption straight out of the SHM ring (default)
///   "shm:copy"                -> Copy     consumer copies each block out of SHM, releases the ring slot early
///   "tcp:<host>:<port>"       -> Tcp      Phase 1: bespoke TcpFrame.h block bytes over a TCP socket (a copy)
///   "arrow:<host>:<port>"     -> ArrowTcp Phase 2 Branch A: a standard Apache Arrow IPC stream over the
///                                         same per-stream TCP socket (D-HC-0205/0207)
enum class ShmTransportMode : uint8_t
{
    Adopt = 0,
    Copy = 1,
    Tcp = 2,
    ArrowTcp = 3,
};

struct ShmTransportSpec
{
    ShmTransportMode mode = ShmTransportMode::Adopt;
    String tcp_host;        /// populated only for Tcp / ArrowTcp
    UInt16 tcp_port = 0;    /// populated only for Tcp / ArrowTcp
};

inline const char * toString(ShmTransportMode mode)
{
    switch (mode)
    {
        case ShmTransportMode::Adopt:    return "shm:adopt";
        case ShmTransportMode::Copy:     return "shm:copy";
        case ShmTransportMode::Tcp:      return "tcp";
        case ShmTransportMode::ArrowTcp: return "arrow";
    }
    return "shm:adopt";
}

/// Parse "<host>:<port>" (host may contain ':' for IPv6, so split on the LAST colon) into out.
inline bool tryParseHostPort(std::string_view rest, ShmTransportSpec & out)
{
    const auto colon = rest.rfind(':');
    if (colon == std::string_view::npos || colon == 0 || colon + 1 >= rest.size())
        return false;

    out.tcp_host = String(rest.substr(0, colon));
    const std::string_view port_str = rest.substr(colon + 1);

    uint64_t port = 0;
    for (const char c : port_str)
    {
        if (c < '0' || c > '9')
            return false;
        port = port * 10 + static_cast<uint64_t>(c - '0');
        if (port > 65535)
            return false;
    }
    if (port == 0)
        return false;

    out.tcp_port = static_cast<UInt16>(port);
    return true;
}

/// Parse the transport-spec literal. Returns false on a malformed spec (the caller raises the
/// typed SQL error). Kept header-inline (no new TU, no ErrorCodes coupling in this header).
inline bool tryParseShmTransportSpec(const String & spec, ShmTransportSpec & out)
{
    out = ShmTransportSpec{};

    if (spec.empty() || spec == "shm" || spec == "shm:adopt")
    {
        out.mode = ShmTransportMode::Adopt;
        return true;
    }
    if (spec == "shm:copy")
    {
        out.mode = ShmTransportMode::Copy;
        return true;
    }

    /// The per-stream socket transports: "tcp:<host>:<port>" (bespoke) / "arrow:<host>:<port>" (Arrow IPC).
    const std::string_view sv{spec};
    static constexpr std::string_view tcp_prefix = "tcp:";
    static constexpr std::string_view arrow_prefix = "arrow:";
    if (sv.size() > tcp_prefix.size() && sv.substr(0, tcp_prefix.size()) == tcp_prefix)
    {
        if (!tryParseHostPort(sv.substr(tcp_prefix.size()), out))
            return false;
        out.mode = ShmTransportMode::Tcp;
        return true;
    }
    if (sv.size() > arrow_prefix.size() && sv.substr(0, arrow_prefix.size()) == arrow_prefix)
    {
        if (!tryParseHostPort(sv.substr(arrow_prefix.size()), out))
            return false;
        out.mode = ShmTransportMode::ArrowTcp;
        return true;
    }

    return false;
}

}

#endif
