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
///   ""  | "shm" | "shm:adopt" -> Adopt  zero-copy adoption straight out of the SHM ring (default)
///   "shm:copy"                -> Copy   consumer copies each block out of SHM, releases the ring slot early
///   "tcp:<host>:<port>"       -> Tcp    Phase 1: block bytes delivered over a TCP socket (still a copy)
enum class ShmTransportMode : uint8_t
{
    Adopt = 0,
    Copy = 1,
    Tcp = 2,
};

struct ShmTransportSpec
{
    ShmTransportMode mode = ShmTransportMode::Adopt;
    String tcp_host;        /// populated only for Tcp
    UInt16 tcp_port = 0;    /// populated only for Tcp
};

inline const char * toString(ShmTransportMode mode)
{
    switch (mode)
    {
        case ShmTransportMode::Adopt: return "shm:adopt";
        case ShmTransportMode::Copy:  return "shm:copy";
        case ShmTransportMode::Tcp:   return "tcp";
    }
    return "shm:adopt";
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

    /// "tcp:<host>:<port>" — host may itself contain ':' (IPv6) so split on the LAST colon.
    static constexpr std::string_view tcp_prefix = "tcp:";
    if (spec.size() > tcp_prefix.size() && std::string_view(spec).substr(0, tcp_prefix.size()) == tcp_prefix)
    {
        const String rest = spec.substr(tcp_prefix.size());     /// "<host>:<port>"
        const auto colon = rest.rfind(':');
        if (colon == String::npos || colon == 0 || colon + 1 >= rest.size())
            return false;

        out.tcp_host = rest.substr(0, colon);
        const String port_str = rest.substr(colon + 1);

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
        out.mode = ShmTransportMode::Tcp;
        return true;
    }

    return false;
}

}

#endif
