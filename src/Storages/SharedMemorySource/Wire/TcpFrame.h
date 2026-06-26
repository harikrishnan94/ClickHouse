#pragma once

#if defined(OS_LINUX)

#include <Storages/SharedMemorySource/Wire/Layout.h>

#include <cstdint>


namespace DB::SharedMemoryWire
{

/// Hot-Cold Phase 1 — TCP stream framing (decision D-HC-0101/0103).
///
/// A producer worker that owns a TCP listener speaks this little-endian wire to the consumer
/// (`TcpStreamSource`). It carries the SAME column bytes as the SHM data plane so the unchanged
/// `adopt()` reconstructs columns identically; the only difference from SHM is that offsets are
/// FRAME-RELATIVE (the recv buffer is the `data_region_base`) and the handshake/slot metadata travel
/// inline instead of via a mapped HandshakeRegion + SlotEntry.
///
/// Stream = HANDSHAKE then a sequence of BLOCK frames, the last with eos_marker=1.
///   HANDSHAKE := TcpHandshakeHeader, then SchemaEntry[schema_count]   (schema names + type_strings:
///                the consumer cross-validates the schema and recovers Decimal scale / DateTime64
///                precision exactly as the SHM handshake does — these are NOT on the per-block wire).
///   BLOCK     := TcpBlockHeader, then `payload_len` bytes = the frame-relative data region:
///                ColumnDescriptor[schema_count] at byte `descriptors_offset` (always 0), followed by
///                each column's value buffer (and, for String, offsets buffer) laid out with the
///                identical alignment / PADDING_FOR_SIMD / offsets[-1]-zero-sentinel rules as SHM, but
///                with value_offset / offsets_offset relative to the payload base (byte 0).
///
/// The PG producer side mirrors these structs byte-for-byte (src/shm_tcp.* in pg_clickhouse).
/// Little-endian only (matches Layout.h SHM_MAGIC convention).

/// LE bytes = ASCII "SHMTCP\0\1" — distinct from SHM_MAGIC so a mis-pointed stream is caught.
inline constexpr uint64_t SHM_TCP_MAGIC = 0x0100'5043'544D'4853ULL;
inline constexpr uint32_t SHM_TCP_ABI_VERSION_1 = 1;

#pragma pack(push, 1)
struct TcpHandshakeHeader
{
    uint64_t magic;          ///< SHM_TCP_MAGIC
    uint32_t abi_version;    ///< SHM_TCP_ABI_VERSION_1
    uint32_t schema_count;   ///< number of SchemaEntry that follow; must equal SQL column count
};

struct TcpBlockHeader
{
    uint64_t payload_len;          ///< bytes of frame-relative data region that follow this header
    uint64_t row_count;            ///< block row count (== value_count / offsets_count per descriptor)
    uint64_t descriptors_offset;   ///< byte offset of ColumnDescriptor[schema_count] within payload (0)
    uint8_t  eos_marker;           ///< 1 if this is the end-of-stream frame (then row_count may be 0)
    uint8_t  reserved[7];          ///< MBZ
};
#pragma pack(pop)

static_assert(sizeof(TcpHandshakeHeader) == 16);
static_assert(sizeof(TcpBlockHeader) == 32);

/// Cap on a single block's payload, to bound a malformed/hostile length before allocation.
/// Generous: K-independent, sized for the widest blocks (IMPL_MAX_ROWS_PER_BLOCK wide rows + pads).
inline constexpr uint64_t TCP_MAX_BLOCK_PAYLOAD = 1ULL << 31;   ///< 2 GiB

}

#endif
