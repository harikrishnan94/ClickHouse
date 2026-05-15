#pragma once

#include <Columns/IColumn.h>
#include <DataTypes/IDataType.h>
#include <Storages/SharedMemorySource/Adoption/RetainToken.h>
#include <Storages/SharedMemorySource/Tracker/ChargeHandle.h>
#include <Storages/SharedMemorySource/Wire/Layout.h>

#include <cstddef>
#include <string>
#include <utility>
#include <vector>


namespace DB
{

/// Single consumer-side seam that turns producer data-plane bytes into IColumns.
/// Owned by the pollable source (T3.2); called once per drainable block.
///
/// Performs descriptor-level validation (`pollable-shm-source.md` Producer-side
/// preconditions enumerated rows 13–20 + 26) before any data-plane read.
/// Content-level validation (preconditions 21–22) is lazy and surfaces via the
/// returned ColumnString's `validateAdoptedOffsets()` — the source invokes it
/// before emitting the chunk.
///
/// On success, the returned columns jointly carry `retain_token` and
/// `charge_handle`: the retain rides every column via std::shared_ptr aliasing;
/// the move-only ChargeHandle is wrapped in a std::shared_ptr<ChargeHandle>
/// here so every column also holds a reference (final-drop ordering is the
/// shared_ptr's responsibility — the ChargeHandle's destructor runs when the
/// last reference drops, releasing both the MemoryTracker charge and the
/// feature-local counter exactly once per `memory-tracker-integration.md` I7).
///
/// On ANY failure path (type rejection, descriptor-level violation, or any
/// other exception before successful return), both `retain_token` and
/// `charge_handle` are released before the exception propagates. The retain
/// drops because the local shared_ptr copy goes out of scope; the
/// ChargeHandle's destructor runs on the local rvalue. This is the
/// `system.md` I10 (Exception safety) obligation that this layer owns.
///
/// Spec authority:
///   `adoption-layer.md` §Interfaces & contracts (Adopt entry point + Retain
///   and charge handle semantics), §Constraints (validation two-tier), I3, I4;
///   `pollable-shm-source.md` Producer-side preconditions enumerated rows 13–22;
///   `system.md` I5, I10.
Columns adopt(
    const std::vector<SharedMemoryWire::ColumnDescriptor> & descriptors,
    const std::vector<std::pair<std::string, DataTypePtr>> & schema,
    const char * data_region_base,
    size_t data_region_size,
    UInt64 row_count,
    RetainToken retain_token,
    ChargeHandle charge_handle);

}
