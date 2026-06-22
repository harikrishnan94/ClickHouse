#pragma once

#if defined(OS_LINUX)

#include <Storages/IStorage.h>

#include <base/types.h>


namespace DB
{

/// IStorage facade over `PollableShmSource`. Reached from SQL via the `streamed_table()`
/// table function (legacy alias `shm()`) (T3.4); not registerable as a real storage engine in phase 1 (the
/// table function constructs a transient instance per query, the storage is
/// dropped at query end).
///
/// `read()` ignores `num_streams` per `pollable-shm-source.md` N11 (single
/// stream per source instance in phase 1). The producer feeds one block at a
/// time; downstream parallelism is the pipeline's concern.
///
/// `isRemote() == false`: the producer is by definition co-located (single-host
/// SHM per `system.md` N1).
///
/// `supportsColumnsWithDynamicStructure()` defaults to false (inherited from
/// IStorage): phase-1 schema is fully static and fixed at SQL parse/resolve
/// time per `shm-block-stream.md` §Schema declaration and negotiation.
///
/// Spec authority: `system.md` §Component map (Source -> downstream pipeline);
/// `pollable-shm-source.md` N11, §Interfaces & contracts (streamed_table() table function);
/// `shm-block-stream.md` §Schema declaration and negotiation.
class StorageShm final : public IStorage
{
public:
    StorageShm(const StorageID & id, const ColumnsDescription & cols, const String & shm_name_);

    String getName() const override { return "Shm"; }
    bool isRemote() const override { return false; }

    /// N11: single stream regardless of num_streams. We don't even resize the
    /// pipe — the produced Pipe holds exactly one PollableShmSource and we
    /// return it as-is. The downstream `parallelize_output_from_storages`
    /// machinery may still add splitters; that's downstream's choice.
    Pipe read(
        const Names & column_names,
        const StorageSnapshotPtr & storage_snapshot,
        SelectQueryInfo & query_info,
        ContextPtr context,
        QueryProcessingStage::Enum processed_stage,
        size_t max_block_size,
        size_t num_streams) override;

private:
    String shm_name;
};

}

#endif
