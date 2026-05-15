#if defined(OS_LINUX)

#include <Storages/SharedMemorySource/Source/StorageShm.h>
#include <Storages/SharedMemorySource/Source/PollableShmSource.h>

#include <Core/Block.h>
#include <Core/Settings.h>
#include <DataTypes/IDataType.h>
#include <Interpreters/Context.h>
#include <QueryPipeline/Pipe.h>
#include <Storages/ColumnsDescription.h>
#include <Storages/StorageInMemoryMetadata.h>
#include <Storages/StorageSnapshot.h>

#include <memory>
#include <utility>
#include <vector>


namespace DB
{

namespace Setting
{
    extern const SettingsUInt64 shm_source_stall_timeout_ms;
}


StorageShm::StorageShm(const StorageID & id, const ColumnsDescription & cols, const String & shm_name_)
    : IStorage(id)
    , shm_name(shm_name_)
{
    StorageInMemoryMetadata storage_metadata;
    storage_metadata.setColumns(cols);
    setInMemoryMetadata(storage_metadata);
}


Pipe StorageShm::read(
    const Names & column_names,
    const StorageSnapshotPtr & storage_snapshot,
    SelectQueryInfo & /*query_info*/,
    ContextPtr context,
    QueryProcessingStage::Enum /*processed_stage*/,
    size_t /*max_block_size*/,
    size_t /*num_streams*/)
{
    storage_snapshot->check(column_names);
    const auto & meta_columns = storage_snapshot->metadata->getColumns();

    /// Build the FULL producer schema in declared (ordinal) order. The SHM-adoption ABI
    /// handshake compares the producer's published schema against this complete list
    /// (preconditions 4–6); a projection (`SELECT count()`, `SELECT s1`) does not change
    /// what the producer publishes, only what we *emit* downstream — Finding 4.
    std::vector<DataTypePtr> full_column_types;
    std::vector<String> full_column_names;
    const auto ordinary = meta_columns.getOrdinary();
    full_column_types.reserve(ordinary.size());
    full_column_names.reserve(ordinary.size());
    for (const auto & nt : ordinary)
    {
        full_column_names.push_back(nt.name);
        full_column_types.push_back(nt.type);
    }

    /// Build the chunk header from the requested column subset (what downstream sees).
    /// Empty `column_names` (e.g. `SELECT count()`) → empty header → zero-column Chunks.
    Block header;
    std::vector<String> requested_names;
    requested_names.reserve(column_names.size());
    for (const auto & name : column_names)
    {
        const auto & col = meta_columns.get(name);
        header.insert({col.type->createColumn(), col.type, col.name});
        requested_names.push_back(col.name);
    }

    auto shared_header = std::make_shared<const Block>(std::move(header));
    const UInt64 stall_ms = context->getSettingsRef()[Setting::shm_source_stall_timeout_ms];

    /// N11: exactly one source per `read()` call.
    return Pipe(std::make_shared<PollableShmSource>(
        std::move(shared_header), shm_name,
        std::move(full_column_types), std::move(full_column_names),
        std::move(requested_names), stall_ms));
}

}

#endif
