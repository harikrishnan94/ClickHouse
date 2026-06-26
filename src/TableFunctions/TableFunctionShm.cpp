#if defined(OS_LINUX)

#include <TableFunctions/TableFunctionShm.h>
#include <TableFunctions/TableFunctionFactory.h>
#include <TableFunctions/registerTableFunctions.h>

#include <Core/Settings.h>
#include <Core/Types_fwd.h>
#include <Interpreters/Context.h>
#include <Interpreters/evaluateConstantExpression.h>
#include <Interpreters/parseColumnsListForTableFunction.h>
#include <Parsers/ASTFunction.h>
#include <Storages/ColumnsDescription.h>
#include <Storages/SharedMemorySource/Source/StorageShm.h>
#include <Storages/SharedMemorySource/Source/TransportMode.h>
#include <Storages/SharedMemorySource/Wire/WireTypeMapping.h>
#include <Storages/checkAndGetLiteralArgument.h>
#include <Common/Exception.h>

#include <memory>


namespace DB
{

namespace ErrorCodes
{
    extern const int NUMBER_OF_ARGUMENTS_DOESNT_MATCH;
    extern const int SUPPORT_IS_DISABLED;
    extern const int SHM_SCHEMA_MISMATCH;
    extern const int BAD_ARGUMENTS;
}

namespace Setting
{
    extern const SettingsBool allow_experimental_streamed_table_function;
}


void TableFunctionShm::parseArguments(const ASTPtr & ast_function, ContextPtr context)
{
    const auto * function = ast_function->as<ASTFunction>();
    if (function == nullptr || function->arguments == nullptr)
        throw Exception(ErrorCodes::NUMBER_OF_ARGUMENTS_DOESNT_MATCH,
            "Table function 'streamed_table' requires 2 or 3 arguments: "
            "streamed_table(name String, columns String[, transport String])");

    auto & args = function->arguments->children;
    if (args.size() < 2 || args.size() > 3)
        throw Exception(ErrorCodes::NUMBER_OF_ARGUMENTS_DOESNT_MATCH,
            "Table function 'streamed_table' requires 2 or 3 arguments (got {}): "
            "streamed_table(name String, columns String[, transport String])",
            args.size());

    shm_name = checkAndGetLiteralArgument<String>(
        evaluateConstantExpressionOrIdentifierAsLiteral(args[0], context), "name");
    columns_str = checkAndGetLiteralArgument<String>(
        evaluateConstantExpressionOrIdentifierAsLiteral(args[1], context), "columns");

    /// Optional 3rd literal arg: the consumer transport spec (D-HC-0001). Absent => zero-copy
    /// adopt (default), preserving the original 2-arg behavior.
    if (args.size() == 3)
    {
        const String transport_str = checkAndGetLiteralArgument<String>(
            evaluateConstantExpressionOrIdentifierAsLiteral(args[2], context), "transport");
        if (!tryParseShmTransportSpec(transport_str, transport_spec))
            throw Exception(ErrorCodes::BAD_ARGUMENTS,
                "Table function 'streamed_table': unrecognised transport spec '{}'. Expected one of "
                "'shm' / 'shm:adopt' (zero-copy adopt, default), 'shm:copy' (consumer copy out of "
                "shared memory), or 'tcp:<host>:<port>'.",
                transport_str);
    }
}


ColumnsDescription TableFunctionShm::getActualTableStructure(ContextPtr context, bool /*is_insert_query*/) const
{
    return parseColumnsListFromString(columns_str, context);
}


StoragePtr TableFunctionShm::executeImpl(
    const ASTPtr & /*ast_function*/,
    ContextPtr context,
    const std::string & table_name,
    ColumnsDescription /*cached_columns*/,
    bool is_insert_query) const
{
    /// `pollable-shm-source.md` AC9: gate at parse/resolve. We raise SUPPORT_IS_DISABLED
    /// (the `feature-gate-disabled` failure class per `pollable-shm-source.md` Failure
    /// classes table) when the experimental setting is off.
    if (!context->getSettingsRef()[Setting::allow_experimental_streamed_table_function])
        throw Exception(ErrorCodes::SUPPORT_IS_DISABLED,
            "Table function 'streamed_table' is experimental. "
            "Set `allow_experimental_streamed_table_function = 1` to enable it.");

    /// `pollable-shm-source.md` Producer-side preconditions enumerated row 6:
    /// SQL-side membership gate (BEFORE attach). Phase-1 set: {UInt64, String}.
    /// This is the `schema-mismatch` failure class.
    auto columns = getActualTableStructure(context, is_insert_query);
    for (const auto & col : columns)
    {
        if (!SharedMemoryWire::isSupportedShmType(col.type->getTypeId()))
            throw Exception(ErrorCodes::SHM_SCHEMA_MISMATCH,
                "Table function 'streamed_table': column '{}' has type '{}' which is not in the supported set "
                "{} for the SHM-adoption ABI",
                col.name, col.type->getName(), SharedMemoryWire::supportedShmTypeList());
    }

    /// SHM transports (adopt/copy) attach by name; TCP transport (Phase 1) connects to the
    /// producer-per-stream listener carried in the transport spec (host:port).
    auto storage = std::make_shared<StorageShm>(
        StorageID(getDatabaseName(), table_name), columns, shm_name,
        transport_spec.mode, transport_spec.tcp_host, transport_spec.tcp_port);
    storage->startup();
    return storage;
}


void registerTableFunctionShm(TableFunctionFactory & factory)
{
    factory.registerFunction<TableFunctionShm>(
    {
        .description = "Reads rows from a producer-published POSIX shared-memory block stream. "
                       "An optional 3rd argument selects the consumer transport: 'shm' / 'shm:adopt' "
                       "(zero-copy SHM adoption, default), 'shm:copy' (consumer copies each block out "
                       "of shared memory and releases the ring slot immediately), or 'tcp:<host>:<port>' "
                       "(reserved). Experimental: requires allow_experimental_streamed_table_function = 1.",
        .examples = {
            {"basic",
             "SELECT count() FROM streamed_table('/my_shm', 'id UInt64, s String') "
             "SETTINGS allow_experimental_streamed_table_function = 1",
             ""},
            {"copy-mode",
             "SELECT count() FROM streamed_table('/my_shm', 'id UInt64, s String', 'shm:copy') "
             "SETTINGS allow_experimental_streamed_table_function = 1",
             ""}},
        .category = FunctionDocumentation::Category::TableFunction
    });

    /// Legacy alias: `shm(...)` continues to resolve to the same table function.
    factory.registerAlias(TableFunctionShm::name_legacy, TableFunctionShm::name);
}

}

#endif
