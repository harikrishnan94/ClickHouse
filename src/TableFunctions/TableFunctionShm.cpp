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
}

namespace Setting
{
    extern const SettingsBool allow_experimental_shm_table_function;
}


void TableFunctionShm::parseArguments(const ASTPtr & ast_function, ContextPtr context)
{
    const auto * function = ast_function->as<ASTFunction>();
    if (function == nullptr || function->arguments == nullptr)
        throw Exception(ErrorCodes::NUMBER_OF_ARGUMENTS_DOESNT_MATCH,
            "Table function 'shm' requires 2 arguments: shm(name String, columns String)");

    auto & args = function->arguments->children;
    if (args.size() != 2)
        throw Exception(ErrorCodes::NUMBER_OF_ARGUMENTS_DOESNT_MATCH,
            "Table function 'shm' requires exactly 2 arguments (got {}): shm(name String, columns String)",
            args.size());

    shm_name = checkAndGetLiteralArgument<String>(
        evaluateConstantExpressionOrIdentifierAsLiteral(args[0], context), "name");
    columns_str = checkAndGetLiteralArgument<String>(
        evaluateConstantExpressionOrIdentifierAsLiteral(args[1], context), "columns");
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
    if (!context->getSettingsRef()[Setting::allow_experimental_shm_table_function])
        throw Exception(ErrorCodes::SUPPORT_IS_DISABLED,
            "Table function 'shm' is experimental. "
            "Set `allow_experimental_shm_table_function = 1` to enable it.");

    /// `pollable-shm-source.md` Producer-side preconditions enumerated row 6:
    /// SQL-side membership gate (BEFORE attach). Phase-1 set: {UInt64, String}.
    /// This is the `schema-mismatch` failure class.
    auto columns = getActualTableStructure(context, is_insert_query);
    for (const auto & col : columns)
    {
        const auto type_id = col.type->getTypeId();
        if (type_id != TypeIndex::UInt64 && type_id != TypeIndex::String)
            throw Exception(ErrorCodes::SHM_SCHEMA_MISMATCH,
                "Table function 'shm': column '{}' has type '{}' which is not in the supported set "
                "{{UInt64, String}} for the phase-1 SHM-adoption ABI",
                col.name, col.type->getName());
    }

    auto storage = std::make_shared<StorageShm>(
        StorageID(getDatabaseName(), table_name), columns, shm_name);
    storage->startup();
    return storage;
}


void registerTableFunctionShm(TableFunctionFactory & factory)
{
    factory.registerFunction<TableFunctionShm>(
    {
        .description = "Reads rows from a producer-published POSIX shared-memory block stream "
                       "via the zero-copy SHM-adoption ABI. Experimental: requires "
                       "allow_experimental_shm_table_function = 1.",
        .examples = {
            {"basic",
             "SELECT count() FROM shm('/my_shm', 'id UInt64, s String') "
             "SETTINGS allow_experimental_shm_table_function = 1",
             ""}},
        .category = FunctionDocumentation::Category::TableFunction
    });
}

}

#endif
