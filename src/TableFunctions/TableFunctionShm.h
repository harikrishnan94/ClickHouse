#pragma once

#if defined(OS_LINUX)

#    include <TableFunctions/ITableFunction.h>

#    include <base/types.h>

#    include <string>


namespace DB
{

/// `streamed_table(name, columns)` — the SQL surface for the zero-copy SHM source
/// feature. The legacy name `shm(...)` is kept as an alias.
///
///   SELECT * FROM streamed_table('/my_shm_object', 'id UInt64, v1 UInt64, s1 String')
///
/// Gated behind the experimental setting `allow_experimental_streamed_table_function`
/// (default false; legacy alias `allow_experimental_shm_table_function`). Membership
/// of every declared type in the supported fixed-width / String set is checked at
/// parse/resolve time as the SQL-side gate of `pollable-shm-source.md` Producer-side
/// preconditions enumerated row 6.
///
/// Spec authority: `pollable-shm-source.md` §streamed_table() table function, AC9,
/// §Attach-time observable failures (gate + schema-mismatch rows).
class TableFunctionShm final : public ITableFunction
{
public:
    static constexpr auto name = "streamed_table";
    /// Legacy name registered as an alias for backward compatibility.
    static constexpr auto name_legacy = "shm";

    std::string getName() const override { return name; }

    void parseArguments(const ASTPtr & ast_function, ContextPtr context) override;
    ColumnsDescription getActualTableStructure(ContextPtr context, bool is_insert_query) const override;

    bool hasStaticStructure() const override { return true; }

private:
    StoragePtr executeImpl(
        const ASTPtr & ast_function,
        ContextPtr context,
        const std::string & table_name,
        ColumnsDescription cached_columns,
        bool is_insert_query) const override;

    /// StorageShm is a transient storage constructed directly by executeImpl(),
    /// not a StorageFactory-registered engine. Returning an empty engine name
    /// keeps ITableFunction::checkSourceAccess() from querying StorageFactory
    /// for an engine that intentionally does not exist.
    const char * getStorageEngineName() const override { return ""; }

    String shm_name;
    String columns_str;
};

}

#endif
