#pragma once

#if defined(OS_LINUX)

#include <TableFunctions/ITableFunction.h>

#include <base/types.h>

#include <string>


namespace DB
{

/// `shm(name, columns)` — the SQL surface for the zero-copy SHM source feature.
///
///   SELECT * FROM shm('/my_shm_object', 'id UInt64, v1 UInt64, s1 String')
///
/// Gated behind the experimental setting `allow_experimental_shm_table_function`
/// (default false). Membership of every declared type in the supported set
/// {UInt64, String} is checked at parse/resolve time as the SQL-side gate of
/// `pollable-shm-source.md` Producer-side preconditions enumerated row 6.
///
/// Spec authority: `pollable-shm-source.md` §shm() table function, AC9,
/// §Attach-time observable failures (gate + schema-mismatch rows).
class TableFunctionShm final : public ITableFunction
{
public:
    static constexpr auto name = "shm";

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

    const char * getStorageEngineName() const override { return "Shm"; }

    String shm_name;
    String columns_str;
};

}

#endif
