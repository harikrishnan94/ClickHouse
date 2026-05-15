-- Tags: no-fasttest
-- Tags rationale:
--   no-fasttest: the new SHM table function is Linux-only and gated by an experimental setting;
--   keep it out of the fasttest lane for now.

-- AC9: with the setting off (default), the function is rejected at parse/resolve.
SELECT * FROM shm('any_name', 'a UInt64'); -- { serverError SUPPORT_IS_DISABLED }

-- Enabling the setting unlocks the function; an attach to a non-existent SHM object then
-- surfaces the attach-failed class.
SET allow_experimental_shm_table_function = 1;
SELECT * FROM shm('this_shm_object_does_not_exist_04229', 'a UInt64'); -- { serverError SHM_ATTACH_FAILED }

-- The SQL-side membership gate rejects unsupported column types before attach.
SELECT * FROM shm('any_name', 'a Int32'); -- { serverError SHM_SCHEMA_MISMATCH }
SELECT * FROM shm('any_name', 'a DateTime'); -- { serverError SHM_SCHEMA_MISMATCH }
SELECT * FROM shm('any_name', 'a Array(UInt64)'); -- { serverError SHM_SCHEMA_MISMATCH }

-- The supported set {UInt64, String} passes the gate; only the attach fails.
SELECT * FROM shm('any_name', 'id UInt64, s String'); -- { serverError SHM_ATTACH_FAILED }
