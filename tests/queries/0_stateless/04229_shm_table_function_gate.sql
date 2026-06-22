-- Tags: no-fasttest
-- Tags rationale:
--   no-fasttest: the SHM-backed streamed_table() function is Linux-only and gated by an
--   experimental setting; keep it out of the fasttest lane for now.

-- AC9: with the setting off (default), the function is rejected at parse/resolve.
SELECT * FROM streamed_table('any_name', 'a UInt64'); -- { serverError SUPPORT_IS_DISABLED }
-- The legacy alias shm() resolves to the same function and is gated identically.
SELECT * FROM shm('any_name', 'a UInt64'); -- { serverError SUPPORT_IS_DISABLED }

-- Enabling the setting unlocks the function; an attach to a non-existent SHM object then
-- surfaces the attach-failed class.
SET allow_experimental_streamed_table_function = 1;
SELECT * FROM streamed_table('this_shm_object_does_not_exist_04229', 'a UInt64'); -- { serverError SHM_ATTACH_FAILED }

-- The legacy setting name is an alias and enables the function just the same.
SET allow_experimental_streamed_table_function = 0;
SET allow_experimental_shm_table_function = 1;
SELECT * FROM streamed_table('this_shm_object_does_not_exist_04229', 'a UInt64'); -- { serverError SHM_ATTACH_FAILED }

-- The SQL-side membership gate rejects unsupported column types before attach.
SELECT * FROM streamed_table('any_name', 'a Array(UInt64)'); -- { serverError SHM_SCHEMA_MISMATCH }
SELECT * FROM streamed_table('any_name', 'a Tuple(UInt64, String)'); -- { serverError SHM_SCHEMA_MISMATCH }
SELECT * FROM streamed_table('any_name', 'a Nullable(UInt64)'); -- { serverError SHM_SCHEMA_MISMATCH }
-- Decimal256 (precision > 38) is outside the adopted-column set (which covers widths up to 16
-- bytes), so it is declined at the gate -- the offload fails closed rather than streaming a type
-- the zero-copy path cannot represent.
SELECT * FROM streamed_table('any_name', 'a Decimal(40, 2)'); -- { serverError SHM_SCHEMA_MISMATCH }
SELECT * FROM streamed_table('any_name', 'a Decimal256(10)'); -- { serverError SHM_SCHEMA_MISMATCH }

-- The supported fixed-width + String set passes the gate; only the attach fails. This covers
-- the types the ClickBench / TPC-H Q1/Q6/Q18 target workloads need (integers, floats, Date,
-- DateTime, decimals, String). Previously Int32, DateTime, and Decimal were rejected at the
-- gate; the extended ABI now accepts them, so they reach attach and fail there instead.
SELECT * FROM streamed_table('any_name', 'id UInt64, s String'); -- { serverError SHM_ATTACH_FAILED }
SELECT * FROM streamed_table('any_name', 'a Int32'); -- { serverError SHM_ATTACH_FAILED }
SELECT * FROM streamed_table('any_name', 'a DateTime'); -- { serverError SHM_ATTACH_FAILED }
SELECT * FROM streamed_table('any_name', 'a UInt8, b Int16, c UInt32, d Int64, e Float32, f Float64, g Date, h Date32'); -- { serverError SHM_ATTACH_FAILED }
-- Decimal32/64/128 and DateTime64 are in the adopted set (the scale rides in the type string).
SELECT * FROM streamed_table('any_name', 'a Decimal(9, 2), b Decimal(18, 4), c Decimal(38, 6)'); -- { serverError SHM_ATTACH_FAILED }
SELECT * FROM streamed_table('any_name', 'a Decimal32(2), b Decimal64(4), c Decimal128(6)'); -- { serverError SHM_ATTACH_FAILED }
SELECT * FROM streamed_table('any_name', 'a DateTime64(3)'); -- { serverError SHM_ATTACH_FAILED }
-- The legacy shm() alias accepts the extended set too.
SELECT * FROM shm('any_name', 'a Int64, b String'); -- { serverError SHM_ATTACH_FAILED }
