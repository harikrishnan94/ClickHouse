#!/usr/bin/env python3
"""check_order.py - assert that a ClickHouse Native stream on stdin is ordered.

Reads a ClickHouse Native-format stream (as emitted with client protocol
version 0, i.e. what `clickhouse local`, `clickhouse client` file output and
the plain HTTP interface produce) block by block and asserts that the single
UInt64 column named 'tag' is non-decreasing WITHIN each block.
With --global it additionally asserts the order across blocks (the first tag
of block N must be >= the last tag of block N-1).

Wire framing (verified empirically against build/reldeb clickhouse local and
against src/Formats/NativeWriter.cpp, see SELFTEST.md):
  per block:
    varuint num_columns
    varuint num_rows
    per column:
      string  name      (varuint length + bytes)
      string  type      (varuint length + bytes)
      data              (for UInt64: num_rows * 8 bytes little-endian)
  NO BlockInfo prefix and NO custom-serialization prefix byte: those are only
  written for client protocol revision > 0 / >= 54454 respectively
  (NativeWriter::write in src/Formats/NativeWriter.cpp), and file/HTTP output
  uses client_protocol_version = 0.

Exit code: 0 if no violations (and no parse errors), 1 otherwise.
Penultimate line (always, machine-checkable): STATS blocks=B rows=R violations=V
Final line (machine-checkable):
  ORDER-BLOCKS OK (B blocks, R rows)
  ORDER-BLOCKS FAIL (V violations in B blocks)
  ORDER-BLOCKS FAIL (parse error)

Server lifecycle helpers in run_order.sh are adapted from
/mnt/data/jbmt_results/jbmt-sweep-20260724/join_memory_bench.py; this parser
is written from scratch against NativeWriter.cpp because that harness never
parses Native streams.
"""

import argparse
import struct
import sys

MAX_VIOLATIONS_PRINTED = 20


class StreamReader:
    """Buffered reader over a binary file object with exact-read semantics."""

    def __init__(self, fileobj):
        self.f = fileobj
        self.offset = 0

    def read_exact(self, n):
        data = self.f.read(n)
        self.offset += len(data)
        if len(data) != n:
            raise EOFError(
                f"unexpected end of stream at offset {self.offset}: "
                f"wanted {n} bytes, got {len(data)}"
            )
        return data

    def read_varuint_or_eof(self):
        """Returns None on clean EOF at a block boundary."""
        first = self.f.read(1)
        if len(first) == 0:
            return None
        self.offset += 1
        b = first[0]
        value = b & 0x7F
        shift = 7
        while b & 0x80:
            b = self.read_exact(1)[0]
            value |= (b & 0x7F) << shift
            shift += 7
            if shift > 63:
                raise ValueError(f"varuint too long at offset {self.offset}")
        return value

    def read_varuint(self):
        v = self.read_varuint_or_eof()
        if v is None:
            raise EOFError(f"unexpected end of stream at offset {self.offset}")
        return v

    def read_binary_string(self):
        length = self.read_varuint()
        if length > 1 << 20:
            raise ValueError(
                f"implausible string length {length} at offset {self.offset}; "
                "framing assumption violated (BlockInfo present? compressed stream?)"
            )
        return self.read_exact(length).decode("utf-8")


def check_stream(reader, global_order, column_name, quiet):
    n_blocks = 0
    n_rows = 0
    n_violations = 0
    n_printed = 0
    prev_block_last = None  # for --global

    while True:
        num_columns = reader.read_varuint_or_eof()
        if num_columns is None:
            break
        num_rows = reader.read_varuint()
        if num_columns != 1:
            raise ValueError(
                f"block {n_blocks}: expected exactly 1 column, got {num_columns} "
                "(the harness queries must SELECT only the tag column)"
            )
        name = reader.read_binary_string()
        type_name = reader.read_binary_string()
        if name != column_name:
            raise ValueError(
                f"block {n_blocks}: expected column '{column_name}', got '{name}'"
            )
        if type_name != "UInt64":
            raise ValueError(
                f"block {n_blocks}: expected type UInt64 for column "
                f"'{column_name}', got '{type_name}'"
            )

        # Zero rows => zero bytes of data (NativeWriter::write).
        if num_rows > 0:
            data = reader.read_exact(num_rows * 8)
            values = struct.unpack(f"<{num_rows}Q", data)

            prev = values[0]
            block_violation_reported = False
            for row_idx in range(1, num_rows):
                v = values[row_idx]
                if v < prev:
                    n_violations += 1
                    if not block_violation_reported and not quiet:
                        if n_printed < MAX_VIOLATIONS_PRINTED:
                            print(
                                f"VIOLATION block={n_blocks} rows={num_rows} "
                                f"first_offending_row={row_idx} "
                                f"tag[{row_idx - 1}]={prev} tag[{row_idx}]={v}"
                            )
                            n_printed += 1
                        block_violation_reported = True
                prev = v

            if global_order and prev_block_last is not None and values[0] < prev_block_last:
                n_violations += 1
                if not quiet and n_printed < MAX_VIOLATIONS_PRINTED:
                    print(
                        f"VIOLATION (global) block={n_blocks} first_row_tag={values[0]} "
                        f"< previous_block_last_tag={prev_block_last}"
                    )
                    n_printed += 1
            prev_block_last = values[-1]

        n_blocks += 1
        n_rows += num_rows

    return n_blocks, n_rows, n_violations


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--global",
        dest="global_order",
        action="store_true",
        help="also assert order across blocks (single-stream runs only)",
    )
    parser.add_argument(
        "--column",
        default="tag",
        help="expected column name (default: tag)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="suppress per-violation lines, print only the final line",
    )
    args = parser.parse_args()

    reader = StreamReader(sys.stdin.buffer)
    try:
        n_blocks, n_rows, n_violations = check_stream(
            reader, args.global_order, args.column, args.quiet
        )
    except (EOFError, ValueError, UnicodeDecodeError) as e:
        print(f"PARSE ERROR: {e}")
        print("STATS blocks=0 rows=0 violations=0")
        print("ORDER-BLOCKS FAIL (parse error)")
        return 1

    print(f"STATS blocks={n_blocks} rows={n_rows} violations={n_violations}")

    if n_blocks == 0 or n_rows == 0:
        # An empty result is vacuous, not a pass: the harness always expects rows.
        print(f"PARSE ERROR: empty stream ({n_blocks} blocks, {n_rows} rows)")
        print("ORDER-BLOCKS FAIL (parse error)")
        return 1

    if n_violations == 0:
        print(f"ORDER-BLOCKS OK ({n_blocks} blocks, {n_rows} rows)")
        return 0
    print(f"ORDER-BLOCKS FAIL ({n_violations} violations in {n_blocks} blocks)")
    return 1


if __name__ == "__main__":
    sys.exit(main())
