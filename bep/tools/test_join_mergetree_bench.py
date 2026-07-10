#!/usr/bin/env python3
"""Unit tests for the persistent `MergeTree` join benchmark driver."""

import contextlib
import dataclasses
import decimal
import hashlib
import io
import os
import re
import sys
import unittest
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, os.path.dirname(__file__))

import join_mergetree_bench as bench


def make_metadata(**overrides):
    fields = dict(
        schema_version=bench.SCHEMA_VERSION,
        max_cardinality=32,
        bucket_width=8,
        max_multiplicity=2,
        max_cycles=3,
        max_build_payload_columns=1,
        max_probe_payload_columns=1,
        generator_signature=bench.GENERATOR_SIGNATURE,
    )
    fields.update(overrides)
    return bench.LoadedMetadata(**fields)


def make_point(**overrides):
    fields = dict(
        cardinality=16,
        bucket_width=8,
        multiplicity=2,
        ratio=decimal.Decimal("1.5"),
        hit_rate=decimal.Decimal("0.5"),
        probe_rows=36,
        hit_rows=13,
    )
    fields.update(overrides)
    return bench.BenchmarkPoint(**fields)


def build_part_pairs(metadata):
    return [(str(i), f"build-hash-{i}") for i in range(metadata.max_multiplicity)]


def probe_part_pairs(metadata):
    return [(str(j), f"probe-hash-{j}") for j in range(metadata.max_cycles)]


def metadata_with_fingerprints(metadata):
    return dataclasses.replace(
        metadata,
        build_part_fingerprint=bench.part_fingerprint(build_part_pairs(metadata)),
        probe_part_fingerprint=bench.part_fingerprint(probe_part_pairs(metadata)),
    )


def make_layout_rows(metadata):
    rows = [
        {"kind": "table", "key": bench.BUILD_TABLE,
         "value": "MergeTree|occurrence|occurrence, card_bucket, shuffle_rank"},
        {"kind": "table", "key": bench.PROBE_TABLE,
         "value": "MergeTree|cycle|cycle, card_bucket, rank"},
        {"kind": "table", "key": bench.METADATA_TABLE, "value": "MergeTree||"},
    ]
    for table, columns in (
        (bench.BUILD_TABLE, bench.build_columns(metadata.max_build_payload_columns)),
        (bench.PROBE_TABLE, bench.probe_columns(metadata.max_probe_payload_columns)),
        (bench.METADATA_TABLE, bench.METADATA_COLUMNS),
    ):
        rows.extend(
            {"kind": "column", "key": f"{table}.{position}",
             "value": f"{name}|{type_name}"}
            for position, (name, type_name) in enumerate(columns, 1)
        )
    rows.extend(
        {"kind": "part", "key": f"{bench.BUILD_TABLE}.{partition}",
         "value": f"1|{part_hash}"}
        for partition, part_hash in build_part_pairs(metadata)
    )
    rows.extend(
        {"kind": "part", "key": f"{bench.PROBE_TABLE}.{partition}",
         "value": f"1|{part_hash}"}
        for partition, part_hash in probe_part_pairs(metadata)
    )
    rows.append(
        {"kind": "part", "key": f"{bench.METADATA_TABLE}.all", "value": "1|metadata-hash"}
    )
    return rows


def make_state_rows(metadata):
    rows = make_layout_rows(metadata)
    rows.extend(
        {"kind": "build_count", "key": str(i), "value": str(metadata.max_cardinality)}
        for i in range(metadata.max_multiplicity)
    )
    rows.extend(
        {"kind": "probe_count", "key": str(j), "value": str(metadata.max_cardinality)}
        for j in range(metadata.max_cycles)
    )
    return rows


def simulated_selected_rows(point, max_cycles, total_buckets):
    """Enumerate probe-table rows the run predicate should select."""
    cycles_full, remainder, full_buckets, remainder_ranks, bucket_count = (
        bench._probe_selection(point)
    )
    rows = []
    for cycle in range(max_cycles):
        for bucket in range(total_buckets):
            for rank in range(point.bucket_width):
                if (cycle < cycles_full and bucket < bucket_count) or (
                    remainder
                    and cycle == cycles_full
                    and (
                        bucket < full_buckets
                        or (bucket == full_buckets and rank < remainder_ranks)
                    )
                ):
                    rows.append((cycle, bucket, rank))
    return rows


class ParseTests(unittest.TestCase):
    def test_integer_list_preserves_exact_order(self):
        self.assertEqual(bench.parse_integer_list("100,7,42", "cardinalities"), [100, 7, 42])

    def test_integer_list_rejects_empty_duplicate_and_nonpositive_values(self):
        for text in ("", "1,,2", "1,1", "0", "-1", "1.0"):
            with self.subTest(text=text), self.assertRaises(ValueError):
                bench.parse_integer_list(text, "cardinalities")

    def test_nonnegative_integer_list_accepts_zero_and_rejects_invalid_values(self):
        self.assertEqual(
            bench.parse_nonnegative_integer_list("0,2,7", "payload columns"),
            [0, 2, 7],
        )
        for text in ("", "0,,2", "1,1", "-1", "1.0", "x"):
            with self.subTest(text=text), self.assertRaises(ValueError):
                bench.parse_nonnegative_integer_list(text, "payload columns")

    def test_decimal_list_accepts_integral_decimal_values(self):
        self.assertEqual(
            bench.parse_decimal_list("2.0,0.5", "ratios"),
            [decimal.Decimal("2.0"), decimal.Decimal("0.5")],
        )

    def test_decimal_list_rejects_nonfinite_nonpositive_and_duplicates(self):
        for text in ("", "1,,2", "NaN", "Infinity", "0", "-0.1", "1,1.0"):
            with self.subTest(text=text), self.assertRaises(ValueError):
                bench.parse_decimal_list(text, "ratios")

    def test_canonical_decimal_is_stable(self):
        self.assertEqual(bench.canonical_decimal(decimal.Decimal("2.000")), "2")
        self.assertEqual(bench.canonical_decimal(decimal.Decimal("0.0500")), "0.05")


class SeedMixingTests(unittest.TestCase):
    def test_python_int_hash64_matches_clickhouse_reference_values(self):
        self.assertEqual(bench.int_hash64(0), 4761183170873013810)
        self.assertEqual(bench.int_hash64(1), 10577349846663553072)
        self.assertEqual(bench.int_hash64(123456789), 16268010412262273259)

    def test_mix_seed_wraps_multiplication_to_uint64(self):
        mixed = bench.mix_seed(bench.AFFINE_SEED, 63, 511)
        self.assertGreaterEqual(mixed, 0)
        self.assertLessEqual(mixed, bench.UINT64_MAX)
        self.assertEqual(
            mixed,
            bench.AFFINE_SEED
            ^ ((63 * bench.OCCURRENCE_MIX) & bench.UINT64_MAX)
            ^ ((511 * bench.BUCKET_MIX) & bench.UINT64_MAX),
        )

    def test_mix_seed_sql_text_appears_in_generated_inserts(self):
        metadata = make_metadata()
        build_sql = bench.build_insert_sql(metadata)
        self.assertIn(
            bench.mix_seed_sql(bench.SHUFFLE_SEED, "occurrence", "card_bucket"),
            build_sql,
        )
        probe_sql = bench.probe_insert_sql(metadata)
        self.assertIn(
            bench.mix_seed_sql(bench.AFFINE_SEED, "cycle", "card_bucket"),
            probe_sql,
        )
        self.assertIn(
            bench.mix_seed_sql(bench.PROBE_CYCLE_SEED, "cycle", "card_bucket"),
            probe_sql,
        )

    def test_affine_permutation_covers_every_bucket_exactly_once_per_cycle(self):
        width = 8
        orders = set()
        for cycle in range(3):
            for bucket in range(4):
                multiplier = (
                    bench.int_hash64(bench.mix_seed(bench.AFFINE_SEED, cycle, bucket)) | 1
                )
                offset = bench.int_hash64(
                    bench.mix_seed(bench.PROBE_CYCLE_SEED, cycle, bucket)
                ) % width
                selectors = [
                    bucket * width + (rank * multiplier + offset) % width
                    for rank in range(width)
                ]
                self.assertEqual(
                    set(selectors),
                    set(range(bucket * width, (bucket + 1) * width)),
                    f"cycle={cycle} bucket={bucket}",
                )
                orders.add(tuple(selector % width for selector in selectors))
        self.assertGreater(len(orders), 1, "shuffles must differ across blocks")


class PointValidationTests(unittest.TestCase):
    def setUp(self):
        self.metadata = make_metadata(
            max_multiplicity=4,
            max_build_payload_columns=3,
            max_probe_payload_columns=2,
        )

    def test_round_hit_count_uses_decimal_half_up(self):
        self.assertEqual(bench.round_hit_count(5, decimal.Decimal("0.5")), 3)
        self.assertEqual(bench.round_hit_count(3, decimal.Decimal("0.5")), 2)
        self.assertEqual(bench.round_hit_count(10, decimal.Decimal("0.05")), 1)

    def test_round_hit_count_ignores_decimal_context_precision(self):
        just_below_half = decimal.Decimal(
            "0.4999999999999999999999999999999999999999"
        )
        just_above_half = decimal.Decimal(
            "0.5000000000000000000000000000000000000001"
        )
        with decimal.localcontext() as context:
            context.prec = 6
            self.assertEqual(bench.round_hit_count(1, just_below_half), 0)
            self.assertEqual(bench.round_hit_count(1, just_above_half), 1)

    def test_validate_points_accepts_bucket_multiples_with_exact_probe_counts(self):
        points = bench.validate_points(
            self.metadata,
            [8, 32],
            [1, 2],
            [decimal.Decimal("0.5"), decimal.Decimal("1.5")],
            [decimal.Decimal("0"), decimal.Decimal("1")],
        )
        self.assertEqual(len(points), 16)
        self.assertEqual(points[0].cardinality, 8)
        self.assertEqual(points[0].bucket_width, 8)
        self.assertEqual(points[0].probe_rows, 4)
        self.assertEqual(points[0].hit_rows, 0)

    def test_validate_points_rejects_cardinality_not_a_bucket_multiple(self):
        for cardinality in (4, 12, 40):
            with self.subTest(cardinality=cardinality), self.assertRaisesRegex(
                ValueError, r"multiple of bucket width 8.*32"
            ):
                bench.validate_points(
                    self.metadata,
                    [cardinality],
                    [1],
                    [decimal.Decimal("1")],
                    [decimal.Decimal("1")],
                )

    def test_validate_points_rejects_probe_rows_beyond_cycle_capacity(self):
        with self.assertRaisesRegex(ValueError, "max cycles 3"):
            bench.validate_points(
                self.metadata,
                [8],
                [2],
                [decimal.Decimal("2")],
                [decimal.Decimal("1")],
            )

    def test_validate_points_accepts_probe_rows_at_exact_cycle_capacity(self):
        points = bench.validate_points(
            self.metadata,
            [8],
            [2],
            [decimal.Decimal("1.5")],
            [decimal.Decimal("1")],
        )
        self.assertEqual(points[0].probe_rows, 24)
        self.assertEqual(
            points[0].probe_rows,
            self.metadata.max_cycles * points[0].cardinality,
        )

    def test_validate_points_adds_payload_cross_product(self):
        points = bench.validate_points(
            self.metadata,
            [8],
            [1],
            [decimal.Decimal("1")],
            [decimal.Decimal("1")],
            [0, 3],
            [0, 1, 2],
        )
        self.assertEqual(len(points), 6)
        self.assertEqual(
            [(point.build_payload_columns, point.probe_payload_columns) for point in points],
            [(0, 0), (0, 1), (0, 2), (3, 0), (3, 1), (3, 2)],
        )
        self.assertIn("bp=3", points[-1].label)
        self.assertIn("pp=2", points[-1].label)

    def test_validate_points_rejects_payload_counts_above_loaded_maxima(self):
        with self.assertRaisesRegex(ValueError, "build payload column count 4.*loaded maximum 3"):
            bench.validate_points(
                self.metadata,
                [8],
                [1],
                [decimal.Decimal("1")],
                [decimal.Decimal("1")],
                [4],
                [1],
            )
        with self.assertRaisesRegex(ValueError, "probe payload column count 3.*loaded maximum 2"):
            bench.validate_points(
                self.metadata,
                [8],
                [1],
                [decimal.Decimal("1")],
                [decimal.Decimal("1")],
                [1],
                [3],
            )

    def test_validate_points_rejects_out_of_range_axes(self):
        bad_cases = [
            ([8], [0], [decimal.Decimal("1")], [decimal.Decimal("1")]),
            ([8], [5], [decimal.Decimal("1")], [decimal.Decimal("1")]),
            ([8], [1], [decimal.Decimal("1")], [decimal.Decimal("1.01")]),
        ]
        for args in bad_cases:
            with self.subTest(args=args), self.assertRaises(ValueError):
                bench.validate_points(self.metadata, *args)

    def test_validate_points_requires_integral_probe_count(self):
        with self.assertRaisesRegex(ValueError, "exact integer"):
            bench.validate_points(
                self.metadata,
                [8],
                [1],
                [decimal.Decimal("0.15")],
                [decimal.Decimal("1")],
            )

    def test_validate_points_rejects_uint64_output_count_overflow(self):
        metadata = make_metadata(
            max_cardinality=1 << 56,
            bucket_width=1 << 56,
            max_multiplicity=64,
            max_cycles=128,
        )
        with self.assertRaisesRegex(ValueError, "joined output.*UInt64"):
            bench.validate_points(
                metadata,
                [1 << 56],
                [64],
                [decimal.Decimal("2")],
                [decimal.Decimal("1")],
            )


class LoadValidationTests(unittest.TestCase):
    @staticmethod
    def load_args(
        max_cardinality=32,
        bucket_width=8,
        max_multiplicity=2,
        max_cycles=3,
        max_build_payload_columns=1,
        max_probe_payload_columns=1,
    ):
        return SimpleNamespace(
            max_cardinality=max_cardinality,
            bucket_width=bucket_width,
            max_multiplicity=max_multiplicity,
            max_cycles=max_cycles,
            max_build_payload_columns=max_build_payload_columns,
            max_probe_payload_columns=max_probe_payload_columns,
        )

    def test_load_accepts_valid_bucketed_dimensions(self):
        metadata = bench._parse_load_metadata(self.load_args())
        self.assertEqual(metadata.schema_version, bench.SCHEMA_VERSION)
        self.assertEqual(metadata.max_cardinality, 32)
        self.assertEqual(metadata.bucket_width, 8)
        self.assertEqual(metadata.max_multiplicity, 2)
        self.assertEqual(metadata.max_cycles, 3)

    def test_load_rejects_cardinality_outside_domain(self):
        for max_cardinality in (0, -1, bench.MISS_DOMAIN_BIT, bench.MISS_DOMAIN_BIT + 1):
            with self.subTest(max_cardinality=max_cardinality), self.assertRaisesRegex(
                ValueError, "max cardinality"
            ):
                bench._parse_load_metadata(
                    self.load_args(max_cardinality=max_cardinality, bucket_width=1, max_cycles=1)
                )

    def test_load_rejects_non_power_of_two_bucket_width(self):
        for bucket_width in (0, -8, 3, 12, 24):
            with self.subTest(bucket_width=bucket_width), self.assertRaisesRegex(
                ValueError, "power of two"
            ):
                bench._parse_load_metadata(
                    self.load_args(max_cardinality=24, bucket_width=bucket_width)
                )

    def test_load_rejects_bucket_width_not_dividing_max_cardinality(self):
        with self.assertRaisesRegex(ValueError, "divide"):
            bench._parse_load_metadata(self.load_args(max_cardinality=36, bucket_width=8))

    def test_load_rejects_multiplicity_and_cycle_caps(self):
        with self.assertRaisesRegex(ValueError, "max multiplicity"):
            bench._parse_load_metadata(self.load_args(max_multiplicity=65))
        with self.assertRaisesRegex(ValueError, "max multiplicity"):
            bench._parse_load_metadata(self.load_args(max_multiplicity=0))
        with self.assertRaisesRegex(ValueError, "max cycles"):
            bench._parse_load_metadata(self.load_args(max_cycles=129))
        with self.assertRaisesRegex(ValueError, "max cycles"):
            bench._parse_load_metadata(self.load_args(max_cycles=0))

    def test_load_accepts_boundary_multiplicity_and_cycles(self):
        metadata = bench._parse_load_metadata(
            self.load_args(max_multiplicity=64, max_cycles=128)
        )
        self.assertEqual(metadata.max_multiplicity, 64)
        self.assertEqual(metadata.max_cycles, 128)

    def test_load_rejects_probe_domain_overflow(self):
        half_domain = bench.MISS_DOMAIN_BIT // 2
        metadata = bench._parse_load_metadata(
            self.load_args(max_cardinality=half_domain, bucket_width=half_domain, max_cycles=2)
        )
        self.assertEqual(
            metadata.max_cardinality * metadata.max_cycles, bench.MISS_DOMAIN_BIT
        )
        with self.assertRaisesRegex(ValueError, "max cycles"):
            bench._parse_load_metadata(
                self.load_args(
                    max_cardinality=half_domain, bucket_width=half_domain, max_cycles=4
                )
            )

    def test_load_rejects_negative_payload_maxima(self):
        for field in ("max_build_payload_columns", "max_probe_payload_columns"):
            args = self.load_args()
            setattr(args, field, -1)
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "nonnegative"):
                bench._parse_load_metadata(args)

    def test_capacity_estimate_follows_raw_byte_formula(self):
        metadata = make_metadata(
            max_build_payload_columns=3, max_probe_payload_columns=2
        )
        expected = 32 * 2 * 8 * (5 + 3) + 32 * 3 * 8 * (5 + 2)
        self.assertEqual(bench.estimate_raw_bytes(metadata), expected)

    def test_capacity_guard_uses_injected_free_bytes(self):
        metadata = make_metadata(
            max_build_payload_columns=3, max_probe_payload_columns=2
        )
        estimate = bench.estimate_raw_bytes(metadata)
        self.assertEqual(estimate, 9472)
        bench.check_capacity(metadata, free_bytes=10_525)
        with self.assertRaisesRegex(ValueError, "90%.*10000"):
            bench.check_capacity(metadata, free_bytes=10_000)


class LoadSqlTests(unittest.TestCase):
    def setUp(self):
        self.metadata = make_metadata()

    def test_schema_uses_bucketed_columns_partitions_and_sorting_keys(self):
        sql = bench.recreate_schema_sql(1, 1)
        for name in ("occurrence", "card_bucket", "selector", "k", "shuffle_rank"):
            self.assertIn(f"{name} UInt64", sql)
        for name in ("cycle", "rank", "hit_k", "miss_k"):
            self.assertIn(f"{name} UInt64", sql)
        self.assertIn("PARTITION BY occurrence", sql)
        self.assertIn("ORDER BY (occurrence, card_bucket, shuffle_rank)", sql)
        self.assertIn("PARTITION BY cycle", sql)
        self.assertIn("ORDER BY (cycle, card_bucket, rank)", sql)
        self.assertNotRegex(sql, r"\n\s+cardinality UInt64")

    def test_schema_has_dynamic_payload_columns_for_zero_and_multiple_counts(self):
        zero_sql = bench.recreate_schema_sql(0, 0)
        self.assertNotIn("b_p", zero_sql)
        self.assertNotIn("p_p", zero_sql)
        many_sql = bench.recreate_schema_sql(3, 2)
        for name in ("b_p0", "b_p1", "b_p2", "p_p0", "p_p1"):
            self.assertIn(f"{name} UInt64", many_sql)
        self.assertNotIn("b_p3", many_sql)
        self.assertNotIn("p_p2", many_sql)
        self.assertEqual(
            bench.build_columns(3)[-3:],
            (("b_p0", "UInt64"), ("b_p1", "UInt64"), ("b_p2", "UInt64")),
        )
        self.assertEqual(
            bench.probe_columns(2)[-2:],
            (("p_p0", "UInt64"), ("p_p1", "UInt64")),
        )

    def test_build_insert_generates_bucketed_rows_with_nesting_keys(self):
        sql = bench.build_insert_sql(self.metadata)
        self.assertIn("FROM numbers(64)", sql)
        self.assertIn("number % 32", sql)
        self.assertIn("intDiv(number, 32)", sql)
        self.assertIn("intDiv(number % 32, 8)", sql)
        self.assertIn(
            f"intHash64(bitXor(selector, toUInt64({bench.KEY_SEED}))) AS k", sql
        )
        self.assertIn("AS shuffle_rank", sql)

    def test_build_key_is_cardinality_independent(self):
        small = bench.build_insert_sql(make_metadata(max_cardinality=16, bucket_width=8))
        large = bench.build_insert_sql(make_metadata(max_cardinality=32, bucket_width=8))
        key_expression = f"intHash64(bitXor(selector, toUInt64({bench.KEY_SEED}))) AS k"
        self.assertIn(key_expression, small)
        self.assertIn(key_expression, large)

    def test_build_insert_generates_zero_or_all_independently_mixed_payloads(self):
        zero_sql = bench.build_insert_sql(
            make_metadata(max_build_payload_columns=0)
        )
        self.assertNotIn("b_p", zero_sql)
        many_sql = bench.build_insert_sql(
            make_metadata(max_build_payload_columns=3)
        )
        payload_lines = [
            line.strip().rstrip(",")
            for line in many_sql.splitlines()
            if re.search(r"AS b_p\d+,?$", line.strip())
        ]
        self.assertEqual(len(payload_lines), 3)
        self.assertEqual(len(set(payload_lines)), 3)
        self.assertTrue(all("intHash64" in line for line in payload_lines))

    def test_probe_insert_uses_uint128_affine_and_disjoint_misses(self):
        sql = bench.probe_insert_sql(self.metadata)
        self.assertIn("FROM numbers(96)", sql)
        self.assertIn("intDiv(number, 32)", sql)
        self.assertIn("intDiv(number % 32, 8)", sql)
        self.assertIn("number % 8", sql)
        self.assertIn("toUInt128(card_bucket) * 8", sql)
        self.assertRegex(
            sql,
            r"toUInt128\(rank\)\s+\*\s+toUInt128\(\s*bitOr\(\s*intHash64\(",
        )
        self.assertIn("toUInt64(1)", sql)
        self.assertIn("% 8", sql)
        self.assertIn(
            f"bitOr(toUInt64({bench.MISS_DOMAIN_BIT}), global_row)", sql
        )
        self.assertIn(str(bench.KEY_SEED), sql)

    def test_probe_insert_generates_zero_or_all_independently_mixed_payloads(self):
        zero_sql = bench.probe_insert_sql(
            make_metadata(max_probe_payload_columns=0)
        )
        self.assertNotIn("p_p", zero_sql)
        many_sql = bench.probe_insert_sql(
            make_metadata(max_probe_payload_columns=3)
        )
        payload_lines = [
            line.strip().rstrip(",")
            for line in many_sql.splitlines()
            if re.search(r"AS p_p\d+,?$", line.strip())
        ]
        self.assertEqual(len(payload_lines), 3)
        self.assertEqual(len(set(payload_lines)), 3)
        self.assertTrue(all("intHash64" in line for line in payload_lines))
        self.assertTrue(all("global_row" in line for line in payload_lines))

    def test_inserts_allow_up_to_max_cycles_partitions_per_block(self):
        for sql in (
            bench.build_insert_sql(self.metadata),
            bench.probe_insert_sql(self.metadata),
        ):
            self.assertIn("max_partitions_per_insert_block", sql)

    def test_validation_sql_checks_counts_layout_and_active_parts(self):
        sql = bench.loaded_state_query()
        self.assertIn("system.columns", sql)
        self.assertIn("system.tables", sql)
        self.assertIn("system.parts", sql)
        self.assertIn("partition_key", sql)
        self.assertIn("partition_id", sql)
        self.assertIn("GROUP BY occurrence", sql)
        self.assertIn("GROUP BY cycle", sql)
        self.assertIn("FORMAT JSONEachRow", sql)

    def test_layout_precheck_does_not_read_malformed_data_tables(self):
        sql = bench.loaded_layout_query()
        self.assertIn("system.columns", sql)
        self.assertIn("system.tables", sql)
        self.assertIn("system.parts", sql)
        self.assertNotIn(f"FROM {bench.BUILD_TABLE}", sql)
        self.assertNotIn(f"FROM {bench.PROBE_TABLE}", sql)


class PartFingerprintTests(unittest.TestCase):
    def test_fingerprint_is_deterministic_and_order_independent(self):
        pairs = [("0", "a"), ("1", "b")]
        expected = hashlib.sha256(b"0\ta\n1\tb\n").hexdigest()
        self.assertEqual(bench.part_fingerprint(pairs), expected)
        self.assertEqual(bench.part_fingerprint(list(reversed(pairs))), expected)
        self.assertNotEqual(
            bench.part_fingerprint([("0", "a"), ("1", "c")]), expected
        )

    def test_parts_query_reads_partition_ids_and_file_hashes(self):
        sql = bench.parts_query()
        self.assertIn("system.parts", sql)
        self.assertIn("partition_id", sql)
        self.assertIn("hash_of_all_files", sql)
        self.assertIn("active", sql)
        self.assertIn("FORMAT JSONEachRow", sql)

    def test_collect_part_fingerprints_requires_one_part_per_partition(self):
        metadata = make_metadata()
        rows = [
            {"table": bench.BUILD_TABLE, "partition_id": partition, "part_hash": part_hash}
            for partition, part_hash in build_part_pairs(metadata)
        ] + [
            {"table": bench.PROBE_TABLE, "partition_id": partition, "part_hash": part_hash}
            for partition, part_hash in probe_part_pairs(metadata)
        ]
        build_fp, probe_fp = bench.collect_part_fingerprints(rows, metadata)
        self.assertEqual(build_fp, bench.part_fingerprint(build_part_pairs(metadata)))
        self.assertEqual(probe_fp, bench.part_fingerprint(probe_part_pairs(metadata)))

        with self.assertRaisesRegex(ValueError, "part"):
            bench.collect_part_fingerprints(
                rows
                + [{"table": bench.BUILD_TABLE, "partition_id": "0", "part_hash": "dup"}],
                metadata,
            )
        with self.assertRaisesRegex(ValueError, "partition"):
            bench.collect_part_fingerprints(rows[1:], metadata)
        with self.assertRaisesRegex(ValueError, "partition"):
            bench.collect_part_fingerprints(
                rows
                + [{"table": bench.PROBE_TABLE, "partition_id": "9", "part_hash": "x"}],
                metadata,
            )


class MetadataTests(unittest.TestCase):
    def test_schema_version_four_with_bucketed_metadata_columns(self):
        self.assertEqual(bench.SCHEMA_VERSION, 4)
        self.assertEqual(bench.GENERATOR_SIGNATURE, "join-mergetree-generator-v4")
        for column in (
            ("schema_version", "UInt64"),
            ("max_cardinality", "UInt64"),
            ("bucket_width", "UInt64"),
            ("max_multiplicity", "UInt64"),
            ("max_cycles", "UInt64"),
            ("max_build_payload_columns", "UInt64"),
            ("max_probe_payload_columns", "UInt64"),
            ("generator_signature", "String"),
            ("build_part_fingerprint", "String"),
            ("probe_part_fingerprint", "String"),
        ):
            self.assertIn(column, bench.METADATA_COLUMNS)

    def test_metadata_insert_embeds_python_fingerprints(self):
        metadata = metadata_with_fingerprints(make_metadata())
        sql = bench.metadata_insert_sql(metadata)
        self.assertIn(f"toUInt64({bench.SCHEMA_VERSION})", sql)
        self.assertIn("toUInt64(32)", sql)
        self.assertIn("toUInt64(8)", sql)
        self.assertIn(bench.GENERATOR_SIGNATURE, sql)
        self.assertIn(metadata.build_part_fingerprint, sql)
        self.assertIn(metadata.probe_part_fingerprint, sql)
        self.assertNotIn("system.parts", sql)
        for field, _ in bench.METADATA_COLUMNS:
            self.assertIn(field, bench.metadata_query())

    def test_metadata_identity_includes_all_bucketed_dimensions(self):
        current = make_metadata()
        for change in (
            {"max_cardinality": 64},
            {"bucket_width": 16},
            {"max_multiplicity": 1},
            {"max_cycles": 2},
            {"max_build_payload_columns": 2},
            {"max_probe_payload_columns": 0},
            {"generator_signature": "stale-generator"},
        ):
            with self.subTest(change=change):
                self.assertFalse(
                    bench._metadata_matches(
                        current, dataclasses.replace(current, **change)
                    )
                )
        self.assertTrue(
            bench._metadata_matches(
                current,
                dataclasses.replace(current, build_part_fingerprint="different"),
            )
        )

    def test_read_metadata_round_trips_v4_fields(self):
        expected = metadata_with_fingerprints(make_metadata())
        layout_rows = make_layout_rows(expected)
        metadata_rows = [
            {
                "schema_version": str(expected.schema_version),
                "max_cardinality": str(expected.max_cardinality),
                "bucket_width": str(expected.bucket_width),
                "max_multiplicity": str(expected.max_multiplicity),
                "max_cycles": str(expected.max_cycles),
                "max_build_payload_columns": str(expected.max_build_payload_columns),
                "max_probe_payload_columns": str(expected.max_probe_payload_columns),
                "generator_signature": expected.generator_signature,
                "build_part_fingerprint": expected.build_part_fingerprint,
                "probe_part_fingerprint": expected.probe_part_fingerprint,
            }
        ]

        def query(_binary, _path, sql):
            if f"FROM {bench.METADATA_TABLE}" in sql:
                return metadata_rows
            return layout_rows

        with mock.patch.object(
            bench,
            "_table_names",
            return_value={bench.BUILD_TABLE, bench.PROBE_TABLE, bench.METADATA_TABLE},
        ), mock.patch.object(bench, "_query_json", side_effect=query):
            self.assertEqual(bench.read_metadata("binary", "path"), expected)

    def test_old_v3_layout_is_rejected_for_rebuild(self):
        old_rows = [
            {
                "kind": "table",
                "key": bench.BUILD_TABLE,
                "value": "MergeTree|cardinality, occurrence, shuffle_rank",
            },
            {
                "kind": "column",
                "key": f"{bench.BUILD_TABLE}.1",
                "value": "cardinality|UInt64",
            },
        ]
        with mock.patch.object(
            bench,
            "_table_names",
            return_value={bench.BUILD_TABLE, bench.PROBE_TABLE, bench.METADATA_TABLE},
        ), mock.patch.object(bench, "_query_json", return_value=old_rows):
            self.assertIsNone(bench.read_metadata("binary", "path"))


class StateValidationTests(unittest.TestCase):
    def setUp(self):
        self.metadata = metadata_with_fingerprints(make_metadata())

    def test_valid_bucketed_state_passes(self):
        rows = make_state_rows(self.metadata)
        self.assertEqual(bench.validate_loaded_state(rows, self.metadata), [])

    def test_state_rejects_multiple_parts_in_one_partition(self):
        rows = make_state_rows(self.metadata)
        for row in rows:
            if row["kind"] == "part" and row["key"] == f"{bench.BUILD_TABLE}.0":
                row["value"] = "2|build-hash-0"
        errors = bench.validate_loaded_state(rows, self.metadata)
        self.assertTrue(
            any("one active part" in error for error in errors), errors
        )

    def test_state_rejects_missing_partition(self):
        rows = [
            row
            for row in make_state_rows(self.metadata)
            if not (
                row["kind"] == "part" and row["key"] == f"{bench.PROBE_TABLE}.2"
            )
        ]
        errors = bench.validate_loaded_state(rows, self.metadata)
        self.assertTrue(
            any("one active part" in error for error in errors), errors
        )

    def test_state_rejects_fingerprint_mismatch(self):
        rows = make_state_rows(self.metadata)
        for row in rows:
            if row["kind"] == "part" and row["key"] == f"{bench.BUILD_TABLE}.1":
                row["value"] = "1|mutated"
        errors = bench.validate_loaded_state(rows, self.metadata)
        self.assertTrue(
            any("fingerprint" in error for error in errors), errors
        )

    def test_state_rejects_empty_metadata_fingerprints(self):
        metadata = dataclasses.replace(
            self.metadata, build_part_fingerprint="", probe_part_fingerprint=""
        )
        rows = make_state_rows(metadata)
        errors = bench.validate_loaded_state(rows, metadata)
        self.assertTrue(
            any("fingerprint" in error for error in errors), errors
        )

    def test_state_rejects_per_partition_count_mismatch(self):
        rows = make_state_rows(self.metadata)
        for row in rows:
            if row["kind"] == "probe_count" and row["key"] == "1":
                row["value"] = "31"
        errors = bench.validate_loaded_state(rows, self.metadata)
        self.assertTrue(
            any("count" in error for error in errors), errors
        )


class BenchmarkSqlTests(unittest.TestCase):
    def setUp(self):
        self.point = make_point()

    def test_probe_selection_decomposes_probe_rows(self):
        cases = {
            32: (2, 0, 0, 0, 2),
            36: (2, 4, 0, 4, 2),
            40: (2, 8, 1, 0, 2),
            44: (2, 12, 1, 4, 2),
            12: (0, 12, 1, 4, 2),
            48: (3, 0, 0, 0, 2),
        }
        for probe_rows, expected in cases.items():
            point = make_point(probe_rows=probe_rows)
            with self.subTest(probe_rows=probe_rows):
                self.assertEqual(bench._probe_selection(point), expected)

    def test_probe_predicate_covers_all_disjunct_shapes(self):
        cases = {
            32: "(cycle < 2 AND card_bucket < 2)",
            36: (
                "(cycle < 2 AND card_bucket < 2) OR "
                "(cycle = 2 AND (card_bucket = 0 AND rank < 4))"
            ),
            40: "(cycle < 2 AND card_bucket < 2) OR (cycle = 2 AND card_bucket < 1)",
            44: (
                "(cycle < 2 AND card_bucket < 2) OR "
                "(cycle = 2 AND (card_bucket < 1 OR (card_bucket = 1 AND rank < 4)))"
            ),
            12: "(cycle = 0 AND (card_bucket < 1 OR (card_bucket = 1 AND rank < 4)))",
            48: "(cycle < 3 AND card_bucket < 2)",
        }
        for probe_rows, expected in cases.items():
            point = make_point(probe_rows=probe_rows)
            with self.subTest(probe_rows=probe_rows):
                self.assertEqual(bench._probe_predicate(point), expected)

    def test_payload_projection_covers_probe_build_both_and_neither(self):
        cases = (
            (0, 2, "SELECT p.p_p0, p.p_p1 ", ("b_p",)),
            (2, 0, "SELECT b.b_p0, b.b_p1 ", ("p_p",)),
            (2, 2, "SELECT p.p_p0, p.p_p1, b.b_p0, b.b_p1 ", ()),
            (0, 0, "SELECT toUInt8(0) AS matched ", ("b_p", "p_p")),
        )
        for build_count, probe_count, prefix, absent in cases:
            point = make_point(
                build_payload_columns=build_count,
                probe_payload_columns=probe_count,
            )
            sql = bench.join_query(
                point, "radix_join", threads=6, max_memory=123456, output_format="Null"
            )
            with self.subTest(build_count=build_count, probe_count=probe_count):
                self.assertTrue(sql.startswith(prefix), sql)
                for name in absent:
                    self.assertNotIn(name, sql)

    def test_final_projection_excludes_key_but_join_subqueries_keep_it(self):
        point = make_point(build_payload_columns=1, probe_payload_columns=1)
        sql = bench.join_query(
            point, "radix_join", threads=6, max_memory=123456, output_format="Null"
        )
        final_select = sql.split(" FROM (", 1)[0]
        self.assertNotIn("p.k", final_select)
        self.assertNotIn("b.k", final_select)
        self.assertNotRegex(final_select, r"\bk\b")
        self.assertGreaterEqual(sql.count(" AS k"), 1)
        self.assertIn("SELECT k, b_p0", sql)
        self.assertIn("USING (k)", sql)

    def test_timing_and_verification_use_same_projection(self):
        point = make_point(build_payload_columns=2, probe_payload_columns=1)
        timed = bench.join_query(
            point, "radix_join", threads=6, max_memory=123456, output_format="Null"
        )
        verified = bench.verification_query(point, "radix_join", 6, 123456)
        self.assertEqual(timed.split(" FROM (", 1)[0], verified.split(" FROM (", 1)[0])
        self.assertIn("ORDER BY ALL", verified)
        self.assertTrue(verified.endswith("FORMAT Hash"))

    def test_query_never_names_unrequested_payload_columns(self):
        point = make_point(build_payload_columns=1, probe_payload_columns=1)
        sql = bench.join_query(
            point, "radix_join", threads=6, max_memory=123456, output_format="Null"
        )
        self.assertIn("p_p0", sql)
        self.assertIn("b_p0", sql)
        for name in ("p_p1", "p_p2", "b_p1", "b_p2"):
            self.assertNotIn(name, sql)

    def test_timed_query_uses_dense_bresenham_and_pins_settings(self):
        sql = bench.join_query(
            self.point, "radix_join", threads=6, max_memory=123456, output_format="Null"
        )
        self.assertEqual(sql.count(bench.PROBE_TABLE), 1)
        self.assertNotIn("UNION ALL", sql)
        self.assertIn("PREWHERE occurrence < 2 AND card_bucket < 2", sql)
        self.assertIn(
            "PREWHERE (cycle < 2 AND card_bucket < 2) OR "
            "(cycle = 2 AND (card_bucket = 0 AND rank < 4))",
            sql,
        )
        self.assertIn(
            "intDiv(toUInt128(cycle * 16 + card_bucket * 8 + rank + 1) * 13, 36)", sql
        )
        self.assertIn(
            "intDiv(toUInt128(cycle * 16 + card_bucket * 8 + rank) * 13, 36)", sql
        )
        self.assertNotIn("selector", sql.split(" FROM (", 1)[1].split("INNER JOIN")[0])
        for setting in (
            "join_algorithm = 'radix_join'",
            "max_threads = 6",
            "query_plan_join_swap_table = false",
            "enable_analyzer = 1",
            "enable_join_runtime_filters = 0",
            "max_bytes_before_external_join = 0",
            "max_bytes_ratio_before_external_join = 0",
            "max_memory_usage = 123456",
        ):
            self.assertIn(setting, sql)
        self.assertTrue(sql.endswith("FORMAT Null"))

    def test_verification_query_sorts_full_output_and_hashes(self):
        sql = bench.verification_query(self.point, "parallel_hash", 4, 999)
        self.assertIn("ORDER BY ALL", sql)
        self.assertTrue(sql.endswith("FORMAT Hash"))

    def test_assertion_query_contains_all_three_counts(self):
        sql = bench.assertion_query(self.point, 4, 999)
        self.assertIn(bench.ASSERT_MARKER, sql)
        self.assertIn("probe_count", sql)
        self.assertIn("build_count", sql)
        self.assertIn("joined_count", sql)
        self.assertIn("AS probe_count_source", sql)
        self.assertIn("AS build_count_source", sql)

    def test_measurement_script_contains_only_warmup_and_timed_joins(self):
        sql = bench.measurement_script(self.point, "radix_join", 4, 999, runs=2)
        self.assertEqual(sql.count("SELECT p.p_p0, b.b_p0"), 3)
        self.assertNotIn("system.events", sql)
        self.assertNotIn("CREATE TEMPORARY TABLE", sql)
        self.assertFalse(hasattr(bench, "EVENT_MARKER"))


class BucketSimulationTests(unittest.TestCase):
    MAX_CYCLES = 3
    TOTAL_BUCKETS = 4  # max_cardinality=32, bucket_width=8

    def test_selected_row_count_equals_probe_rows(self):
        for probe_rows in (12, 32, 36, 40, 44, 48):
            point = make_point(probe_rows=probe_rows)
            selected = simulated_selected_rows(
                point, self.MAX_CYCLES, self.TOTAL_BUCKETS
            )
            with self.subTest(probe_rows=probe_rows):
                self.assertEqual(len(selected), probe_rows)

    def test_dense_index_is_a_bijection_onto_probe_rows(self):
        for probe_rows in (12, 32, 36, 44, 48):
            point = make_point(probe_rows=probe_rows)
            selected = simulated_selected_rows(
                point, self.MAX_CYCLES, self.TOTAL_BUCKETS
            )
            dense = {
                cycle * point.cardinality + bucket * point.bucket_width + rank
                for cycle, bucket, rank in selected
            }
            with self.subTest(probe_rows=probe_rows):
                self.assertEqual(dense, set(range(probe_rows)))

    def test_bresenham_hits_are_exact_for_any_hit_count(self):
        for probe_rows in (12, 36, 48):
            for hit_rows in (0, 1, probe_rows // 3, probe_rows // 2, probe_rows):
                hits = sum(
                    (dense + 1) * hit_rows // probe_rows
                    > dense * hit_rows // probe_rows
                    for dense in range(probe_rows)
                )
                with self.subTest(probe_rows=probe_rows, hit_rows=hit_rows):
                    self.assertEqual(hits, hit_rows)

    def test_each_selected_block_covers_its_bucket_exactly_once(self):
        point = make_point(probe_rows=44)
        selected = simulated_selected_rows(point, self.MAX_CYCLES, self.TOTAL_BUCKETS)
        width = point.bucket_width
        blocks = {}
        for cycle, bucket, rank in selected:
            multiplier = (
                bench.int_hash64(bench.mix_seed(bench.AFFINE_SEED, cycle, bucket)) | 1
            )
            offset = bench.int_hash64(
                bench.mix_seed(bench.PROBE_CYCLE_SEED, cycle, bucket)
            ) % width
            blocks.setdefault((cycle, bucket), []).append(
                bucket * width + (rank * multiplier + offset) % width
            )
        for (cycle, bucket), selectors in blocks.items():
            if len(selectors) < width:
                continue  # partial remainder block covers a strict prefix
            with self.subTest(cycle=cycle, bucket=bucket):
                self.assertEqual(
                    set(selectors),
                    set(range(bucket * width, (bucket + 1) * width)),
                )
        full_blocks = [key for key, value in blocks.items() if len(value) == width]
        self.assertGreater(len(full_blocks), 1)

    def test_small_valid_points_integrate_selection_dither_and_affine_permutation(self):
        metadata = make_metadata()
        cases = (
            (1, 1, "0.5", "0.5"),
            (2, 2, "1.5", "0.5"),
            (3, 1, "1.125", "0.3"),
            (4, 1, "2.25", "0.75"),
        )
        for bucket_count, multiplicity, ratio, hit_rate in cases:
            point = bench.validate_points(
                metadata,
                [bucket_count * metadata.bucket_width],
                [multiplicity],
                [decimal.Decimal(ratio)],
                [decimal.Decimal(hit_rate)],
            )[0]
            rows = simulated_selected_rows(
                point, metadata.max_cycles, metadata.max_cardinality // metadata.bucket_width
            )
            dense = [
                cycle * point.cardinality
                + bucket * point.bucket_width
                + rank
                for cycle, bucket, rank in rows
            ]
            hits = sum(
                (value + 1) * point.hit_rows // point.probe_rows
                > value * point.hit_rows // point.probe_rows
                for value in dense
            )
            with self.subTest(
                c=bucket_count, m=multiplicity, r=ratio, h=hit_rate
            ):
                self.assertEqual(len(rows), point.probe_rows)
                self.assertEqual(dense, list(range(point.probe_rows)))
                self.assertEqual(hits, point.hit_rows)
                for cycle in range(point.probe_rows // point.cardinality):
                    selectors = {
                        bucket * point.bucket_width
                        + (
                            rank
                            * (
                                bench.int_hash64(
                                    bench.mix_seed(
                                        bench.AFFINE_SEED, cycle, bucket
                                    )
                                )
                                | 1
                            )
                            + bench.int_hash64(
                                bench.mix_seed(
                                    bench.PROBE_CYCLE_SEED, cycle, bucket
                                )
                            )
                            % point.bucket_width
                        )
                        % point.bucket_width
                        for selected_cycle, bucket, rank in rows
                        if selected_cycle == cycle
                    }
                    self.assertEqual(selectors, set(range(point.cardinality)))


class VerificationTests(unittest.TestCase):
    def setUp(self):
        self.point = make_point()

    def test_verify_point_passes_when_hashes_match(self):
        with mock.patch.object(
            bench, "_execute_bytes", return_value=(b"same", None)
        ):
            status, detail, errors = bench._verify_point(
                "binary", "path", self.point, 4, 999, False, 10**9
            )
        self.assertEqual(status, "PASS")
        self.assertEqual(errors, {})

    def test_verify_point_reports_hash_mismatch(self):
        with mock.patch.object(
            bench, "_execute_bytes", side_effect=[(b"one", None), (b"two", None)]
        ):
            status, detail, errors = bench._verify_point(
                "binary", "path", self.point, 4, 999, False, 10**9
            )
        self.assertEqual(status, "FAIL")
        self.assertEqual(errors, {})

    def test_verify_point_isolates_per_algorithm_errors(self):
        def execute(_binary, _path, sql, *, purpose):
            if "radix_join" in sql:
                return None, f"{purpose} failed: NOT_IMPLEMENTED"
            return b"hash", None

        with mock.patch.object(bench, "_execute_bytes", side_effect=execute):
            status, detail, errors = bench._verify_point(
                "binary", "path", self.point, 4, 999, False, 10**9
            )
        self.assertEqual(status, "ERROR")
        self.assertIn("radix_join", detail)
        self.assertEqual(set(errors), {"radix_join"})

    def test_verify_point_skip_paths_report_no_errors(self):
        status, detail, errors = bench._verify_point(
            "binary", "path", self.point, 4, 999, True, 10**9
        )
        self.assertEqual((status, errors), ("SKIP", {}))
        status, detail, errors = bench._verify_point(
            "binary", "path", self.point, 4, 999, False, 1
        )
        self.assertEqual(status, "SKIP")
        self.assertIn("exceed", detail)

    def test_result_table_marks_successful_algorithm_verification_pass(self):
        output = io.StringIO()
        results = [
            bench.AlgorithmResult("radix_join", "ERROR", detail="NOT_IMPLEMENTED"),
            bench.AlgorithmResult("parallel_hash", "OK"),
        ]
        with contextlib.redirect_stdout(output):
            bench._print_point_results(
                self.point,
                ("ERROR", "radix_join failed"),
                results,
                {"radix_join": "NOT_IMPLEMENTED"},
            )
        self.assertRegex(output.getvalue(), r"radix_join\s+ERROR\s+ERROR")
        self.assertRegex(output.getvalue(), r"parallel_hash\s+OK\s+PASS")


class CommandInterfaceTests(unittest.TestCase):
    def test_matching_valid_load_is_noop_even_when_free_space_is_low(self):
        parser = bench.build_parser()
        args = parser.parse_args(
            [
                "load",
                "--binary",
                sys.executable,
                "--path",
                ".",
                "--max-cardinality",
                "32",
                "--bucket-width",
                "8",
                "--max-multiplicity",
                "2",
                "--max-cycles",
                "3",
            ]
        )
        metadata = make_metadata()
        output = io.StringIO()
        with mock.patch.object(
            bench, "read_metadata", return_value=metadata
        ), mock.patch.object(
            bench, "inspect_loaded_data", return_value=[]
        ), mock.patch.object(
            bench, "_filesystem_free_bytes", return_value=0
        ), contextlib.redirect_stdout(output):
            self.assertEqual(bench.load_command(args), 0)
        self.assertIn("READY", output.getvalue())

    def test_load_cli_exposes_bucketed_dataset_options(self):
        parser = bench.build_parser()
        load = parser.parse_args(
            ["load", "--max-cardinality", "32", "--bucket-width", "8"]
        )
        self.assertEqual(load.max_cardinality, 32)
        self.assertEqual(load.bucket_width, 8)
        self.assertEqual(load.max_multiplicity, 1)
        self.assertEqual(load.max_cycles, 1)
        self.assertEqual(load.max_build_payload_columns, 1)
        self.assertEqual(load.max_probe_payload_columns, 1)
        output = io.StringIO()
        with contextlib.redirect_stdout(output), self.assertRaises(SystemExit):
            parser.parse_args(["load", "--help"])
        for option in (
            "--max-cardinality",
            "--bucket-width",
            "--max-multiplicity",
            "--max-cycles",
            "--max-build-payload-columns",
            "--max-probe-payload-columns",
        ):
            self.assertIn(option, output.getvalue())
        for option in ("--cardinalities", "--max-ratio"):
            self.assertNotIn(option, output.getvalue())

    def test_run_cli_defaults_preserve_one_payload_column_per_side(self):
        parser = bench.build_parser()
        run = parser.parse_args(["run", "--multiplicities", "1", "--ratios", "1"])
        self.assertEqual(run.build_payload_columns, "1")
        self.assertEqual(run.probe_payload_columns, "1")
        output = io.StringIO()
        with contextlib.redirect_stdout(output), self.assertRaises(SystemExit):
            parser.parse_args(["run", "--help"])
        for option in (
            "--cardinalities",
            "--build-payload-columns",
            "--probe-payload-columns",
        ):
            self.assertIn(option, output.getvalue())

    def test_run_rejects_malformed_payload_lists_before_metadata_queries(self):
        parser = bench.build_parser()
        for option in ("--build-payload-columns", "--probe-payload-columns"):
            for value in ("-1", "1.0", "1,1"):
                args = parser.parse_args(
                    [
                        "run",
                        "--binary",
                        sys.executable,
                        "--multiplicities",
                        "1",
                        "--ratios",
                        "1",
                        option,
                        value,
                    ]
                )
                with self.subTest(option=option, value=value), mock.patch.object(
                    bench, "read_metadata"
                ) as read_metadata, contextlib.redirect_stderr(io.StringIO()):
                    self.assertEqual(bench.run_command(args), 2)
                    read_metadata.assert_not_called()

    def test_run_rejects_payload_bounds_before_state_validation(self):
        metadata = make_metadata(
            max_build_payload_columns=2, max_probe_payload_columns=1
        )
        args = bench.build_parser().parse_args(
            [
                "run",
                "--binary",
                sys.executable,
                "--multiplicities",
                "1",
                "--ratios",
                "1",
                "--build-payload-columns",
                "3",
            ]
        )
        with mock.patch.object(bench, "read_metadata", return_value=metadata), mock.patch.object(
            bench, "inspect_loaded_data"
        ) as inspect, contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(bench.run_command(args), 2)
            inspect.assert_not_called()

    def test_run_rejects_non_bucket_multiple_cardinality_before_state_validation(self):
        metadata = make_metadata()
        args = bench.build_parser().parse_args(
            [
                "run",
                "--binary",
                sys.executable,
                "--multiplicities",
                "1",
                "--ratios",
                "1",
                "--cardinalities",
                "12",
            ]
        )
        errors = io.StringIO()
        with mock.patch.object(bench, "read_metadata", return_value=metadata), mock.patch.object(
            bench, "inspect_loaded_data"
        ) as inspect, contextlib.redirect_stderr(errors):
            self.assertEqual(bench.run_command(args), 2)
            inspect.assert_not_called()
        self.assertIn("bucket width 8", errors.getvalue())
        self.assertIn("32", errors.getvalue())

    def test_run_rejects_cycle_overflow_before_state_validation(self):
        metadata = make_metadata()
        args = bench.build_parser().parse_args(
            [
                "run",
                "--binary",
                sys.executable,
                "--multiplicities",
                "2",
                "--ratios",
                "2",
            ]
        )
        errors = io.StringIO()
        with mock.patch.object(bench, "read_metadata", return_value=metadata), mock.patch.object(
            bench, "inspect_loaded_data"
        ) as inspect, contextlib.redirect_stderr(errors):
            self.assertEqual(bench.run_command(args), 2)
            inspect.assert_not_called()
        self.assertIn("max cycles", errors.getvalue())


class OutputParsingTests(unittest.TestCase):
    def test_parse_assertion_output(self):
        output = f"{bench.ASSERT_MARKER}\t15\t20\t16\n".encode()
        self.assertEqual(bench.parse_assertion_output(output), (15, 20, 16))

    def test_parse_assertion_output_rejects_missing_or_malformed_marker(self):
        for output in (b"", b"noise\n", f"{bench.ASSERT_MARKER}\tx\t2\t3\n".encode()):
            with self.subTest(output=output), self.assertRaises(ValueError):
                bench.parse_assertion_output(output)

    def test_parse_profile_events_groups_packets_and_fills_omitted_zeroes(self):
        stderr = "\n".join(
            [
                "noise before packets",
                "2026.01.01 [ 0 ] RealTimeMicroseconds: 1000 (increment)",
                "2026.01.01 [ 0 ] Query: 1 (increment)",
                "2026.01.01 [ 0 ] SelectedRows: 42 (increment)",
                "unrelated diagnostic",
                "0.0012345",
                "2026.01.01 [ 0 ] RealTimeMicroseconds: 800 (increment)",
                "2026.01.01 [ 0 ] Query: 1 (increment)",
                "0.000800",
            ]
        )
        packets = bench.parse_profile_events(stderr, expected_packets=2)
        self.assertEqual(packets[0]["SelectedRows"], 42)
        self.assertEqual(packets[1]["SelectedRows"], 0)
        self.assertEqual(packets[1]["RadixHashJoinBuildMicroseconds"], 0)
        self.assertEqual(packets[0]["WallTimeMicroseconds"], 1235)
        self.assertEqual(packets[1]["WallTimeMicroseconds"], 800)

    def test_parse_profile_events_rejects_malformed_tracked_data(self):
        cases = [
            (
                "2026 [ 0 ] Query: 1 (increment)\n"
                "2026 [ 0 ] SelectedRows: 1 (increment)\n"
                "2026 [ 0 ] SelectedRows: 2 (increment)\n"
                "2026 [ 0 ] RealTimeMicroseconds: 3 (increment)\n"
                "0.001"
            ),
            (
                "2026 [ 0 ] Query: 1 (increment)\n"
                "2026 [ 0 ] SelectedRows: 0 (increment)\n"
                "2026 [ 0 ] SelectedRows: 0 (increment)\n"
                "2026 [ 0 ] RealTimeMicroseconds: 3 (increment)\n"
                "0.001"
            ),
            (
                "2026 [ 0 ] Query: 2 (increment)\n"
                "2026 [ 0 ] RealTimeMicroseconds: 3 (increment)\n"
                "0.001"
            ),
            (
                "2026 [ 0 ] Query: 1 (increment)\n"
                "2026 [ 0 ] SelectedRows: nope (increment)\n"
                "2026 [ 0 ] RealTimeMicroseconds: 3 (increment)\n"
                "0.001"
            ),
            (
                "2026 [ 0 ] SelectedRows: 1 (increment)\n"
                "2026 [ 0 ] RealTimeMicroseconds: 3 (increment)\n"
                "0.001"
            ),
        ]
        for stderr in cases:
            with self.subTest(stderr=stderr), self.assertRaises(ValueError):
                bench.parse_profile_events(stderr, expected_packets=1)

    def test_parse_profile_events_requires_exact_packet_count(self):
        stderr = (
            "2026 [ 0 ] Query: 1 (increment)\n"
            "2026 [ 0 ] RealTimeMicroseconds: 3 (increment)\n"
            "0.001"
        )
        with self.assertRaisesRegex(ValueError, "expected 2.*got 1"):
            bench.parse_profile_events(stderr, expected_packets=2)

    def test_parse_profile_events_rejects_missing_malformed_and_extra_times(self):
        packet = (
            "2026 [ 0 ] Query: 1 (increment)\n"
            "2026 [ 0 ] RealTimeMicroseconds: 3 (increment)"
        )
        cases = [
            packet,
            packet + "\nnot-a-time",
            packet + "\n0.001\n0.002",
        ]
        for stderr in cases:
            with self.subTest(stderr=stderr), self.assertRaises(ValueError):
                bench.parse_profile_events(stderr, expected_packets=1)

    def test_timed_profile_parser_discards_warmup_group(self):
        def group(rows, seconds):
            return (
                "2026 [ 0 ] Query: 1 (increment)\n"
                f"2026 [ 0 ] SelectedRows: {rows} (increment)\n"
                f"{seconds}"
            )

        stderr = "\n".join(
            [group(100, "0.010"), group(20, "0.002"), group(30, "0.003")]
        )
        runs = bench.parse_timed_profile_events(stderr, runs=2)
        self.assertEqual([run["SelectedRows"] for run in runs], [20, 30])
        self.assertEqual(
            [run["WallTimeMicroseconds"] for run in runs],
            [2000, 3000],
        )

    def test_measurements_use_median_time_and_representative_run(self):
        runs = []
        for elapsed, build in ((3000, 30), (1000, 10), (2000, 20)):
            row = dict.fromkeys(bench.EVENTS, 0)
            row["WallTimeMicroseconds"] = elapsed
            row["RealTimeMicroseconds"] = 1_000_000 - elapsed
            row["RadixHashJoinBuildMicroseconds"] = build
            runs.append(row)
        result = bench.summarize_measurements(runs)
        self.assertEqual(result.median_us, 2000)
        self.assertEqual(result.min_us, 1000)
        self.assertEqual(result.events["RadixHashJoinBuildMicroseconds"], 20)

    def test_fallback_detection_uses_leaf_group_execution_evidence(self):
        radix = []
        parallel = []
        for _ in range(2):
            r = dict.fromkeys(bench.EVENTS, 0)
            r["RadixHashJoinLeafGroupBuilds"] = 1
            radix.append(r)
            parallel.append(dict.fromkeys(bench.EVENTS, 0))
        self.assertIsNone(bench.fallback_reason("radix_join", radix))
        self.assertIsNone(bench.fallback_reason("parallel_hash", parallel))
        radix[1]["RadixHashJoinLeafGroupBuilds"] = 0
        self.assertIn("run 2", bench.fallback_reason("radix_join", radix))
        parallel[1]["RadixHashJoinLeafGroupBuilds"] = 1
        self.assertIn("run 2", bench.fallback_reason("parallel_hash", parallel))

    def test_error_point_with_no_algorithm_rows_prints_cleanly(self):
        point = make_point()
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            winner, speedup = bench._print_point_results(
                point, ("ERROR", "query failed"), []
            )
        self.assertIsNone(winner)
        self.assertIsNone(speedup)
        self.assertIn("verify", output.getvalue())
        self.assertIn("Winner: excluded", output.getvalue())


@unittest.skipUnless(
    os.path.isfile(bench.DEFAULT_BINARY) and os.access(bench.DEFAULT_BINARY, os.X_OK),
    "real build/reldeb clickhouse binary is unavailable",
)
class ProfileEventsIntegrationTests(unittest.TestCase):
    def test_multiquery_packets_keep_query_scoped_selected_rows(self):
        self.assertTrue(hasattr(bench, "parse_profile_events"))
        sql = (
            "SELECT number FROM numbers(3) FORMAT Null;\n"
            "SELECT sum(number) FROM numbers(10000000) FORMAT Null;\n"
        )
        returncode, stdout, stderr = bench._run_local(
            bench.DEFAULT_BINARY,
            "build/reldeb/join_mergetree_profile_test_data",
            sql,
            profile_events=True,
        )
        self.assertEqual(returncode, 0, stderr)
        self.assertEqual(stdout, b"")
        packets = bench.parse_profile_events(stderr, expected_packets=2)
        self.assertEqual(len(packets), 2)
        self.assertEqual(
            [packet["SelectedRows"] for packet in packets],
            [3, 10000000],
        )
        self.assertTrue(
            all(packet["RealTimeMicroseconds"] > 0 for packet in packets)
        )
        self.assertTrue(
            all(packet["WallTimeMicroseconds"] >= 0 for packet in packets)
        )

    def test_sql_seed_mixing_matches_python_simulation(self):
        pairs = ((0, 0), (1, 2), (63, 511))
        expressions = []
        expected = []
        for cycle, bucket in pairs:
            mixed_sql = bench.mix_seed_sql(
                bench.AFFINE_SEED, f"toUInt64({cycle})", f"toUInt64({bucket})"
            )
            expressions.append(f"toString(intHash64({mixed_sql}))")
            expected.append(
                str(bench.int_hash64(bench.mix_seed(bench.AFFINE_SEED, cycle, bucket)))
            )
        sql = "SELECT " + ", ".join(expressions) + " FORMAT TSV"
        returncode, stdout, stderr = bench._run_local(
            bench.DEFAULT_BINARY,
            "build/reldeb/join_mergetree_parity_test_data",
            sql,
        )
        self.assertEqual(returncode, 0, stderr)
        self.assertEqual(stdout.decode("utf-8").strip().split("\t"), expected)


class ProfileEventsCommandTests(unittest.TestCase):
    def test_profile_command_requests_final_totals_and_wall_time(self):
        completed = SimpleNamespace(returncode=0, stdout=b"", stderr=b"")
        with mock.patch.object(bench.subprocess, "run", return_value=completed) as run:
            bench._run_local("clickhouse", "data", "SELECT 1", profile_events=True)
        command = run.call_args.args[0]
        self.assertIn("--time", command)
        self.assertIn("--profile-events-delay-ms=-1", command)
        self.assertNotIn("--profile-events-delay-ms=0", command)


if __name__ == "__main__":
    unittest.main()
