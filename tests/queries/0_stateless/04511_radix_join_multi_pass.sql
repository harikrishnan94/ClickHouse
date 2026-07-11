-- Tags: no-random-settings
-- `max_threads = 4` requires at least four final leaf partitions, while the per-pass fanout cap of
-- two forces a 1+1-bit plan and exercises the refine scatter for both simple and composite keys.
SET enable_analyzer = 1;
SET max_threads = 4;
SET radix_join_max_partitions_per_pass = 2;

SELECT
    'single_u64',
    (
        SELECT (count(), sum(cityHash64(p.probe_payload, b.build_payload)))
        FROM
            (SELECT number AS probe_payload, number % 150 AS key FROM numbers(400)) AS p
        INNER JOIN
            (SELECT number AS build_payload, number % 100 AS key FROM numbers(600)) AS b
        ON p.key = b.key
        SETTINGS join_algorithm = 'radix_join'
    )
    =
    (
        SELECT (count(), sum(cityHash64(p.probe_payload, b.build_payload)))
        FROM
            (SELECT number AS probe_payload, number % 150 AS key FROM numbers(400)) AS p
        INNER JOIN
            (SELECT number AS build_payload, number % 100 AS key FROM numbers(600)) AS b
        ON p.key = b.key
        SETTINGS join_algorithm = 'hash'
    );

SELECT
    'composite_u64',
    (
        SELECT (count(), sum(cityHash64(p.probe_payload, b.build_payload)))
        FROM
            (SELECT number AS probe_payload, number % 150 AS key1, number % 17 AS key2 FROM numbers(400)) AS p
        INNER JOIN
            (SELECT number AS build_payload, number % 100 AS key1, number % 17 AS key2 FROM numbers(600)) AS b
        ON p.key1 = b.key1 AND p.key2 = b.key2
        SETTINGS join_algorithm = 'radix_join'
    )
    =
    (
        SELECT (count(), sum(cityHash64(p.probe_payload, b.build_payload)))
        FROM
            (SELECT number AS probe_payload, number % 150 AS key1, number % 17 AS key2 FROM numbers(400)) AS p
        INNER JOIN
            (SELECT number AS build_payload, number % 100 AS key1, number % 17 AS key2 FROM numbers(600)) AS b
        ON p.key1 = b.key1 AND p.key2 = b.key2
        SETTINGS join_algorithm = 'hash'
    );
