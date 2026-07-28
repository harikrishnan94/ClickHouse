#!/usr/bin/env python3
"""Parallel range-read the largest data files to finish EBS snapshot hydration.

The sequential warm-read pass in jbmt_prep_phj_ph.sh degrades to ~14 MB/s on its
tail, because by then a single `cat` stream is left and a snapshot-backed volume
serves each first-touch block with an S3 round trip that only concurrency hides
(the same pass ran at 473 MB/s while 64 streams were in flight). This reads the
big files as many concurrent 64 MiB ranges instead, so the tail hydrates at
device speed rather than at one stream's latency.

Additive and idempotent: it only reads, so it can run alongside the warm-read
pass, and re-reading an already-hydrated block is a local read.
"""
import concurrent.futures
import os
import pathlib
import sys

CHUNK = 64 << 20
ROOT = sys.argv[1] if len(sys.argv) > 1 else "/mnt/data/jbmt_server/data"
MIN_SIZE = 256 << 20
WORKERS = 64


def ranges():
    files = []
    for dirpath, _dirnames, filenames in os.walk(ROOT):
        for name in filenames:
            p = pathlib.Path(dirpath) / name
            try:
                size = p.stat().st_size
            except OSError:
                continue
            if size >= MIN_SIZE:
                files.append((size, p))
    files.sort(reverse=True)
    total = sum(s for s, _ in files)
    print(f"{len(files)} files >= {MIN_SIZE >> 20} MiB, {total / 1e9:.1f} GB", flush=True)
    for size, p in files:
        for off in range(0, size, CHUNK):
            yield p, off


def read_range(job):
    path, off = job
    try:
        with open(path, "rb", buffering=0) as fh:
            fh.seek(off)
            got = 0
            while got < CHUNK:
                b = fh.read(min(1 << 22, CHUNK - got))
                if not b:
                    break
                got += len(b)
        return got
    except OSError:
        return 0


def main():
    done = 0
    with concurrent.futures.ThreadPoolExecutor(WORKERS) as ex:
        for n in ex.map(read_range, ranges(), chunksize=1):
            done += n
    print(f"hydrated {done / 1e9:.1f} GB of large-file ranges", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
