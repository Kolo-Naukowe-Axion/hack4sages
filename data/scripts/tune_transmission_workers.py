#!/usr/bin/env python3
"""Run a pilot sweep to choose the best worker count for transmission generation."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from prt_transmission_benchmark.constants import DEFAULT_TUNING_SAMPLE_COUNT, DEFAULT_WORKER_CANDIDATES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python-bin", type=Path, default=Path(sys.executable))
    parser.add_argument("--input-data-path", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--sample-count", type=int, default=DEFAULT_TUNING_SAMPLE_COUNT)
    parser.add_argument("--worker-candidates", type=int, nargs="+", default=list(DEFAULT_WORKER_CANDIDATES))
    parser.add_argument("--seed", type=int, default=10032026)
    parser.add_argument("--keep-runs", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    script_path = Path(__file__).resolve().parent / "generate_transmission_benchmark.py"
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    results = []

    base_env = os.environ.copy()
    base_env.setdefault("OMP_NUM_THREADS", "1")
    base_env.setdefault("MKL_NUM_THREADS", "1")
    base_env.setdefault("OPENBLAS_NUM_THREADS", "1")
    base_env.setdefault("NUMEXPR_NUM_THREADS", "1")

    for workers in args.worker_candidates:
        run_root = output_root / ("workers_%02d" % workers)
        if run_root.exists():
            shutil.rmtree(run_root)
        command = [
            str(args.python_bin),
            str(script_path),
            "--output-root",
            str(run_root),
            "--p-rt-input-data-path",
            str(args.input_data_path.resolve()),
            "--sample-count",
            str(args.sample_count),
            "--shard-size",
            "512",
            "--workers",
            str(workers),
            "--seed",
            str(args.seed),
            "--generate-only",
        ]
        started = time.perf_counter()
        subprocess.run(command, check=True, env=base_env)
        elapsed = time.perf_counter() - started
        rate = (float(args.sample_count) / elapsed) * 60.0
        results.append({"workers": int(workers), "elapsed_seconds": elapsed, "samples_per_minute": rate})
        if not args.keep_runs:
            shutil.rmtree(run_root, ignore_errors=True)

    results.sort(key=lambda item: item["samples_per_minute"], reverse=True)
    summary = {
        "recommended_workers": results[0]["workers"],
        "results": results,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
