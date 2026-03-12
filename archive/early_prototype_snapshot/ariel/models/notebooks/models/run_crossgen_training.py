#!/usr/bin/env python3
"""Run the cross-generator hybrid training loop with live stdout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_batch_probe_sizes(raw: str) -> tuple[int, ...]:
    values = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        values.append(int(item))
    if not values:
        raise argparse.ArgumentTypeError("batch probe sizes must contain at least one integer.")
    return tuple(values)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the live-streaming crossgen hybrid training loop.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--data-root", default="data/generated-data/crossgen_biosignatures_20260311")
    parser.add_argument("--output-dir", default="outputs/model_quant_sketch_crossgen_live")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-batch-size", type=int, default=1024)
    parser.add_argument("--eval-batch-size", type=int, default=8192)
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--early-stop-patience", type=int, default=6)
    parser.add_argument("--scheduler-patience", type=int, default=2)
    parser.add_argument("--classical-lr", type=float, default=2.0e-3)
    parser.add_argument("--quantum-lr", type=float, default=8.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--gradient-clip-norm", type=float, default=5.0)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--qnn-qubits", type=int, default=12)
    parser.add_argument("--qnn-depth", type=int, default=2)
    parser.add_argument("--batch-probe-sizes", type=parse_batch_probe_sizes, default=(1024, 1536, 2048, 3072))
    parser.add_argument("--batch-probe-steps", type=int, default=2)
    parser.add_argument("--autotune-batch-size", dest="autotune_batch_size", action="store_true")
    parser.add_argument("--no-autotune-batch-size", dest="autotune_batch_size", action="store_false")
    parser.set_defaults(autotune_batch_size=False)
    parser.add_argument("--log-every-batches", type=int, default=1)
    parser.add_argument("--quantum-device", default="lightning.gpu")
    parser.add_argument("--train-pool-limit", type=int, default=None)
    parser.add_argument("--tau-test-limit", type=int, default=None)
    parser.add_argument("--poseidon-limit", type=int, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    from models.crossgen_hybrid_training import (
        TrainingConfig,
        default_crossgen_data_root,
        run_training_experiment,
    )

    project_root = Path(args.project_root).resolve()
    data_root = Path(args.data_root)
    if not data_root.is_absolute():
        default_root = default_crossgen_data_root(project_root)
        if args.data_root == "data/generated-data/crossgen_biosignatures_20260311":
            data_root = default_root
        else:
            data_root = project_root / data_root

    config = TrainingConfig(
        project_root=str(project_root),
        data_root=str(data_root),
        output_dir=args.output_dir,
        seed=args.seed,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        max_epochs=args.max_epochs,
        early_stop_patience=args.early_stop_patience,
        scheduler_patience=args.scheduler_patience,
        classical_lr=args.classical_lr,
        quantum_lr=args.quantum_lr,
        weight_decay=args.weight_decay,
        gradient_clip_norm=args.gradient_clip_norm,
        dropout=args.dropout,
        qnn_qubits=args.qnn_qubits,
        qnn_depth=args.qnn_depth,
        autotune_batch_size=args.autotune_batch_size,
        batch_probe_sizes=args.batch_probe_sizes,
        batch_probe_steps=args.batch_probe_steps,
        log_every_batches=args.log_every_batches,
        quantum_device=args.quantum_device,
        train_pool_limit=args.train_pool_limit,
        tau_test_limit=args.tau_test_limit,
        poseidon_limit=args.poseidon_limit,
    )
    print(json.dumps(config.to_json_dict(), indent=2), flush=True)
    result = run_training_experiment(config)
    print(json.dumps(result["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
