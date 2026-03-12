#!/usr/bin/env python3
"""Run the JAX/Catalyst cross-generator hybrid training loop with live stdout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_batch_ladder(raw: str) -> tuple[int, ...]:
    values = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        values.append(int(item))
    if not values:
        raise argparse.ArgumentTypeError("batch ladder must contain at least one integer.")
    return tuple(values)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the live-streaming JAX/Catalyst crossgen hybrid training loop.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--data-root", default="data/generated-data/crossgen_biosignatures_20260311")
    parser.add_argument("--output-dir", default="outputs/model_quant_sketch_crossgen_jax_live")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size-train", type=int, default=1024)
    parser.add_argument("--batch-size-eval", type=int, default=8192)
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--early-stop-patience", type=int, default=6)
    parser.add_argument("--scheduler-patience", type=int, default=2)
    parser.add_argument("--scheduler-factor", type=float, default=0.5)
    parser.add_argument("--classical-lr", type=float, default=2.0e-3)
    parser.add_argument("--quantum-lr", type=float, default=6.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--aux-hidden-dim", type=int, default=64)
    parser.add_argument("--aux-out-dim", type=int, default=32)
    parser.add_argument("--spectral-hidden-dim", type=int, default=64)
    parser.add_argument("--spectral-out-dim", type=int, default=32)
    parser.add_argument("--fusion-hidden-dim", type=int, default=48)
    parser.add_argument("--head-hidden-dim", type=int, default=96)
    parser.add_argument("--qnn-qubits", type=int, default=16)
    parser.add_argument("--qnn-depth", type=int, default=2)
    parser.add_argument("--device-backend", default="lightning.gpu")
    parser.add_argument("--precision", default="float32")
    parser.add_argument("--compile-warmup-steps", type=int, default=2)
    parser.add_argument("--batch-ladder", type=parse_batch_ladder, default=(256, 512, 1024, 2048))
    parser.add_argument("--autotune-batch", dest="autotune_batch", action="store_true")
    parser.add_argument("--no-autotune-batch", dest="autotune_batch", action="store_false")
    parser.set_defaults(autotune_batch=True)
    parser.add_argument("--log-every-batches", type=int, default=1)
    parser.add_argument("--train-pool-limit", type=int, default=None)
    parser.add_argument("--tau-test-limit", type=int, default=None)
    parser.add_argument("--poseidon-limit", type=int, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    from models.crossgen_hybrid_training_jax import (
        TrainingConfigJAX,
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

    config = TrainingConfigJAX(
        project_root=str(project_root),
        data_root=str(data_root),
        output_dir=args.output_dir,
        seed=args.seed,
        batch_size_train=args.batch_size_train,
        batch_size_eval=args.batch_size_eval,
        max_epochs=args.max_epochs,
        early_stop_patience=args.early_stop_patience,
        scheduler_patience=args.scheduler_patience,
        scheduler_factor=args.scheduler_factor,
        classical_lr=args.classical_lr,
        quantum_lr=args.quantum_lr,
        weight_decay=args.weight_decay,
        gradient_clip_norm=args.gradient_clip_norm,
        dropout=args.dropout,
        aux_hidden_dim=args.aux_hidden_dim,
        aux_out_dim=args.aux_out_dim,
        spectral_hidden_dim=args.spectral_hidden_dim,
        spectral_out_dim=args.spectral_out_dim,
        fusion_hidden_dim=args.fusion_hidden_dim,
        head_hidden_dim=args.head_hidden_dim,
        qnn_qubits=args.qnn_qubits,
        qnn_depth=args.qnn_depth,
        device_backend=args.device_backend,
        precision=args.precision,
        compile_warmup_steps=args.compile_warmup_steps,
        autotune_batch=args.autotune_batch,
        performance_batch_ladder=args.batch_ladder,
        log_every_batches=args.log_every_batches,
        train_pool_limit=args.train_pool_limit,
        tau_test_limit=args.tau_test_limit,
        poseidon_limit=args.poseidon_limit,
    )
    print(json.dumps(config.to_json_dict(), indent=2), flush=True)
    result = run_training_experiment(config)
    print(json.dumps(result["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
