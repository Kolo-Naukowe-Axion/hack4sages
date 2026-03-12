#!/usr/bin/env python3
"""Run the Five-qubit ExoBiome Ariel hybrid quantum regressor."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--data-root", default="data/ariel-ml-dataset")
    parser.add_argument("--output-dir", default="outputs/five_qubit_exobiome")
    parser.add_argument("--prepared-cache-dir", default=None)
    parser.add_argument("--init-checkpoint-path", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--early-stop-patience", type=int, default=6)
    parser.add_argument("--scheduler-patience", type=int, default=2)
    parser.add_argument("--scheduler-factor", type=float, default=0.5)
    parser.add_argument("--classical-lr", type=float, default=2.0e-3)
    parser.add_argument("--quantum-lr", type=float, default=8.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--gradient-clip-norm", type=float, default=5.0)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--loss-name", default="mse", choices=("mse", "huber"))
    parser.add_argument("--qnn-qubits", type=int, default=5)
    parser.add_argument("--qnn-depth", type=int, default=2)
    parser.add_argument("--qnn-init-scale", type=float, default=0.1)
    parser.add_argument("--quantum-device", default=None)
    parser.add_argument("--quantum-use-async", action="store_true")
    parser.add_argument("--quantum-warmup-epochs", type=int, default=5)
    parser.add_argument("--quantum-ramp-epochs", type=int, default=4)
    parser.add_argument("--quantum-backbone-freeze-epochs", type=int, default=0)
    parser.add_argument("--log-every-batches", type=int, default=20)
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--val-limit", type=int, default=None)
    parser.add_argument("--holdout-limit", type=int, default=None)
    parser.add_argument("--test-limit", type=int, default=None)
    parser.add_argument("--classical-only", action="store_true")
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.cpu_only:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    from models.five_qubit_exobiome.training import TrainingConfig, run_training_experiment

    config = TrainingConfig(
        project_root=str(Path(args.project_root).resolve()),
        data_root=args.data_root,
        output_dir=args.output_dir,
        prepared_cache_dir=args.prepared_cache_dir,
        init_checkpoint_path=args.init_checkpoint_path,
        seed=args.seed,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        max_epochs=args.max_epochs,
        early_stop_patience=args.early_stop_patience,
        scheduler_patience=args.scheduler_patience,
        scheduler_factor=args.scheduler_factor,
        classical_lr=args.classical_lr,
        quantum_lr=args.quantum_lr,
        weight_decay=args.weight_decay,
        gradient_clip_norm=args.gradient_clip_norm,
        dropout=args.dropout,
        loss_name=args.loss_name,
        qnn_qubits=args.qnn_qubits,
        qnn_depth=args.qnn_depth,
        qnn_init_scale=args.qnn_init_scale,
        quantum_use_async=args.quantum_use_async,
        classical_only=args.classical_only,
        quantum_warmup_epochs=args.quantum_warmup_epochs,
        quantum_ramp_epochs=args.quantum_ramp_epochs,
        quantum_backbone_freeze_epochs=args.quantum_backbone_freeze_epochs,
        use_amp=not args.no_amp,
        log_every_batches=args.log_every_batches,
        train_limit=args.train_limit,
        val_limit=args.val_limit,
        holdout_limit=args.holdout_limit,
        test_limit=args.test_limit,
    )
    if args.quantum_device is not None:
        config.quantum_device = args.quantum_device

    print(json.dumps(config.to_json_dict(), indent=2), flush=True)
    result = run_training_experiment(config)
    print(json.dumps(result["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
