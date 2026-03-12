#!/usr/bin/env python3
"""Run the TensorFlow ADC baseline adaptation on the cross-generator dataset."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--data-root", default="data/generated-data/crossgen_biosignatures_20260311")
    parser.add_argument("--output-dir", default="outputs/adc_crossgen_tf")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--augment-repeat", type=int, default=5)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--mc-samples", type=int, default=64)
    parser.add_argument("--qnn-qubits", type=int, default=4)
    parser.add_argument("--qnn-depth", type=int, default=2)
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--val-limit", type=int, default=None)
    parser.add_argument("--poseidon-limit", type=int, default=None)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--quantum-device-name", default="default.qubit")
    parser.add_argument("--save-mc-samples", action="store_true")
    parser.add_argument("--cpu-only", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.cpu_only:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    from models.adc_crossgen_tf.training import TrainingConfig, train_and_evaluate

    config = TrainingConfig(
        project_root=str(Path(args.project_root).resolve()),
        data_root=args.data_root,
        output_dir=args.output_dir,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        augment_repeat=args.augment_repeat,
        dropout=args.dropout,
        mc_samples=args.mc_samples,
        qnn_qubits=args.qnn_qubits,
        qnn_depth=args.qnn_depth,
        train_limit=args.train_limit,
        val_limit=args.val_limit,
        poseidon_limit=args.poseidon_limit,
        patience=args.patience,
        quantum_device_name=args.quantum_device_name,
        save_mc_samples=args.save_mc_samples,
    )
    print(json.dumps(config.to_json_dict(), indent=2), flush=True)
    result = train_and_evaluate(config)
    print(json.dumps(result["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
