"""Run controlled TauREx-to-POSEIDON quantum-vs-classical residual experiments."""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data/TauREx set")
    parser.add_argument("--output-dir", default="outputs/controlled_quantum_residual")
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--stage1-epochs", type=int, default=15)
    parser.add_argument("--residual-epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--val-limit", type=int, default=None)
    parser.add_argument("--holdout-limit", type=int, default=None)
    parser.add_argument("--test-limit", type=int, default=None)
    parser.add_argument("--classical-lr-stage1", type=float, default=1.0e-3)
    parser.add_argument("--classical-lr-residual", type=float, default=5.0e-5)
    parser.add_argument("--residual-lr", type=float, default=2.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--qnn-qubits", type=int, default=8)
    parser.add_argument("--qnn-depth", type=int, default=2)
    parser.add_argument("--quantum-device", default="auto")
    parser.add_argument("--quantum-use-async", action="store_true")
    parser.add_argument("--warmup-epochs", type=int, default=0)
    parser.add_argument("--ramp-epochs", type=int, default=8)
    parser.add_argument("--freeze-epochs", type=int, default=4)
    parser.add_argument("--gpu-memory-fraction", type=float, default=0.65)
    parser.add_argument("--torch-threads", type=int, default=6)
    parser.add_argument(
        "--stop-after", choices=("stage1", "first_pair", "all"), default="all"
    )
    parser.add_argument("--skip-existing", action="store_true", default=True)
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def append_manifest(payload: dict[str, Any]) -> None:
    path = PROJECT_ROOT / "_script_manifest.jsonl"
    with path.open("a") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def resolve_quantum_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if not torch.cuda.is_available():
        return "lightning.qubit"
    try:
        import importlib.util

        for name in (
            "pennylane_lightning.lightning_gpu",
            "lightning_gpu_ops",
            "pennylane_lightning_gpu",
        ):
            if importlib.util.find_spec(name) is not None:
                return "lightning.gpu"
    except Exception:
        return "lightning.qubit"
    return "lightning.qubit"


def configure_resources(args: argparse.Namespace) -> None:
    os.environ.setdefault("OMP_NUM_THREADS", str(args.torch_threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(args.torch_threads))
    torch.set_num_threads(max(1, args.torch_threads))
    try:
        torch.set_num_interop_threads(max(1, args.torch_threads // 2))
    except RuntimeError:
        pass
    if torch.cuda.is_available():
        fraction = min(max(float(args.gpu_memory_fraction), 0.1), 0.95)
        torch.cuda.set_per_process_memory_fraction(fraction, device=0)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")


def maybe_log_mlflow(run_name: str, output_dir: Path, params: dict[str, Any]) -> None:
    try:
        import mlflow
    except Exception:
        return
    try:
        mlflow.set_tracking_uri(f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}")
        mlflow.set_experiment("controlled_quantum_residual")
        with mlflow.start_run(run_name=run_name):
            mlflow.log_params(
                {
                    k: v
                    for k, v in params.items()
                    if isinstance(v, (str, int, float, bool))
                }
            )
            for metrics_name in (
                "validation_metrics.json",
                "holdout_metrics.json",
                "run_summary.json",
            ):
                metrics_path = output_dir / metrics_name
                if not metrics_path.exists():
                    continue
                payload = load_json(metrics_path)
                for key, value in payload.items():
                    if isinstance(value, (int, float)):
                        mlflow.log_metric(
                            f"{metrics_name.removesuffix('.json')}.{key}", float(value)
                        )
            for artifact in (
                "config.json",
                "history.csv",
                "validation_metrics.json",
                "holdout_metrics.json",
                "run_summary.json",
            ):
                path = output_dir / artifact
                if path.exists():
                    mlflow.log_artifact(str(path))
    except Exception as exc:
        warning_path = output_dir / "mlflow_warning.txt"
        warning_path.write_text(f"MLflow logging failed: {type(exc).__name__}: {exc}\n")


def cleanup_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_stage1(
    args: argparse.Namespace, seed: int, out_dir: Path, cache_dir: Path
) -> Path:
    from models.taurex_exobiome.training import TrainingConfig, run_training_experiment

    done = out_dir / "run_summary.json"
    if args.skip_existing and done.exists():
        print(f"[skip] stage1 seed={seed}: {done}", flush=True)
        return out_dir / "best_model.pt"

    config = TrainingConfig(
        project_root=str(PROJECT_ROOT),
        data_root=args.data_root,
        dataset_format="taurex",
        output_dir=str(out_dir),
        prepared_cache_dir=str(cache_dir),
        seed=seed,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        max_epochs=args.stage1_epochs,
        early_stop_patience=max(4, min(8, args.stage1_epochs)),
        scheduler_patience=3,
        classical_lr=args.classical_lr_stage1,
        quantum_lr=args.residual_lr,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
        loss_name="mse",
        qnn_qubits=args.qnn_qubits,
        qnn_depth=args.qnn_depth,
        quantum_device="lightning.qubit",
        classical_only=True,
        quantum_warmup_epochs=0,
        quantum_ramp_epochs=0,
        quantum_backbone_freeze_epochs=0,
        use_amp=not args.no_amp,
        log_every_batches=100,
        train_limit=args.train_limit,
        val_limit=args.val_limit,
        holdout_limit=args.holdout_limit,
        test_limit=args.test_limit,
    )
    print(f"[run] stage1 classical seed={seed} out={out_dir}", flush=True)
    run_training_experiment(config)
    maybe_log_mlflow(f"seed{seed}_stage1_classical", out_dir, config.to_json_dict())
    cleanup_cuda()
    return out_dir / "best_model.pt"


def run_quantum(
    args: argparse.Namespace,
    seed: int,
    init_checkpoint: Path,
    out_dir: Path,
    cache_dir: Path,
) -> None:
    from models.taurex_exobiome.training import TrainingConfig, run_training_experiment

    if args.skip_existing and (out_dir / "run_summary.json").exists():
        print(f"[skip] quantum seed={seed}: {out_dir}", flush=True)
        return

    q_device = resolve_quantum_device(args.quantum_device)
    config = TrainingConfig(
        project_root=str(PROJECT_ROOT),
        data_root=args.data_root,
        dataset_format="taurex",
        output_dir=str(out_dir),
        prepared_cache_dir=str(cache_dir),
        init_checkpoint_path=str(init_checkpoint),
        seed=seed,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        max_epochs=args.residual_epochs,
        early_stop_patience=max(4, min(8, args.residual_epochs)),
        scheduler_patience=3,
        classical_lr=args.classical_lr_residual,
        quantum_lr=args.residual_lr,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
        loss_name="mse",
        qnn_qubits=args.qnn_qubits,
        qnn_depth=args.qnn_depth,
        qnn_init_scale=0.1,
        quantum_device=q_device,
        quantum_use_async=args.quantum_use_async,
        classical_only=False,
        quantum_warmup_epochs=args.warmup_epochs,
        quantum_ramp_epochs=args.ramp_epochs,
        quantum_backbone_freeze_epochs=args.freeze_epochs,
        use_amp=not args.no_amp,
        log_every_batches=100,
        train_limit=args.train_limit,
        val_limit=args.val_limit,
        holdout_limit=args.holdout_limit,
        test_limit=args.test_limit,
    )
    print(
        f"[run] gated quantum residual seed={seed} q_device={q_device} out={out_dir}",
        flush=True,
    )
    run_training_experiment(config)
    maybe_log_mlflow(
        f"seed{seed}_gated_quantum_residual", out_dir, config.to_json_dict()
    )
    cleanup_cuda()


def run_noquant(
    args: argparse.Namespace,
    seed: int,
    init_checkpoint: Path,
    out_dir: Path,
    cache_dir: Path,
) -> None:
    from models.taurex_exobiome_without_quant.training import (
        TrainingConfig,
        run_training_experiment,
    )

    if args.skip_existing and (out_dir / "run_summary.json").exists():
        print(f"[skip] classical residual seed={seed}: {out_dir}", flush=True)
        return

    config = TrainingConfig(
        project_root=str(PROJECT_ROOT),
        data_root=args.data_root,
        dataset_format="taurex",
        device="cuda" if torch.cuda.is_available() else "cpu",
        feature_recipe="legacy",
        output_dir=str(out_dir),
        prepared_cache_dir=str(cache_dir),
        init_checkpoint_path=str(init_checkpoint),
        seed=seed,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        max_epochs=args.residual_epochs,
        early_stop_patience=max(4, min(8, args.residual_epochs)),
        scheduler_patience=3,
        classical_lr=args.classical_lr_residual,
        quantum_lr=args.residual_lr,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
        loss_name="mse",
        qnn_qubits=args.qnn_qubits,
        qnn_depth=args.qnn_depth,
        classical_only=False,
        architecture="legacy_conv_refiner",
        spectral_width=48,
        quantum_warmup_epochs=args.warmup_epochs,
        quantum_ramp_epochs=args.ramp_epochs,
        quantum_backbone_freeze_epochs=args.freeze_epochs,
        use_amp=not args.no_amp,
        log_every_batches=100,
        train_limit=args.train_limit,
        val_limit=args.val_limit,
        holdout_limit=args.holdout_limit,
        test_limit=args.test_limit,
        taurex_ignore_poseidon=False,
    )
    print(f"[run] classical residual seed={seed} out={out_dir}", flush=True)
    run_training_experiment(config)
    maybe_log_mlflow(f"seed{seed}_classical_residual", out_dir, config.to_json_dict())
    cleanup_cuda()


def collect_summary(base_dir: Path, seeds: list[int]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        for variant in (
            "stage1_classical",
            "gated_quantum_residual",
            "classical_residual",
        ):
            run_dir = base_dir / f"seed_{seed}" / variant
            summary_path = run_dir / "run_summary.json"
            if not summary_path.exists():
                continue
            summary = load_json(summary_path)
            rows.append(
                {
                    "seed": seed,
                    "variant": variant,
                    "validation_rmse_mean": summary.get("validation_rmse_mean"),
                    "holdout_rmse_mean": summary.get("holdout_rmse_mean"),
                    "best_epoch": summary.get("best_epoch"),
                    "output_dir": str(run_dir),
                }
            )

    paired = []
    for seed in seeds:
        q = next(
            (
                r
                for r in rows
                if r["seed"] == seed and r["variant"] == "gated_quantum_residual"
            ),
            None,
        )
        c = next(
            (
                r
                for r in rows
                if r["seed"] == seed and r["variant"] == "classical_residual"
            ),
            None,
        )
        if q and c:
            paired.append(c["holdout_rmse_mean"] - q["holdout_rmse_mean"])
    return {
        "rows": rows,
        "paired_noquant_minus_quantum_holdout_rmse": paired,
        "paired_mean": float(np.mean(paired)) if paired else None,
        "paired_std": float(np.std(paired, ddof=1)) if len(paired) > 1 else None,
    }


def main() -> None:
    args = parse_args()
    start = time.time()
    configure_resources(args)

    seeds = [int(item.strip()) for item in args.seeds.split(",") if item.strip()]
    base_dir = (PROJECT_ROOT / args.output_dir).resolve()
    cache_dir = base_dir / "prepared_cache"
    base_dir.mkdir(parents=True, exist_ok=True)
    save_json(
        base_dir / "campaign_config.json",
        {
            **vars(args),
            "project_root": str(PROJECT_ROOT),
            "git_commit": git_commit(),
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device": torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else None,
            "resolved_quantum_device": resolve_quantum_device(args.quantum_device),
        },
    )

    completed_pairs = 0
    try:
        for seed in seeds:
            seed_dir = base_dir / f"seed_{seed}"
            stage1 = seed_dir / "stage1_classical"
            checkpoint = run_stage1(args, seed, stage1, cache_dir)
            if args.stop_after == "stage1":
                break
            run_quantum(
                args, seed, checkpoint, seed_dir / "gated_quantum_residual", cache_dir
            )
            run_noquant(
                args, seed, checkpoint, seed_dir / "classical_residual", cache_dir
            )
            completed_pairs += 1
            save_json(
                base_dir / "campaign_summary.json", collect_summary(base_dir, seeds)
            )
            if args.stop_after == "first_pair" and completed_pairs >= 1:
                break
    finally:
        summary = collect_summary(base_dir, seeds)
        save_json(base_dir / "campaign_summary.json", summary)
        append_manifest(
            {
                "timestamp_unix": time.time(),
                "command": " ".join([sys.executable, *sys.argv]),
                "cwd": str(PROJECT_ROOT),
                "duration_seconds": time.time() - start,
                "outputs": [
                    str(base_dir / "campaign_config.json"),
                    str(base_dir / "campaign_summary.json"),
                ],
                "status": "finished_or_interrupted",
            }
        )
        print(
            json.dumps({"campaign_dir": str(base_dir), "summary": summary}, indent=2),
            flush=True,
        )


if __name__ == "__main__":
    main()
