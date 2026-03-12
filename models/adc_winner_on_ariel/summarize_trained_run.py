"""Summarize an archived ADC winner-style run for model-to-model comparisons."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import torch
import yaml

from .model import IndependentNSF, ModelConfig


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _load_history(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _flatten_metrics(prefix: str, metrics: dict[str, Any]) -> dict[str, Any]:
    flat = {
        f"{prefix}_rows": int(metrics["rows"]),
        f"{prefix}_num_samples": int(metrics["num_samples"]),
        f"{prefix}_point_estimate": str(metrics["point_estimate"]),
        f"{prefix}_mrmse": float(metrics["rmse_mean"]),
    }
    for name, value in metrics["rmse"].items():
        flat[f"{prefix}_{name}_rmse"] = float(value)
    return flat


def _count_parameters(model: torch.nn.Module) -> tuple[int, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return int(total), int(trainable)


def build_summary(run_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    settings = yaml.safe_load((run_dir / "settings_resolved.yaml").read_text())
    history = _load_history(run_dir / "history.jsonl")
    validation_metrics = _load_json(run_dir / "validation_metrics.json")
    holdout_metrics = _load_json(run_dir / "holdout_metrics.json")
    prepared_manifest = _load_json(run_dir / "prepared_manifest.json")
    saved_split_manifest = _load_json(run_dir / "saved_split_manifest.json")

    model = IndependentNSF(ModelConfig(**settings["model"]))
    best_mrmse_ckpt = torch.load(run_dir / "best_model_by_mrmse.pt", map_location="cpu")
    best_nll_ckpt = torch.load(run_dir / "best_model_by_nll.pt", map_location="cpu")
    model.load_state_dict(best_mrmse_ckpt["model"])
    total_params, trainable_params = _count_parameters(model)

    best_val_nll_row = min(history, key=lambda row: float(row["val_nll"]))
    preview_rows = [row for row in history if "val_rmse_mean" in row]
    best_preview_row = min(preview_rows, key=lambda row: float(row["val_rmse_mean"])) if preview_rows else None
    final_row = history[-1]

    summary = {
        "model_name": "adc_winner_on_ariel",
        "model_family": "independent_conditional_neural_spline_flows",
        "package_root": str(run_dir.parent),
        "run_dir": str(run_dir),
        "comparison_ready": True,
        "seed": int(settings.get("seed", 42)),
        "data_root": prepared_manifest["data_root"],
        "split_source": prepared_manifest["split_source"],
        "split_rows": prepared_manifest["split_sizes"],
        "saved_split_rows": saved_split_manifest["rows"],
        "target_columns": prepared_manifest["target_columns"],
        "context_layout": prepared_manifest["context_layout"],
        "architecture": {
            "context_dim": int(settings["model"]["context_dim"]),
            "hidden_features": int(settings["model"]["hidden_features"]),
            "hidden_layers": int(settings["model"]["hidden_layers"]),
            "transforms": int(settings["model"]["transforms"]),
            "bins": int(settings["model"]["bins"]),
            "flow_count": len(prepared_manifest["target_columns"]),
        },
        "training": {
            "device": str(settings["training"]["device"]),
            "batch_size": int(settings["training"]["batch_size"]),
            "eval_batch_size": int(settings["training"]["eval_batch_size"]),
            "learning_rate": float(settings["training"]["learning_rate"]),
            "epochs_max": int(settings["training"]["epochs"]),
            "patience": int(settings["training"]["patience"]),
            "scheduler_factor": float(settings["training"]["scheduler_factor"]),
            "scheduler_patience": int(settings["training"]["scheduler_patience"]),
            "gradient_clip_norm": float(settings["training"]["gradient_clip_norm"]),
        },
        "evaluation": {
            "preview_every_epochs": int(settings["evaluation"]["metric_every_epochs"]),
            "preview_max_rows": int(settings["evaluation"]["metric_max_rows"]),
            "preview_num_samples": int(settings["evaluation"]["metric_num_samples"]),
            "final_num_samples": int(settings["evaluation"]["final_num_samples"]),
            "point_estimate": str(settings["evaluation"]["point_estimate"]),
        },
        "parameters": {
            "total": total_params,
            "trainable": trainable_params,
        },
        "checkpoints": {
            "best_model_by_mrmse": {
                "path": str(run_dir / "best_model_by_mrmse.pt"),
                "epoch": int(best_mrmse_ckpt["state"]["epoch"]),
                "bytes": int((run_dir / "best_model_by_mrmse.pt").stat().st_size),
            },
            "best_model_by_nll": {
                "path": str(run_dir / "best_model_by_nll.pt"),
                "epoch": int(best_nll_ckpt["state"]["epoch"]),
                "bytes": int((run_dir / "best_model_by_nll.pt").stat().st_size),
            },
            "resume_latest": {
                "path": str(run_dir / "resume_latest.pt"),
                "bytes": int((run_dir / "resume_latest.pt").stat().st_size),
            },
        },
        "history": {
            "epochs_completed": int(final_row["epoch"]),
            "early_stopped": int(final_row["epoch"]) < int(settings["training"]["epochs"]),
            "final_train_nll": float(final_row["train_nll"]),
            "final_val_nll": float(final_row["val_nll"]),
            "final_lr": float(final_row["lr"]),
            "best_val_nll": float(best_val_nll_row["val_nll"]),
            "best_val_nll_epoch": int(best_val_nll_row["epoch"]),
            "preview_points_recorded": len(preview_rows),
            "best_preview_mrmse": float(best_preview_row["val_rmse_mean"]) if best_preview_row is not None else None,
            "best_preview_mrmse_epoch": int(best_preview_row["epoch"]) if best_preview_row is not None else None,
        },
        "metrics": {
            "validation": validation_metrics,
            "holdout": holdout_metrics,
        },
    }

    flat = {
        "model_name": summary["model_name"],
        "model_family": summary["model_family"],
        "seed": summary["seed"],
        "context_dim": summary["architecture"]["context_dim"],
        "hidden_features": summary["architecture"]["hidden_features"],
        "hidden_layers": summary["architecture"]["hidden_layers"],
        "transforms": summary["architecture"]["transforms"],
        "bins": summary["architecture"]["bins"],
        "flow_count": summary["architecture"]["flow_count"],
        "total_params": summary["parameters"]["total"],
        "trainable_params": summary["parameters"]["trainable"],
        "train_rows": summary["split_rows"]["train"],
        "validation_rows": summary["split_rows"]["validation"],
        "holdout_rows": summary["split_rows"]["holdout"],
        "test_rows": summary["split_rows"]["testdata"],
        "epochs_completed": summary["history"]["epochs_completed"],
        "epochs_max": summary["training"]["epochs_max"],
        "early_stopped": summary["history"]["early_stopped"],
        "best_val_nll": summary["history"]["best_val_nll"],
        "best_val_nll_epoch": summary["history"]["best_val_nll_epoch"],
        "best_preview_mrmse": summary["history"]["best_preview_mrmse"],
        "best_preview_mrmse_epoch": summary["history"]["best_preview_mrmse_epoch"],
        "final_train_nll": summary["history"]["final_train_nll"],
        "final_val_nll": summary["history"]["final_val_nll"],
        "final_lr": summary["history"]["final_lr"],
        "best_model_by_mrmse_epoch": summary["checkpoints"]["best_model_by_mrmse"]["epoch"],
        "best_model_by_nll_epoch": summary["checkpoints"]["best_model_by_nll"]["epoch"],
        "best_model_by_mrmse_bytes": summary["checkpoints"]["best_model_by_mrmse"]["bytes"],
        "best_model_by_nll_bytes": summary["checkpoints"]["best_model_by_nll"]["bytes"],
        "resume_latest_bytes": summary["checkpoints"]["resume_latest"]["bytes"],
    }
    flat.update(_flatten_metrics("validation", validation_metrics))
    flat.update(_flatten_metrics("holdout", holdout_metrics))
    return summary, flat


def save_flat_csv(path: Path, row: dict[str, Any]) -> None:
    fieldnames = list(row.keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=Path(__file__).resolve().parent / "trained_run")
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-csv", type=Path, default=None)
    args = parser.parse_args()

    run_dir = args.run_dir.expanduser().resolve()
    output_json = args.output_json.expanduser().resolve() if args.output_json else run_dir / "comparison_metrics.json"
    output_csv = args.output_csv.expanduser().resolve() if args.output_csv else run_dir / "comparison_metrics.csv"

    summary, flat = build_summary(run_dir)
    output_json.write_text(json.dumps(summary, indent=2) + "\n")
    save_flat_csv(output_csv, flat)
    print(json.dumps({"comparison_metrics_json": str(output_json), "comparison_metrics_csv": str(output_csv)}, indent=2))


if __name__ == "__main__":
    main()
