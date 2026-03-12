"""Train winner-family independent tracedata flows on the local Ariel split."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from .constants import TARGET_COLUMNS
from .dataset import build_trace_batch, load_prepared_data
from .evaluate import evaluate_bundle, evaluate_testdata_medians, save_metrics
from .model import IndependentNSF, ModelConfig


@dataclass
class TrainState:
    epoch: int = 0
    best_val_nll: float = float("inf")
    epochs_since_improvement: int = 0


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(settings: dict[str, Any]) -> torch.device:
    requested = str(settings["training"].get("device", "auto"))
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")
    if device.type == "mps" and not (getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()):
        raise RuntimeError("MPS requested but unavailable.")
    return device


def configure_runtime(device: torch.device) -> None:
    cpu_threads = max(1, min(os.cpu_count() or 1, 16))
    try:
        torch.set_num_threads(cpu_threads)
    except RuntimeError:
        pass
    try:
        torch.set_num_interop_threads(max(1, min(cpu_threads // 2, 8)))
    except RuntimeError:
        pass
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")


def batch_permutation(length: int, batch_size: int, seed: int, epoch: int) -> list[torch.Tensor]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + epoch)
    order = torch.randperm(length, generator=generator, device="cpu")
    return [order[start : start + batch_size] for start in range(0, length, batch_size)]


def weighted_nll(flow, context: torch.Tensor, trace_targets: torch.Tensor, trace_weights: torch.Tensor) -> torch.Tensor:
    log_prob = flow(context.unsqueeze(1)).log_prob(trace_targets)
    return -(log_prob * trace_weights).sum() / max(1, context.shape[0] * trace_targets.shape[-1])


def evaluate_target_nll(flow, split, scalers, *, device: torch.device, batch_size: int, target_index: int) -> float:
    losses = []
    flow.eval()
    with torch.inference_mode():
        for start in range(0, split.rows, batch_size):
            rows = torch.arange(start, min(start + batch_size, split.rows), dtype=torch.long)
            context, trace_targets, trace_weights = build_trace_batch(
                split,
                rows,
                scalers,
                device=device,
                target_index=target_index,
            )
            losses.append(float(weighted_nll(flow, context, trace_targets, trace_weights).detach().cpu().item()))
    return float(np.mean(losses)) if losses else float("inf")


def save_checkpoint(path: Path, *, flow, optimizer, scheduler, state: TrainState, settings: dict[str, Any], target_name: str, target_index: int) -> None:
    torch.save(
        {
            "target_name": target_name,
            "target_index": target_index,
            "model": flow.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "state": state.__dict__,
            "settings": settings,
        },
        path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--settings", type=Path, required=True)
    parser.add_argument("--prepared-data", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()

    settings = yaml.safe_load(args.settings.read_text())
    set_seed(int(settings.get("seed", 42)))
    device = resolve_device(settings)
    configure_runtime(device)

    run_dir = args.run_dir.expanduser().resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "settings_resolved.yaml").write_text(yaml.safe_dump(settings, sort_keys=False))

    data = load_prepared_data(args.prepared_data)
    if data.manifest:
        (run_dir / "prepared_manifest.json").write_text(json.dumps(data.manifest, indent=2) + "\n")

    model = IndependentNSF(ModelConfig(**settings["model"])).to(device)
    batch_size = int(settings["training"]["batch_size"])
    eval_batch_size = int(settings["training"]["eval_batch_size"])
    max_epochs = int(settings["training"]["epochs"])
    patience = int(settings["training"]["patience"])
    learning_rate = float(settings["training"]["learning_rate"])
    gradient_clip_norm = float(settings["training"].get("gradient_clip_norm", 5.0))

    target_summaries = []
    for target_index, target_name in enumerate(TARGET_COLUMNS):
        print(f"Training target {target_index + 1}/{len(TARGET_COLUMNS)}: {target_name}", flush=True)
        target_dir = run_dir / "targets" / target_name
        target_dir.mkdir(parents=True, exist_ok=True)

        flow = model.flows[target_index]
        optimizer = torch.optim.Adam(flow.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=float(settings["training"].get("scheduler_factor", 0.5)),
            patience=int(settings["training"].get("scheduler_patience", 4)),
        )
        state = TrainState()
        history_path = target_dir / "history.jsonl"

        for epoch in range(1, max_epochs + 1):
            state.epoch = epoch
            flow.train()
            batch_losses = []
            for batch_rows in batch_permutation(data.train.rows, batch_size, int(settings.get("seed", 42)) + target_index * 10_000, epoch):
                optimizer.zero_grad(set_to_none=True)
                context, trace_targets, trace_weights = build_trace_batch(
                    data.train,
                    batch_rows,
                    data.scalers,
                    device=device,
                    target_index=target_index,
                )
                loss = weighted_nll(flow, context, trace_targets, trace_weights)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(flow.parameters(), gradient_clip_norm)
                optimizer.step()
                batch_losses.append(float(loss.detach().cpu().item()))

            val_nll = evaluate_target_nll(
                flow,
                data.validation,
                data.scalers,
                device=device,
                batch_size=eval_batch_size,
                target_index=target_index,
            )
            scheduler.step(val_nll)
            train_nll = float(np.mean(batch_losses)) if batch_losses else float("nan")
            log_row = {
                "epoch": epoch,
                "target": target_name,
                "train_nll": train_nll,
                "val_nll": val_nll,
                "lr": float(optimizer.param_groups[0]["lr"]),
            }
            print(
                f"[{target_name}] epoch {epoch} complete | train_nll={train_nll:.6f} val_nll={val_nll:.6f} lr={optimizer.param_groups[0]['lr']:.6e}",
                flush=True,
            )
            with history_path.open("a") as handle:
                handle.write(json.dumps(log_row) + "\n")

            save_checkpoint(
                target_dir / "resume_latest.pt",
                flow=flow,
                optimizer=optimizer,
                scheduler=scheduler,
                state=state,
                settings=settings,
                target_name=target_name,
                target_index=target_index,
            )
            if val_nll < state.best_val_nll:
                state.best_val_nll = val_nll
                state.epochs_since_improvement = 0
                save_checkpoint(
                    target_dir / "best_model_by_nll.pt",
                    flow=flow,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    state=state,
                    settings=settings,
                    target_name=target_name,
                    target_index=target_index,
                )
            else:
                state.epochs_since_improvement += 1

            if state.epochs_since_improvement >= patience:
                print(f"[{target_name}] early stopping after epoch {epoch} due to validation NLL patience.", flush=True)
                break

        best_payload = torch.load(target_dir / "best_model_by_nll.pt", map_location=device)
        flow.load_state_dict(best_payload["model"])
        target_summary = {
            "target_name": target_name,
            "best_val_nll": float(best_payload["state"]["best_val_nll"]),
            "best_epoch": int(best_payload["state"]["epoch"]),
        }
        target_summaries.append(target_summary)
        (target_dir / "summary.json").write_text(json.dumps(target_summary, indent=2) + "\n")

    bundle_path = run_dir / "best_independent_bundle.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "target_summaries": target_summaries,
            "settings": settings,
        },
        bundle_path,
    )

    evaluation_settings = settings["evaluation"]
    final_num_samples = int(evaluation_settings["final_num_samples"])
    point_estimate = str(evaluation_settings.get("point_estimate", "median"))
    trace_seed = int(evaluation_settings.get("trace_seed", 42))

    validation_metrics, validation_predictions = evaluate_bundle(
        model,
        data.validation,
        data,
        device=device,
        num_samples=final_num_samples,
        batch_size=eval_batch_size,
        point_estimate=point_estimate,
        trace_seed=trace_seed,
        log_prefix="validation final",
    )
    holdout_metrics, holdout_predictions = evaluate_bundle(
        model,
        data.holdout,
        data,
        device=device,
        num_samples=final_num_samples,
        batch_size=eval_batch_size,
        point_estimate=point_estimate,
        trace_seed=trace_seed,
        log_prefix="holdout final",
    )
    test_predictions = evaluate_testdata_medians(
        model,
        data.testdata,
        data,
        device=device,
        num_samples=final_num_samples,
        batch_size=eval_batch_size,
        point_estimate=point_estimate,
    )

    save_metrics(run_dir / "validation_metrics.json", validation_metrics)
    save_metrics(run_dir / "holdout_metrics.json", holdout_metrics)
    validation_predictions.to_csv(run_dir / "validation_predictions.csv", index=False)
    holdout_predictions.to_csv(run_dir / "holdout_predictions.csv", index=False)
    test_predictions.to_csv(run_dir / "testdata_predictions.csv", index=False)

    summary = {
        "run_dir": str(run_dir),
        "device": str(device),
        "prepared_data": str(Path(args.prepared_data).expanduser().resolve()),
        "targets": target_summaries,
        "validation_metrics": validation_metrics,
        "holdout_metrics": holdout_metrics,
    }
    (run_dir / "run_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

