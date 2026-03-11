"""Training entrypoint for the winner-style five-gas independent NSF model."""

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

from .dataset import build_context_batch, load_prepared_data, move_prepared_data_to_device
from .evaluate import evaluate_point_metric, save_metrics
from .model import IndependentNSF, ModelConfig


@dataclass
class TrainState:
    epoch: int = 0
    best_val_nll: float = float("inf")
    best_val_rmse: float = float("inf")
    epochs_since_improvement: int = 0


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(settings: dict[str, Any]) -> torch.device:
    requested = settings["training"].get("device", "cuda")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")
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


def batch_permutation(length: int, batch_size: int, seed: int, epoch: int, device: torch.device) -> list[torch.Tensor]:
    generator = torch.Generator(device=device.type if device.type == "cuda" else "cpu")
    generator.manual_seed(seed + epoch)
    order = torch.randperm(length, generator=generator, device=device if device.type == "cuda" else "cpu")
    return [order[start : start + batch_size] for start in range(0, length, batch_size)]


def evaluate_nll(model, split, scalers, *, device: torch.device, batch_size: int, sample_noise: bool, noise_seed: int) -> float:
    losses = []
    total_batches = int(math.ceil(split.rows / batch_size))
    noise_generator = torch.Generator(device=device.type)
    noise_generator.manual_seed(int(noise_seed))
    model.eval()
    with torch.inference_mode():
        for batch_number, start in enumerate(range(0, split.rows, batch_size), start=1):
            rows = torch.arange(start, min(start + batch_size, split.rows), dtype=torch.long)
            context = build_context_batch(
                split,
                rows,
                scalers,
                device=device,
                sample_noise=sample_noise,
                noise_generator=noise_generator,
            )
            targets = split.targets_scaled.index_select(0, rows.to(split.targets_scaled.device, non_blocking=device.type == "cuda"))
            if targets.device != device:
                targets = targets.to(device, non_blocking=device.type == "cuda")
            loss, _ = model.negative_log_likelihood(context, targets)
            losses.append(float(loss.detach().cpu().item()))
    return float(np.mean(losses)) if losses else float("inf")


def save_checkpoint(path: Path, *, model, optimizer, scheduler, state: TrainState, settings: dict[str, Any]) -> None:
    torch.save(
        {
            "model": model.state_dict(),
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
    parser.add_argument("--resume", default="auto")
    args = parser.parse_args()

    settings = yaml.safe_load(args.settings.read_text())
    set_seed(int(settings.get("seed", 42)))
    run_dir = args.run_dir.expanduser().resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "settings_resolved.yaml").write_text(yaml.safe_dump(settings, sort_keys=False))

    device = resolve_device(settings)
    configure_runtime(device)
    data = load_prepared_data(args.prepared_data)
    data = move_prepared_data_to_device(data, device)
    model = IndependentNSF(ModelConfig(**settings["model"])).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(settings["training"]["learning_rate"]))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=float(settings["training"].get("scheduler_factor", 0.5)),
        patience=int(settings["training"].get("scheduler_patience", 4)),
    )

    state = TrainState()
    history_path = run_dir / "history.jsonl"
    max_epochs = int(settings["training"]["epochs"])
    patience = int(settings["training"]["patience"])
    batch_size = int(settings["training"]["batch_size"])
    eval_batch_size = int(settings["training"]["eval_batch_size"])
    metric_every = int(settings["evaluation"]["metric_every_epochs"])
    metric_max_rows = int(settings["evaluation"]["metric_max_rows"])
    metric_samples = int(settings["evaluation"]["metric_num_samples"])
    metric_point = str(settings["evaluation"].get("point_estimate", "median"))
    metric_noise_seed = int(settings["evaluation"].get("noise_seed", 42))

    for epoch in range(1, max_epochs + 1):
        state.epoch = epoch
        model.train()
        noise_generator = torch.Generator(device=device.type)
        noise_generator.manual_seed(int(settings.get("seed", 42)) + epoch)
        batch_losses = []
        for batch_rows in batch_permutation(data.train.rows, batch_size, int(settings.get("seed", 42)), epoch, device):
            optimizer.zero_grad(set_to_none=True)
            context = build_context_batch(
                data.train,
                batch_rows,
                data.runtime_scalers or data.scalers,
                device=device,
                sample_noise=True,
                noise_generator=noise_generator,
            )
            targets = data.train.targets_scaled.index_select(
                0, batch_rows.to(data.train.targets_scaled.device, non_blocking=device.type == "cuda")
            )
            if targets.device != device:
                targets = targets.to(device, non_blocking=device.type == "cuda")
            loss, _ = model.negative_log_likelihood(context, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(settings["training"].get("gradient_clip_norm", 5.0)))
            optimizer.step()
            batch_losses.append(float(loss.detach().cpu().item()))

        val_nll = evaluate_nll(
            model,
            data.validation,
            data.runtime_scalers or data.scalers,
            device=device,
            batch_size=eval_batch_size,
            sample_noise=True,
            noise_seed=metric_noise_seed,
        )
        scheduler.step(val_nll)
        train_loss = float(np.mean(batch_losses)) if batch_losses else float("nan")
        log_row = {
            "epoch": epoch,
            "train_nll": train_loss,
            "val_nll": val_nll,
            "lr": float(optimizer.param_groups[0]["lr"]),
        }
        print(
            "Epoch {epoch} complete | train_nll={train_nll:.6f} val_nll={val_nll:.6f} lr={lr:.6e}".format(**log_row),
            flush=True,
        )

        if val_nll < state.best_val_nll:
            state.best_val_nll = val_nll
            state.epochs_since_improvement = 0
            save_checkpoint(run_dir / "best_model_by_nll.pt", model=model, optimizer=optimizer, scheduler=scheduler, state=state, settings=settings)
        else:
            state.epochs_since_improvement += 1

        if metric_every > 0 and epoch % metric_every == 0:
            print(
                f"Epoch {epoch} validation mRMSE | rows={metric_max_rows}/{data.validation.rows} samples={metric_samples} point={metric_point}",
                flush=True,
            )
            metrics = evaluate_point_metric(
                model,
                data.validation,
                data.scalers,
                device=device,
                num_samples=metric_samples,
                point_estimate=metric_point,
                batch_size=eval_batch_size,
                max_rows=metric_max_rows,
                row_seed=int(settings["evaluation"].get("row_seed", 42)),
                sample_noise=True,
                noise_seed=metric_noise_seed,
                log_prefix=f"epoch {epoch} validation",
            )
            log_row["val_rmse_mean"] = metrics["rmse_mean"]
            print(
                f"Epoch {epoch} validation mRMSE | mean={metrics['rmse_mean']:.6f} point={metric_point}",
                flush=True,
            )
            if metrics["rmse_mean"] < state.best_val_rmse:
                state.best_val_rmse = metrics["rmse_mean"]
                save_checkpoint(run_dir / "best_model_by_mrmse.pt", model=model, optimizer=optimizer, scheduler=scheduler, state=state, settings=settings)

        with history_path.open("a") as handle:
            handle.write(json.dumps(log_row) + "\n")
        save_checkpoint(run_dir / "resume_latest.pt", model=model, optimizer=optimizer, scheduler=scheduler, state=state, settings=settings)

        if state.epochs_since_improvement >= patience:
            print(f"Early stopping after epoch {epoch} due to validation NLL patience.", flush=True)
            break

    if (run_dir / "best_model_by_mrmse.pt").exists():
        best_ckpt = torch.load(run_dir / "best_model_by_mrmse.pt", map_location=device)
        model.load_state_dict(best_ckpt["model"])
        val_metrics = evaluate_point_metric(
            model,
            data.validation,
            data.scalers,
            device=device,
            num_samples=int(settings["evaluation"]["final_num_samples"]),
            point_estimate=metric_point,
            batch_size=eval_batch_size,
            max_rows=None,
            row_seed=int(settings["evaluation"].get("row_seed", 42)),
            sample_noise=True,
            noise_seed=metric_noise_seed,
            log_prefix="validation final",
        )
        holdout_metrics = evaluate_point_metric(
            model,
            data.holdout,
            data.scalers,
            device=device,
            num_samples=int(settings["evaluation"]["final_num_samples"]),
            point_estimate=metric_point,
            batch_size=eval_batch_size,
            max_rows=None,
            row_seed=int(settings["evaluation"].get("row_seed", 42)),
            sample_noise=True,
            noise_seed=metric_noise_seed,
            log_prefix="holdout final",
        )
        save_metrics(run_dir / "validation_metrics.json", val_metrics)
        save_metrics(run_dir / "holdout_metrics.json", holdout_metrics)


if __name__ == "__main__":
    main()
