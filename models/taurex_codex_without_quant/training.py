"""Training loop for the non-quantum TauREx atmospheric regressor."""

from __future__ import annotations

import json
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from .constants import AUX_COLUMNS, DEFAULT_DATA_ROOT, DEFAULT_OUTPUT_DIR, TARGET_COLUMNS
from .dataset import InferenceSplit, LabeledSplit, PreparedData, prepare_data
from .model import ModelConfig, TauRExNoQuantRegressor, build_model


def resolve_project_root(path_hint: Optional[Path] = None) -> Path:
    candidate = (path_hint or Path.cwd()).resolve()
    if (candidate / "data").exists():
        return candidate
    if candidate.name == "models" and (candidate.parent / "data").exists():
        return candidate.parent
    return candidate


@dataclass
class TrainingConfig:
    project_root: str = "."
    data_root: str = str(DEFAULT_DATA_ROOT)
    output_dir: str = str(DEFAULT_OUTPUT_DIR)
    prepared_cache_dir: Optional[str] = None
    dataset_format: str = "auto"
    init_checkpoint_path: Optional[str] = None
    seed: int = 42
    batch_size: int = 256
    eval_batch_size: int = 512
    max_epochs: int = 60
    early_stop_patience: int = 12
    scheduler_patience: int = 4
    scheduler_factor: float = 0.5
    classical_lr: float = 1.0e-3
    quantum_lr: float = 5.0e-4
    weight_decay: float = 1.0e-4
    gradient_clip_norm: float = 5.0
    dropout: float = 0.05
    loss_name: str = "mse"
    qnn_qubits: int = 8
    qnn_depth: int = 2
    qnn_init_scale: float = 0.1
    quantum_device: str = "classical_refiner"
    quantum_use_async: bool = False
    classical_only: bool = False
    quantum_warmup_epochs: int = 3
    quantum_ramp_epochs: int = 6
    quantum_backbone_freeze_epochs: int = 0
    use_amp: bool = True
    log_every_batches: int = 20
    train_limit: Optional[int] = None
    val_limit: Optional[int] = None
    holdout_limit: Optional[int] = None
    test_limit: Optional[int] = None
    taurex_ignore_poseidon: bool = False

    def resolved_project_root(self) -> Path:
        return resolve_project_root(Path(self.project_root))

    def resolved_data_root(self) -> Path:
        root = Path(self.data_root).expanduser()
        if root.is_absolute():
            return root
        return self.resolved_project_root() / root

    def resolved_output_dir(self) -> Path:
        root = Path(self.output_dir).expanduser()
        if root.is_absolute():
            return root
        return self.resolved_project_root() / root

    def resolved_init_checkpoint_path(self) -> Optional[str]:
        if self.init_checkpoint_path is None:
            return None
        path = Path(self.init_checkpoint_path).expanduser()
        if path.is_absolute():
            return str(path)
        return str((self.resolved_project_root() / path).resolve())

    def resolved_prepared_cache_dir(self) -> Optional[str]:
        if self.prepared_cache_dir is None:
            return None
        root = Path(self.prepared_cache_dir).expanduser()
        if root.is_absolute():
            return str(root)
        return str(self.resolved_output_dir() / root)

    def to_json_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        refinement_width = max(32, int(self.qnn_qubits) * 4)
        payload["project_root"] = str(self.resolved_project_root())
        payload["data_root"] = str(self.resolved_data_root())
        payload["output_dir"] = str(self.resolved_output_dir())
        payload["prepared_cache_dir"] = self.resolved_prepared_cache_dir()
        payload["init_checkpoint_path"] = self.resolved_init_checkpoint_path()
        payload["aux_columns"] = AUX_COLUMNS
        payload["target_columns"] = TARGET_COLUMNS
        payload["refinement_width"] = refinement_width
        payload["refinement_layers"] = int(self.qnn_depth)
        payload["refinement_lr"] = float(self.quantum_lr)
        return payload


def set_runtime_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_runtime() -> None:
    requested_threads = os.environ.get("OMP_NUM_THREADS")
    if requested_threads:
        cpu_threads = max(1, int(requested_threads))
    elif torch.cuda.is_available():
        cpu_threads = max(1, min(os.cpu_count() or 1, 16))
    else:
        cpu_threads = max(1, min(os.cpu_count() or 1, 16))
    try:
        torch.set_num_threads(cpu_threads)
    except RuntimeError:
        pass
    try:
        torch.set_num_interop_threads(max(1, min(cpu_threads // 2, 8)))
    except RuntimeError:
        pass
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")


def resolve_training_device(config: TrainingConfig) -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def move_split_to_device(split: LabeledSplit | InferenceSplit, device: torch.device) -> LabeledSplit | InferenceSplit:
    kwargs = {"non_blocking": device.type == "cuda"}
    split.aux = split.aux.to(device, **kwargs)
    split.spectra = split.spectra.to(device, **kwargs)
    if isinstance(split, LabeledSplit):
        split.targets = split.targets.to(device, **kwargs)
    return split


def move_prepared_data_to_device(data: PreparedData, device: torch.device) -> PreparedData:
    move_split_to_device(data.train, device)
    move_split_to_device(data.val, device)
    move_split_to_device(data.holdout, device)
    move_split_to_device(data.testdata, device)
    return data


def batch_indices(length: int, batch_size: int, seed: int, epoch: int, device: torch.device) -> Iterable[torch.Tensor]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + epoch)
    permutation = torch.randperm(length, generator=generator)
    if device.type == "cuda":
        permutation = permutation.to(device, non_blocking=True)
    for start in range(0, length, batch_size):
        yield permutation[start : start + batch_size]


def maybe_sync_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def gather_labeled_batch(split: LabeledSplit, indices: torch.Tensor, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if indices.device != split.aux.device:
        indices = indices.to(split.aux.device, non_blocking=split.aux.device.type == "cuda")
    aux = split.aux.index_select(0, indices)
    spectra = split.spectra.index_select(0, indices)
    targets = split.targets.index_select(0, indices)
    if aux.device != device:
        aux = aux.to(device, non_blocking=device.type == "cuda")
        spectra = spectra.to(device, non_blocking=device.type == "cuda")
        targets = targets.to(device, non_blocking=device.type == "cuda")
    return aux, spectra, targets


def gather_inference_batch(split: InferenceSplit, start: int, stop: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    indices = torch.arange(start, stop, device=split.aux.device)
    aux = split.aux.index_select(0, indices)
    spectra = split.spectra.index_select(0, indices)
    if aux.device != device:
        aux = aux.to(device, non_blocking=device.type == "cuda")
        spectra = spectra.to(device, non_blocking=device.type == "cuda")
    return aux, spectra


def format_cuda_memory(device: torch.device) -> str:
    if device.type != "cuda":
        return "CUDA memory: disabled"
    allocated = torch.cuda.memory_allocated(device) / (1024 * 1024)
    reserved = torch.cuda.memory_reserved(device) / (1024 * 1024)
    peak_allocated = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
    peak_reserved = torch.cuda.max_memory_reserved(device) / (1024 * 1024)
    return (
        f"CUDA memory | allocated={allocated:.1f}MB | reserved={reserved:.1f}MB | "
        f"peak_allocated={peak_allocated:.1f}MB | peak_reserved={peak_reserved:.1f}MB"
    )


class OriginalScaleMSELoss(nn.Module):
    def __init__(self, target_scale: np.ndarray) -> None:
        super().__init__()
        self.register_buffer("target_scale", torch.as_tensor(target_scale, dtype=torch.float32))

    def forward(self, pred_scaled: torch.Tensor, target_scaled: torch.Tensor) -> torch.Tensor:
        delta = (pred_scaled - target_scaled) * self.target_scale
        return delta.square().mean()


class OriginalScaleHuberLoss(nn.Module):
    def __init__(self, target_scale: np.ndarray, delta: float = 1.0) -> None:
        super().__init__()
        self.register_buffer("target_scale", torch.as_tensor(target_scale, dtype=torch.float32))
        self.delta = float(delta)

    def forward(self, pred_scaled: torch.Tensor, target_scaled: torch.Tensor) -> torch.Tensor:
        delta = torch.abs((pred_scaled - target_scaled) * self.target_scale)
        quadratic = torch.minimum(delta, torch.full_like(delta, self.delta))
        linear = delta - quadratic
        return (0.5 * quadratic.square() + self.delta * linear).mean()


class OriginalScaleMRMSELoss(nn.Module):
    def __init__(self, target_scale: np.ndarray, eps: float = 1.0e-8) -> None:
        super().__init__()
        self.register_buffer("target_scale", torch.as_tensor(target_scale, dtype=torch.float32))
        self.eps = float(eps)

    def forward(self, pred_scaled: torch.Tensor, target_scaled: torch.Tensor) -> torch.Tensor:
        delta = (pred_scaled - target_scaled) * self.target_scale
        per_target_rmse = torch.sqrt(delta.square().mean(dim=0) + self.eps)
        return per_target_rmse.mean()


def build_loss_fn(config: TrainingConfig, data: PreparedData) -> nn.Module:
    loss_name = config.loss_name.strip().lower()
    if loss_name == "mse":
        return OriginalScaleMSELoss(data.target_scaler.scale)
    if loss_name == "huber":
        return OriginalScaleHuberLoss(data.target_scaler.scale, delta=1.0)
    if loss_name == "mrmse":
        return OriginalScaleMRMSELoss(data.target_scaler.scale)
    raise ValueError(f"Unsupported loss_name={config.loss_name!r}. Expected one of: mse, huber, mrmse.")


def evaluate_labeled_split(
    model: TauRExNoQuantRegressor,
    split: LabeledSplit,
    target_scaler,
    batch_size: int,
    loss_fn: nn.Module,
    enable_refinement: bool = True,
    refinement_scale: float = 1.0,
) -> dict[str, Any]:
    model.eval()
    pred_batches = []
    target_batches = []
    losses = []

    with torch.inference_mode():
        for start in range(0, split.rows, batch_size):
            stop = min(start + batch_size, split.rows)
            aux, spectra, targets = gather_labeled_batch(
                split,
                torch.arange(start, stop, device=split.aux.device),
                model.device_ref,
            )
            pred = model(aux, spectra, enable_refinement=enable_refinement, refinement_scale=refinement_scale)
            pred_batches.append(pred.detach().cpu().numpy())
            target_batches.append(targets.detach().cpu().numpy())
            losses.append(float(loss_fn(pred, targets).item()))

    pred_scaled = np.concatenate(pred_batches, axis=0)
    true_scaled = np.concatenate(target_batches, axis=0)
    pred_orig = target_scaler.inverse_transform(pred_scaled)
    true_orig = target_scaler.inverse_transform(true_scaled)
    rmse_orig = np.sqrt(np.mean((pred_orig - true_orig) ** 2, axis=0))
    mae_orig = np.mean(np.abs(pred_orig - true_orig), axis=0)
    return {
        "loss": float(np.mean(losses)),
        "pred_scaled": pred_scaled,
        "true_scaled": true_scaled,
        "pred_orig": pred_orig,
        "true_orig": true_orig,
        "rmse_orig": rmse_orig,
        "mae_orig": mae_orig,
        "rmse_mean": float(rmse_orig.mean()),
        "mae_mean": float(mae_orig.mean()),
    }


def predict_inference_split(
    model: TauRExNoQuantRegressor,
    split: InferenceSplit,
    target_scaler,
    batch_size: int,
    enable_refinement: bool = True,
    refinement_scale: float = 1.0,
) -> np.ndarray:
    model.eval()
    batches = []
    with torch.inference_mode():
        for start in range(0, split.rows, batch_size):
            stop = min(start + batch_size, split.rows)
            aux, spectra = gather_inference_batch(split, start, stop, model.device_ref)
            batches.append(
                model(aux, spectra, enable_refinement=enable_refinement, refinement_scale=refinement_scale)
                .detach()
                .cpu()
                .numpy()
            )
    pred_scaled = np.concatenate(batches, axis=0)
    return target_scaler.inverse_transform(pred_scaled)


def format_target_vector(values: np.ndarray) -> str:
    return " | ".join(f"{name}={value:.4f}" for name, value in zip(TARGET_COLUMNS, values))


def summarize_epoch_predictions(target_scaler, pred_batches: list[np.ndarray], true_batches: list[np.ndarray]) -> dict[str, Any]:
    pred_scaled = np.concatenate(pred_batches, axis=0)
    true_scaled = np.concatenate(true_batches, axis=0)
    pred_orig = target_scaler.inverse_transform(pred_scaled)
    true_orig = target_scaler.inverse_transform(true_scaled)
    rmse_orig = np.sqrt(np.mean((pred_orig - true_orig) ** 2, axis=0))
    mae_orig = np.mean(np.abs(pred_orig - true_orig), axis=0)
    return {
        "pred_orig": pred_orig,
        "true_orig": true_orig,
        "rmse_orig": rmse_orig,
        "mae_orig": mae_orig,
        "rmse_mean": float(rmse_orig.mean()),
        "mae_mean": float(mae_orig.mean()),
    }


def resolve_refinement_scale(config: TrainingConfig, epoch_index: int) -> float:
    if config.classical_only or epoch_index < config.quantum_warmup_epochs:
        return 0.0
    if config.quantum_ramp_epochs <= 1:
        return 1.0
    active_epoch = epoch_index - config.quantum_warmup_epochs + 1
    return float(min(1.0, active_epoch / config.quantum_ramp_epochs))


def resolve_refinement_active_epoch(config: TrainingConfig, epoch_index: int) -> int:
    if config.classical_only or epoch_index < config.quantum_warmup_epochs:
        return 0
    return int(epoch_index - config.quantum_warmup_epochs + 1)


def save_training_progress(
    output_dir: Path,
    config: TrainingConfig,
    history: list[dict[str, Any]],
    best_state: Optional[dict[str, torch.Tensor]],
    last_state: dict[str, torch.Tensor],
    best_epoch: int,
    best_val_rmse: float,
    best_refinement_scale: float,
    epoch: int,
    refinement_active: bool,
    refinement_scale: float,
    backbone_frozen: bool,
) -> None:
    history_frame = pd.DataFrame(history)
    history_frame.to_csv(output_dir / "history.csv", index=False)
    save_json(
        output_dir / "training_state.json",
        {
            "current_epoch": int(epoch),
            "best_epoch": int(best_epoch),
            "best_val_rmse_mean": float(best_val_rmse),
            "best_refinement_scale": float(best_refinement_scale),
            "refinement_active": bool(refinement_active),
            "refinement_scale": float(refinement_scale),
            "backbone_frozen": bool(backbone_frozen),
        },
    )
    torch.save(
        {
            "config": config.to_json_dict(),
            "feature_cols": AUX_COLUMNS,
            "target_cols": TARGET_COLUMNS,
            "model_state_dict": last_state,
        },
        output_dir / "last_model.pt",
    )
    if best_state is not None:
        torch.save(
            {
                "config": config.to_json_dict(),
                "feature_cols": AUX_COLUMNS,
                "target_cols": TARGET_COLUMNS,
                "best_epoch": int(best_epoch),
                "best_val_rmse": float(best_val_rmse),
                "best_refinement_scale": float(best_refinement_scale),
                "model_state_dict": best_state,
            },
            output_dir / "best_model.pt",
        )


def maybe_initialize_from_checkpoint(model: TauRExNoQuantRegressor, config: TrainingConfig) -> None:
    checkpoint_path = config.resolved_init_checkpoint_path()
    if checkpoint_path is None:
        return

    payload = torch.load(checkpoint_path, map_location="cpu")
    state_dict = payload.get("model_state_dict", payload)
    model_state = model.state_dict()
    filtered_state: dict[str, torch.Tensor] = {}
    skipped_shape_keys: list[str] = []
    for key, value in state_dict.items():
        target = model_state.get(key)
        if target is None:
            continue
        if target.shape != value.shape:
            skipped_shape_keys.append(key)
            continue
        filtered_state[key] = value

    load_result = model.load_state_dict(filtered_state, strict=False)
    print(
        f"Initialized from checkpoint: {checkpoint_path} | loaded_keys={len(filtered_state)} | "
        f"missing={len(load_result.missing_keys)} | unexpected={len(load_result.unexpected_keys)}",
        flush=True,
    )
    if skipped_shape_keys:
        print(f"Skipped shape-mismatched checkpoint keys: {', '.join(skipped_shape_keys[:12])}", flush=True)


def train_model(config: TrainingConfig, data: PreparedData, device: torch.device, output_dir: Optional[Path] = None) -> dict[str, Any]:
    model = build_model(
        ModelConfig(
            spectral_input_channels=int(data.train.spectra.shape[1]),
            dropout=config.dropout,
            refinement_width=max(32, int(config.qnn_qubits) * 4),
            refinement_layers=max(1, int(config.qnn_depth)),
            classical_only=config.classical_only,
            use_amp=config.use_amp,
        ),
        device,
    )
    maybe_initialize_from_checkpoint(model, config)
    backbone_params = list(model.backbone_parameters())
    refinement_params = list(model.refinement_parameters())
    loss_fn = build_loss_fn(config, data).to(device)

    optimizer_param_groups = [
        {"params": backbone_params, "lr": config.classical_lr, "weight_decay": config.weight_decay},
    ]
    if refinement_params:
        optimizer_param_groups.append(
            {"params": refinement_params, "lr": config.quantum_lr, "weight_decay": config.weight_decay}
        )
    optimizer = torch.optim.AdamW(optimizer_param_groups, betas=(0.9, 0.95))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config.scheduler_factor,
        patience=config.scheduler_patience,
    )

    history: list[dict[str, Any]] = []
    best_val_rmse = float("inf")
    best_epoch = -1
    best_refinement_scale = 0.0
    best_state: Optional[dict[str, torch.Tensor]] = None
    last_state = {name: value.detach().cpu() for name, value in model.state_dict().items()}
    patience_left = config.early_stop_patience
    total_batches = (len(data.train.aux) + config.batch_size - 1) // config.batch_size

    print(f"Torch device: {device} | classical_only={config.classical_only}", flush=True)
    print(
        f"Loss: {config.loss_name} | refinement_width={max(32, int(config.qnn_qubits) * 4)} | refinement_layers={config.qnn_depth} | "
        f"warmup_epochs={0 if config.classical_only else config.quantum_warmup_epochs} | "
        f"ramp_epochs={0 if config.classical_only else config.quantum_ramp_epochs} | "
        f"freeze_epochs={0 if config.classical_only else config.quantum_backbone_freeze_epochs}",
        flush=True,
    )
    if device.type == "cuda":
        print(f"Torch CUDA device: {torch.cuda.get_device_name(device)}", flush=True)
        print(format_cuda_memory(device), flush=True)

    for epoch in range(config.max_epochs):
        model.train()
        refinement_active_epoch = resolve_refinement_active_epoch(config, epoch)
        refinement_scale = resolve_refinement_scale(config, epoch)
        refinement_active = refinement_scale > 0.0
        backbone_frozen = (
            not config.classical_only
            and refinement_active_epoch > 0
            and refinement_active_epoch <= config.quantum_backbone_freeze_epochs
        )
        model.set_backbone_trainable(not backbone_frozen)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        epoch_start = time.perf_counter()
        batch_losses = []
        train_pred_batches: list[np.ndarray] = []
        train_true_batches: list[np.ndarray] = []

        for batch_index, indices in enumerate(
            batch_indices(data.train.rows, config.batch_size, config.seed, epoch, data.train.aux.device),
            start=1,
        ):
            batch_start = time.perf_counter()
            aux, spectra, targets = gather_labeled_batch(data.train, indices, device)
            optimizer.zero_grad(set_to_none=True)
            maybe_sync_cuda(device)
            pred = model(aux, spectra, enable_refinement=refinement_active, refinement_scale=refinement_scale)
            loss = loss_fn(pred, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(backbone_params, config.gradient_clip_norm)
            if refinement_params:
                torch.nn.utils.clip_grad_norm_(refinement_params, config.gradient_clip_norm)
            optimizer.step()
            maybe_sync_cuda(device)
            batch_losses.append(float(loss.item()))
            train_pred_batches.append(pred.detach().cpu().numpy())
            train_true_batches.append(targets.detach().cpu().numpy())

            if config.log_every_batches > 0 and (
                batch_index % config.log_every_batches == 0 or batch_index == total_batches
            ):
                print(
                    f"Epoch {epoch + 1}/{config.max_epochs} | Batch {batch_index}/{total_batches} | "
                    f"batch_loss={batch_losses[-1]:.5f} | avg_loss={np.mean(batch_losses):.5f} | "
                    f"batch_time={time.perf_counter() - batch_start:.2f}s",
                    flush=True,
                )

        train_metrics = summarize_epoch_predictions(data.target_scaler, train_pred_batches, train_true_batches)
        val_metrics = evaluate_labeled_split(
            model,
            data.val,
            data.target_scaler,
            config.eval_batch_size,
            loss_fn,
            enable_refinement=refinement_active,
            refinement_scale=refinement_scale,
        )
        scheduler.step(val_metrics["rmse_mean"])
        epoch_seconds = time.perf_counter() - epoch_start

        backbone_lr = optimizer.param_groups[0]["lr"]
        refinement_lr = optimizer.param_groups[1]["lr"] if len(optimizer.param_groups) > 1 else 0.0
        history_row = {
            "epoch": epoch + 1,
            "train_loss": float(np.mean(batch_losses)),
            "train_rmse_mean": float(train_metrics["rmse_mean"]),
            "train_mae_mean": float(train_metrics["mae_mean"]),
            "val_loss": float(val_metrics["loss"]),
            "val_rmse_mean": float(val_metrics["rmse_mean"]),
            "val_mae_mean": float(val_metrics["mae_mean"]),
            "epoch_seconds": epoch_seconds,
            "backbone_lr": backbone_lr,
            "refinement_lr": refinement_lr,
            "refinement_active": int(refinement_active),
            "refinement_scale": float(refinement_scale),
            "backbone_frozen": int(backbone_frozen),
        }
        for target_name, rmse_value in zip(TARGET_COLUMNS, train_metrics["rmse_orig"]):
            history_row[f"train_rmse_{target_name}"] = float(rmse_value)
        for target_name, mae_value in zip(TARGET_COLUMNS, train_metrics["mae_orig"]):
            history_row[f"train_mae_{target_name}"] = float(mae_value)
        for target_name, rmse_value in zip(TARGET_COLUMNS, val_metrics["rmse_orig"]):
            history_row[f"val_rmse_{target_name}"] = float(rmse_value)
        for target_name, mae_value in zip(TARGET_COLUMNS, val_metrics["mae_orig"]):
            history_row[f"val_mae_{target_name}"] = float(mae_value)
        history.append(history_row)

        print(
            f"Epoch {epoch + 1}/{config.max_epochs} | train_loss={history_row['train_loss']:.5f} | "
            f"train_rmse_mean={history_row['train_rmse_mean']:.5f} | "
            f"val_rmse_mean={history_row['val_rmse_mean']:.5f} | "
            f"val_mae_mean={history_row['val_mae_mean']:.5f} | "
            f"time={epoch_seconds:.1f}s | lr=({backbone_lr:.2e}, {refinement_lr:.2e}) | "
            f"refinement_active={refinement_active} | refinement_scale={refinement_scale:.2f} | "
            f"backbone_frozen={backbone_frozen}",
            flush=True,
        )
        print(f"Train RMSE      | {format_target_vector(train_metrics['rmse_orig'])}", flush=True)
        print(f"Validation RMSE | {format_target_vector(val_metrics['rmse_orig'])}", flush=True)
        print(f"Validation MAE  | {format_target_vector(val_metrics['mae_orig'])}", flush=True)
        if device.type == "cuda":
            print(f"Epoch {epoch + 1} memory | {format_cuda_memory(device)}", flush=True)

        if val_metrics["rmse_mean"] < best_val_rmse:
            best_val_rmse = float(val_metrics["rmse_mean"])
            best_epoch = epoch + 1
            best_refinement_scale = float(refinement_scale)
            best_state = {name: value.detach().cpu() for name, value in model.state_dict().items()}
            patience_left = config.early_stop_patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"Early stopping at epoch {epoch + 1}.", flush=True)
                last_state = {name: value.detach().cpu() for name, value in model.state_dict().items()}
                if output_dir is not None:
                    save_training_progress(
                        output_dir,
                        config,
                        history,
                        best_state,
                        last_state,
                        best_epoch,
                        best_val_rmse,
                        best_refinement_scale,
                        epoch + 1,
                        refinement_active,
                        refinement_scale,
                        backbone_frozen,
                    )
                break

        last_state = {name: value.detach().cpu() for name, value in model.state_dict().items()}
        if output_dir is not None:
            save_training_progress(
                output_dir,
                config,
                history,
                best_state,
                last_state,
                best_epoch,
                best_val_rmse,
                best_refinement_scale,
                epoch + 1,
                refinement_active,
                refinement_scale,
                backbone_frozen,
            )

    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint.")

    model.set_backbone_trainable(True)
    model.load_state_dict(best_state)
    return {
        "model": model,
        "history": history,
        "best_epoch": best_epoch,
        "best_val_rmse": best_val_rmse,
        "best_refinement_scale": best_refinement_scale,
        "best_state": best_state,
        "last_state": last_state,
    }


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def metrics_payload(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "rows": int(metrics["pred_orig"].shape[0]),
        "rmse_mean": float(metrics["rmse_mean"]),
        "mae_mean": float(metrics["mae_mean"]),
        "rmse": {name: float(value) for name, value in zip(TARGET_COLUMNS, metrics["rmse_orig"])},
        "mae": {name: float(value) for name, value in zip(TARGET_COLUMNS, metrics["mae_orig"])},
    }


def save_labeled_predictions(path: Path, split: LabeledSplit, metrics: dict[str, Any]) -> None:
    frame = pd.DataFrame({"planet_ID": split.planet_ids.tolist()})
    for index, target_name in enumerate(TARGET_COLUMNS):
        frame[f"true_{target_name}"] = metrics["true_orig"][:, index]
        frame[f"pred_{target_name}"] = metrics["pred_orig"][:, index]
    frame.to_csv(path, index=False)


def save_inference_predictions(path: Path, split: InferenceSplit, predictions: np.ndarray) -> None:
    frame = pd.DataFrame({"planet_ID": split.planet_ids.tolist()})
    for index, target_name in enumerate(TARGET_COLUMNS):
        frame[target_name] = predictions[:, index]
    frame.to_csv(path, index=False)


def run_training_experiment(config: Optional[TrainingConfig] = None) -> dict[str, Any]:
    cfg = config or TrainingConfig()
    cfg.project_root = str(cfg.resolved_project_root())

    set_runtime_seed(cfg.seed)
    configure_runtime()
    device = resolve_training_device(cfg)

    output_dir = cfg.resolved_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Project root: {cfg.resolved_project_root()}", flush=True)
    print(f"Data root: {cfg.resolved_data_root()}", flush=True)
    print(f"Output dir: {output_dir}", flush=True)
    print(f"Dataset format: {cfg.dataset_format}", flush=True)
    print(f"Batch size: {cfg.batch_size}", flush=True)
    print(f"Eval batch size: {cfg.eval_batch_size}", flush=True)
    print(f"Refinement width / layers: {max(32, int(cfg.qnn_qubits) * 4)}/{cfg.qnn_depth}", flush=True)
    print(f"Loss: {cfg.loss_name}", flush=True)
    print(f"Ignore poseidon: {cfg.taurex_ignore_poseidon}", flush=True)
    print(f"Prepared cache dir: {cfg.resolved_prepared_cache_dir() or str(output_dir / 'prepared_cache')}", flush=True)

    save_json(output_dir / "config.json", cfg.to_json_dict())

    data = prepare_data(
        data_root=cfg.resolved_data_root(),
        output_dir=output_dir,
        prepared_cache_dir=cfg.resolved_prepared_cache_dir(),
        dataset_format=cfg.dataset_format,
        seed=cfg.seed,
        train_limit=cfg.train_limit,
        val_limit=cfg.val_limit,
        holdout_limit=cfg.holdout_limit,
        test_limit=cfg.test_limit,
        taurex_ignore_poseidon=cfg.taurex_ignore_poseidon,
    )
    save_json(output_dir / "split_manifest.json", data.split_manifest)
    save_json(output_dir / "prepared_manifest.json", data.prepared_manifest)

    data = move_prepared_data_to_device(data, device)
    training = train_model(cfg, data, device, output_dir=output_dir)
    model: TauRExNoQuantRegressor = training["model"]
    loss_fn = build_loss_fn(cfg, data).to(device)

    save_json(
        output_dir / "scalers.json",
        {
            "aux_scaler": data.aux_scaler.state_dict(),
            "target_scaler": data.target_scaler.state_dict(),
            "spectral_scaler": data.spectral_scaler.state_dict(),
        },
    )

    final_refinement_scale = 0.0 if cfg.classical_only else float(training["best_refinement_scale"])
    validation_metrics = evaluate_labeled_split(
        model, data.val, data.target_scaler, cfg.eval_batch_size, loss_fn, enable_refinement=not cfg.classical_only, refinement_scale=final_refinement_scale
    )
    holdout_metrics = evaluate_labeled_split(
        model, data.holdout, data.target_scaler, cfg.eval_batch_size, loss_fn, enable_refinement=not cfg.classical_only, refinement_scale=final_refinement_scale
    )
    test_predictions = predict_inference_split(
        model, data.testdata, data.target_scaler, cfg.eval_batch_size, enable_refinement=not cfg.classical_only, refinement_scale=final_refinement_scale
    )

    save_json(output_dir / "validation_metrics.json", metrics_payload(validation_metrics))
    save_json(output_dir / "holdout_metrics.json", metrics_payload(holdout_metrics))
    save_labeled_predictions(output_dir / "validation_predictions.csv", data.val, validation_metrics)
    save_labeled_predictions(output_dir / "holdout_predictions.csv", data.holdout, holdout_metrics)
    save_inference_predictions(output_dir / "testdata_predictions.csv", data.testdata, test_predictions)

    summary = {
        "best_epoch": training["best_epoch"],
        "best_val_rmse_mean": float(training["best_val_rmse"]),
        "best_refinement_scale": float(training["best_refinement_scale"]),
        "validation_rmse_mean": float(validation_metrics["rmse_mean"]),
        "validation_mae_mean": float(validation_metrics["mae_mean"]),
        "holdout_rmse_mean": float(holdout_metrics["rmse_mean"]),
        "holdout_mae_mean": float(holdout_metrics["mae_mean"]),
        "testdata_rows": int(data.testdata.rows),
        "output_dir": str(output_dir),
        "dataset": data.split_manifest,
    }
    save_json(output_dir / "run_summary.json", summary)

    return {
        "summary": summary,
        "validation_metrics": validation_metrics,
        "holdout_metrics": holdout_metrics,
        "test_predictions": test_predictions,
    }
