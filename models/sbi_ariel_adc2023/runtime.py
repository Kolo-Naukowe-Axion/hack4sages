"""Shared runtime helpers for ADC2023 FMPE training."""

from __future__ import annotations

import gc
import json
import math
import os
import random
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler


class IndexBatchSampler(Sampler[list[int]]):
    def __init__(self, indices: np.ndarray, batch_size: int, start_batch: int = 0) -> None:
        self.indices = np.asarray(indices, dtype=np.int64)
        self.batch_size = int(batch_size)
        self.start_batch = int(start_batch)

    def __iter__(self) -> Iterable[list[int]]:
        start_index = self.start_batch * self.batch_size
        for offset in range(start_index, len(self.indices), self.batch_size):
            yield self.indices[offset : offset + self.batch_size].tolist()

    def __len__(self) -> int:
        remaining = max(0, len(self.indices) - self.start_batch * self.batch_size)
        return math.ceil(remaining / self.batch_size)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_runtime(device: torch.device) -> None:
    if os.name != "nt":
        try:
            torch.multiprocessing.set_sharing_strategy("file_system")
        except (AttributeError, RuntimeError):
            pass
    if device.type != "cuda":
        return
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")


def resolve_device(device_name: str) -> torch.device:
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False.")
    if device_name == "mps":
        if not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but torch.backends.mps.is_available() is False.")
    return torch.device(device_name)


def build_loader_kwargs(
    training_cfg: dict[str, Any],
    *,
    persistent_workers: Optional[bool] = None,
) -> dict[str, Any]:
    num_workers = int(training_cfg.get("num_workers", 0))
    kwargs: dict[str, Any] = {
        "num_workers": num_workers,
        "pin_memory": bool(training_cfg.get("pin_memory", False)),
    }
    if num_workers > 0:
        if persistent_workers is None:
            persistent_workers = bool(training_cfg.get("persistent_workers", False))
        kwargs["persistent_workers"] = bool(persistent_workers)
        if training_cfg.get("prefetch_factor") is not None:
            kwargs["prefetch_factor"] = int(training_cfg["prefetch_factor"])
    return kwargs


def build_eval_loader(dataset, training_cfg: dict[str, Any]) -> DataLoader:
    batch_size = int(training_cfg.get("eval_batch_size", training_cfg["batch_size"]))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        **build_loader_kwargs(training_cfg, persistent_workers=False),
    )


def build_train_loader(dataset, training_cfg: dict[str, Any], seed: int, epoch: int, start_batch: int) -> DataLoader:
    permutation = np.random.default_rng(seed + epoch).permutation(len(dataset)).astype(np.int64)
    batch_sampler = IndexBatchSampler(permutation, int(training_cfg["batch_size"]), start_batch=start_batch)
    return DataLoader(dataset, batch_sampler=batch_sampler, **build_loader_kwargs(training_cfg))


def close_dataloader(dataloader: Optional[DataLoader]) -> None:
    if dataloader is None:
        return
    iterator = getattr(dataloader, "_iterator", None)
    if iterator is not None:
        shutdown_workers = getattr(iterator, "_shutdown_workers", None)
        if shutdown_workers is not None:
            shutdown_workers()
        dataloader._iterator = None
    gc.collect()


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def write_history_row(path: Path, epoch: int, train_loss: float, val_loss: float, lr: float) -> None:
    mode = "a" if path.exists() else "w"
    with path.open(mode, encoding="utf-8") as handle:
        handle.write(f"{epoch}\t{train_loss}\t{val_loss}\t{lr}\n")


def compute_grad_norm(parameters: Iterable[torch.nn.Parameter]) -> float:
    norms: list[torch.Tensor] = []
    for parameter in parameters:
        if parameter.grad is None:
            continue
        norms.append(parameter.grad.detach().norm(2))
    if not norms:
        return 0.0
    return float(torch.norm(torch.stack(norms), 2).item())


def current_lr(optimizer: torch.optim.Optimizer) -> float:
    return float(optimizer.param_groups[0]["lr"])


def step_scheduler(scheduler: Optional[Any], validation_loss: float) -> None:
    if scheduler is None:
        return
    if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
        scheduler.step(validation_loss)
        return
    scheduler.step()


def evaluate_loss(model: Any, dataloader: DataLoader, device: torch.device) -> float:
    total_loss = 0.0
    total_examples = 0
    model.network.eval()
    with torch.no_grad():
        for theta, context in dataloader:
            theta = theta.to(device, non_blocking=device.type == "cuda")
            context = context.to(device, non_blocking=device.type == "cuda")
            loss = model.loss(theta, context)
            batch_examples = len(theta)
            total_loss += float(loss.item()) * batch_examples
            total_examples += batch_examples
    model.network.train()
    return total_loss / max(total_examples, 1)
