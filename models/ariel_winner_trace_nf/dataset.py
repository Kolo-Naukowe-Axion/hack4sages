"""Prepared-data loading and batch helpers for the Ariel winner-family rerun."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .preprocessing import PreparedScalers, inverse_transform_targets_numpy, transform_targets_torch


@dataclass
class LoadedSplit:
    planet_id: np.ndarray
    context: torch.Tensor
    radius_reference: torch.Tensor
    fm_targets_raw: np.ndarray | None
    quartiles_raw: np.ndarray | None
    trace_targets_raw: torch.Tensor | None
    trace_weights: torch.Tensor | None

    @property
    def rows(self) -> int:
        return int(self.context.shape[0])


@dataclass
class PreparedData:
    train: LoadedSplit
    validation: LoadedSplit
    holdout: LoadedSplit
    testdata: LoadedSplit
    scalers: PreparedScalers
    manifest: dict


def _load_split(path: Path) -> LoadedSplit:
    payload = np.load(path, allow_pickle=False)
    fm_targets_raw = payload["fm_targets_raw"].astype(np.float32) if "fm_targets_raw" in payload else None
    quartiles_raw = payload["quartiles_raw"].astype(np.float32) if "quartiles_raw" in payload else None
    trace_targets_raw = torch.from_numpy(payload["trace_targets_raw"].astype(np.float32)) if "trace_targets_raw" in payload else None
    trace_weights = torch.from_numpy(payload["trace_weights"].astype(np.float32)) if "trace_weights" in payload else None
    return LoadedSplit(
        planet_id=payload["planet_id"].astype("U32"),
        context=torch.from_numpy(payload["context"].astype(np.float32)),
        radius_reference=torch.from_numpy(payload["radius_reference"].astype(np.float32)),
        fm_targets_raw=fm_targets_raw,
        quartiles_raw=quartiles_raw,
        trace_targets_raw=trace_targets_raw,
        trace_weights=trace_weights,
    )


def load_prepared_data(prepared_root: Path) -> PreparedData:
    prepared_root = prepared_root.expanduser().resolve()
    manifest = {}
    manifest_path = prepared_root / "manifest.json"
    if manifest_path.is_file():
        import json

        manifest = json.loads(manifest_path.read_text())
    return PreparedData(
        train=_load_split(prepared_root / "train.npz"),
        validation=_load_split(prepared_root / "validation.npz"),
        holdout=_load_split(prepared_root / "holdout.npz"),
        testdata=_load_split(prepared_root / "testdata.npz"),
        scalers=PreparedScalers.load(prepared_root / "scalers.npz"),
        manifest=manifest,
    )


def build_trace_batch(
    split: LoadedSplit,
    indices: torch.Tensor,
    scalers: PreparedScalers,
    *,
    device: torch.device,
    target_index: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    cpu_indices = indices.to(dtype=torch.long, device="cpu")
    non_blocking = device.type == "cuda"

    context = split.context.index_select(0, cpu_indices).to(device, non_blocking=non_blocking)
    radius_reference = split.radius_reference.index_select(0, cpu_indices).to(device, non_blocking=non_blocking)

    if split.trace_targets_raw is None or split.trace_weights is None:
        raise ValueError("Requested trace batch from an unlabeled split.")

    trace_targets = split.trace_targets_raw.index_select(0, cpu_indices).to(device, non_blocking=non_blocking)
    trace_weights = split.trace_weights.index_select(0, cpu_indices).to(device, non_blocking=non_blocking)
    trace_targets = transform_targets_torch(trace_targets, radius_reference, scalers)
    if target_index is not None:
        trace_targets = trace_targets[..., target_index : target_index + 1]
    return context, trace_targets, trace_weights


def inverse_targets_batch(values: np.ndarray, radius_reference: np.ndarray, scalers: PreparedScalers) -> np.ndarray:
    return inverse_transform_targets_numpy(values, radius_reference, scalers)

