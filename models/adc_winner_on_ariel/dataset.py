"""Prepared-data loader for the winner-style five-gas independent NSF model."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .preprocessing import PreparedScalers, RuntimeScalers, build_context, inverse_transform_targets, scalers_to_device, transform_targets


@dataclass
class LoadedSplit:
    planet_id: np.ndarray
    aux_raw: torch.Tensor
    spectra_raw: torch.Tensor
    noise_raw: torch.Tensor
    targets_raw: np.ndarray | None
    targets_scaled: torch.Tensor | None

    @property
    def rows(self) -> int:
        return int(self.aux_raw.shape[0])

    def to_device(self, device: torch.device) -> "LoadedSplit":
        kwargs = {"non_blocking": device.type == "cuda"}
        self.aux_raw = self.aux_raw.to(device, **kwargs)
        self.spectra_raw = self.spectra_raw.to(device, **kwargs)
        self.noise_raw = self.noise_raw.to(device, **kwargs)
        if self.targets_scaled is not None:
            self.targets_scaled = self.targets_scaled.to(device, **kwargs)
        return self


@dataclass
class PreparedData:
    train: LoadedSplit
    validation: LoadedSplit
    holdout: LoadedSplit
    testdata: LoadedSplit
    scalers: PreparedScalers
    runtime_scalers: RuntimeScalers | None = None


def _load_split(path: Path, scalers: PreparedScalers) -> LoadedSplit:
    payload = np.load(path)
    targets_raw = payload["targets_raw"].astype(np.float32) if "targets_raw" in payload else None
    targets_scaled = (
        torch.from_numpy(transform_targets(targets_raw, scalers)).to(dtype=torch.float32) if targets_raw is not None else None
    )
    return LoadedSplit(
        planet_id=payload["planet_id"].astype(np.int64),
        aux_raw=torch.from_numpy(payload["aux_raw"].astype(np.float32)),
        spectra_raw=torch.from_numpy(payload["spectra_raw"].astype(np.float32)),
        noise_raw=torch.from_numpy(payload["noise_raw"].astype(np.float32)),
        targets_raw=targets_raw,
        targets_scaled=targets_scaled,
    )


def load_prepared_data(prepared_root: Path) -> PreparedData:
    prepared_root = prepared_root.expanduser().resolve()
    scalers = PreparedScalers.load(prepared_root / "scalers.npz")
    return PreparedData(
        train=_load_split(prepared_root / "train.npz", scalers),
        validation=_load_split(prepared_root / "validation.npz", scalers),
        holdout=_load_split(prepared_root / "holdout.npz", scalers),
        testdata=_load_split(prepared_root / "testdata.npz", scalers),
        scalers=scalers,
    )


def move_prepared_data_to_device(data: PreparedData, device: torch.device) -> PreparedData:
    data.train.to_device(device)
    data.validation.to_device(device)
    data.holdout.to_device(device)
    data.testdata.to_device(device)
    data.runtime_scalers = scalers_to_device(data.scalers, device)
    return data


def build_context_batch(
    split: LoadedSplit,
    indices: torch.Tensor,
    scalers: PreparedScalers | RuntimeScalers,
    *,
    device: torch.device,
    sample_noise: bool,
    noise_generator: torch.Generator | None = None,
) -> torch.Tensor:
    indices = indices.to(split.aux_raw.device, non_blocking=device.type == "cuda")
    aux = split.aux_raw.index_select(0, indices)
    spectra = split.spectra_raw.index_select(0, indices)
    noise = split.noise_raw.index_select(0, indices)
    if aux.device != device:
        aux = aux.to(device, non_blocking=device.type == "cuda")
        spectra = spectra.to(device, non_blocking=device.type == "cuda")
        noise = noise.to(device, non_blocking=device.type == "cuda")
    return build_context(aux, spectra, noise, scalers, sample_noise=sample_noise, noise_generator=noise_generator)


def inverse_targets_batch(values: np.ndarray, scalers: PreparedScalers) -> np.ndarray:
    return inverse_transform_targets(values, scalers)
