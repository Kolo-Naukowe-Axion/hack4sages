"""Prepared split datasets for cross-generator FMPE training."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .constants import (
    AUX_FILENAME_TEMPLATE,
    CONTEXT_DIM,
    DATASET_TYPE,
    MANIFEST_FILENAME,
    METADATA_FILENAME_TEMPLATE,
    NOISE_FILENAME_TEMPLATE,
    NORMALIZATION_MODE,
    NORMALIZATION_FILENAME,
    SAFE_AUX_FEATURE_COLS,
    SPECTRA_FILENAME_TEMPLATE,
    SPLIT_SPECS,
    TARGET_COLS,
    TARGET_FILENAME_TEMPLATE,
    THETA_DIM,
)


def _safe_denominator(minimum: np.ndarray, maximum: np.ndarray) -> np.ndarray:
    denominator = maximum - minimum
    return np.where(denominator == 0.0, 1.0, denominator).astype(np.float32)


def _safe_scale(scale: np.ndarray) -> np.ndarray:
    return np.where(scale == 0.0, 1.0, scale).astype(np.float32)


@dataclass(frozen=True)
class NormalizationStats:
    spectra_min: np.ndarray
    spectra_scale: np.ndarray
    noise_min: np.ndarray
    noise_max: np.ndarray
    aux_min: np.ndarray
    aux_max: np.ndarray
    targets_min: np.ndarray
    targets_max: np.ndarray

    @classmethod
    def load(cls, prepared_dir: Path) -> "NormalizationStats":
        manifest = json.loads((prepared_dir / MANIFEST_FILENAME).read_text())
        if manifest.get("dataset_type") != DATASET_TYPE or manifest.get("normalization_mode") != NORMALIZATION_MODE:
            raise RuntimeError(
                "Prepared dataset normalization does not match the upstream sbi-ariel-compatible adapter. "
                "Re-run prepare_dataset with the patched code to rebuild the prepared split directory."
            )
        payload = np.load(prepared_dir / NORMALIZATION_FILENAME)
        if "spectra_scale" not in payload:
            raise RuntimeError(
                "Prepared dataset is missing spectra_scale statistics. Re-run prepare_dataset with the patched code."
            )
        return cls(
            spectra_min=payload["spectra_min"].astype(np.float32),
            spectra_scale=payload["spectra_scale"].astype(np.float32),
            noise_min=payload["noise_min"].astype(np.float32),
            noise_max=payload["noise_max"].astype(np.float32),
            aux_min=payload["aux_min"].astype(np.float32),
            aux_max=payload["aux_max"].astype(np.float32),
            targets_min=payload["targets_min"].astype(np.float32),
            targets_max=payload["targets_max"].astype(np.float32),
        )

    def normalize_spectra(self, values: np.ndarray) -> np.ndarray:
        return ((values - self.spectra_min) / _safe_scale(self.spectra_scale)).astype(np.float32)

    def normalize_noise(self, values: np.ndarray) -> np.ndarray:
        return ((values - self.noise_min) / _safe_denominator(self.noise_min, self.noise_max)).astype(np.float32)

    def normalize_aux(self, values: np.ndarray) -> np.ndarray:
        return ((values - self.aux_min) / _safe_denominator(self.aux_min, self.aux_max)).astype(np.float32)

    def normalize_targets(self, values: np.ndarray) -> np.ndarray:
        return ((values - self.targets_min) / _safe_denominator(self.targets_min, self.targets_max)).astype(np.float32)

    def inverse_targets(self, values: np.ndarray) -> np.ndarray:
        denominator = _safe_denominator(self.targets_min, self.targets_max)
        return (values * denominator + self.targets_min).astype(np.float32)


class CrossGenRealFullNormalizedArielDataset(Dataset):
    """Explicit split dataset with upstream sbi-ariel RealFullNormalized-style normalization."""

    def __init__(self, prepared_dir: str | Path, split_name: str) -> None:
        if split_name not in SPLIT_SPECS:
            raise ValueError(f"Unsupported split '{split_name}'. Expected one of {sorted(SPLIT_SPECS)}.")

        self.prepared_dir = Path(prepared_dir).expanduser().resolve()
        self.split_name = split_name
        self.manifest = json.loads((self.prepared_dir / MANIFEST_FILENAME).read_text())
        self.stats = NormalizationStats.load(self.prepared_dir)
        self.metadata = pd.read_csv(self.prepared_dir / METADATA_FILENAME_TEMPLATE.format(split_name=split_name))

        spectra = np.load(self.prepared_dir / SPECTRA_FILENAME_TEMPLATE.format(split_name=split_name)).astype(np.float32)
        noise_scalar = np.load(self.prepared_dir / NOISE_FILENAME_TEMPLATE.format(split_name=split_name)).astype(np.float32)
        aux = np.load(self.prepared_dir / AUX_FILENAME_TEMPLATE.format(split_name=split_name)).astype(np.float32)
        targets = np.load(self.prepared_dir / TARGET_FILENAME_TEMPLATE.format(split_name=split_name)).astype(np.float32)

        spectra_norm = self.stats.normalize_spectra(spectra)
        noise_norm = self.stats.normalize_noise(noise_scalar)
        aux_norm = self.stats.normalize_aux(aux)
        context = np.concatenate([spectra_norm, noise_norm, aux_norm], axis=1).astype(np.float32)
        theta = self.stats.normalize_targets(targets)

        self.context_raw = context
        self.targets_raw = targets.astype(np.float32)
        self.theta = torch.from_numpy(theta)
        self.context = torch.from_numpy(context)

    def __len__(self) -> int:
        return len(self.theta)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.theta[idx], self.context[idx]

    def inverse_transform_theta(self, values: np.ndarray | torch.Tensor) -> np.ndarray:
        array = values.detach().cpu().numpy() if isinstance(values, torch.Tensor) else np.asarray(values)
        return self.stats.inverse_targets(array.astype(np.float32, copy=False))

    @property
    def sample_ids(self) -> np.ndarray:
        return self.metadata["sample_id"].to_numpy(dtype="U64")


CrossGenRealScalarNoiseNormalizedDataset = CrossGenRealFullNormalizedArielDataset


def resolve_prepared_dir(settings: dict[str, Any], prepared_data_override: Optional[str | Path] = None) -> Path:
    if prepared_data_override is not None:
        return Path(prepared_data_override).expanduser().resolve()
    return Path(settings["dataset"]["path"]).expanduser().resolve()


def load_datasets(
    settings: dict[str, Any],
    prepared_data_override: Optional[str | Path] = None,
) -> dict[str, CrossGenRealFullNormalizedArielDataset]:
    prepared_dir = resolve_prepared_dir(settings, prepared_data_override)
    dataset_type = settings["dataset"].get("type", DATASET_TYPE)
    if dataset_type not in {DATASET_TYPE, "CrossGenRealScalarNoiseNormalizedDataset"}:
        raise ValueError(
            f"Unsupported dataset type '{dataset_type}'. Expected '{DATASET_TYPE}'."
        )
    datasets = {
        "train": CrossGenRealFullNormalizedArielDataset(prepared_dir, settings["dataset"]["train_split"]),
        "validation": CrossGenRealFullNormalizedArielDataset(prepared_dir, settings["dataset"]["validation_split"]),
        "holdout": CrossGenRealFullNormalizedArielDataset(prepared_dir, settings["dataset"]["holdout_split"]),
    }
    settings.setdefault("task", {})
    settings["task"]["dim_theta"] = THETA_DIM
    settings["task"]["dim_x"] = CONTEXT_DIM
    settings["task"]["target_columns"] = TARGET_COLS
    settings["task"]["safe_aux_feature_cols"] = SAFE_AUX_FEATURE_COLS
    settings["dataset"]["path"] = str(prepared_dir)
    return datasets
