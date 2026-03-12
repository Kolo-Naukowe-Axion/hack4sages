"""Dataset loading for prepared TauREx-only five-gas FMPE arrays."""

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
    CONTEXT_DIM,
    CONTEXT_FILENAME_TEMPLATE,
    DATASET_TYPE,
    DEFAULT_PREPARED_DIR,
    HOLDOUT_SPLIT,
    MANIFEST_FILENAME,
    METADATA_FILENAME_TEMPLATE,
    NORMALIZATION_FILENAME,
    NORMALIZATION_MODE,
    RAW_TARGET_FILENAME_TEMPLATE,
    TARGET_COLS,
    TARGET_FILENAME_TEMPLATE,
    TESTDATA_SPLIT,
    THETA_DIM,
    TRAIN_SPLIT,
    VALIDATION_SPLIT,
)


@dataclass(frozen=True)
class NormalizationStats:
    target_mean: np.ndarray
    target_scale: np.ndarray

    @classmethod
    def load(cls, prepared_dir: Path) -> "NormalizationStats":
        manifest = json.loads((prepared_dir / MANIFEST_FILENAME).read_text())
        if manifest.get("dataset_type") != DATASET_TYPE or manifest.get("normalization_mode") != NORMALIZATION_MODE:
            raise RuntimeError(
                "Prepared dataset normalization does not match the TauREx-only five-gas FMPE adapter. "
                "Re-run prepare_dataset with the current code."
            )
        payload = np.load(prepared_dir / NORMALIZATION_FILENAME)
        return cls(
            target_mean=payload["target_mean"].astype(np.float32),
            target_scale=np.where(payload["target_scale"] == 0.0, 1.0, payload["target_scale"]).astype(np.float32),
        )

    def inverse_targets(self, values: np.ndarray) -> np.ndarray:
        return (values * self.target_scale + self.target_mean).astype(np.float32)


class TauRExFiveGasPreparedDataset(Dataset):
    """Prepared labeled split for TauREx-only five-gas FMPE training."""

    def __init__(self, prepared_dir: str | Path, split_name: str) -> None:
        if split_name not in {TRAIN_SPLIT, VALIDATION_SPLIT, HOLDOUT_SPLIT}:
            raise ValueError(f"Unsupported split '{split_name}'.")

        self.prepared_dir = Path(prepared_dir).expanduser().resolve()
        self.split_name = split_name
        self.manifest = json.loads((self.prepared_dir / MANIFEST_FILENAME).read_text())
        self.stats = NormalizationStats.load(self.prepared_dir)
        self.metadata = pd.read_csv(self.prepared_dir / METADATA_FILENAME_TEMPLATE.format(split_name=split_name))
        self.context_raw = np.load(self.prepared_dir / CONTEXT_FILENAME_TEMPLATE.format(split_name=split_name)).astype(np.float32)
        self.theta_raw = np.load(self.prepared_dir / TARGET_FILENAME_TEMPLATE.format(split_name=split_name)).astype(np.float32)
        self.targets_raw = np.load(self.prepared_dir / RAW_TARGET_FILENAME_TEMPLATE.format(split_name=split_name)).astype(
            np.float32
        )
        self.context = torch.from_numpy(self.context_raw)
        self.theta = torch.from_numpy(self.theta_raw)

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


class TauRExFiveGasInferenceDataset:
    """Prepared unlabeled test split for TauREx-only five-gas FMPE inference."""

    def __init__(self, prepared_dir: str | Path) -> None:
        self.prepared_dir = Path(prepared_dir).expanduser().resolve()
        self.manifest = json.loads((self.prepared_dir / MANIFEST_FILENAME).read_text())
        self.stats = NormalizationStats.load(self.prepared_dir)
        self.metadata = pd.read_csv(self.prepared_dir / METADATA_FILENAME_TEMPLATE.format(split_name=TESTDATA_SPLIT))
        self.context_raw = np.load(self.prepared_dir / CONTEXT_FILENAME_TEMPLATE.format(split_name=TESTDATA_SPLIT)).astype(
            np.float32
        )
        self.context = torch.from_numpy(self.context_raw)

    def __len__(self) -> int:
        return int(self.context.shape[0])

    @property
    def sample_ids(self) -> np.ndarray:
        return self.metadata["sample_id"].to_numpy(dtype="U64")

    def inverse_transform_theta(self, values: np.ndarray | torch.Tensor) -> np.ndarray:
        array = values.detach().cpu().numpy() if isinstance(values, torch.Tensor) else np.asarray(values)
        return self.stats.inverse_targets(array.astype(np.float32, copy=False))


def resolve_prepared_dir(settings: dict[str, Any], prepared_data_override: Optional[str | Path] = None) -> Path:
    if prepared_data_override is not None:
        return Path(prepared_data_override).expanduser().resolve()
    dataset_path = settings.get("dataset", {}).get("path", str(DEFAULT_PREPARED_DIR))
    return Path(dataset_path).expanduser().resolve()


def load_datasets(
    settings: dict[str, Any],
    prepared_data_override: Optional[str | Path] = None,
) -> dict[str, Any]:
    prepared_dir = resolve_prepared_dir(settings, prepared_data_override=prepared_data_override)
    manifest = json.loads((prepared_dir / MANIFEST_FILENAME).read_text())
    if manifest.get("dataset_type") != DATASET_TYPE:
        raise ValueError(f"Unsupported dataset type '{manifest.get('dataset_type')}'.")

    datasets = {
        "train": TauRExFiveGasPreparedDataset(prepared_dir, settings["dataset"].get("train_split", TRAIN_SPLIT)),
        "validation": TauRExFiveGasPreparedDataset(
            prepared_dir, settings["dataset"].get("validation_split", VALIDATION_SPLIT)
        ),
        "holdout": TauRExFiveGasPreparedDataset(prepared_dir, settings["dataset"].get("holdout_split", HOLDOUT_SPLIT)),
        "testdata": TauRExFiveGasInferenceDataset(prepared_dir),
    }
    settings.setdefault("task", {})
    settings["task"]["dim_theta"] = THETA_DIM
    settings["task"]["dim_x"] = CONTEXT_DIM
    settings["task"]["target_columns"] = TARGET_COLS
    settings.setdefault("dataset", {})
    settings["dataset"]["path"] = str(prepared_dir)
    return datasets
