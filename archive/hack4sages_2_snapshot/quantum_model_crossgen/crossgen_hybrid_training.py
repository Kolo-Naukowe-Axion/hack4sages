"""Hybrid quantum training helpers for the crossgen biosignatures dataset (rebinned to Ariel grid)."""

from __future__ import annotations

import json
import math
import os
import random
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import h5py
import matplotlib
import numpy as np
import pandas as pd
import spectres
import torch
import torch.nn as nn

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SAFE_AUX_FEATURE_COLS = [
    "planet_radius_rjup",
    "log_g_cgs",
    "temperature_k",
    "star_radius_rsun",
    "log10_sigma_ppm",
]

TARGET_COLS = [
    "log10_vmr_h2o",
    "log10_vmr_co2",
    "log10_vmr_co",
    "log10_vmr_ch4",
    "log10_vmr_nh3",
]

ARIEL_WAVELENGTH_GRID = np.array([
    0.9500000000, 1.1563750000, 1.2749034375, 1.4055810398,
    1.5496530964, 1.7084925388, 1.8836130240, 1.9695975000, 2.0091864097,
    2.0495710566, 2.0907674348, 2.1327918603, 2.1756609767, 2.2193917623,
    2.2640015367, 2.3095079676, 2.3559290777, 2.4032832522, 2.4515892456,
    2.5008661894, 2.5511335998, 2.6024113852, 2.6547198540, 2.7080797231,
    2.7625121255, 2.8180386192, 2.8746811955, 2.9324622875, 2.9914047795,
    3.0515320156, 3.1128678091, 3.1754364520, 3.2392627247, 3.3043719055,
    3.3707897808, 3.4385426554, 3.5076573628, 3.5781612758, 3.6500823174,
    3.7234489720, 4.0321666667, 4.3054579630, 4.5972723360, 4.9088652388,
], dtype=np.float64)


def rebin_spectra(old_wavelengths: np.ndarray, spectra: np.ndarray,
                  new_wavelengths: np.ndarray = ARIEL_WAVELENGTH_GRID) -> tuple[np.ndarray, np.ndarray]:
    rebinned = spectres.spectres(new_wavelengths, old_wavelengths, spectra, verbose=False)
    return new_wavelengths.astype(np.float32), rebinned.astype(np.float32)


def resolve_project_root(path_hint: Optional[Path] = None) -> Path:
    candidate = (path_hint or Path.cwd()).resolve()
    if (candidate / "data").exists():
        return candidate
    if candidate.name == "models" and (candidate.parent / "data").exists():
        return candidate.parent
    return candidate


def default_data_root(project_root: Path) -> Path:
    candidate = project_root / "crossgen_biosignatures_20260311"
    if candidate.exists():
        return candidate
    return project_root.parent / "quantum_model_crossgen" / "crossgen_biosignatures_20260311"


def default_output_dir(project_root: Path) -> Path:
    return project_root / "outputs" / "model_crossgen_rebinned"


def default_quantum_device() -> str:
    return "lightning.gpu" if torch.cuda.is_available() else "lightning.qubit"


@dataclass
class TrainingConfig:
    project_root: str = "."
    data_root: str = ""
    output_dir: str = "outputs/model_crossgen_rebinned"
    seed: int = 42
    internal_val_fraction: float = 0.10
    train_batch_size: int = 256
    eval_batch_size: int = 8192
    max_epochs: int = 30
    early_stop_patience: int = 6
    scheduler_patience: int = 5
    scheduler_factor: float = 0.5
    classical_lr: float = 2.0e-3
    quantum_lr: float = 6.0e-4
    weight_decay: float = 1.0e-4
    gradient_clip_norm: float = 5.0
    dropout: float = 0.05
    aux_hidden_dim: int = 64
    aux_out_dim: int = 32
    spectral_hidden_dim: int = 64
    spectral_out_dim: int = 32
    fusion_hidden_dim: int = 48
    head_hidden_dim: int = 96
    qnn_qubits: int = 12
    qnn_depth: int = 2
    quantum_device: str = default_quantum_device()
    log_every_batches: int = 1
    use_amp: bool = True
    train_pool_limit: Optional[int] = None
    test_limit: Optional[int] = None

    def resolved_project_root(self) -> Path:
        return resolve_project_root(Path(self.project_root))

    def resolved_data_root(self) -> Path:
        if self.data_root:
            root = Path(self.data_root).expanduser()
            if root.is_absolute():
                return root
            return self.resolved_project_root() / root
        return default_data_root(self.resolved_project_root())

    def resolved_output_dir(self) -> Path:
        root = Path(self.output_dir).expanduser()
        if root.is_absolute():
            return root
        return self.resolved_project_root() / root

    def to_json_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["project_root"] = str(self.resolved_project_root())
        payload["data_root"] = str(self.resolved_data_root())
        payload["output_dir"] = str(self.resolved_output_dir())
        payload["safe_aux_feature_cols"] = SAFE_AUX_FEATURE_COLS
        payload["target_cols"] = TARGET_COLS
        return payload


@dataclass
class ArrayStandardizer:
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, values: np.ndarray) -> "ArrayStandardizer":
        values64 = values.astype(np.float64, copy=False)
        mean = values64.mean(axis=0)
        scale = values64.std(axis=0)
        scale = np.where(scale == 0.0, 1.0, scale)
        return cls(mean=mean.astype(np.float32), scale=scale.astype(np.float32))

    def transform(self, values: np.ndarray) -> np.ndarray:
        return ((values - self.mean) / self.scale).astype(np.float32)

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        return (values * self.scale + self.mean).astype(np.float32)

    def state_dict(self) -> Dict[str, Any]:
        return {"mean": self.mean.tolist(), "scale": self.scale.tolist()}


@dataclass
class SpectralStandardizer:
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, spectra: np.ndarray) -> "SpectralStandardizer":
        spectra64 = spectra.astype(np.float64, copy=False)
        mean = spectra64.mean(axis=0)
        scale = spectra64.std(axis=0)
        scale = np.where(scale == 0.0, 1.0, scale)
        return cls(mean=mean.astype(np.float32), scale=scale.astype(np.float32))

    def transform(self, spectra: np.ndarray) -> np.ndarray:
        return ((spectra - self.mean[None, None, :]) / self.scale[None, None, :]).astype(np.float32)

    def state_dict(self) -> Dict[str, Any]:
        return {"mean": self.mean.tolist(), "scale": self.scale.tolist()}


@dataclass
class SplitTensors:
    sample_ids: np.ndarray
    aux: torch.Tensor
    spectra: torch.Tensor
    targets: torch.Tensor
    raw_targets: np.ndarray


@dataclass
class PreparedData:
    train: SplitTensors
    inner_val: SplitTensors
    test: SplitTensors
    aux_scaler: ArrayStandardizer
    target_scaler: ArrayStandardizer
    spectral_scaler: SpectralStandardizer
    wavelength_um: np.ndarray
    preflight: Dict[str, Any]


def set_runtime_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_runtime() -> None:
    cpu_threads = max(1, min(os.cpu_count() or 1, 32))
    torch.set_num_threads(cpu_threads)
    torch.set_num_interop_threads(max(1, min(cpu_threads // 2, 8)))
    if not torch.cuda.is_available():
        return
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")


def resolve_training_device(config: TrainingConfig) -> torch.device:
    if config.quantum_device.lower() == "lightning.gpu":
        if not torch.cuda.is_available():
            raise RuntimeError("QUANTUM_DEVICE=lightning.gpu requires CUDA, but torch.cuda.is_available() is False.")
        return torch.device("cuda")
    return torch.device("cpu")


def resolve_amp_dtype(device: torch.device) -> Optional[torch.dtype]:
    if device.type != "cuda":
        return None
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def maybe_limit_indices(indices: np.ndarray, limit: Optional[int], seed: int) -> np.ndarray:
    if limit is None or limit >= len(indices):
        return np.sort(indices)
    rng = np.random.default_rng(seed)
    chosen = rng.choice(indices, size=limit, replace=False)
    return np.sort(chosen.astype(np.int64))


def load_crossgen_dataset(data_root: Path) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    labels = pd.read_parquet(data_root / "labels.parquet")

    with h5py.File(data_root / "spectra.h5", "r") as handle:
        wavelength_um_raw = np.asarray(handle["wavelength_um"][:], dtype=np.float64)
        noisy_spectra_raw = np.asarray(handle["transit_depth_noisy"][:], dtype=np.float32)
        sigma_ppm = np.asarray(handle["sigma_ppm"][:], dtype=np.float32)

    wavelength_um, noisy_spectra = rebin_spectra(wavelength_um_raw, noisy_spectra_raw)
    print(f"Rebinned spectra: {noisy_spectra_raw.shape[1]} bins -> {noisy_spectra.shape[1]} bins "
          f"({wavelength_um[0]:.2f} - {wavelength_um[-1]:.2f} um)", flush=True)

    labels["log10_sigma_ppm"] = np.log10(np.clip(sigma_ppm, 1e-10, None))

    return labels, noisy_spectra, sigma_ppm, wavelength_um


def build_raw_arrays(
    labels: pd.DataFrame,
    noisy_spectra: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    aux = labels[SAFE_AUX_FEATURE_COLS].to_numpy(dtype=np.float32, copy=True)
    per_sample_mean = noisy_spectra.mean(axis=1, keepdims=True)
    per_sample_mean = np.where(per_sample_mean == 0, 1.0, per_sample_mean)
    spectra = (noisy_spectra / per_sample_mean)[:, None, :].astype(np.float32)
    return aux, spectra


def split_summary(labels: pd.DataFrame, wavelength_um: np.ndarray) -> Dict[str, Any]:
    return {
        "row_count": int(len(labels)),
        "feature_count": len(SAFE_AUX_FEATURE_COLS),
        "target_count": len(TARGET_COLS),
        "wavelength_bins": int(len(wavelength_um)),
        "wavelength_min_um": float(wavelength_um.min()),
        "wavelength_max_um": float(wavelength_um.max()),
        "safe_aux_feature_cols": SAFE_AUX_FEATURE_COLS,
        "target_cols": TARGET_COLS,
    }


def make_split_tensors(
    labels: pd.DataFrame,
    aux_values: np.ndarray,
    spectra_values: np.ndarray,
    targets_scaled: np.ndarray,
    raw_targets: np.ndarray,
) -> SplitTensors:
    return SplitTensors(
        sample_ids=labels["sample_id"].to_numpy(dtype="U32"),
        aux=torch.from_numpy(aux_values.astype(np.float32, copy=False)),
        spectra=torch.from_numpy(spectra_values.astype(np.float32, copy=False)),
        targets=torch.from_numpy(targets_scaled.astype(np.float32, copy=False)),
        raw_targets=raw_targets.astype(np.float32, copy=True),
    )


def prepare_data(config: TrainingConfig) -> PreparedData:
    labels, noisy_spectra, sigma_ppm, wavelength_um = load_crossgen_dataset(config.resolved_data_root())
    aux_raw, spectra_raw = build_raw_arrays(labels, noisy_spectra)
    targets_raw = labels[TARGET_COLS].to_numpy(dtype=np.float32, copy=True)

    inner_train_indices = np.where(labels["split"].values == "train")[0]
    inner_val_indices = np.where(labels["split"].values == "val")[0]
    test_indices = np.where(labels["split"].values == "test")[0]

    if config.train_pool_limit is not None:
        inner_train_indices = maybe_limit_indices(inner_train_indices, config.train_pool_limit, config.seed)

    aux_scaler = ArrayStandardizer.fit(aux_raw[inner_train_indices])
    target_scaler = ArrayStandardizer.fit(targets_raw[inner_train_indices])
    spectral_scaler = SpectralStandardizer.fit(spectra_raw[inner_train_indices, 0, :])

    def transform(indices: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        aux_scaled = aux_scaler.transform(aux_raw[indices])
        spectra_scaled = spectral_scaler.transform(spectra_raw[indices])
        targets_scaled = target_scaler.transform(targets_raw[indices])
        return aux_scaled, spectra_scaled, targets_scaled

    train_aux, train_spectra, train_targets = transform(inner_train_indices)
    inner_val_aux, inner_val_spectra, inner_val_targets = transform(inner_val_indices)
    test_aux, test_spectra, test_targets = transform(test_indices)

    preflight = split_summary(labels, wavelength_um)
    preflight["inner_train_rows"] = int(len(inner_train_indices))
    preflight["inner_val_rows"] = int(len(inner_val_indices))
    preflight["test_rows"] = int(len(test_indices))

    return PreparedData(
        train=make_split_tensors(
            labels.iloc[inner_train_indices],
            train_aux, train_spectra, train_targets, targets_raw[inner_train_indices],
        ),
        inner_val=make_split_tensors(
            labels.iloc[inner_val_indices],
            inner_val_aux, inner_val_spectra, inner_val_targets, targets_raw[inner_val_indices],
        ),
        test=make_split_tensors(
            labels.iloc[test_indices],
            test_aux, test_spectra, test_targets, targets_raw[test_indices],
        ),
        aux_scaler=aux_scaler,
        target_scaler=target_scaler,
        spectral_scaler=spectral_scaler,
        wavelength_um=wavelength_um,
        preflight=preflight,
    )


class AuxEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SpectralEncoder(nn.Module):
    def __init__(self, in_channels: int, hidden_dim: int, out_dim: int, dropout: float) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=7, padding=3),
            nn.GELU(),
            nn.Conv1d(32, hidden_dim, kernel_size=5, stride=2, padding=2),
            nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_dim, out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.conv(x))


class FusionEncoder(nn.Module):
    def __init__(self, aux_dim: int, spec_dim: int, hidden_dim: int, out_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(aux_dim + spec_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(self, aux_feat: torch.Tensor, spectral_feat: torch.Tensor) -> torch.Tensor:
        fused = torch.cat([aux_feat, spectral_feat], dim=-1)
        return torch.tanh(self.net(fused)) * math.pi


def import_pennylane() -> Any:
    try:
        import pennylane as qml
    except ImportError as exc:
        raise ImportError(
            "PennyLane is required for the hybrid quantum model. Install the training dependencies first."
        ) from exc
    return qml


class QuantumBlock(nn.Module):
    """A lighter 12-qubit circuit than the original sketch, kept trainable on full data."""

    def __init__(self, n_qubits: int, depth: int, quantum_device_name: str) -> None:
        super().__init__()
        if depth % 2 != 0:
            raise ValueError("qnn_depth must be even for the current ansatz.")

        self.n_qubits = n_qubits
        self.depth = depth
        self.quantum_device_name = quantum_device_name
        self.qml = import_pennylane()
        device_kwargs: Dict[str, Any] = {"wires": n_qubits}
        if quantum_device_name.startswith("lightning."):
            device_kwargs["c_dtype"] = np.complex64
        if quantum_device_name == "lightning.gpu":
            device_kwargs["use_async"] = True
        self.device = self.qml.device(quantum_device_name, **device_kwargs)
        self.num_blocks = depth // 2
        self.num_weights = 3 * n_qubits * self.num_blocks
        self.weights = nn.Parameter(0.5 * torch.randn(self.num_weights, dtype=torch.float32))
        self.qnode = self._build_qnode()

    def _build_qnode(self) -> Any:
        qml = self.qml
        n_qubits = self.n_qubits
        num_blocks = self.num_blocks
        device = self.device

        @qml.qnode(device, interface="torch", diff_method="adjoint")
        def circuit(inputs: torch.Tensor, weights: torch.Tensor) -> list[torch.Tensor]:
            for qubit in range(n_qubits):
                qml.RY(inputs[..., qubit], wires=qubit)

            param_idx = 0
            for _ in range(num_blocks):
                for qubit in range(n_qubits):
                    qml.RY(weights[param_idx], wires=qubit)
                    param_idx += 1
                for qubit in range(n_qubits):
                    qml.CNOT(wires=[qubit, (qubit + 1) % n_qubits])
                for qubit in range(n_qubits):
                    qml.RZ(weights[param_idx], wires=qubit)
                    param_idx += 1
                for qubit in range(n_qubits):
                    qml.CRX(weights[param_idx], wires=[qubit, (qubit + 1) % n_qubits])
                    param_idx += 1

            return [qml.expval(qml.PauliZ(qubit)) for qubit in range(n_qubits)]

        return circuit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = self.qnode(x.to(dtype=torch.float32), self.weights)
        if isinstance(outputs, (list, tuple)):
            outputs = torch.stack(tuple(outputs), dim=-1)
        return outputs.to(dtype=torch.float32)


class AtmosphereHead(nn.Module):
    def __init__(self, in_dim: int, latent_dim: int, hidden_dim: int, n_targets: int, dropout: float) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, n_targets),
        )
        self.residual = nn.Linear(latent_dim, n_targets)

    def forward(self, head_in: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
        return self.mlp(head_in) + self.residual(latent)


class HybridAtmosphereModel(nn.Module):
    def __init__(
        self,
        aux_encoder: AuxEncoder,
        spectral_encoder: SpectralEncoder,
        fusion_encoder: FusionEncoder,
        quantum_block: QuantumBlock,
        head: AtmosphereHead,
        classical_device: torch.device,
        amp_dtype: Optional[torch.dtype],
    ) -> None:
        super().__init__()
        self.aux_encoder = aux_encoder.to(classical_device)
        self.spectral_encoder = spectral_encoder.to(classical_device)
        self.fusion_encoder = fusion_encoder.to(classical_device)
        self.quantum_block = quantum_block.to(classical_device)
        self.head = head.to(classical_device)
        self.classical_device = classical_device
        self.amp_dtype = amp_dtype

    def classical_parameters(self) -> Iterable[nn.Parameter]:
        for module in (self.aux_encoder, self.spectral_encoder, self.fusion_encoder, self.head):
            yield from module.parameters()

    def quantum_parameters(self) -> Iterable[nn.Parameter]:
        yield from self.quantum_block.parameters()

    def forward(self, aux: torch.Tensor, spectra: torch.Tensor) -> torch.Tensor:
        autocast_enabled = self.classical_device.type == "cuda" and self.amp_dtype is not None
        autocast_ctx = (
            torch.autocast(device_type="cuda", dtype=self.amp_dtype)
            if autocast_enabled
            else nullcontext()
        )
        with autocast_ctx:
            aux_feat = self.aux_encoder(aux)
            spectral_feat = self.spectral_encoder(spectra)
            latent = self.fusion_encoder(aux_feat, spectral_feat)

        latent = latent.float()
        quantum_feat = self.quantum_block(latent)
        head_in = torch.cat([quantum_feat, latent, aux_feat.float(), spectral_feat.float()], dim=-1)
        return self.head(head_in, quantum_feat)


def build_model(config: TrainingConfig, device: torch.device) -> HybridAtmosphereModel:
    return HybridAtmosphereModel(
        aux_encoder=AuxEncoder(len(SAFE_AUX_FEATURE_COLS), config.aux_hidden_dim, config.aux_out_dim, config.dropout),
        spectral_encoder=SpectralEncoder(1, config.spectral_hidden_dim, config.spectral_out_dim, config.dropout),
        fusion_encoder=FusionEncoder(
            config.aux_out_dim,
            config.spectral_out_dim,
            config.fusion_hidden_dim,
            config.qnn_qubits,
        ),
        quantum_block=QuantumBlock(config.qnn_qubits, config.qnn_depth, config.quantum_device),
        head=AtmosphereHead(
            in_dim=config.qnn_qubits * 2 + config.aux_out_dim + config.spectral_out_dim,
            latent_dim=config.qnn_qubits,
            hidden_dim=config.head_hidden_dim,
            n_targets=len(TARGET_COLS),
            dropout=config.dropout,
        ),
        classical_device=device,
        amp_dtype=resolve_amp_dtype(device) if config.use_amp else None,
    )


def move_split_to_device(split: SplitTensors, device: torch.device) -> SplitTensors:
    kwargs = {"non_blocking": device.type == "cuda"}
    split.aux = split.aux.to(device, **kwargs)
    split.spectra = split.spectra.to(device, **kwargs)
    split.targets = split.targets.to(device, **kwargs)
    return split


def move_prepared_data_to_device(data: PreparedData, device: torch.device) -> PreparedData:
    move_split_to_device(data.train, device)
    move_split_to_device(data.inner_val, device)
    move_split_to_device(data.test, device)
    return data


def dataset_devices(data: PreparedData) -> Dict[str, str]:
    return {
        "train": str(data.train.aux.device),
        "inner_val": str(data.inner_val.aux.device),
        "test": str(data.test.aux.device),
    }


def batch_indices(length: int, batch_size: int, seed: int, epoch: int, device: torch.device) -> Iterable[torch.Tensor]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + epoch)
    permutation = torch.randperm(length, generator=generator)
    if device.type == "cuda":
        permutation = permutation.to(device, non_blocking=True)
    for start in range(0, length, batch_size):
        yield permutation[start : start + batch_size]


def gather_batch(
    split: SplitTensors,
    indices: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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


def evaluate_split(
    model: HybridAtmosphereModel,
    split: SplitTensors,
    target_scaler: ArrayStandardizer,
    batch_size: int,
) -> Dict[str, Any]:
    model.eval()
    loss_fn = nn.MSELoss()
    pred_batches = []
    target_batches = []
    losses = []

    with torch.inference_mode():
        for start in range(0, len(split.aux), batch_size):
            stop = min(start + batch_size, len(split.aux))
            indices = torch.arange(start, stop, device=split.aux.device)
            aux, spectra, targets = gather_batch(split, indices, model.classical_device)
            pred = model(aux, spectra)
            pred_batches.append(pred.detach().cpu().numpy())
            target_batches.append(targets.detach().cpu().numpy())
            losses.append(float(loss_fn(pred, targets).item()))

    pred_scaled = np.concatenate(pred_batches, axis=0)
    true_scaled = np.concatenate(target_batches, axis=0)
    pred_orig = target_scaler.inverse_transform(pred_scaled)
    true_orig = target_scaler.inverse_transform(true_scaled)
    rmse_orig = np.sqrt(np.mean((pred_orig - true_orig) ** 2, axis=0))

    return {
        "loss": float(np.mean(losses)),
        "pred_scaled": pred_scaled,
        "true_scaled": true_scaled,
        "pred_orig": pred_orig,
        "true_orig": true_orig,
        "rmse_orig": rmse_orig,
        "rmse_mean": float(rmse_orig.mean()),
    }


def format_target_rmse(metrics: Dict[str, Any]) -> str:
    return " | ".join(
        f"{target_name}={rmse_value:.4f}"
        for target_name, rmse_value in zip(TARGET_COLS, metrics["rmse_orig"])
    )


def train_model(config: TrainingConfig, data: PreparedData, device: torch.device, output_dir: Optional[Path] = None, resume_from: Optional[Path] = None) -> Dict[str, Any]:
    model = build_model(config, device)
    classical_params = list(model.classical_parameters())
    quantum_params = list(model.quantum_parameters())
    loss_fn = nn.MSELoss()

    optimizer = torch.optim.AdamW(
        [
            {"params": classical_params, "lr": config.classical_lr, "weight_decay": config.weight_decay},
            {"params": quantum_params, "lr": config.quantum_lr, "weight_decay": 0.0},
        ]
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config.scheduler_factor,
        patience=config.scheduler_patience,
    )

    history: list[Dict[str, Any]] = []
    best_val_loss = float("inf")
    best_epoch = -1
    best_state: Optional[Dict[str, torch.Tensor]] = None
    patience_left = config.early_stop_patience
    total_batches = math.ceil(len(data.train.aux) / config.train_batch_size)
    start_epoch = 0

    if resume_from is not None and resume_from.exists():
        ckpt = torch.load(resume_from, weights_only=False, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        start_epoch = ckpt["epoch"]
        best_val_loss = ckpt["val_loss"]
        best_epoch = ckpt["epoch"]
        best_state = ckpt["model_state_dict"]
        print(f"Resumed from checkpoint: epoch {start_epoch}, val_loss={best_val_loss:.5f}", flush=True)

    print(
        f"Torch device: {device} | Quantum device: {model.quantum_block.quantum_device_name} | "
        f"Dataset devices: {dataset_devices(data)}",
        flush=True,
    )
    if device.type == "cuda":
        print(f"Torch CUDA device: {torch.cuda.get_device_name(device)}", flush=True)
        print(format_cuda_memory(device), flush=True)

    for epoch in range(start_epoch, config.max_epochs):
        model.train()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        epoch_start = time.perf_counter()
        batch_losses = []
        first_batch_logged = False

        for batch_idx, indices in enumerate(
            batch_indices(len(data.train.aux), config.train_batch_size, config.seed, epoch, data.train.aux.device),
            start=1,
        ):
            batch_start = time.perf_counter()
            aux, spectra, targets = gather_batch(data.train, indices, device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(aux, spectra)
            loss = loss_fn(pred, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(classical_params, config.gradient_clip_norm)
            torch.nn.utils.clip_grad_norm_(quantum_params, 1.0)  # kept tight for quantum stability
            optimizer.step()
            batch_losses.append(float(loss.item()))

            if device.type == "cuda" and not first_batch_logged:
                print(f"After first batch | {format_cuda_memory(device)}", flush=True)
                first_batch_logged = True

            if config.log_every_batches > 0 and (
                batch_idx % config.log_every_batches == 0 or batch_idx == total_batches
            ):
                print(
                    f"Epoch {epoch + 1}/{config.max_epochs} | Batch {batch_idx}/{total_batches} | "
                    f"batch_loss={batch_losses[-1]:.5f} | avg_loss={np.mean(batch_losses):.5f} | "
                    f"batch_time={time.perf_counter() - batch_start:.2f}s",
                    flush=True,
                )

        inner_val_metrics = evaluate_split(model, data.inner_val, data.target_scaler, config.eval_batch_size)
        scheduler.step(inner_val_metrics["loss"])
        epoch_seconds = time.perf_counter() - epoch_start

        history_row = {
            "epoch": epoch + 1,
            "train_loss": float(np.mean(batch_losses)),
            "inner_val_loss": float(inner_val_metrics["loss"]),
            "inner_val_rmse_mean": float(inner_val_metrics["rmse_mean"]),
            "epoch_seconds": epoch_seconds,
            "classical_lr": optimizer.param_groups[0]["lr"],
            "quantum_lr": optimizer.param_groups[1]["lr"],
        }
        for target_name, rmse_value in zip(TARGET_COLS, inner_val_metrics["rmse_orig"]):
            history_row[f"inner_val_rmse_{target_name}"] = float(rmse_value)
        history.append(history_row)

        print(
            f"Epoch {epoch + 1}/{config.max_epochs} | train_loss={history_row['train_loss']:.5f} | "
            f"inner_val_loss={history_row['inner_val_loss']:.5f} | "
            f"inner_val_rmse_mean={history_row['inner_val_rmse_mean']:.5f} | "
            f"time={epoch_seconds:.1f}s | "
            f"lr=({history_row['classical_lr']:.2e}, {history_row['quantum_lr']:.2e})",
            flush=True,
        )
        print(
            f"Epoch {epoch + 1} inner-val target RMSE | {format_target_rmse(inner_val_metrics)}",
            flush=True,
        )
        if device.type == "cuda":
            print(f"Epoch {epoch + 1} memory | {format_cuda_memory(device)}", flush=True)

        if inner_val_metrics["loss"] < best_val_loss:
            best_val_loss = float(inner_val_metrics["loss"])
            best_epoch = epoch + 1
            best_state = {name: value.detach().cpu() for name, value in model.state_dict().items()}
            patience_left = config.early_stop_patience
            if output_dir is not None:
                output_dir.mkdir(parents=True, exist_ok=True)
                ckpt_path = output_dir / "best_model.pt"
                torch.save({"epoch": best_epoch, "val_loss": best_val_loss, "model_state_dict": best_state}, ckpt_path)
                print(f"  Checkpoint saved: {ckpt_path} (epoch {best_epoch}, val_loss={best_val_loss:.5f})", flush=True)
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"Early stopping at epoch {epoch + 1}.", flush=True)
                break

    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint.")

    last_state = {name: value.detach().cpu() for name, value in model.state_dict().items()}
    model.load_state_dict(best_state)
    return {
        "model": model,
        "history": history,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "best_state": best_state,
        "last_state": last_state,
    }


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def metrics_frame(split_name: str, metrics: Dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "split": split_name,
            "target": TARGET_COLS,
            "rmse": metrics["rmse_orig"],
        }
    )


def save_predictions(path: Path, split: SplitTensors, metrics: Dict[str, Any]) -> None:
    frame = pd.DataFrame({"sample_id": split.sample_ids})
    for idx, target_name in enumerate(TARGET_COLS):
        frame[f"true_{target_name}"] = metrics["true_orig"][:, idx]
        frame[f"pred_{target_name}"] = metrics["pred_orig"][:, idx]
    frame.to_csv(path, index=False)


def save_history_plots(output_dir: Path, history_frame: pd.DataFrame) -> Dict[str, str]:
    paths: Dict[str, str] = {}

    loss_path = output_dir / "loss_curve.png"
    plt.figure(figsize=(8, 4))
    plt.plot(history_frame["epoch"], history_frame["train_loss"], label="Train")
    plt.plot(history_frame["epoch"], history_frame["inner_val_loss"], label="Inner val")
    plt.xlabel("Epoch")
    plt.ylabel("MSE loss")
    plt.title("Training and Inner-Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(loss_path, dpi=150)
    plt.close()
    paths["loss_curve_png"] = str(loss_path)

    rmse_path = output_dir / "rmse_curve.png"
    plt.figure(figsize=(8, 4))
    plt.plot(history_frame["epoch"], history_frame["inner_val_rmse_mean"])
    plt.xlabel("Epoch")
    plt.ylabel("Mean RMSE")
    plt.title("Inner-Validation Mean RMSE")
    plt.tight_layout()
    plt.savefig(rmse_path, dpi=150)
    plt.close()
    paths["rmse_curve_png"] = str(rmse_path)

    return paths


def run_training_experiment(config: Optional[TrainingConfig] = None, resume: bool = False) -> Dict[str, Any]:
    cfg = config or TrainingConfig()
    cfg.project_root = str(cfg.resolved_project_root())

    set_runtime_seed(cfg.seed)
    configure_runtime()
    device = resolve_training_device(cfg)

    print(f"Project root: {cfg.resolved_project_root()}", flush=True)
    print(f"Data root: {cfg.resolved_data_root()}", flush=True)
    print(f"Output dir: {cfg.resolved_output_dir()}", flush=True)
    print(f"Train batch size: {cfg.train_batch_size}", flush=True)
    print(f"Eval batch size: {cfg.eval_batch_size}", flush=True)
    print(f"Quantum device: {cfg.quantum_device}", flush=True)
    print(f"Quantum width/depth: {cfg.qnn_qubits}/{cfg.qnn_depth}", flush=True)

    data = prepare_data(cfg)
    output_dir = cfg.resolved_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)

    save_json(output_dir / "config.json", cfg.to_json_dict())
    save_json(output_dir / "preflight.json", data.preflight)

    data = move_prepared_data_to_device(data, device)
    resume_from = (output_dir / "best_model.pt") if resume else None
    training = train_model(cfg, data, device, output_dir=output_dir, resume_from=resume_from)
    model: HybridAtmosphereModel = training["model"]

    torch.save(
        {
            "config": cfg.to_json_dict(),
            "feature_cols": SAFE_AUX_FEATURE_COLS,
            "target_cols": TARGET_COLS,
            "best_epoch": training["best_epoch"],
            "best_val_loss": training["best_val_loss"],
            "model_state_dict": training["best_state"],
        },
        output_dir / "best_model.pt",
    )
    torch.save(
        {
            "config": cfg.to_json_dict(),
            "feature_cols": SAFE_AUX_FEATURE_COLS,
            "target_cols": TARGET_COLS,
            "model_state_dict": training["last_state"],
        },
        output_dir / "last_model.pt",
    )

    save_json(
        output_dir / "scalers.json",
        {
            "aux_scaler": data.aux_scaler.state_dict(),
            "target_scaler": data.target_scaler.state_dict(),
            "spectral_scaler": data.spectral_scaler.state_dict(),
        },
    )

    history_frame = pd.DataFrame(training["history"])
    history_frame.to_csv(output_dir / "history.csv", index=False)
    plot_paths = save_history_plots(output_dir, history_frame)

    inner_val_metrics = evaluate_split(model, data.inner_val, data.target_scaler, cfg.eval_batch_size)
    test_metrics = evaluate_split(model, data.test, data.target_scaler, cfg.eval_batch_size)

    def metrics_payload(metrics: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "rmse_mean": float(metrics["rmse_mean"]),
            "rmse": {name: float(value) for name, value in zip(TARGET_COLS, metrics["rmse_orig"])},
        }

    save_json(output_dir / "inner_val_metrics.json", metrics_payload(inner_val_metrics))
    save_json(output_dir / "test_metrics.json", metrics_payload(test_metrics))

    save_predictions(output_dir / "test_predictions.csv", data.test, test_metrics)

    test_frame = metrics_frame("test", test_metrics)
    test_frame.to_csv(output_dir / "metrics_summary.csv", index=False)

    summary = {
        "best_epoch": training["best_epoch"],
        "best_inner_val_loss": float(training["best_val_loss"]),
        "test_rmse_mean": float(test_metrics["rmse_mean"]),
        "output_dir": str(output_dir),
    }
    save_json(output_dir / "run_summary.json", summary)

    return {
        "config": cfg.to_json_dict(),
        "preflight": data.preflight,
        "history_frame": history_frame,
        "inner_val_metrics_frame": metrics_frame("inner_val", inner_val_metrics),
        "test_metrics_frame": test_frame,
        "summary": summary,
        "artifacts": {
            "best_model_pt": str(output_dir / "best_model.pt"),
            "last_model_pt": str(output_dir / "last_model.pt"),
            "scalers_json": str(output_dir / "scalers.json"),
            "history_csv": str(output_dir / "history.csv"),
            "test_predictions_csv": str(output_dir / "test_predictions.csv"),
            "metrics_summary_csv": str(output_dir / "metrics_summary.csv"),
            **plot_paths,
        },
    }
