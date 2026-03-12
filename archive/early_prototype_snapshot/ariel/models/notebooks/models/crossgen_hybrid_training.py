"""Hybrid quantum training helpers for the cross-generator biosignature dataset."""

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
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split

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


def resolve_project_root(path_hint: Optional[Path] = None) -> Path:
    candidate = (path_hint or Path.cwd()).resolve()
    if (candidate / "data").exists():
        return candidate
    if candidate.name == "models" and (candidate.parent / "data").exists():
        return candidate.parent
    return candidate


def default_crossgen_data_root(project_root: Path) -> Path:
    local_candidate = project_root / "data" / "generated-data" / "crossgen_biosignatures_20260311"
    if local_candidate.exists():
        return local_candidate

    remote_candidate = Path.home() / "hack4sages-output" / "crossgen-biosignatures"
    if remote_candidate.exists():
        return remote_candidate

    return local_candidate


def default_output_dir(project_root: Path) -> Path:
    return project_root / "outputs" / "model_quant_sketch_crossgen"


def default_quantum_device() -> str:
    return "lightning.gpu" if torch.cuda.is_available() else "lightning.qubit"


@dataclass
class TrainingConfig:
    project_root: str = "."
    data_root: str = "data/generated-data/crossgen_biosignatures_20260311"
    output_dir: str = "outputs/model_quant_sketch_crossgen"
    seed: int = 42
    internal_val_fraction: float = 0.10
    train_batch_size: int = 1024
    eval_batch_size: int = 8192
    max_epochs: int = 30
    early_stop_patience: int = 6
    scheduler_patience: int = 2
    scheduler_factor: float = 0.5
    classical_lr: float = 2.0e-3
    quantum_lr: float = 8.0e-4
    weight_decay: float = 1.0e-4
    gradient_clip_norm: float = 5.0
    dropout: float = 0.0
    aux_hidden_dim: int = 64
    aux_out_dim: int = 32
    spectral_hidden_dim: int = 96
    spectral_out_dim: int = 96
    fusion_hidden_dim: int = 192
    fusion_out_dim: int = 128
    classical_head_hidden_dim: int = 128
    quantum_head_hidden_dim: int = 96
    qnn_qubits: int = 12
    qnn_depth: int = 2
    quantum_device: str = default_quantum_device()
    log_every_batches: int = 1
    use_amp: bool = True
    autotune_batch_size: bool = False
    batch_probe_sizes: tuple[int, ...] = (1024, 1536, 2048, 3072)
    batch_probe_steps: int = 2
    train_pool_limit: Optional[int] = None
    tau_test_limit: Optional[int] = None
    poseidon_limit: Optional[int] = None

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
    generators: np.ndarray
    splits: np.ndarray
    aux: torch.Tensor
    spectra: torch.Tensor
    targets: torch.Tensor
    raw_targets: np.ndarray


@dataclass
class PreparedData:
    train: SplitTensors
    inner_val: SplitTensors
    tau_test: SplitTensors
    poseidon: SplitTensors
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
    requested_threads = os.environ.get("OMP_NUM_THREADS")
    if requested_threads:
        cpu_threads = max(1, int(requested_threads))
    elif torch.cuda.is_available():
        cpu_threads = max(1, min(os.cpu_count() or 1, 8))
    else:
        cpu_threads = max(1, min(os.cpu_count() or 1, 32))
    torch.set_num_threads(cpu_threads)
    torch.set_num_interop_threads(max(1, min(cpu_threads // 2, 4)))
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
    labels_path = data_root / "labels.parquet"
    spectra_path = data_root / "spectra.h5"

    labels = pd.read_parquet(labels_path).reset_index(drop=True)
    with h5py.File(spectra_path, "r") as handle:
        h5_sample_id = handle["sample_id"][:].astype("U32")
        h5_generator = handle["generator"][:].astype("U16")
        h5_split = handle["split"][:].astype("U16")
        wavelength_um = np.asarray(handle["wavelength_um"][:], dtype=np.float32)
        noisy_spectra = np.asarray(handle["transit_depth_noisy"][:], dtype=np.float32)
        sigma_ppm = np.asarray(handle["sigma_ppm"][:], dtype=np.float32)

    index_frame = pd.DataFrame(
        {
            "sample_id": h5_sample_id,
            "_row_index": np.arange(len(h5_sample_id), dtype=np.int64),
            "_generator_h5": h5_generator,
            "_split_h5": h5_split,
        }
    )
    merged = labels.merge(index_frame, on="sample_id", how="inner", validate="one_to_one")
    if len(merged) != len(labels):
        raise AssertionError("labels.parquet and spectra.h5 have inconsistent sample_id coverage.")
    if not np.array_equal(merged["generator"].to_numpy(dtype="U16"), merged["_generator_h5"].to_numpy(dtype="U16")):
        raise AssertionError("Generator columns do not align between labels.parquet and spectra.h5.")
    if not np.array_equal(merged["split"].to_numpy(dtype="U16"), merged["_split_h5"].to_numpy(dtype="U16")):
        raise AssertionError("Split columns do not align between labels.parquet and spectra.h5.")

    row_index = merged["_row_index"].to_numpy(dtype=np.int64)
    labels_aligned = merged.drop(columns=["_row_index", "_generator_h5", "_split_h5"]).reset_index(drop=True)
    return labels_aligned, noisy_spectra[row_index], sigma_ppm[row_index], wavelength_um


def build_raw_arrays(
    labels: pd.DataFrame,
    noisy_spectra: np.ndarray,
    sigma_ppm: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    safe_aux = labels[SAFE_AUX_FEATURE_COLS[:-1]].to_numpy(dtype=np.float32, copy=True)
    log_sigma = np.log10(np.clip(sigma_ppm.astype(np.float32, copy=False), 1.0, None)).reshape(-1, 1)
    aux = np.concatenate([safe_aux, log_sigma], axis=1).astype(np.float32)
    spectra = noisy_spectra[:, None, :].astype(np.float32)
    return aux, spectra


def split_summary(labels: pd.DataFrame, wavelength_um: np.ndarray) -> Dict[str, Any]:
    def rows_for(generator: str, split: Optional[str] = None) -> int:
        mask = labels["generator"].eq(generator)
        if split is not None:
            mask &= labels["split"].eq(split)
        return int(mask.sum())

    return {
        "row_count": int(len(labels)),
        "feature_count": len(SAFE_AUX_FEATURE_COLS),
        "target_count": len(TARGET_COLS),
        "wavelength_bins": int(len(wavelength_um)),
        "wavelength_min_um": float(wavelength_um.min()),
        "wavelength_max_um": float(wavelength_um.max()),
        "tau_train_rows": rows_for("tau", "train"),
        "tau_val_rows": rows_for("tau", "val"),
        "poseidon_rows": rows_for("poseidon"),
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
        generators=labels["generator"].to_numpy(dtype="U16"),
        splits=labels["split"].to_numpy(dtype="U16"),
        aux=torch.from_numpy(aux_values.astype(np.float32, copy=False)),
        spectra=torch.from_numpy(spectra_values.astype(np.float32, copy=False)),
        targets=torch.from_numpy(targets_scaled.astype(np.float32, copy=False)),
        raw_targets=raw_targets.astype(np.float32, copy=True),
    )


def prepare_data(config: TrainingConfig) -> PreparedData:
    labels, noisy_spectra, sigma_ppm, wavelength_um = load_crossgen_dataset(config.resolved_data_root())
    aux_raw, spectra_raw = build_raw_arrays(labels, noisy_spectra, sigma_ppm)
    targets_raw = labels[TARGET_COLS].to_numpy(dtype=np.float32, copy=True)

    tau_train_pool = np.flatnonzero((labels["generator"] == "tau") & (labels["split"] == "train"))
    tau_test_indices = np.flatnonzero((labels["generator"] == "tau") & (labels["split"] == "val"))
    poseidon_indices = np.flatnonzero(labels["generator"] == "poseidon")

    tau_train_pool = maybe_limit_indices(tau_train_pool, config.train_pool_limit, config.seed)
    tau_test_indices = maybe_limit_indices(tau_test_indices, config.tau_test_limit, config.seed + 1)
    poseidon_indices = maybe_limit_indices(poseidon_indices, config.poseidon_limit, config.seed + 2)

    pool_positions = np.arange(len(tau_train_pool))
    inner_train_pos, inner_val_pos = train_test_split(
        pool_positions,
        test_size=config.internal_val_fraction,
        random_state=config.seed,
        shuffle=True,
    )
    inner_train_indices = np.sort(tau_train_pool[inner_train_pos])
    inner_val_indices = np.sort(tau_train_pool[inner_val_pos])

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
    tau_test_aux, tau_test_spectra, tau_test_targets = transform(tau_test_indices)
    poseidon_aux, poseidon_spectra, poseidon_targets = transform(poseidon_indices)

    preflight = split_summary(labels, wavelength_um)
    preflight["inner_train_rows"] = int(len(inner_train_indices))
    preflight["inner_val_rows"] = int(len(inner_val_indices))
    preflight["tau_test_eval_rows"] = int(len(tau_test_indices))
    preflight["poseidon_eval_rows"] = int(len(poseidon_indices))

    return PreparedData(
        train=make_split_tensors(
            labels.iloc[inner_train_indices],
            train_aux,
            train_spectra,
            train_targets,
            targets_raw[inner_train_indices],
        ),
        inner_val=make_split_tensors(
            labels.iloc[inner_val_indices],
            inner_val_aux,
            inner_val_spectra,
            inner_val_targets,
            targets_raw[inner_val_indices],
        ),
        tau_test=make_split_tensors(
            labels.iloc[tau_test_indices],
            tau_test_aux,
            tau_test_spectra,
            tau_test_targets,
            targets_raw[tau_test_indices],
        ),
        poseidon=make_split_tensors(
            labels.iloc[poseidon_indices],
            poseidon_aux,
            poseidon_spectra,
            poseidon_targets,
            targets_raw[poseidon_indices],
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
        if in_channels != 1:
            raise ValueError("SpectralEncoder expects a single spectral channel.")
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
            nn.GELU(),
            nn.LayerNorm(out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class FusionEncoder(nn.Module):
    def __init__(self, aux_dim: int, spec_dim: int, hidden_dim: int, out_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(aux_dim + spec_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
            nn.GELU(),
            nn.LayerNorm(out_dim),
        )

    def forward(self, aux_feat: torch.Tensor, spectral_feat: torch.Tensor) -> torch.Tensor:
        fused = torch.cat([aux_feat, spectral_feat], dim=-1)
        return self.net(fused)


def import_pennylane() -> Any:
    try:
        import pennylane as qml
    except ImportError as exc:
        raise ImportError(
            "PennyLane is required for the hybrid quantum model. Install the training dependencies first."
        ) from exc
    return qml


class QuantumBlock(nn.Module):
    """Quantum correction block used as an adapter on top of the classical trunk."""

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
        self.weights = nn.Parameter(0.01 * torch.randn(self.num_weights, dtype=torch.float32))
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
    def __init__(self, in_dim: int, hidden_dim: int, n_targets: int, dropout: float) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, n_targets),
        )

    def forward(self, head_in: torch.Tensor) -> torch.Tensor:
        return self.mlp(head_in)


class QuantumProjector(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, n_qubits: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_qubits),
        )

    def forward(self, fused_rep: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.net(fused_rep)) * math.pi


class HybridAtmosphereModel(nn.Module):
    def __init__(
        self,
        aux_encoder: AuxEncoder,
        spectral_encoder: SpectralEncoder,
        fusion_encoder: FusionEncoder,
        quantum_projector: QuantumProjector,
        quantum_block: QuantumBlock,
        classical_head: AtmosphereHead,
        quantum_head: AtmosphereHead,
        classical_device: torch.device,
        amp_dtype: Optional[torch.dtype],
    ) -> None:
        super().__init__()
        self.aux_encoder = aux_encoder.to(classical_device)
        self.spectral_encoder = spectral_encoder.to(classical_device)
        self.fusion_encoder = fusion_encoder.to(classical_device)
        self.quantum_projector = quantum_projector.to(classical_device)
        self.quantum_block = quantum_block.to(classical_device)
        self.classical_head = classical_head.to(classical_device)
        self.quantum_head = quantum_head.to(classical_device)
        self.quantum_gate = nn.Parameter(torch.tensor(0.05, dtype=torch.float32, device=classical_device))
        self.classical_device = classical_device
        self.amp_dtype = amp_dtype

    def classical_parameters(self) -> Iterable[nn.Parameter]:
        for module in (
            self.aux_encoder,
            self.spectral_encoder,
            self.fusion_encoder,
            self.quantum_projector,
            self.classical_head,
            self.quantum_head,
        ):
            yield from module.parameters()
        yield self.quantum_gate

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
            fused_rep = self.fusion_encoder(aux_feat, spectral_feat)
            base_pred = self.classical_head(fused_rep)
            quantum_angles = self.quantum_projector(fused_rep)

        fused_rep = fused_rep.float()
        base_pred = base_pred.float()
        quantum_feat = self.quantum_block(quantum_angles.float())
        quantum_correction = self.quantum_head(torch.cat([fused_rep, quantum_feat], dim=-1))
        return base_pred + torch.tanh(self.quantum_gate) * quantum_correction.float()


def build_model(config: TrainingConfig, device: torch.device) -> HybridAtmosphereModel:
    return HybridAtmosphereModel(
        aux_encoder=AuxEncoder(len(SAFE_AUX_FEATURE_COLS), config.aux_hidden_dim, config.aux_out_dim, config.dropout),
        spectral_encoder=SpectralEncoder(1, config.spectral_hidden_dim, config.spectral_out_dim, config.dropout),
        fusion_encoder=FusionEncoder(
            config.aux_out_dim,
            config.spectral_out_dim,
            config.fusion_hidden_dim,
            config.fusion_out_dim,
            config.dropout,
        ),
        quantum_projector=QuantumProjector(
            config.fusion_out_dim,
            config.classical_head_hidden_dim,
            config.qnn_qubits,
            config.dropout,
        ),
        quantum_block=QuantumBlock(config.qnn_qubits, config.qnn_depth, config.quantum_device),
        classical_head=AtmosphereHead(
            in_dim=config.fusion_out_dim,
            hidden_dim=config.classical_head_hidden_dim,
            n_targets=len(TARGET_COLS),
            dropout=config.dropout,
        ),
        quantum_head=AtmosphereHead(
            in_dim=config.fusion_out_dim + config.qnn_qubits,
            hidden_dim=config.quantum_head_hidden_dim,
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
    move_split_to_device(data.tau_test, device)
    move_split_to_device(data.poseidon, device)
    return data


def dataset_devices(data: PreparedData) -> Dict[str, str]:
    return {
        "train": str(data.train.aux.device),
        "inner_val": str(data.inner_val.aux.device),
        "tau_test": str(data.tau_test.aux.device),
        "poseidon": str(data.poseidon.aux.device),
    }


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


def benchmark_batch_sizes(
    config: TrainingConfig,
    data: PreparedData,
    device: torch.device,
) -> Dict[str, Any]:
    if not config.autotune_batch_size:
        return {
            "enabled": False,
            "selected_train_batch_size": config.train_batch_size,
            "results": [],
        }

    candidates = sorted({int(size) for size in config.batch_probe_sizes if int(size) > 0})
    candidates = [size for size in candidates if size <= len(data.train.aux)]
    if not candidates:
        return {
            "enabled": False,
            "selected_train_batch_size": config.train_batch_size,
            "results": [],
        }

    results = []
    for batch_size in candidates:
        model = build_model(config, device)
        classical_params = list(model.classical_parameters())
        quantum_params = list(model.quantum_parameters())
        optimizer = torch.optim.AdamW(
            [
                {"params": classical_params, "lr": config.classical_lr, "weight_decay": config.weight_decay},
                {"params": quantum_params, "lr": config.quantum_lr, "weight_decay": 0.0},
            ]
        )
        loss_fn = nn.MSELoss()
        warmup_steps = 1
        measured_steps = max(1, config.batch_probe_steps)
        times = []
        try:
            for step_idx, indices in enumerate(batch_indices(len(data.train.aux), batch_size, config.seed, 0, data.train.aux.device)):
                aux, spectra, targets = gather_batch(data.train, indices, device)
                optimizer.zero_grad(set_to_none=True)
                maybe_sync_cuda(device)
                started = time.perf_counter()
                pred = model(aux, spectra)
                loss = loss_fn(pred, targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(classical_params, config.gradient_clip_norm)
                torch.nn.utils.clip_grad_norm_(quantum_params, config.gradient_clip_norm)
                optimizer.step()
                maybe_sync_cuda(device)
                elapsed = time.perf_counter() - started
                if step_idx >= warmup_steps:
                    times.append(elapsed)
                if len(times) >= measured_steps:
                    break
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower() and device.type == "cuda":
                torch.cuda.empty_cache()
                results.append({"batch_size": batch_size, "status": "oom"})
                continue
            raise
        finally:
            del model
            del optimizer
            if device.type == "cuda":
                torch.cuda.empty_cache()

        avg_seconds = float(np.mean(times))
        results.append(
            {
                "batch_size": batch_size,
                "status": "ok",
                "avg_batch_seconds": avg_seconds,
                "samples_per_second": float(batch_size / avg_seconds),
            }
        )

    successful = [item for item in results if item["status"] == "ok"]
    if successful:
        best = max(successful, key=lambda item: item["samples_per_second"])
        config.train_batch_size = int(best["batch_size"])

    return {
        "enabled": True,
        "selected_train_batch_size": config.train_batch_size,
        "results": results,
    }


def train_model(config: TrainingConfig, data: PreparedData, device: torch.device) -> Dict[str, Any]:
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

    print(
        f"Torch device: {device} | Quantum device: {model.quantum_block.quantum_device_name} | "
        f"Dataset devices: {dataset_devices(data)}",
        flush=True,
    )
    if device.type == "cuda":
        print(f"Torch CUDA device: {torch.cuda.get_device_name(device)}", flush=True)
        print(format_cuda_memory(device), flush=True)

    for epoch in range(config.max_epochs):
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
            maybe_sync_cuda(device)
            pred = model(aux, spectra)
            loss = loss_fn(pred, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(classical_params, config.gradient_clip_norm)
            torch.nn.utils.clip_grad_norm_(quantum_params, config.gradient_clip_norm)
            optimizer.step()
            maybe_sync_cuda(device)
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
    frame = pd.DataFrame(
        {
            "sample_id": split.sample_ids,
            "generator": split.generators,
            "split": split.splits,
        }
    )
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


def run_training_experiment(config: Optional[TrainingConfig] = None) -> Dict[str, Any]:
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
    batch_probe = benchmark_batch_sizes(cfg, data, device)
    if batch_probe["enabled"]:
        save_json(output_dir / "batch_probe.json", batch_probe)
        save_json(output_dir / "config.json", cfg.to_json_dict())
        print(f"Batch probe selected train batch size: {cfg.train_batch_size}", flush=True)
    training = train_model(cfg, data, device)
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
    tau_test_metrics = evaluate_split(model, data.tau_test, data.target_scaler, cfg.eval_batch_size)
    poseidon_metrics = evaluate_split(model, data.poseidon, data.target_scaler, cfg.eval_batch_size)

    def metrics_payload(metrics: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "rmse_mean": float(metrics["rmse_mean"]),
            "rmse": {name: float(value) for name, value in zip(TARGET_COLS, metrics["rmse_orig"])},
        }

    save_json(output_dir / "inner_val_metrics.json", metrics_payload(inner_val_metrics))
    save_json(output_dir / "tau_test_metrics.json", metrics_payload(tau_test_metrics))
    save_json(output_dir / "poseidon_metrics.json", metrics_payload(poseidon_metrics))

    save_predictions(output_dir / "tau_test_predictions.csv", data.tau_test, tau_test_metrics)
    save_predictions(output_dir / "poseidon_predictions.csv", data.poseidon, poseidon_metrics)

    tau_frame = metrics_frame("tau_test", tau_test_metrics)
    poseidon_frame = metrics_frame("poseidon", poseidon_metrics)
    combined_metrics = pd.concat([tau_frame, poseidon_frame], ignore_index=True)
    combined_metrics.to_csv(output_dir / "metrics_summary.csv", index=False)

    summary = {
        "best_epoch": training["best_epoch"],
        "best_inner_val_loss": float(training["best_val_loss"]),
        "tau_test_rmse_mean": float(tau_test_metrics["rmse_mean"]),
        "poseidon_rmse_mean": float(poseidon_metrics["rmse_mean"]),
        "output_dir": str(output_dir),
    }
    save_json(output_dir / "run_summary.json", summary)

    return {
        "config": cfg.to_json_dict(),
        "preflight": data.preflight,
        "history_frame": history_frame,
        "inner_val_metrics_frame": metrics_frame("inner_val", inner_val_metrics),
        "tau_test_metrics_frame": tau_frame,
        "poseidon_metrics_frame": poseidon_frame,
        "summary": summary,
        "artifacts": {
            "best_model_pt": str(output_dir / "best_model.pt"),
            "last_model_pt": str(output_dir / "last_model.pt"),
            "scalers_json": str(output_dir / "scalers.json"),
            "batch_probe_json": str(output_dir / "batch_probe.json") if batch_probe["enabled"] else "",
            "history_csv": str(output_dir / "history.csv"),
            "tau_test_predictions_csv": str(output_dir / "tau_test_predictions.csv"),
            "poseidon_predictions_csv": str(output_dir / "poseidon_predictions.csv"),
            "metrics_summary_csv": str(output_dir / "metrics_summary.csv"),
            **plot_paths,
        },
    }
