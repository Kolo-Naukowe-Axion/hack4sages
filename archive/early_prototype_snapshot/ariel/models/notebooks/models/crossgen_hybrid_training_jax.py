"""JAX/Catalyst hybrid quantum training helpers for the cross-generator biosignature dataset."""

from __future__ import annotations

import json
import math
import os
import pickle
import random
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence

import h5py
import matplotlib
import numpy as np
import pandas as pd
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
    return project_root / "outputs" / "model_quant_sketch_crossgen_jax"


@dataclass
class TrainingConfigJAX:
    project_root: str = "."
    data_root: str = "data/generated-data/crossgen_biosignatures_20260311"
    output_dir: str = "outputs/model_quant_sketch_crossgen_jax"
    seed: int = 42
    internal_val_fraction: float = 0.10
    batch_size_train: int = 1024
    batch_size_eval: int = 8192
    max_epochs: int = 30
    early_stop_patience: int = 6
    scheduler_patience: int = 2
    scheduler_factor: float = 0.5
    classical_lr: float = 2.0e-3
    quantum_lr: float = 6.0e-4
    weight_decay: float = 1.0e-4
    gradient_clip_norm: float = 1.0
    dropout: float = 0.0
    aux_hidden_dim: int = 64
    aux_out_dim: int = 32
    spectral_hidden_dim: int = 64
    spectral_out_dim: int = 32
    fusion_hidden_dim: int = 48
    head_hidden_dim: int = 96
    qnn_qubits: int = 16
    qnn_depth: int = 2
    device_backend: str = "lightning.gpu"
    precision: str = "float32"
    compile_warmup_steps: int = 2
    autotune_batch: bool = True
    performance_batch_ladder: tuple[int, ...] = field(default_factory=lambda: (256, 512, 1024, 2048))
    log_every_batches: int = 1
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
        payload["performance_batch_ladder"] = list(self.performance_batch_ladder)
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
class SplitArrays:
    sample_ids: np.ndarray
    generators: np.ndarray
    splits: np.ndarray
    aux_host: np.ndarray
    spectra_host: np.ndarray
    targets_host: np.ndarray
    raw_targets: np.ndarray
    aux: Any = None
    spectra: Any = None
    targets: Any = None


@dataclass
class PreparedDataJAX:
    train: SplitArrays
    inner_val: SplitArrays
    tau_test: SplitArrays
    poseidon: SplitArrays
    aux_scaler: ArrayStandardizer
    target_scaler: ArrayStandardizer
    spectral_scaler: SpectralStandardizer
    wavelength_um: np.ndarray
    preflight: Dict[str, Any]


@dataclass
class RuntimeImports:
    jax: Any
    jnp: Any
    qml: Any
    catalyst: Any


@dataclass
class JAXTrainingRuntime:
    libs: RuntimeImports
    device: Any
    dtype: Any
    train_step: Callable[..., Any]
    predict_step: Callable[..., Any]
    init_params: Callable[[int], Dict[str, Any]]
    init_optimizer_state: Callable[[Dict[str, Any]], Dict[str, Any]]
    parameter_count: Callable[[Dict[str, Any]], int]
    metadata: Dict[str, Any]


def import_jax_stack() -> RuntimeImports:
    try:
        import jax
        import jax.numpy as jnp
    except ImportError as exc:
        raise ImportError("The JAX/Catalyst training path requires JAX. Install the JAX requirements first.") from exc

    try:
        import pennylane as qml
    except ImportError as exc:
        raise ImportError(
            "The JAX/Catalyst training path requires PennyLane. Install the JAX requirements first."
        ) from exc

    try:
        import catalyst
    except ImportError as exc:
        raise ImportError(
            "The JAX/Catalyst training path requires pennylane-catalyst. Install the JAX requirements first."
        ) from exc

    return RuntimeImports(jax=jax, jnp=jnp, qml=qml, catalyst=catalyst)


def set_runtime_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def configure_runtime(libs: RuntimeImports, config: TrainingConfigJAX) -> None:
    libs.jax.config.update("jax_enable_x64", config.precision == "float64")


def resolve_jax_device(libs: RuntimeImports, config: TrainingConfigJAX) -> Any:
    if config.device_backend == "lightning.gpu":
        gpu_devices = libs.jax.devices("gpu")
        if not gpu_devices:
            raise RuntimeError("device_backend=lightning.gpu requires a JAX-visible GPU device.")
        return gpu_devices[0]

    devices = libs.jax.devices()
    if not devices:
        raise RuntimeError("JAX did not report any runtime devices.")
    return devices[0]


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


def make_split_arrays(
    labels: pd.DataFrame,
    aux_values: np.ndarray,
    spectra_values: np.ndarray,
    targets_scaled: np.ndarray,
    raw_targets: np.ndarray,
) -> SplitArrays:
    return SplitArrays(
        sample_ids=labels["sample_id"].to_numpy(dtype="U32"),
        generators=labels["generator"].to_numpy(dtype="U16"),
        splits=labels["split"].to_numpy(dtype="U16"),
        aux_host=aux_values.astype(np.float32, copy=False),
        spectra_host=spectra_values.astype(np.float32, copy=False),
        targets_host=targets_scaled.astype(np.float32, copy=False),
        raw_targets=raw_targets.astype(np.float32, copy=True),
    )


def prepare_data(config: TrainingConfigJAX) -> PreparedDataJAX:
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

    return PreparedDataJAX(
        train=make_split_arrays(
            labels.iloc[inner_train_indices],
            train_aux,
            train_spectra,
            train_targets,
            targets_raw[inner_train_indices],
        ),
        inner_val=make_split_arrays(
            labels.iloc[inner_val_indices],
            inner_val_aux,
            inner_val_spectra,
            inner_val_targets,
            targets_raw[inner_val_indices],
        ),
        tau_test=make_split_arrays(
            labels.iloc[tau_test_indices],
            tau_test_aux,
            tau_test_spectra,
            tau_test_targets,
            targets_raw[tau_test_indices],
        ),
        poseidon=make_split_arrays(
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


def move_split_to_device(split: SplitArrays, libs: RuntimeImports, device: Any) -> SplitArrays:
    split.aux = libs.jax.device_put(split.aux_host.astype(np.float32, copy=False), device)
    spectra_nwc = np.moveaxis(split.spectra_host.astype(np.float32, copy=False), 1, -1)
    split.spectra = libs.jax.device_put(spectra_nwc, device)
    split.targets = libs.jax.device_put(split.targets_host.astype(np.float32, copy=False), device)
    return split


def move_prepared_data_to_device(data: PreparedDataJAX, libs: RuntimeImports, device: Any) -> PreparedDataJAX:
    move_split_to_device(data.train, libs, device)
    move_split_to_device(data.inner_val, libs, device)
    move_split_to_device(data.tau_test, libs, device)
    move_split_to_device(data.poseidon, libs, device)
    return data


def _array_device_name(array: Any) -> str:
    try:
        devices = sorted(str(dev) for dev in array.devices())
        return ",".join(devices)
    except Exception:
        try:
            return str(array.device())
        except Exception:
            return "unknown"


def dataset_devices(data: PreparedDataJAX) -> Dict[str, str]:
    return {
        "train": _array_device_name(data.train.aux),
        "inner_val": _array_device_name(data.inner_val.aux),
        "tau_test": _array_device_name(data.tau_test.aux),
        "poseidon": _array_device_name(data.poseidon.aux),
    }


def _sample_system_metrics() -> Dict[str, Optional[float]]:
    metrics: Dict[str, Optional[float]] = {"gpu_util": None, "gpu_mem_mb": None, "cpu_percent": None}

    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        first_line = result.stdout.strip().splitlines()[0]
        util, mem = [item.strip() for item in first_line.split(",", maxsplit=1)]
        metrics["gpu_util"] = float(util)
        metrics["gpu_mem_mb"] = float(mem)
    except Exception:
        pass

    try:
        result = subprocess.run(
            ["ps", "-p", str(os.getpid()), "-o", "%cpu="],
            check=True,
            capture_output=True,
            text=True,
        )
        metrics["cpu_percent"] = float(result.stdout.strip())
    except Exception:
        pass

    return metrics


def build_runtime(config: TrainingConfigJAX, libs: RuntimeImports, device: Any) -> JAXTrainingRuntime:
    jax = libs.jax
    jnp = libs.jnp
    qml = libs.qml
    catalyst = libs.catalyst

    if config.qnn_depth % 2 != 0:
        raise ValueError("qnn_depth must be even for the current ansatz.")

    dtype = getattr(jnp, config.precision)
    complex_dtype = np.complex64 if config.precision == "float32" else np.complex128
    quantum_device_kwargs: Dict[str, Any] = {"wires": config.qnn_qubits}
    if config.device_backend.startswith("lightning."):
        quantum_device_kwargs["c_dtype"] = complex_dtype
    if config.device_backend == "lightning.gpu":
        quantum_device_kwargs["use_async"] = True

    q_device = qml.device(config.device_backend, **quantum_device_kwargs)
    num_quantum_blocks = config.qnn_depth // 2
    num_quantum_weights = 3 * config.qnn_qubits * num_quantum_blocks

    def glorot_uniform(key: Any, shape: Sequence[int]) -> Any:
        if len(shape) < 2:
            fan_in = shape[0]
            fan_out = shape[0]
        elif len(shape) == 2:
            fan_in, fan_out = shape[0], shape[1]
        else:
            receptive_field = int(np.prod(shape[:-2]))
            fan_in = shape[-2] * receptive_field
            fan_out = shape[-1] * receptive_field
        limit = math.sqrt(6.0 / float(fan_in + fan_out))
        return jax.random.uniform(key, shape, minval=-limit, maxval=limit, dtype=dtype)

    def init_dense(key: Any, in_dim: int, out_dim: int) -> Dict[str, Any]:
        weight_key, _ = jax.random.split(key)
        return {"w": glorot_uniform(weight_key, (in_dim, out_dim)), "b": jnp.zeros((out_dim,), dtype=dtype)}

    def init_conv1d(key: Any, in_channels: int, out_channels: int, kernel_size: int) -> Dict[str, Any]:
        weight_key, _ = jax.random.split(key)
        return {
            "w": glorot_uniform(weight_key, (kernel_size, in_channels, out_channels)),
            "b": jnp.zeros((out_channels,), dtype=dtype),
        }

    def init_params(seed: int) -> Dict[str, Any]:
        keys = iter(jax.random.split(jax.random.PRNGKey(seed), 16))
        classical = {
            "aux_l1": init_dense(next(keys), len(SAFE_AUX_FEATURE_COLS), config.aux_hidden_dim),
            "aux_l2": init_dense(next(keys), config.aux_hidden_dim, config.aux_hidden_dim),
            "aux_l3": init_dense(next(keys), config.aux_hidden_dim, config.aux_out_dim),
            "spec_c1": init_conv1d(next(keys), 1, 32, 7),
            "spec_c2": init_conv1d(next(keys), 32, config.spectral_hidden_dim, 5),
            "spec_c3": init_conv1d(next(keys), config.spectral_hidden_dim, config.spectral_hidden_dim, 3),
            "spec_proj": init_dense(next(keys), config.spectral_hidden_dim, config.spectral_out_dim),
            "fusion_l1": init_dense(next(keys), config.aux_out_dim + config.spectral_out_dim, config.fusion_hidden_dim),
            "fusion_l2": init_dense(next(keys), config.fusion_hidden_dim, config.qnn_qubits),
            "head_l1": init_dense(next(keys), config.qnn_qubits * 2, config.head_hidden_dim),
            "head_l2": init_dense(next(keys), config.head_hidden_dim, config.head_hidden_dim),
            "head_l3": init_dense(next(keys), config.head_hidden_dim, len(TARGET_COLS)),
            "head_residual": init_dense(next(keys), config.qnn_qubits, len(TARGET_COLS)),
        }
        quantum = {"weights": 0.01 * jax.random.normal(next(keys), (num_quantum_weights,), dtype=dtype)}
        return {"classical": classical, "quantum": quantum}

    def init_optimizer_state(params: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "step": jnp.array(0, dtype=jnp.int32),
            "classical_m": jax.tree_util.tree_map(jnp.zeros_like, params["classical"]),
            "classical_v": jax.tree_util.tree_map(jnp.zeros_like, params["classical"]),
            "quantum_m": jax.tree_util.tree_map(jnp.zeros_like, params["quantum"]),
            "quantum_v": jax.tree_util.tree_map(jnp.zeros_like, params["quantum"]),
        }

    def linear(params: Dict[str, Any], x: Any) -> Any:
        return jnp.matmul(x, params["w"]) + params["b"]

    def conv1d(params: Dict[str, Any], x: Any, stride: int, padding: int) -> Any:
        y = jax.lax.conv_general_dilated(
            x,
            params["w"],
            window_strides=(stride,),
            padding=((padding, padding),),
            dimension_numbers=("NWC", "WIO", "NWC"),
        )
        return y + params["b"][None, None, :]

    def maybe_dropout(x: Any, key: Any) -> Any:
        if config.dropout <= 0.0:
            return x
        keep_prob = 1.0 - config.dropout
        mask = jax.random.bernoulli(key, keep_prob, x.shape)
        return jnp.where(mask, x / keep_prob, 0.0)

    def aux_encoder_train(classical_params: Dict[str, Any], aux: Any, key: Any) -> Any:
        k1, _ = jax.random.split(key)
        aux = jax.nn.gelu(linear(classical_params["aux_l1"], aux))
        aux = maybe_dropout(aux, k1)
        aux = jax.nn.gelu(linear(classical_params["aux_l2"], aux))
        aux = jax.nn.gelu(linear(classical_params["aux_l3"], aux))
        return aux

    def aux_encoder_eval(classical_params: Dict[str, Any], aux: Any) -> Any:
        aux = jax.nn.gelu(linear(classical_params["aux_l1"], aux))
        aux = jax.nn.gelu(linear(classical_params["aux_l2"], aux))
        aux = jax.nn.gelu(linear(classical_params["aux_l3"], aux))
        return aux

    def spectral_encoder_train(classical_params: Dict[str, Any], spectra: Any, key: Any) -> Any:
        k1, _ = jax.random.split(key)
        spectra = jax.nn.gelu(conv1d(classical_params["spec_c1"], spectra, stride=1, padding=3))
        spectra = jax.nn.gelu(conv1d(classical_params["spec_c2"], spectra, stride=2, padding=2))
        spectra = jax.nn.gelu(conv1d(classical_params["spec_c3"], spectra, stride=1, padding=1))
        spectra = jnp.mean(spectra, axis=1)
        spectra = jax.nn.gelu(linear(classical_params["spec_proj"], spectra))
        spectra = maybe_dropout(spectra, k1)
        return spectra

    def spectral_encoder_eval(classical_params: Dict[str, Any], spectra: Any) -> Any:
        spectra = jax.nn.gelu(conv1d(classical_params["spec_c1"], spectra, stride=1, padding=3))
        spectra = jax.nn.gelu(conv1d(classical_params["spec_c2"], spectra, stride=2, padding=2))
        spectra = jax.nn.gelu(conv1d(classical_params["spec_c3"], spectra, stride=1, padding=1))
        spectra = jnp.mean(spectra, axis=1)
        spectra = jax.nn.gelu(linear(classical_params["spec_proj"], spectra))
        return spectra

    def fusion_encoder(classical_params: Dict[str, Any], aux_feat: Any, spectral_feat: Any) -> Any:
        fused = jnp.concatenate([aux_feat, spectral_feat], axis=-1)
        fused = jax.nn.gelu(linear(classical_params["fusion_l1"], fused))
        fused = linear(classical_params["fusion_l2"], fused)
        return jnp.tanh(fused) * math.pi

    def classical_trunk_train(classical_params: Dict[str, Any], aux: Any, spectra: Any, key: Any) -> Any:
        aux_key, spec_key = jax.random.split(key)
        aux_feat = aux_encoder_train(classical_params, aux, aux_key)
        spectral_feat = spectral_encoder_train(classical_params, spectra, spec_key)
        return fusion_encoder(classical_params, aux_feat, spectral_feat)

    def classical_trunk_eval(classical_params: Dict[str, Any], aux: Any, spectra: Any) -> Any:
        aux_feat = aux_encoder_eval(classical_params, aux)
        spectral_feat = spectral_encoder_eval(classical_params, spectra)
        return fusion_encoder(classical_params, aux_feat, spectral_feat)

    def classical_head_train(classical_params: Dict[str, Any], quantum_feat: Any, latent: Any, key: Any) -> Any:
        head_in = jnp.concatenate([quantum_feat, latent], axis=-1)
        head = jax.nn.gelu(linear(classical_params["head_l1"], head_in))
        head = maybe_dropout(head, key)
        head = jax.nn.gelu(linear(classical_params["head_l2"], head))
        head = linear(classical_params["head_l3"], head)
        return head + linear(classical_params["head_residual"], latent)

    def classical_head_eval(classical_params: Dict[str, Any], quantum_feat: Any, latent: Any) -> Any:
        head_in = jnp.concatenate([quantum_feat, latent], axis=-1)
        head = jax.nn.gelu(linear(classical_params["head_l1"], head_in))
        head = jax.nn.gelu(linear(classical_params["head_l2"], head))
        head = linear(classical_params["head_l3"], head)
        return head + linear(classical_params["head_residual"], latent)

    accelerated_trunk_train = catalyst.accelerate(classical_trunk_train, dev=device)
    accelerated_trunk_eval = catalyst.accelerate(classical_trunk_eval, dev=device)
    accelerated_head_train = catalyst.accelerate(classical_head_train, dev=device)
    accelerated_head_eval = catalyst.accelerate(classical_head_eval, dev=device)

    @qml.qnode(q_device, interface="jax", diff_method="adjoint")
    def quantum_circuit(latent: Any, weights: Any) -> Any:
        for qubit in range(config.qnn_qubits):
            qml.RY(latent[qubit], wires=qubit)

        param_idx = 0
        for _ in range(num_quantum_blocks):
            for qubit in range(config.qnn_qubits):
                qml.RY(weights[param_idx], wires=qubit)
                param_idx += 1
            for qubit in range(config.qnn_qubits):
                qml.CNOT(wires=[qubit, (qubit + 1) % config.qnn_qubits])
            for qubit in range(config.qnn_qubits):
                qml.RZ(weights[param_idx], wires=qubit)
                param_idx += 1
            for qubit in range(config.qnn_qubits):
                qml.CRX(weights[param_idx], wires=[qubit, (qubit + 1) % config.qnn_qubits])
                param_idx += 1

        return [qml.expval(qml.PauliZ(qubit)) for qubit in range(config.qnn_qubits)]

    def quantum_batch(latent_batch: Any, weights: Any) -> Any:
        batch_size = latent_batch.shape[0]
        outputs = jnp.zeros((batch_size, config.qnn_qubits), dtype=dtype)

        @catalyst.for_loop(0, batch_size, 1)
        def loop(i: int, running: Any) -> Any:
            quantum_row = jnp.asarray(quantum_circuit(latent_batch[i], weights), dtype=dtype)
            return running.at[i].set(quantum_row)

        return loop(outputs)

    def global_norm(tree: Dict[str, Any]) -> Any:
        leaves = jax.tree_util.tree_leaves(tree)
        squared_norm = sum(jnp.sum(jnp.square(leaf)) for leaf in leaves)
        return jnp.sqrt(squared_norm + jnp.array(1.0e-12, dtype=dtype))

    def clip_tree(tree: Dict[str, Any]) -> tuple[Dict[str, Any], Any]:
        norm = global_norm(tree)
        scale = jnp.minimum(jnp.array(1.0, dtype=dtype), jnp.array(config.gradient_clip_norm, dtype=dtype) / (norm + 1.0e-6))
        clipped = jax.tree_util.tree_map(lambda leaf: leaf * scale, tree)
        return clipped, norm

    def adamw_group(
        params_group: Dict[str, Any],
        grads_group: Dict[str, Any],
        m_group: Dict[str, Any],
        v_group: Dict[str, Any],
        learning_rate: Any,
        weight_decay: float,
        step: Any,
    ) -> tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        beta1 = 0.9
        beta2 = 0.999
        eps = 1.0e-8

        next_m = jax.tree_util.tree_map(lambda m, g: beta1 * m + (1.0 - beta1) * g, m_group, grads_group)
        next_v = jax.tree_util.tree_map(lambda v, g: beta2 * v + (1.0 - beta2) * (g * g), v_group, grads_group)
        bias_correction1 = 1.0 - jnp.power(beta1, step.astype(dtype))
        bias_correction2 = 1.0 - jnp.power(beta2, step.astype(dtype))

        next_params = jax.tree_util.tree_map(
            lambda p, g, m, v: p
            - learning_rate
            * ((m / bias_correction1) / (jnp.sqrt(v / bias_correction2) + eps) + weight_decay * p),
            params_group,
            grads_group,
            next_m,
            next_v,
        )
        return next_params, next_m, next_v

    def forward_train(params: Dict[str, Any], aux_batch: Any, spectra_batch: Any, dropout_seed: Any) -> Any:
        trunk_key, head_key = jax.random.split(jax.random.PRNGKey(dropout_seed))
        latent = accelerated_trunk_train(params["classical"], aux_batch, spectra_batch, trunk_key)
        quantum_feat = quantum_batch(latent, params["quantum"]["weights"])
        return accelerated_head_train(params["classical"], quantum_feat, latent, head_key)

    def forward_eval(params: Dict[str, Any], aux_batch: Any, spectra_batch: Any) -> Any:
        latent = accelerated_trunk_eval(params["classical"], aux_batch, spectra_batch)
        quantum_feat = quantum_batch(latent, params["quantum"]["weights"])
        return accelerated_head_eval(params["classical"], quantum_feat, latent)

    @qml.qjit
    def train_step(
        params: Dict[str, Any],
        optimizer_state: Dict[str, Any],
        aux_batch: Any,
        spectra_batch: Any,
        targets_batch: Any,
        dropout_seed: Any,
        classical_lr: Any,
        quantum_lr: Any,
    ) -> Any:
        def loss_fn(model_params: Dict[str, Any]) -> Any:
            pred = forward_train(model_params, aux_batch, spectra_batch, dropout_seed)
            residual = pred - targets_batch
            return jnp.mean(residual * residual)

        loss, grads = catalyst.value_and_grad(loss_fn, method="auto")(params)
        classical_grads, classical_grad_norm = clip_tree(grads["classical"])
        quantum_grads, quantum_grad_norm = clip_tree(grads["quantum"])
        step = optimizer_state["step"] + jnp.array(1, dtype=jnp.int32)

        next_classical, next_classical_m, next_classical_v = adamw_group(
            params["classical"],
            classical_grads,
            optimizer_state["classical_m"],
            optimizer_state["classical_v"],
            classical_lr,
            config.weight_decay,
            step,
        )
        next_quantum, next_quantum_m, next_quantum_v = adamw_group(
            params["quantum"],
            quantum_grads,
            optimizer_state["quantum_m"],
            optimizer_state["quantum_v"],
            quantum_lr,
            0.0,
            step,
        )

        next_params = {"classical": next_classical, "quantum": next_quantum}
        next_optimizer_state = {
            "step": step,
            "classical_m": next_classical_m,
            "classical_v": next_classical_v,
            "quantum_m": next_quantum_m,
            "quantum_v": next_quantum_v,
        }
        return next_params, next_optimizer_state, loss, classical_grad_norm, quantum_grad_norm

    @qml.qjit
    def predict_step(params: Dict[str, Any], aux_batch: Any, spectra_batch: Any) -> Any:
        return forward_eval(params, aux_batch, spectra_batch)

    def parameter_count(params: Dict[str, Any]) -> int:
        leaves = jax.tree_util.tree_leaves(params)
        return int(sum(int(np.prod(np.asarray(leaf).shape)) for leaf in leaves))

    return JAXTrainingRuntime(
        libs=libs,
        device=device,
        dtype=dtype,
        train_step=train_step,
        predict_step=predict_step,
        init_params=init_params,
        init_optimizer_state=init_optimizer_state,
        parameter_count=parameter_count,
        metadata={
            "device_backend": config.device_backend,
            "jax_device": str(device),
            "qnn_qubits": config.qnn_qubits,
            "qnn_depth": config.qnn_depth,
            "num_quantum_weights": num_quantum_weights,
        },
    )


def _tree_to_numpy(tree: Any, libs: RuntimeImports) -> Any:
    return libs.jax.tree_util.tree_map(lambda leaf: np.asarray(leaf), tree)


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def save_pickle(path: Path, payload: Dict[str, Any]) -> None:
    with path.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)


def metrics_frame(split_name: str, metrics: Dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame({"split": split_name, "target": TARGET_COLS, "rmse": metrics["rmse_orig"]})


def save_predictions(path: Path, split: SplitArrays, metrics: Dict[str, Any]) -> None:
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


def _epoch_index_matrix(runtime: JAXTrainingRuntime, size: int, batch_size: int, seed: int) -> Any:
    jnp = runtime.libs.jnp
    permutation = runtime.libs.jax.random.permutation(runtime.libs.jax.random.PRNGKey(seed), size)
    steps = math.ceil(size / batch_size)
    padding = steps * batch_size - size
    if padding > 0:
        permutation = jnp.concatenate([permutation, permutation[:padding]], axis=0)
    return permutation.reshape(steps, batch_size)


def _eval_batch_plan(size: int, batch_size: int) -> list[tuple[int, int]]:
    return [(start, min(start + batch_size, size)) for start in range(0, size, batch_size)]


def _gather_batch(split: SplitArrays, indices: Any, libs: RuntimeImports) -> tuple[Any, Any, Any]:
    aux = libs.jnp.take(split.aux, indices, axis=0)
    spectra = libs.jnp.take(split.spectra, indices, axis=0)
    targets = libs.jnp.take(split.targets, indices, axis=0)
    return aux, spectra, targets


def _warmup_compile(
    config: TrainingConfigJAX,
    data: PreparedDataJAX,
    runtime: JAXTrainingRuntime,
    params: Dict[str, Any],
    optimizer_state: Dict[str, Any],
) -> Dict[str, float]:
    if config.compile_warmup_steps <= 0:
        return {"compile_warmup_seconds": 0.0}

    index_matrix = _epoch_index_matrix(runtime, len(data.train.sample_ids), config.batch_size_train, config.seed)
    warmup_seconds = 0.0
    for step_idx in range(config.compile_warmup_steps):
        aux_batch, spectra_batch, targets_batch = _gather_batch(data.train, index_matrix[step_idx], runtime.libs)
        step_start = time.perf_counter()
        _, _, loss, _, _ = runtime.train_step(
            params,
            optimizer_state,
            aux_batch,
            spectra_batch,
            targets_batch,
            runtime.libs.jnp.array(config.seed + step_idx, dtype=runtime.libs.jnp.int32),
            runtime.libs.jnp.array(config.classical_lr, dtype=runtime.dtype),
            runtime.libs.jnp.array(config.quantum_lr, dtype=runtime.dtype),
        )
        float(np.asarray(loss))
        warmup_seconds += time.perf_counter() - step_start
    return {"compile_warmup_seconds": warmup_seconds}


def autotune_batch_size(
    config: TrainingConfigJAX,
    data: PreparedDataJAX,
    runtime: JAXTrainingRuntime,
) -> Dict[str, Any]:
    candidates = sorted({int(candidate) for candidate in config.performance_batch_ladder if candidate > 0})
    candidates = [candidate for candidate in candidates if candidate <= len(data.train.sample_ids)]
    if not candidates:
        return {
            "enabled": False,
            "selected_train_batch_size": config.batch_size_train,
            "selected_eval_batch_size": config.batch_size_eval,
            "results": [],
        }

    benchmark_results: list[Dict[str, Any]] = []
    for batch_size in candidates:
        params = runtime.init_params(config.seed)
        optimizer_state = runtime.init_optimizer_state(params)
        index_matrix = _epoch_index_matrix(runtime, len(data.train.sample_ids), batch_size, config.seed)
        steps_to_run = min(int(index_matrix.shape[0]), max(config.compile_warmup_steps + 2, 3))
        compile_seconds = 0.0
        steady_seconds = 0.0
        measured_samples = 0
        telemetry_samples: list[Dict[str, Optional[float]]] = []

        try:
            for step_idx in range(steps_to_run):
                aux_batch, spectra_batch, targets_batch = _gather_batch(data.train, index_matrix[step_idx], runtime.libs)
                step_start = time.perf_counter()
                params, optimizer_state, loss, _, _ = runtime.train_step(
                    params,
                    optimizer_state,
                    aux_batch,
                    spectra_batch,
                    targets_batch,
                    runtime.libs.jnp.array(config.seed + step_idx, dtype=runtime.libs.jnp.int32),
                    runtime.libs.jnp.array(config.classical_lr, dtype=runtime.dtype),
                    runtime.libs.jnp.array(config.quantum_lr, dtype=runtime.dtype),
                )
                float(np.asarray(loss))
                elapsed = time.perf_counter() - step_start
                telemetry_samples.append(_sample_system_metrics())
                if step_idx < config.compile_warmup_steps:
                    compile_seconds += elapsed
                else:
                    steady_seconds += elapsed
                    measured_samples += batch_size
        except Exception as exc:
            benchmark_results.append({"batch_size": batch_size, "status": "failed", "error": str(exc)})
            continue

        gpu_utils = [sample["gpu_util"] for sample in telemetry_samples if sample["gpu_util"] is not None]
        gpu_mems = [sample["gpu_mem_mb"] for sample in telemetry_samples if sample["gpu_mem_mb"] is not None]
        cpu_pcts = [sample["cpu_percent"] for sample in telemetry_samples if sample["cpu_percent"] is not None]

        samples_per_second = 0.0 if steady_seconds <= 0 else measured_samples / steady_seconds
        benchmark_results.append(
            {
                "batch_size": batch_size,
                "status": "ok",
                "compile_warmup_seconds": compile_seconds,
                "steady_state_seconds": steady_seconds,
                "samples_per_second": samples_per_second,
                "mean_gpu_util": None if not gpu_utils else float(np.mean(gpu_utils)),
                "mean_gpu_mem_mb": None if not gpu_mems else float(np.mean(gpu_mems)),
                "mean_cpu_percent": None if not cpu_pcts else float(np.mean(cpu_pcts)),
            }
        )

    successful = [result for result in benchmark_results if result.get("status") == "ok"]
    if not successful:
        return {
            "enabled": True,
            "selected_train_batch_size": config.batch_size_train,
            "selected_eval_batch_size": config.batch_size_eval,
            "results": benchmark_results,
        }

    best = max(successful, key=lambda item: item["samples_per_second"])
    config.batch_size_train = int(best["batch_size"])
    config.batch_size_eval = max(config.batch_size_eval, config.batch_size_train * 4)
    return {
        "enabled": True,
        "selected_train_batch_size": config.batch_size_train,
        "selected_eval_batch_size": config.batch_size_eval,
        "results": benchmark_results,
    }


def evaluate_split(
    runtime: JAXTrainingRuntime,
    params: Dict[str, Any],
    split: SplitArrays,
    target_scaler: ArrayStandardizer,
    batch_size: int,
) -> Dict[str, Any]:
    pred_batches = []
    target_batches = []
    losses = []

    for start, stop in _eval_batch_plan(len(split.sample_ids), batch_size):
        indices = runtime.libs.jnp.arange(start, stop, dtype=runtime.libs.jnp.int32)
        aux_batch, spectra_batch, targets_batch = _gather_batch(split, indices, runtime.libs)
        pred = runtime.predict_step(params, aux_batch, spectra_batch)
        pred_np = np.asarray(pred)
        target_np = np.asarray(targets_batch)
        pred_batches.append(pred_np)
        target_batches.append(target_np)
        losses.append(float(np.mean((pred_np - target_np) ** 2)))

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
        f"{target_name}={rmse_value:.4f}" for target_name, rmse_value in zip(TARGET_COLS, metrics["rmse_orig"])
    )


def train_model(
    config: TrainingConfigJAX,
    data: PreparedDataJAX,
    runtime: JAXTrainingRuntime,
) -> Dict[str, Any]:
    params = runtime.init_params(config.seed)
    optimizer_state = runtime.init_optimizer_state(params)
    total_batches = math.ceil(len(data.train.sample_ids) / config.batch_size_train)

    print(
        f"JAX device: {runtime.device} | Quantum device: {config.device_backend} | "
        f"Dataset devices: {dataset_devices(data)} | Parameters: {runtime.parameter_count(params)}",
        flush=True,
    )

    warmup_profile = _warmup_compile(config, data, runtime, params, optimizer_state)
    if warmup_profile["compile_warmup_seconds"] > 0:
        print(
            f"Compile warmup | calls={config.compile_warmup_steps} | "
            f"seconds={warmup_profile['compile_warmup_seconds']:.2f}",
            flush=True,
        )

    history: list[Dict[str, Any]] = []
    batch_history: list[Dict[str, Any]] = []
    best_val_loss = float("inf")
    best_epoch = -1
    best_params: Optional[Dict[str, Any]] = None
    patience_left = config.early_stop_patience
    plateau_left = config.scheduler_patience
    classical_lr = config.classical_lr
    quantum_lr = config.quantum_lr

    for epoch in range(config.max_epochs):
        epoch_start = time.perf_counter()
        epoch_losses = []
        epoch_classical_grad_norms = []
        epoch_quantum_grad_norms = []
        index_matrix = _epoch_index_matrix(runtime, len(data.train.sample_ids), config.batch_size_train, config.seed + epoch)

        for batch_idx in range(total_batches):
            batch_start = time.perf_counter()
            aux_batch, spectra_batch, targets_batch = _gather_batch(data.train, index_matrix[batch_idx], runtime.libs)
            params, optimizer_state, loss, classical_grad_norm, quantum_grad_norm = runtime.train_step(
                params,
                optimizer_state,
                aux_batch,
                spectra_batch,
                targets_batch,
                runtime.libs.jnp.array(config.seed + epoch * total_batches + batch_idx, dtype=runtime.libs.jnp.int32),
                runtime.libs.jnp.array(classical_lr, dtype=runtime.dtype),
                runtime.libs.jnp.array(quantum_lr, dtype=runtime.dtype),
            )
            batch_loss = float(np.asarray(loss))
            class_grad = float(np.asarray(classical_grad_norm))
            quant_grad = float(np.asarray(quantum_grad_norm))
            telemetry = _sample_system_metrics()
            batch_seconds = time.perf_counter() - batch_start
            samples_per_second = config.batch_size_train / batch_seconds if batch_seconds > 0 else 0.0

            epoch_losses.append(batch_loss)
            epoch_classical_grad_norms.append(class_grad)
            epoch_quantum_grad_norms.append(quant_grad)
            batch_history.append(
                {
                    "epoch": epoch + 1,
                    "batch": batch_idx + 1,
                    "batch_loss": batch_loss,
                    "classical_grad_norm": class_grad,
                    "quantum_grad_norm": quant_grad,
                    "batch_seconds": batch_seconds,
                    "samples_per_second": samples_per_second,
                    **telemetry,
                }
            )

            if config.log_every_batches > 0 and (
                (batch_idx + 1) % config.log_every_batches == 0 or (batch_idx + 1) == total_batches
            ):
                print(
                    f"Epoch {epoch + 1}/{config.max_epochs} | Batch {batch_idx + 1}/{total_batches} | "
                    f"batch_loss={batch_loss:.5f} | avg_loss={np.mean(epoch_losses):.5f} | "
                    f"class_grad={class_grad:.3f} | quant_grad={quant_grad:.3f} | "
                    f"samples_per_sec={samples_per_second:.1f} | batch_time={batch_seconds:.2f}s | "
                    f"gpu_util={telemetry['gpu_util']} | gpu_mem_mb={telemetry['gpu_mem_mb']} | cpu_pct={telemetry['cpu_percent']}",
                    flush=True,
                )

        inner_val_metrics = evaluate_split(runtime, params, data.inner_val, data.target_scaler, config.batch_size_eval)
        epoch_seconds = time.perf_counter() - epoch_start
        history_row = {
            "epoch": epoch + 1,
            "train_loss": float(np.mean(epoch_losses)),
            "inner_val_loss": float(inner_val_metrics["loss"]),
            "inner_val_rmse_mean": float(inner_val_metrics["rmse_mean"]),
            "epoch_seconds": epoch_seconds,
            "classical_lr": classical_lr,
            "quantum_lr": quantum_lr,
            "classical_grad_norm_mean": float(np.mean(epoch_classical_grad_norms)),
            "quantum_grad_norm_mean": float(np.mean(epoch_quantum_grad_norms)),
        }
        for target_name, rmse_value in zip(TARGET_COLS, inner_val_metrics["rmse_orig"]):
            history_row[f"inner_val_rmse_{target_name}"] = float(rmse_value)
        history.append(history_row)

        print(
            f"Epoch {epoch + 1}/{config.max_epochs} | train_loss={history_row['train_loss']:.5f} | "
            f"inner_val_loss={history_row['inner_val_loss']:.5f} | "
            f"inner_val_rmse_mean={history_row['inner_val_rmse_mean']:.5f} | "
            f"time={epoch_seconds:.1f}s | lr=({classical_lr:.2e}, {quantum_lr:.2e})",
            flush=True,
        )
        print(f"Epoch {epoch + 1} inner-val target RMSE | {format_target_rmse(inner_val_metrics)}", flush=True)

        if inner_val_metrics["loss"] < best_val_loss:
            best_val_loss = float(inner_val_metrics["loss"])
            best_epoch = epoch + 1
            best_params = _tree_to_numpy(params, runtime.libs)
            patience_left = config.early_stop_patience
            plateau_left = config.scheduler_patience
        else:
            patience_left -= 1
            plateau_left -= 1
            if plateau_left <= 0:
                classical_lr *= config.scheduler_factor
                quantum_lr *= config.scheduler_factor
                plateau_left = config.scheduler_patience
                print(
                    f"ReduceLROnPlateau | new_lrs=({classical_lr:.2e}, {quantum_lr:.2e})",
                    flush=True,
                )
            if patience_left <= 0:
                print(f"Early stopping at epoch {epoch + 1}.", flush=True)
                break

    if best_params is None:
        raise RuntimeError("Training did not produce a checkpoint.")

    return {
        "history": history,
        "batch_history": batch_history,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "best_params": best_params,
        "last_params": _tree_to_numpy(params, runtime.libs),
        "warmup_profile": warmup_profile,
    }


def run_training_experiment(config: Optional[TrainingConfigJAX] = None) -> Dict[str, Any]:
    cfg = config or TrainingConfigJAX()
    cfg.project_root = str(cfg.resolved_project_root())

    libs = import_jax_stack()
    set_runtime_seed(cfg.seed)
    configure_runtime(libs, cfg)
    device = resolve_jax_device(libs, cfg)

    print(f"Project root: {cfg.resolved_project_root()}", flush=True)
    print(f"Data root: {cfg.resolved_data_root()}", flush=True)
    print(f"Output dir: {cfg.resolved_output_dir()}", flush=True)
    print(f"Train batch size: {cfg.batch_size_train}", flush=True)
    print(f"Eval batch size: {cfg.batch_size_eval}", flush=True)
    print(f"Quantum device: {cfg.device_backend}", flush=True)
    print(f"Quantum width/depth: {cfg.qnn_qubits}/{cfg.qnn_depth}", flush=True)
    print(f"Precision: {cfg.precision}", flush=True)

    data = prepare_data(cfg)
    output_dir = cfg.resolved_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)

    save_json(output_dir / "config.json", cfg.to_json_dict())
    save_json(output_dir / "preflight.json", data.preflight)

    data = move_prepared_data_to_device(data, libs, device)
    runtime = build_runtime(cfg, libs, device)

    autotune_summary = {
        "enabled": False,
        "selected_train_batch_size": cfg.batch_size_train,
        "selected_eval_batch_size": cfg.batch_size_eval,
        "results": [],
    }
    if cfg.autotune_batch:
        autotune_summary = autotune_batch_size(cfg, data, runtime)
        save_json(output_dir / "config.json", cfg.to_json_dict())
        save_json(output_dir / "autotune.json", autotune_summary)
        print(
            f"Autotune | train_batch={cfg.batch_size_train} | eval_batch={cfg.batch_size_eval}",
            flush=True,
        )

    training = train_model(cfg, data, runtime)
    best_params = training["best_params"]
    last_params = training["last_params"]

    best_model_path = output_dir / "best_model.pkl"
    last_model_path = output_dir / "last_model.pkl"
    save_pickle(
        best_model_path,
        {
            "config": cfg.to_json_dict(),
            "feature_cols": SAFE_AUX_FEATURE_COLS,
            "target_cols": TARGET_COLS,
            "best_epoch": training["best_epoch"],
            "best_val_loss": training["best_val_loss"],
            "params": best_params,
            "runtime": runtime.metadata,
        },
    )
    save_pickle(
        last_model_path,
        {
            "config": cfg.to_json_dict(),
            "feature_cols": SAFE_AUX_FEATURE_COLS,
            "target_cols": TARGET_COLS,
            "params": last_params,
            "runtime": runtime.metadata,
        },
    )

    save_json(
        output_dir / "scalers.json",
        {
            "aux_scaler": data.aux_scaler.state_dict(),
            "target_scaler": data.target_scaler.state_dict(),
            "spectral_scaler": data.spectral_scaler.state_dict(),
        },
    )
    save_json(
        output_dir / "runtime_metadata.json",
        {
            **runtime.metadata,
            "dataset_devices": dataset_devices(data),
            **training["warmup_profile"],
        },
    )

    history_frame = pd.DataFrame(training["history"])
    history_frame.to_csv(output_dir / "history.csv", index=False)
    batch_history_frame = pd.DataFrame(training["batch_history"])
    batch_history_frame.to_csv(output_dir / "batch_history.csv", index=False)
    plot_paths = save_history_plots(output_dir, history_frame)

    best_params_device = libs.jax.device_put(best_params, device)
    inner_val_metrics = evaluate_split(
        runtime, best_params_device, data.inner_val, data.target_scaler, cfg.batch_size_eval
    )
    tau_test_metrics = evaluate_split(
        runtime, best_params_device, data.tau_test, data.target_scaler, cfg.batch_size_eval
    )
    poseidon_metrics = evaluate_split(
        runtime, best_params_device, data.poseidon, data.target_scaler, cfg.batch_size_eval
    )

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
        "autotuned_train_batch_size": cfg.batch_size_train,
        "autotuned_eval_batch_size": cfg.batch_size_eval,
    }
    save_json(output_dir / "run_summary.json", summary)

    return {
        "config": cfg.to_json_dict(),
        "preflight": data.preflight,
        "autotune_summary": autotune_summary,
        "history_frame": history_frame,
        "batch_history_frame": batch_history_frame,
        "inner_val_metrics_frame": metrics_frame("inner_val", inner_val_metrics),
        "tau_test_metrics_frame": tau_frame,
        "poseidon_metrics_frame": poseidon_frame,
        "summary": summary,
        "artifacts": {
            "best_model_pkl": str(best_model_path),
            "last_model_pkl": str(last_model_path),
            "scalers_json": str(output_dir / "scalers.json"),
            "runtime_metadata_json": str(output_dir / "runtime_metadata.json"),
            "history_csv": str(output_dir / "history.csv"),
            "batch_history_csv": str(output_dir / "batch_history.csv"),
            "tau_test_predictions_csv": str(output_dir / "tau_test_predictions.csv"),
            "poseidon_predictions_csv": str(output_dir / "poseidon_predictions.csv"),
            "metrics_summary_csv": str(output_dir / "metrics_summary.csv"),
            **plot_paths,
        },
    }
