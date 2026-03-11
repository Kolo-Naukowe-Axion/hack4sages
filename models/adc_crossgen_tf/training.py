"""Training loop for the TensorFlow ADC baseline adaptation."""

from __future__ import annotations

import json
import logging
import os
import random
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import tensorflow as tf

from .constants import (
    AUX_COLUMNS,
    DEFAULT_DATA_ROOT,
    DEFAULT_OUTPUT_DIR,
    PRESENCE_COLUMNS,
    PRESENCE_THRESHOLD_LOG10_VMR,
    TARGET_COLUMNS,
)
from .dataset import PreparedData, prepare_data
from .model import ScaleLayer, build_model
from .quantum import QuantumCircuitLayer


_TF_COMPLEX_CAST_WARNING = (
    "You are casting an input of type complex128 to an incompatible dtype float32."
)


class _TensorFlowWarningFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return _TF_COMPLEX_CAST_WARNING not in record.getMessage()


@dataclass
class TrainingConfig:
    project_root: str = "."
    data_root: str = str(DEFAULT_DATA_ROOT)
    output_dir: str = str(DEFAULT_OUTPUT_DIR)
    seed: int = 42
    epochs: int = 30
    batch_size: int = 32
    lr: float = 1.0e-3
    augment_repeat: int = 5
    dropout: float = 0.1
    mc_samples: int = 64
    qnn_qubits: int = 4
    qnn_depth: int = 2
    train_limit: int | None = None
    val_limit: int | None = None
    poseidon_limit: int | None = None
    patience: int = 6
    quantum_device_name: str = "default.qubit"
    save_mc_samples: bool = False

    def resolved_project_root(self) -> Path:
        return Path(self.project_root).expanduser().resolve()

    def resolved_data_root(self) -> Path:
        data_root = Path(self.data_root).expanduser()
        if data_root.is_absolute():
            return data_root
        return self.resolved_project_root() / data_root

    def resolved_output_dir(self) -> Path:
        output_dir = Path(self.output_dir).expanduser()
        if output_dir.is_absolute():
            return output_dir
        return self.resolved_project_root() / output_dir

    def to_json_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["project_root"] = str(self.resolved_project_root())
        payload["data_root"] = str(self.resolved_data_root())
        payload["output_dir"] = str(self.resolved_output_dir())
        payload["aux_columns"] = AUX_COLUMNS
        payload["target_columns"] = TARGET_COLUMNS
        payload["presence_columns"] = PRESENCE_COLUMNS
        return payload


def set_global_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    warnings.filterwarnings(
        "ignore",
        message=r"Support for the TensorFlow interface is deprecated.*",
        category=Warning,
    )
    logger = tf.get_logger()
    if not any(isinstance(existing, _TensorFlowWarningFilter) for existing in logger.filters):
        logger.addFilter(_TensorFlowWarningFilter())


def _predict_mc(model: tf.keras.Model, split, mc_samples: int, batch_size: int) -> np.ndarray:
    predictions = []
    for _ in range(int(mc_samples)):
        batch_predictions = []
        for start in range(0, split.rows, batch_size):
            stop = min(start + batch_size, split.rows)
            batch = model(
                [split.spectra[start:stop], split.aux[start:stop]],
                training=True,
            )
            batch_predictions.append(np.asarray(batch.numpy(), dtype=np.float32))
        predictions.append(np.concatenate(batch_predictions, axis=0))
    return np.stack(predictions, axis=0)


def predict_mc_mean(
    model: tf.keras.Model,
    split,
    target_scaler,
    mc_samples: int,
    batch_size: int,
) -> np.ndarray:
    mc_predictions = _predict_mc(model, split, mc_samples=mc_samples, batch_size=batch_size)
    return target_scaler.inverse_transform(mc_predictions.mean(axis=0))


def _evaluate_split(
    split,
    mean_predictions: np.ndarray,
) -> dict[str, Any]:
    rmse = np.sqrt(np.mean((mean_predictions - split.raw_targets) ** 2, axis=0))
    predicted_presence = (mean_predictions >= PRESENCE_THRESHOLD_LOG10_VMR).astype(np.int64)
    presence_accuracy = (predicted_presence == split.presence).mean(axis=0)
    return {
        "rows": split.rows,
        "rmse_mean": float(rmse.mean()),
        "rmse": {name: float(value) for name, value in zip(TARGET_COLUMNS, rmse)},
        "presence_accuracy": {name: float(value) for name, value in zip(PRESENCE_COLUMNS, presence_accuracy)},
    }


def evaluate_model_split(
    model: tf.keras.Model,
    split,
    target_scaler,
    mc_samples: int,
    batch_size: int,
) -> dict[str, Any]:
    mean_predictions = predict_mc_mean(
        model=model,
        split=split,
        target_scaler=target_scaler,
        mc_samples=mc_samples,
        batch_size=batch_size,
    )
    return _evaluate_split(split, mean_predictions)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _write_prediction_csv(path: Path, sample_ids: np.ndarray, predictions: np.ndarray) -> None:
    frame = pd.DataFrame(predictions, columns=[f"pred_{name}" for name in TARGET_COLUMNS])
    frame.insert(0, "sample_id", sample_ids.tolist())
    frame.to_csv(path, index=False)


def _load_best_model(model_path: Path) -> tf.keras.Model:
    return tf.keras.models.load_model(
        model_path,
        custom_objects={"QuantumCircuitLayer": QuantumCircuitLayer, "ScaleLayer": ScaleLayer},
    )


def _batch_slices(length: int, batch_size: int) -> list[tuple[int, int]]:
    return [(start, min(start + batch_size, length)) for start in range(0, length, batch_size)]


def train_and_evaluate(config: TrainingConfig) -> dict[str, Any]:
    set_global_seed(config.seed)

    output_dir = config.resolved_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "config.json", config.to_json_dict())

    prepared = prepare_data(
        data_root=config.resolved_data_root(),
        seed=config.seed,
        augment_repeat=config.augment_repeat,
        train_limit=config.train_limit,
        val_limit=config.val_limit,
        poseidon_limit=config.poseidon_limit,
    )

    model = build_model(
        spectrum_length=prepared.train.spectra.shape[1],
        aux_dim=prepared.train.aux.shape[1],
        target_dim=prepared.train.targets.shape[1],
        dropout=config.dropout,
        qnn_qubits=config.qnn_qubits,
        qnn_depth=config.qnn_depth,
        quantum_device_name=config.quantum_device_name,
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.lr),
        loss="mse",
        run_eagerly=True,
    )

    best_model_path = output_dir / "best_model.keras"
    history_rows: list[dict[str, float | int]] = []
    best_val_loss = float("inf")
    epochs_without_improvement = 0

    train_batches = _batch_slices(prepared.train.rows, config.batch_size)
    val_batches = _batch_slices(prepared.tau_val.rows, config.batch_size)

    for epoch in range(1, config.epochs + 1):
        train_losses = []
        for start, stop in train_batches:
            loss = model.train_on_batch(
                [prepared.train.spectra[start:stop], prepared.train.aux[start:stop]],
                prepared.train.targets[start:stop],
            )
            train_losses.append(float(loss))

        val_losses = []
        for start, stop in val_batches:
            loss = model.test_on_batch(
                [prepared.tau_val.spectra[start:stop], prepared.tau_val.aux[start:stop]],
                prepared.tau_val.targets[start:stop],
            )
            val_losses.append(float(loss))

        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses))
        history_rows.append({"epoch": epoch, "loss": train_loss, "val_loss": val_loss})
        print(f"Epoch {epoch}/{config.epochs} - loss={train_loss:.6f} - val_loss={val_loss:.6f}", flush=True)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            model.save(best_model_path)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.patience:
                break

    pd.DataFrame(history_rows).to_csv(output_dir / "history.csv", index=False)

    if best_model_path.exists():
        model = _load_best_model(best_model_path)
    else:
        model.save(best_model_path)

    scalers_payload = {
        "spectral": prepared.spectral_scaler.to_dict(),
        "aux": prepared.aux_scaler.to_dict(),
        "targets": prepared.target_scaler.to_dict(),
    }
    _write_json(output_dir / "scalers.json", scalers_payload)

    tau_val_mc = _predict_mc(model, prepared.tau_val, config.mc_samples, config.batch_size)
    poseidon_mc = _predict_mc(model, prepared.poseidon, config.mc_samples, config.batch_size)

    tau_val_mean = prepared.target_scaler.inverse_transform(tau_val_mc.mean(axis=0))
    poseidon_mean = prepared.target_scaler.inverse_transform(poseidon_mc.mean(axis=0))

    tau_val_metrics = _evaluate_split(prepared.tau_val, tau_val_mean)
    poseidon_metrics = _evaluate_split(prepared.poseidon, poseidon_mean)

    _write_json(output_dir / "tau_val_metrics.json", tau_val_metrics)
    _write_json(output_dir / "poseidon_metrics.json", poseidon_metrics)
    _write_prediction_csv(output_dir / "tau_val_predictions.csv", prepared.tau_val.sample_ids, tau_val_mean)
    _write_prediction_csv(output_dir / "poseidon_predictions.csv", prepared.poseidon.sample_ids, poseidon_mean)

    if config.save_mc_samples:
        np.savez_compressed(output_dir / "tau_val_mc_samples.npz", predictions=tau_val_mc)
        np.savez_compressed(output_dir / "poseidon_mc_samples.npz", predictions=poseidon_mc)

    summary = {
        "best_val_loss": float(best_val_loss),
        "epochs_ran": int(len(history_rows)),
        "tau_val_rmse_mean": float(tau_val_metrics["rmse_mean"]),
        "poseidon_rmse_mean": float(poseidon_metrics["rmse_mean"]),
        "output_dir": str(output_dir),
        "dataset": prepared.metadata,
        "gpus_visible": [device.name for device in tf.config.list_physical_devices("GPU")],
    }
    _write_json(output_dir / "run_summary.json", summary)

    result = {
        "summary": summary,
        "tau_val_metrics": tau_val_metrics,
        "poseidon_metrics": poseidon_metrics,
        "prepared": prepared,
    }
    tf.keras.backend.clear_session()
    return result
