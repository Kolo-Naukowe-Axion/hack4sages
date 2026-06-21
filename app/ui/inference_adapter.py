"""UI-local inference seam (Osoba 4).

Keeps the Streamlit UI decoupled from model internals. Wraps the frozen
checkpoint bridge plus the functional data pipeline into a single
``predict(record) -> Prediction``, and reads the checkpoint's precomputed
full-quantum predictions and ground truth for the comparison view.

This is the ONLY place ``app/ui`` touches the model layer, so when the planned
``app/models`` (Osoba 1) or ``app/inference.py`` (Osoba 3) land, only this file
changes. Heavy imports (torch, the checkpoint package) are deferred so the pure
helpers stay importable and testable without them.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd

from app.data.comparison import aggregate, build_comparison
from app.data.pipeline import make_quantum_input
from app.data.preprocessing import denormalize_targets, vmr_to_dict
from app.data.types import GASES, ComparisonRow, GroundTruth, PlanetRecord, Prediction

CLASSICAL_MODEL_NAME = "klasyczny"
QUANTUM_MODEL_NAME = "kwantowy"
BASELINE_MODEL_NAME = "baseline"
VALIDATION_CSV = "validation_predictions.csv"

# Per-gas mean log-VMR over the ADC2023 training set (FM_Parameter_Table). A
# constant "null model": predict the prior mean for every planet, ignoring the
# spectrum entirely. If the trained model beats it, it actually read the
# spectrum. mRMSE of this baseline on the validation set is ~1.49 (model: ~0.31).
BASELINE_LOG_VMR: dict[str, float] = {
    "log_H2O": -5.983,
    "log_CO2": -6.522,
    "log_CO": -4.492,
    "log_CH4": -5.997,
    "log_NH3": -6.494,
}


def baseline_prediction() -> Prediction:
    """The constant prior-mean baseline (null model) as a Prediction."""
    return Prediction(BASELINE_MODEL_NAME, dict(BASELINE_LOG_VMR))


@dataclass(frozen=True)
class InferenceEngine:
    """Loaded checkpoint bundle, frozen bridge, and the model-input transform.

    Build once (cache it in the UI with ``st.cache_resource``) and reuse across
    reruns. ``transform`` is the closure returned by ``make_quantum_input`` that
    maps a ``PlanetRecord`` to ``(aux_n (1, 8), spectra_n (1, 4, 52))``.
    """

    bundle: Any
    bridge: Any
    transform: Any

    @property
    def checkpoint_dir(self) -> Path:
        return Path(self.bundle.checkpoint_dir)

    def predict(self, record: PlanetRecord) -> Prediction:
        """Run live classical-head inference -> physical log-VMR ``Prediction``."""
        import torch

        aux_n, spectra_n = self.transform(record)
        with torch.inference_mode():
            encoded = self.bridge.encode_features(aux_n, spectra_n)
            pred_norm = self.bridge.classical_predict(encoded["head_context"]).cpu().numpy()
        pred_phys = denormalize_targets(pred_norm, self.bundle.target_scaler)
        return Prediction(CLASSICAL_MODEL_NAME, vmr_to_dict(pred_phys[0]))


def load_engine(checkpoint_dir: str | Path | None = None) -> InferenceEngine:
    """Load the frozen hybrid checkpoint (classical head, no PennyLane needed)."""
    from models.garnet_ariel_quantum_regression.checkpoint import (
        FrozenArielHybridBridge,
        load_checkpoint_bundle,
    )

    bundle = (
        load_checkpoint_bundle()
        if checkpoint_dir is None
        else load_checkpoint_bundle(checkpoint_dir)
    )
    bridge = FrozenArielHybridBridge(bundle)
    transform = make_quantum_input(bundle.aux_scaler, bundle.spectral_scaler)
    return InferenceEngine(bundle=bundle, bridge=bridge, transform=transform)


@lru_cache(maxsize=4)
def _reference_table(csv_path: str) -> pd.DataFrame:
    return pd.read_csv(csv_path).set_index("planet_ID")


def _reference_row(checkpoint_dir: str | Path, planet_id: str) -> "pd.Series | None":
    csv = Path(checkpoint_dir) / VALIDATION_CSV
    if not csv.exists():
        return None
    table = _reference_table(str(csv))
    if planet_id not in table.index:
        return None
    return table.loc[planet_id]


def full_quantum_prediction(checkpoint_dir: str | Path, planet_id: str) -> Prediction | None:
    """Real full-quantum-model prediction for ``planet_id`` from the checkpoint's
    ``validation_predictions.csv`` (``pred_<gas>`` columns). ``None`` if absent."""
    row = _reference_row(checkpoint_dir, planet_id)
    if row is None:
        return None
    try:
        log_vmr = {gas: float(row[f"pred_{gas}"]) for gas in GASES}
    except KeyError:
        return None
    return Prediction(QUANTUM_MODEL_NAME, log_vmr)


def reference_truth(checkpoint_dir: str | Path, planet_id: str) -> GroundTruth | None:
    """Ground truth from the checkpoint CSV (``true_<gas>``), for planets whose
    record carries no truth (e.g. uploads). ``None`` if absent."""
    row = _reference_row(checkpoint_dir, planet_id)
    if row is None:
        return None
    try:
        return GroundTruth({gas: float(row[f"true_{gas}"]) for gas in GASES})
    except KeyError:
        return None


def compare(
    truth: GroundTruth | None, preds: list[Prediction]
) -> tuple[list[ComparisonRow], dict[str, dict[str, float]]]:
    """Per-gas comparison rows plus per-model RMSE, from in-memory predictions."""
    mapping = {pred.model_name: pred for pred in preds}
    rows = build_comparison(truth, mapping)
    return rows, aggregate(rows)
