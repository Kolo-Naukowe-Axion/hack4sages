"""Shared infrastructure for the ExoBiome methodological audit harness.

Design rules (deliberate, mirroring scripts/reeval_lib.py):
  * every check writes a self-describing JSON run: git revision, env, input sha256, verdict;
  * no check modifies model/train code — they only import and evaluate;

INVARIANTS — this harness never mutates the repository:
  * the only filesystem writes are the JSON records, which go to `<repo>/reports/audit/<UTC-date>/`.
    Nothing else in the repository is created, modified or deleted. Redirect the records with
    --out or EXOBIOME_AUDIT_OUT if you want them outside the checkout.
  * every check returns one of PASS / FAIL / WARN / INFO with a machine-readable payload,
    so `run_all.py` can assemble an inventory table that a reviewer can re-derive.

Run from the MAIN checkout (the one that has data/ and .venv-qml):
    cd ../hack4sages
    ./.venv-qml/bin/python <path-to>/audit/<needed-test>
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(os.environ.get("EXOBIOME_REPO", "/Users/mariaplatek/projects/AXION/hack4sages")).resolve()
TARGETS = ["log_H2O", "log_CO2", "log_CO", "log_CH4", "log_NH3"]
CROSSGEN_TARGETS = ["log10_vmr_h2o", "log10_vmr_co2", "log10_vmr_co", "log10_vmr_ch4", "log10_vmr_nh3"]

STATUS_ORDER = {"FAIL": 0, "WARN": 1, "INFO": 2, "PASS": 3}


# ---------------------------------------------------------------- provenance


def git_revision() -> str:
    """READ-ONLY. Returns the current HEAD hash, suffixed `-dirty` if the tree has changes.

    Both subprocess calls are read-only git plumbing (`rev-parse`, `status --porcelain`).
    This function records provenance; it does not stage, commit or modify anything.
    """
    try:
        out = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                             capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(REPO), "status", "--porcelain"],
                               capture_output=True, text=True, check=True).stdout.strip()
        return out + ("-dirty" if dirty else "")
    except Exception as exc:  # pragma: no cover
        return f"unavailable:{exc}"


def sha256(path: Path, limit_bytes: int | None = 64 << 20) -> str:
    """sha256 of a file; large files are hashed over their first `limit_bytes` (noted in the key)."""
    h = hashlib.sha256()
    read = 0
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            if limit_bytes is not None and read >= limit_bytes:
                return h.hexdigest() + f"-first{limit_bytes}B"
            h.update(chunk)
            read += len(chunk)
    return h.hexdigest()


def env_versions() -> dict[str, str]:
    def v(mod: str) -> str:
        try:
            return __import__(mod).__version__
        except Exception:
            return "not-installed"
    return {"python": sys.version.split()[0], "platform": platform.platform(),
            "numpy": v("numpy"), "pandas": v("pandas"), "torch": v("torch"),
            "pennylane": v("pennylane"), "zuko": v("zuko"), "h5py": v("h5py"), "scipy": v("scipy")}


AUDIT_DIR = Path(__file__).resolve().parent


def out_dir(explicit: str | None = None) -> Path:
    """Where the JSON records go. This is the ONLY thing the harness writes."""
    if explicit:
        d = Path(explicit)
    elif os.environ.get("EXOBIOME_AUDIT_OUT"):
        d = Path(os.environ["EXOBIOME_AUDIT_OUT"]) / datetime.now(timezone.utc).strftime("%Y%m%d")
    else:
        d = REPO / "reports" / "audit" / datetime.now(timezone.utc).strftime("%Y%m%d")
    d.mkdir(parents=True, exist_ok=True)
    return d


@dataclass
class Check:
    """One audit check. `finding` maps to the section id in docs/METHODOLOGICAL_AUDIT.md."""
    name: str
    finding: str
    question: str
    criterion: str

    def emit(self, status: str, payload: dict[str, Any], inputs: list[Path] | None = None,
             out: str | None = None) -> Path:
        assert status in STATUS_ORDER, status
        record = {
            "check": self.name,
            "finding": self.finding,
            "question": self.question,
            "pass_criterion": self.criterion,
            "status": status,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "git_revision": git_revision(),
            "repo": str(REPO),
            "env": env_versions(),
            "inputs": [{"path": str(p.relative_to(REPO)) if str(p).startswith(str(REPO)) else str(p),
                        "sha256": sha256(p)} for p in (inputs or []) if Path(p).is_file()],
            "payload": payload,
        }
        # NaN / Infinity to NIE jest poprawny JSON (RFC 8259). json.dumps Pythona pisze je
        # domyslnie jako bare NaN / Infinity, co czyta tylko parser Pythona — kazdy scisly
        # odrzuca plik. Zamieniamy na null i ZAPISUJEMY, gdzie to sie stalo, zeby informacja
        # nie zginela po cichu: pole niefinitowe jest wynikiem, nie brakiem danych.
        nonfinite: list[str] = []
        record["payload"] = _sanitize_nonfinite(payload, nonfinite)
        if nonfinite:
            record["nonfinite_fields"] = sorted(nonfinite)
        d = out_dir(out)
        path = d / f"{self.name}.json"
        # allow_nan=False -> jesli sanityzacja czegos nie zlapala, padamy zamiast pisac zly plik
        path.write_text(json.dumps(record, indent=2, default=_jsonable, allow_nan=False) + "\n")
        print(f"\n[{status}] {self.name} :: {self.finding} -> {path}")
        return path


def _sanitize_nonfinite(obj: Any, found: list[str], path: str = "") -> Any:
    """Zamienia NaN / +-Infinity na None, zbierajac sciezki do `found`.

    Nie uzywamy math.isnan bezposrednio na dowolnym obiekcie, bo payloady zawieraja stringi,
    boole i listy niejednorodne. Boole sprawdzamy PRZED liczbami, bo bool jest podklasa int.
    """
    import math
    if isinstance(obj, dict):
        return {k: _sanitize_nonfinite(v, found, f"{path}/{k}") for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_nonfinite(v, found, f"{path}/{i}") for i, v in enumerate(obj)]
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, (int, np.integer)):
        return obj
    if isinstance(obj, (float, np.floating)):
        if not math.isfinite(float(obj)):
            found.append(path or "/")
            return None
        return obj
    if isinstance(obj, np.ndarray):
        return _sanitize_nonfinite(obj.tolist(), found, path)
    return obj


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(type(obj))


# ---------------------------------------------------------------- metrics


def per_gas_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """The repo's metric convention: RMSE over the sample axis, per target column."""
    return np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2, axis=0))


def mrmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Unweighted mean over targets of per-target RMSE (matches training.py:269-280)."""
    return float(per_gas_rmse(y_true, y_pred).mean())


def constant_predictor_mrmse(y_true: np.ndarray, constant: np.ndarray | None = None) -> tuple[float, np.ndarray]:
    """mRMSE of the best single-vector predictor. If `constant` is None, uses the per-column mean
    of y_true itself, i.e. the *most favourable* trivial baseline (an upper bound on model credit)."""
    y_true = np.asarray(y_true, dtype=np.float64)
    c = y_true.mean(axis=0) if constant is None else np.asarray(constant, dtype=np.float64)
    r = per_gas_rmse(y_true, np.broadcast_to(c, y_true.shape))
    return float(r.mean()), r


def skill(model_mrmse: float, baseline_mrmse: float) -> float:
    """1 - model/baseline. <= 0 means the model is no better than a constant.

    baseline == 0 znaczy, ze predyktor staly jest DOKLADNY (wszystkie etykiety identyczne),
    wiec skill jest nieokreslony, a nie zerowy — zwracamy nan zamiast dzielic przez zero.
    Nieosiagalne na obecnych zbiorach (najmniejszy baseline to 1.4404 na ADC holdout), ale
    skill() jest wolane przez cztery checki i nie powinno wysadzac przebiegu.
    """
    if baseline_mrmse == 0:
        return float("nan")
    return float(1.0 - model_mrmse / baseline_mrmse)


def paired_bootstrap(y_true, pred_a, pred_b, n_boot: int = 10000, seed: int = 0) -> dict[str, float]:
    """Bootstrap over ROWS of the mRMSE difference (a - b). This measures test-set sampling
    precision for two FIXED checkpoints; it is NOT evidence about architectures (see a12)."""
    y_true, pred_a, pred_b = map(np.asarray, (y_true, pred_a, pred_b))
    rng = np.random.default_rng(seed)
    n = len(y_true)
    d = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        d[i] = mrmse(y_true[idx], pred_a[idx]) - mrmse(y_true[idx], pred_b[idx])
    return {"delta": mrmse(y_true, pred_a) - mrmse(y_true, pred_b),
            "boot_mean": float(d.mean()), "ci95_lo": float(np.percentile(d, 2.5)),
            "ci95_hi": float(np.percentile(d, 97.5)), "boot_sd": float(d.std(ddof=1)),
            "p_delta_gt_0": float((d > 0).mean()),
            "n_rows": int(n), "n_boot": int(n_boot), "resampling_unit": "test rows (NOT seeds)"}


# Two-sided normal critical values, spelled out so nobody has to guess where they come from.
#   Z_ALPHA_2 = Phi^-1(1 - alpha/2)  with alpha = 0.05  -> 1.9599639845...
#       "how far from zero must the estimate sit before we call it non-zero", i.e. the type-I
#       guard: at most 5 % chance of declaring an effect that is not there.
#   Z_POWER   = Phi^-1(power)        with power = 0.80  -> 0.8416212336...
#       "how far beyond that must the TRUE effect sit before we reliably notice it", i.e. the
#       type-II guard: at least 80 % chance of detecting an effect that IS there.
# They add because the two requirements stack on the same standard-error scale: the true effect
# has to clear the significance boundary AND still leave enough room that the sampling
# distribution mostly falls on the far side of it.
Z_ALPHA_2_05 = 1.9599639845400545
Z_POWER_80 = 0.8416212335729143


def min_detectable_effect(sd_per_row: float, n: int, alpha: float = 0.05, power: float = 0.80,
                          use_t: bool = True) -> float:
    """Smallest true effect a paired design of size `n` can detect, on the scale of `sd_per_row`.

    MDE = (z_{1-alpha/2} + z_{power}) * SE,      SE = sd_per_row / sqrt(n)

    Derivation, so the constants are not floating in the air. Let d-hat be the estimated paired
    difference, with standard error SE. A two-sided test at level `alpha` rejects when
    |d-hat| > z_{1-alpha/2} * SE. If the true effect is delta, then d-hat ~ N(delta, SE^2), and

        power = P(reject | delta) ~= Phi( delta/SE - z_{1-alpha/2} )

    (the far tail is neglected; it contributes < 1e-4 at these settings). Setting that equal to
    the target power and solving for delta gives delta = (z_{1-alpha/2} + z_{power}) * SE.
    With alpha = 0.05 and power = 0.80 the bracket is 1.95996 + 0.84162 = 2.80159.

    ASSUMPTIONS — each one is a way this number can mislead:
      1. `sd_per_row` is the sd of the PER-ROW paired difference of the quantity you actually
         report. If you pass the sd of one estimand and compare the MDE against a different
         estimand, the number is indicative at best. See `mde_from_bootstrap` for the
         assumption-light route, and note that a12 reports both precisely because of this.
      2. Rows are independent and the difference is roughly symmetric. Heavy tails or clustering
         (e.g. several rows from the same planet) inflate the true SE.
      3. `use_t=True` swaps z_{1-alpha/2} for the t critical value with n-1 df, which matters at
         small n: at n = 64 the two-sided 5 % value is 1.998 rather than 1.960, so the normal
         version understates the MDE by about 2 %. At n = 685 the difference is 0.1 %.
      4. This is a PRE-DATA quantity: what the design could resolve. It says nothing about
         whether the effect you measured is real — for that you need the interval, and for an
         architectural claim you need variance across training seeds, not across rows.
    """
    if n <= 1:
        return float("inf")
    if use_t:
        from scipy import stats
        crit = float(stats.t.ppf(1.0 - alpha / 2.0, df=n - 1))
    else:
        crit = float(stats.norm.ppf(1.0 - alpha / 2.0)) if alpha != 0.05 else Z_ALPHA_2_05
    from scipy import stats
    zp = Z_POWER_80 if power == 0.80 else float(stats.norm.ppf(power))
    return float((crit + zp) * sd_per_row / np.sqrt(n))


def mde_from_bootstrap(boot_sd: float, alpha: float = 0.05, power: float = 0.80) -> float:
    """MDE for whatever estimand the bootstrap resampled — no per-row assumption needed.

    `boot_sd` is the standard deviation of the bootstrap distribution of the statistic, which IS
    an estimate of that statistic's standard error. Use this when the reported quantity is an
    aggregate that is not a mean over rows (mRMSE is sqrt-of-mean-of-squares, so the mean of
    per-row RMSE differences is a DIFFERENT functional by Jensen's inequality).
    """
    from scipy import stats
    return float((stats.norm.ppf(1.0 - alpha / 2.0) + stats.norm.ppf(power)) * boot_sd)


# ---------------------------------------------------------------- data access


def adc_root() -> Path:
    return REPO / "data" / "ariel-ml-dataset"


def load_adc_targets() -> "Any":
    import pandas as pd
    fm = pd.read_csv(adc_root() / "TrainingData/Ground Truth Package/FM_Parameter_Table.csv")
    return fm.set_index("planet_ID")


def adc_split_ids(split: str) -> np.ndarray:
    """Planet ids of the canonical seed-42 split, as frozen in data/val_dataset/.
    `saved_split_manifest.json` documents these as identical to the ExoBiome cached split."""
    import pandas as pd
    name = {"holdout": "holdout", "validation": "validation", "train": "train"}[split]
    return pd.read_csv(REPO / f"data/val_dataset/{name}_planet_ids.csv")["planet_ID"].astype(str).to_numpy()


def load_adc_raw(split: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Returns (planet_ids, aux_raw_log10_applied, spectra_raw[N,52,4], targets_raw[N,5])."""
    import h5py
    import pandas as pd
    sys.path.insert(0, str(REPO))
    from models.ariel_exobiome.constants import (AUX_COLUMNS, LOG10_AUX_COLUMNS,
                                                 RAW_SPECTRAL_CHANNELS)
    ids = adc_split_ids(split)
    aux = pd.read_csv(adc_root() / "TrainingData/AuxillaryTable.csv")
    aux = aux.drop(columns=[c for c in aux.columns if c.startswith("Unnamed:")]).set_index("planet_ID")
    tgt = load_adc_targets()
    # .loc z lista etykiet rzuca KeyError, gdy ktorakolwiek nie istnieje — ten sam wzorzec,
    # ktory w a29 zamienial niedopasowanie w crash. Dzis 0 brakujacych na wszystkich trzech
    # splitach (33138 / 4142 / 4143), ale komunikat ma mowic, CO nie pasuje, a nie sam KeyError.
    missing = [i for i in ids if i not in aux.index]
    if missing:
        raise KeyError(f"{len(missing)} planet_ID ze splitu '{split}' nie ma w AuxillaryTable "
                       f"(pierwsze 5: {missing[:5]})")
    aux_raw = aux.loc[ids, AUX_COLUMNS].to_numpy(dtype=np.float32).copy()
    for j, c in enumerate(AUX_COLUMNS):
        if c in LOG10_AUX_COLUMNS:
            aux_raw[:, j] = np.log10(np.clip(aux_raw[:, j], 1e-12, None))
    targets = tgt.loc[ids, TARGETS].to_numpy(dtype=np.float32)
    spec = np.empty((len(ids), 52, len(RAW_SPECTRAL_CHANNELS)), dtype=np.float32)
    with h5py.File(adc_root() / "TrainingData/SpectralData.hdf5", "r") as h:
        for i, pid in enumerate(ids):
            g = h[f"Planet_{pid}"]
            for k, ch in enumerate(RAW_SPECTRAL_CHANNELS):
                spec[i, :, k] = np.asarray(g[ch][:], dtype=np.float32)
    return ids, aux_raw, spec, targets


# ---------------------------------------------------------------- ExoBiome model access

EXOBIOME_ARTIFACT = REPO / "artifacts/ariel_quantum_best_v4_epoch6"


def exobiome_scalers(artifact: Path = EXOBIOME_ARTIFACT):
    sys.path.insert(0, str(REPO))
    from models.ariel_exobiome.dataset import ArrayStandardizer, SpectralStandardizer
    s = json.loads((artifact / "scalers.json").read_text())
    return (ArrayStandardizer.from_state_dict(s["aux_scaler"]),
            ArrayStandardizer.from_state_dict(s["target_scaler"]),
            SpectralStandardizer.from_state_dict(s["spectral_scaler"]))


def exobiome_inputs(spec_raw: np.ndarray, aux_raw: np.ndarray, aux_scaler, spectral_scaler,
                    noise_rng: "np.random.Generator | None" = None):
    """Reproduces models/ariel_exobiome/dataset.py preparation exactly:
    channels [instrument_spectrum, instrument_noise] -> divide by the sample mean of channel 0
    -> per-bin standardize -> append the 2 fixed channels.
    If `noise_rng` is given, N(0, instrument_noise) is added to the spectrum FIRST, i.e. the
    input convention used for the NSF baseline in scripts/reeval_sota.py (sample_noise=True).
    """
    import torch
    s = spec_raw[:, :, 0].astype(np.float64).copy()
    sig = spec_raw[:, :, 1].astype(np.float64)
    if noise_rng is not None:
        s = s + noise_rng.normal(0.0, 1.0, size=s.shape) * sig
    sample = np.stack([s, sig], axis=1).astype(np.float32)
    ref = np.clip(sample[:, 0, :].mean(axis=1, keepdims=True), 1e-12, None)
    sample = (sample / ref[:, None, :]).astype(np.float32)
    return torch.from_numpy(aux_scaler.transform(aux_raw)), torch.from_numpy(spectral_scaler.transform(sample))


def load_exobiome(artifact: Path = EXOBIOME_ARTIFACT, device: str = "cpu"):
    import torch
    sys.path.insert(0, str(REPO))
    from models.ariel_exobiome.model import ModelConfig, build_model
    ck = torch.load(artifact / "best_model.pt", map_location=device, weights_only=False)
    cfg = ck["config"]
    model = build_model(
        ModelConfig(spectral_input_channels=4, dropout=float(cfg["dropout"]),
                    qnn_qubits=int(cfg["qnn_qubits"]), qnn_depth=int(cfg["qnn_depth"]),
                    qnn_init_scale=float(cfg["qnn_init_scale"]), quantum_device="lightning.qubit",
                    quantum_use_async=False, classical_only=False, use_amp=False),
        __import__("torch").device(device))
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    return model, ck


def exobiome_predict(model, aux, spectra, target_scaler, quantum_scale: float, batch: int = 256) -> np.ndarray:
    import torch
    outs = []
    with torch.inference_mode():
        for i in range(0, aux.shape[0], batch):
            outs.append(model(aux[i:i + batch], spectra[i:i + batch],
                              enable_quantum=quantum_scale > 0.0,
                              quantum_scale=quantum_scale).cpu().numpy())
    return target_scaler.inverse_transform(np.concatenate(outs, axis=0))
