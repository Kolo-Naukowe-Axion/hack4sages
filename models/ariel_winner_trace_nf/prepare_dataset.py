"""Prepare tracedata-supervised Ariel data for winner-family independent flows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from .constants import (
    AUX_COLUMNS,
    DEFAULT_DATA_ROOT,
    DEFAULT_PREPARED_ROOT,
    DEFAULT_SPLIT_ROOT,
    FIVE_GAS_TARGET_COLUMNS,
    FM_TARGETS_FILE,
    QUARTILES_FILE,
    QUARTILE_COLUMN_GROUPS,
    SPECTRAL_LENGTH,
    TARGET_COLUMNS,
    TEST_AUX_FILE,
    TEST_SPECTRA_FILE,
    TRACE_FILE,
    TRAIN_AUX_FILE,
    TRAIN_SPECTRA_FILE,
)
from .preprocessing import build_context_numpy, fit_scalers


def _drop_unnamed_columns(frame: pd.DataFrame) -> pd.DataFrame:
    unnamed = [column for column in frame.columns if column.startswith("Unnamed:")]
    if unnamed:
        frame = frame.drop(columns=unnamed)
    return frame


def _load_trace_metadata(trace_path: Path) -> tuple[set[str], int]:
    valid_ids: set[str] = set()
    max_points = 0
    with h5py.File(trace_path, "r") as handle:
        for group_name in handle.keys():
            group = handle[group_name]
            if "tracedata" not in group or "weights" not in group:
                continue
            tracedata = group["tracedata"]
            weights = group["weights"]
            if len(tracedata.shape) != 2 or tracedata.shape[1] != len(TARGET_COLUMNS):
                continue
            if weights.shape != (tracedata.shape[0],):
                continue
            valid_ids.add(group_name.replace("Planet_", ""))
            max_points = max(max_points, int(tracedata.shape[0]))
    if not valid_ids:
        raise RuntimeError(f"No valid tracedata rows found in {trace_path}.")
    return valid_ids, max_points


def _load_split_ids(split_source: Path, valid_trace_ids: set[str]) -> dict[str, list[str]]:
    split_ids: dict[str, list[str]] = {}
    for name in ("train", "validation", "holdout"):
        frame = pd.read_csv(split_source / f"{name}_planet_ids.csv")
        if "planet_ID" not in frame.columns:
            raise KeyError(f"Missing planet_ID column in {split_source / f'{name}_planet_ids.csv'}")
        ids = frame["planet_ID"].astype(str).tolist()
        split_ids[name] = [planet_id for planet_id in ids if planet_id in valid_trace_ids]
    if len(set(split_ids["train"]) & set(split_ids["validation"])) != 0:
        raise ValueError("Saved tracedata-aware train and validation splits overlap.")
    if len(set(split_ids["train"]) & set(split_ids["holdout"])) != 0:
        raise ValueError("Saved tracedata-aware train and holdout splits overlap.")
    if len(set(split_ids["validation"]) & set(split_ids["holdout"])) != 0:
        raise ValueError("Saved tracedata-aware validation and holdout splits overlap.")
    return split_ids


def _load_training_tables(data_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    aux = _drop_unnamed_columns(pd.read_csv(data_root / TRAIN_AUX_FILE))
    fm_targets = _drop_unnamed_columns(pd.read_csv(data_root / FM_TARGETS_FILE))
    quartiles = _drop_unnamed_columns(pd.read_csv(data_root / QUARTILES_FILE))

    merged = aux.merge(fm_targets[["planet_ID", *TARGET_COLUMNS]], on="planet_ID", how="inner", validate="one_to_one")
    merged = merged.merge(quartiles[["planet_ID", *sum((list(QUARTILE_COLUMN_GROUPS[name]) for name in TARGET_COLUMNS), [])]], on="planet_ID", how="left")
    if list(aux.columns) != ["planet_ID", *AUX_COLUMNS]:
        raise AssertionError(f"Unexpected auxiliary columns: {list(aux.columns)}")
    return merged, quartiles


def _load_test_aux(data_root: Path) -> pd.DataFrame:
    aux = _drop_unnamed_columns(pd.read_csv(data_root / TEST_AUX_FILE))
    if list(aux.columns) != ["planet_ID", *AUX_COLUMNS]:
        raise AssertionError(f"Unexpected test auxiliary columns: {list(aux.columns)}")
    return aux


def _load_selected_spectra(hdf5_path: Path, planet_ids: list[str]) -> tuple[np.ndarray, np.ndarray]:
    spectra = np.empty((len(planet_ids), SPECTRAL_LENGTH), dtype=np.float32)
    noise = np.empty((len(planet_ids), SPECTRAL_LENGTH), dtype=np.float32)
    with h5py.File(hdf5_path, "r") as handle:
        for row_index, planet_id in enumerate(planet_ids):
            group = handle[f"Planet_{planet_id}"]
            spectra[row_index] = np.asarray(group["instrument_spectrum"][:], dtype=np.float32)
            noise[row_index] = np.asarray(group["instrument_noise"][:], dtype=np.float32)
    return spectra, noise


def _load_selected_traces(trace_path: Path, planet_ids: list[str], max_points: int) -> tuple[np.ndarray, np.ndarray]:
    targets = np.zeros((len(planet_ids), max_points, len(TARGET_COLUMNS)), dtype=np.float32)
    weights = np.zeros((len(planet_ids), max_points), dtype=np.float32)
    with h5py.File(trace_path, "r") as handle:
        for row_index, planet_id in enumerate(planet_ids):
            group = handle[f"Planet_{planet_id}"]
            tracedata = np.asarray(group["tracedata"][:], dtype=np.float32)
            trace_weights = np.asarray(group["weights"][:], dtype=np.float32)
            trace_len = tracedata.shape[0]
            targets[row_index, :trace_len] = tracedata
            weights[row_index, :trace_len] = trace_weights
    return targets, weights


def _quartiles_array(frame: pd.DataFrame) -> np.ndarray:
    arrays = []
    for target_name in TARGET_COLUMNS:
        q1, q2, q3 = QUARTILE_COLUMN_GROUPS[target_name]
        arrays.append(frame[[q1, q2, q3]].to_numpy(dtype=np.float32))
    return np.stack(arrays, axis=1).astype(np.float32)


def _build_labeled_payload(
    frame: pd.DataFrame,
    *,
    data_root: Path,
    trace_path: Path,
    max_points: int,
    scalers,
) -> dict[str, np.ndarray]:
    planet_ids = frame["planet_ID"].astype(str).tolist()
    aux = frame[AUX_COLUMNS].to_numpy(dtype=np.float32)
    spectra, noise = _load_selected_spectra(data_root / TRAIN_SPECTRA_FILE, planet_ids)
    context, radius_reference = build_context_numpy(aux, spectra, noise, scalers)
    trace_targets, trace_weights = _load_selected_traces(trace_path, planet_ids, max_points)
    return {
        "planet_id": np.asarray(planet_ids, dtype="U32"),
        "context": context.astype(np.float32),
        "radius_reference": radius_reference.astype(np.float32),
        "fm_targets_raw": frame[TARGET_COLUMNS].to_numpy(dtype=np.float32),
        "quartiles_raw": _quartiles_array(frame),
        "trace_targets_raw": trace_targets.astype(np.float32),
        "trace_weights": trace_weights.astype(np.float32),
    }


def _build_unlabeled_payload(frame: pd.DataFrame, *, data_root: Path, scalers) -> dict[str, np.ndarray]:
    planet_ids = frame["planet_ID"].astype(str).tolist()
    aux = frame[AUX_COLUMNS].to_numpy(dtype=np.float32)
    spectra, noise = _load_selected_spectra(data_root / TEST_SPECTRA_FILE, planet_ids)
    context, radius_reference = build_context_numpy(aux, spectra, noise, scalers)
    return {
        "planet_id": np.asarray(planet_ids, dtype="U32"),
        "context": context.astype(np.float32),
        "radius_reference": radius_reference.astype(np.float32),
    }


def _save_split(path: Path, payload: dict[str, np.ndarray]) -> None:
    np.savez_compressed(path, **payload)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--split-source", type=Path, default=DEFAULT_SPLIT_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_PREPARED_ROOT)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    data_root = args.data_root.expanduser().resolve()
    split_source = args.split_source.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output directory already exists and is non-empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    trace_path = data_root / TRACE_FILE
    valid_trace_ids, max_points = _load_trace_metadata(trace_path)
    split_ids = _load_split_ids(split_source, valid_trace_ids)

    merged, _quartiles = _load_training_tables(data_root)
    merged = merged[merged["planet_ID"].isin(valid_trace_ids)].copy()
    merged = merged.set_index("planet_ID", drop=False)

    train_frame = merged.loc[split_ids["train"]].reset_index(drop=True)
    validation_frame = merged.loc[split_ids["validation"]].reset_index(drop=True)
    holdout_frame = merged.loc[split_ids["holdout"]].reset_index(drop=True)
    test_frame = _load_test_aux(data_root)

    train_aux = train_frame[AUX_COLUMNS].to_numpy(dtype=np.float32)
    train_spectra, train_noise = _load_selected_spectra(data_root / TRAIN_SPECTRA_FILE, train_frame["planet_ID"].astype(str).tolist())
    train_trace_targets, train_trace_weights = _load_selected_traces(
        trace_path,
        train_frame["planet_ID"].astype(str).tolist(),
        max_points=max_points,
    )
    scalers, train_radius_reference = fit_scalers(
        train_aux,
        train_spectra,
        train_noise,
        train_trace_targets,
        train_trace_weights,
    )
    scalers.save(output / "scalers.npz")

    train_payload = _build_labeled_payload(
        train_frame,
        data_root=data_root,
        trace_path=trace_path,
        max_points=max_points,
        scalers=scalers,
    )
    validation_payload = _build_labeled_payload(
        validation_frame,
        data_root=data_root,
        trace_path=trace_path,
        max_points=max_points,
        scalers=scalers,
    )
    holdout_payload = _build_labeled_payload(
        holdout_frame,
        data_root=data_root,
        trace_path=trace_path,
        max_points=max_points,
        scalers=scalers,
    )
    test_payload = _build_unlabeled_payload(test_frame, data_root=data_root, scalers=scalers)

    _save_split(output / "train.npz", train_payload)
    _save_split(output / "validation.npz", validation_payload)
    _save_split(output / "holdout.npz", holdout_payload)
    _save_split(output / "testdata.npz", test_payload)

    manifest = {
        "data_root": str(data_root),
        "split_source": str(split_source),
        "package": "ariel_winner_trace_nf",
        "architecture_family": "winner_family_independent_conditional_neural_spline_flows",
        "comparison_note": "This is a winner-family local rerun with ensemble_size=1, not a claim of exact leaderboard reproduction.",
        "target_columns": TARGET_COLUMNS,
        "five_gas_targets": FIVE_GAS_TARGET_COLUMNS,
        "context_layout": {
            "aux_engineered": 12,
            "spectrum_stats": 2,
            "spectrum_normalized": 52,
            "noise_scaled": 52,
            "total": 118,
        },
        "trace_rows_available": int(len(valid_trace_ids)),
        "max_trace_points": int(max_points),
        "split_sizes": {
            "train": int(train_payload["planet_id"].shape[0]),
            "validation": int(validation_payload["planet_id"].shape[0]),
            "holdout": int(holdout_payload["planet_id"].shape[0]),
            "testdata": int(test_payload["planet_id"].shape[0]),
        },
        "trace_points": {
            "train": int((train_payload["trace_weights"] > 0).sum()),
            "validation": int((validation_payload["trace_weights"] > 0).sum()),
            "holdout": int((holdout_payload["trace_weights"] > 0).sum()),
        },
        "references": {
            "official_repo": "https://github.com/AstroAI-CfA/Ariel_Data_Challenge_2023_solution",
            "winner_paper": "https://arxiv.org/abs/2309.09337",
        },
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
