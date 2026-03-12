"""Cross-validation runner for the Ariel five-gas hybrid regressor."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split

from .constants import (
    COARSE_ABUNDANCE_QUANTILES,
    COARSE_STRATIFY_MIN_COUNT,
    PRESENCE_THRESHOLD_LOG10_VMR,
    PRIMARY_STRATIFY_MIN_COUNT,
    TARGET_COLUMNS,
)


@dataclass(frozen=True)
class FoldSpec:
    fold_index: int
    train_indices: np.ndarray
    val_indices: np.ndarray
    holdout_indices: np.ndarray
    outer_splitter: str
    inner_splitter: str


@dataclass(frozen=True)
class CrossValidationConfig:
    num_folds: int = 5
    val_fraction: float = 0.1

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RawArielData:
    data_root: Path
    labeled_frame: pd.DataFrame
    labeled_aux_raw: np.ndarray
    labeled_targets_raw: np.ndarray
    labeled_sample_spectra: np.ndarray
    test_frame: pd.DataFrame
    test_aux_raw: np.ndarray
    test_sample_spectra: np.ndarray
    fixed_channels: np.ndarray
    wavelength_um: np.ndarray


def _presence_signature(targets: np.ndarray) -> np.ndarray:
    presence = (targets >= PRESENCE_THRESHOLD_LOG10_VMR).astype(np.int64)
    bit_weights = (1 << np.arange(presence.shape[1], dtype=np.int64)).reshape(1, -1)
    return (presence * bit_weights).sum(axis=1).astype(np.int64)


def _coarse_abundance_labels(targets: np.ndarray) -> np.ndarray:
    mean_abundance = targets.mean(axis=1)
    quantiles = np.quantile(mean_abundance, COARSE_ABUNDANCE_QUANTILES)
    edges = np.unique(np.asarray(quantiles, dtype=np.float64))
    abundance_bin = np.digitize(mean_abundance, edges, right=False)
    presence_count = (targets >= PRESENCE_THRESHOLD_LOG10_VMR).sum(axis=1)
    return np.asarray([f"{count}_{bin_index}" for count, bin_index in zip(presence_count, abundance_bin)], dtype="U16")


def _can_stratify(labels: np.ndarray, min_count: int) -> bool:
    counts = pd.Series(labels).value_counts()
    return len(counts) > 1 and int(counts.min()) >= int(min_count)


def _build_stratify_labels(targets: np.ndarray) -> tuple[Optional[np.ndarray], str]:
    primary = _presence_signature(targets)
    if _can_stratify(primary, PRIMARY_STRATIFY_MIN_COUNT):
        return primary, "presence_signature"

    coarse = _coarse_abundance_labels(targets)
    if _can_stratify(coarse, COARSE_STRATIFY_MIN_COUNT):
        return coarse, "coarse_abundance"

    return None, "unstratified"


def _can_use_stratified_kfold(labels: Optional[np.ndarray], n_splits: int) -> bool:
    if labels is None:
        return False
    counts = pd.Series(labels).value_counts()
    return len(counts) > 1 and int(counts.min()) >= int(n_splits)


def _can_use_stratified_holdout(labels: Optional[np.ndarray], val_rows: int) -> bool:
    if labels is None or val_rows <= 0:
        return False
    counts = pd.Series(labels).value_counts()
    return len(counts) > 1 and int(counts.min()) >= 2 and val_rows >= len(counts)


def _limit_indices(indices: np.ndarray, limit: Optional[int]) -> np.ndarray:
    if limit is None or limit >= len(indices):
        return np.sort(indices.astype(np.int64))
    return np.sort(indices[: int(limit)].astype(np.int64))


def build_cross_validation_folds(
    targets: np.ndarray,
    num_folds: int = 5,
    seed: int = 42,
    val_fraction: float = 0.1,
    train_limit: Optional[int] = None,
    val_limit: Optional[int] = None,
    holdout_limit: Optional[int] = None,
) -> list[FoldSpec]:
    if num_folds < 2:
        raise ValueError(f"num_folds must be at least 2, got {num_folds}.")
    if not 0.0 < val_fraction < 1.0:
        raise ValueError(f"val_fraction must be between 0 and 1, got {val_fraction}.")

    all_indices = np.arange(len(targets), dtype=np.int64)
    outer_labels, outer_mode = _build_stratify_labels(targets)
    if _can_use_stratified_kfold(outer_labels, num_folds):
        outer_splitter = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=seed)
        outer_name = f"stratified_{outer_mode}"
        outer_splits = outer_splitter.split(all_indices, outer_labels)
    else:
        outer_splitter = KFold(n_splits=num_folds, shuffle=True, random_state=seed)
        outer_name = "kfold"
        outer_splits = outer_splitter.split(all_indices)

    folds: list[FoldSpec] = []
    for fold_index, (train_pool_indices, holdout_indices) in enumerate(outer_splits):
        train_pool_indices = np.asarray(train_pool_indices, dtype=np.int64)
        holdout_indices = np.asarray(holdout_indices, dtype=np.int64)
        if len(train_pool_indices) < 2:
            raise ValueError(f"Fold {fold_index + 1} has insufficient training rows: {len(train_pool_indices)}.")

        inner_targets = targets[train_pool_indices]
        inner_labels, inner_mode = _build_stratify_labels(inner_targets)
        desired_val_rows = int(round(len(train_pool_indices) * val_fraction))
        val_rows = min(max(desired_val_rows, 1), len(train_pool_indices) - 1)
        test_size: float | int = val_rows
        inner_name = "shuffle_split"
        stratify_labels = inner_labels if _can_use_stratified_holdout(inner_labels, val_rows) else None
        if stratify_labels is not None:
            inner_name = f"stratified_{inner_mode}"

        relative_positions = np.arange(len(train_pool_indices), dtype=np.int64)
        try:
            train_positions, val_positions = train_test_split(
                relative_positions,
                test_size=test_size,
                random_state=seed + fold_index + 1,
                shuffle=True,
                stratify=stratify_labels,
            )
        except ValueError:
            train_positions, val_positions = train_test_split(
                relative_positions,
                test_size=test_size,
                random_state=seed + fold_index + 1,
                shuffle=True,
                stratify=None,
            )
            inner_name = "shuffle_split"

        train_indices = _limit_indices(train_pool_indices[train_positions], train_limit)
        val_indices = _limit_indices(train_pool_indices[val_positions], val_limit)
        holdout_indices = _limit_indices(holdout_indices, holdout_limit)
        folds.append(
            FoldSpec(
                fold_index=fold_index,
                train_indices=train_indices,
                val_indices=val_indices,
                holdout_indices=holdout_indices,
                outer_splitter=outer_name,
                inner_splitter=inner_name,
            )
        )

    return folds


def compute_regression_metrics(true_values: np.ndarray, pred_values: np.ndarray) -> dict[str, Any]:
    if true_values.shape != pred_values.shape:
        raise ValueError(f"Shape mismatch: true_values={true_values.shape}, pred_values={pred_values.shape}.")
    rmse = np.sqrt(np.mean((pred_values - true_values) ** 2, axis=0))
    mae = np.mean(np.abs(pred_values - true_values), axis=0)
    return {
        "rows": int(true_values.shape[0]),
        "rmse_mean": float(rmse.mean()),
        "mae_mean": float(mae.mean()),
        "rmse": {name: float(value) for name, value in zip(TARGET_COLUMNS, rmse)},
        "mae": {name: float(value) for name, value in zip(TARGET_COLUMNS, mae)},
    }


def load_raw_ariel_data(data_root: str | Path) -> RawArielData:
    from .constants import RAW_SPECTRAL_CHANNELS, SAMPLE_SPECTRAL_CHANNELS
    from .dataset import (
        _normalize_fixed_channel,
        _normalize_sample_spectra,
        load_test_dataset,
        load_training_dataset,
        transform_aux_features,
    )

    root = Path(data_root).expanduser().resolve()
    labeled_frame, labeled_spectra_raw, wavelength_um = load_training_dataset(root)
    test_frame, test_spectra_raw, test_wavelength_um = load_test_dataset(root)
    if not np.allclose(test_wavelength_um, wavelength_um, atol=1.0e-8):
        raise AssertionError("Training and test wavelength grids do not match.")

    labeled_aux_raw = transform_aux_features(labeled_frame)
    test_aux_raw = transform_aux_features(test_frame)
    labeled_targets_raw = labeled_frame[TARGET_COLUMNS].to_numpy(dtype=np.float32, copy=True)

    sample_channel_indices = [RAW_SPECTRAL_CHANNELS.index(name) for name in SAMPLE_SPECTRAL_CHANNELS]
    width_channel_index = RAW_SPECTRAL_CHANNELS.index("instrument_width")

    labeled_sample_spectra = np.transpose(labeled_spectra_raw[:, :, sample_channel_indices], (0, 2, 1)).astype(np.float32)
    test_sample_spectra = np.transpose(test_spectra_raw[:, :, sample_channel_indices], (0, 2, 1)).astype(np.float32)
    labeled_sample_spectra = _normalize_sample_spectra(labeled_sample_spectra)
    test_sample_spectra = _normalize_sample_spectra(test_sample_spectra)
    fixed_channels = np.stack(
        [
            _normalize_fixed_channel(labeled_spectra_raw[0, :, width_channel_index]),
            _normalize_fixed_channel(wavelength_um),
        ],
        axis=0,
    ).astype(np.float32)

    return RawArielData(
        data_root=root,
        labeled_frame=labeled_frame.reset_index(drop=True),
        labeled_aux_raw=labeled_aux_raw,
        labeled_targets_raw=labeled_targets_raw,
        labeled_sample_spectra=labeled_sample_spectra,
        test_frame=test_frame.reset_index(drop=True),
        test_aux_raw=test_aux_raw,
        test_sample_spectra=test_sample_spectra,
        fixed_channels=fixed_channels,
        wavelength_um=wavelength_um.astype(np.float32),
    )


def prepare_fold_data(
    raw_data: RawArielData,
    fold: FoldSpec,
    output_dir: str | Path,
    seed: int,
    test_limit: Optional[int] = None,
):
    from .constants import (
        AUX_COLUMNS,
        FIXED_SPECTRAL_CHANNELS,
        LOG10_AUX_COLUMNS,
        MODEL_SPECTRAL_CHANNELS,
        RAW_SPECTRAL_CHANNELS,
        SAMPLE_SPECTRAL_CHANNELS,
    )
    from .dataset import (
        ArrayStandardizer,
        PreparedData,
        SpectralStandardizer,
        _make_inference_split,
        _make_labeled_split,
    )

    train_indices = np.asarray(fold.train_indices, dtype=np.int64)
    val_indices = np.asarray(fold.val_indices, dtype=np.int64)
    holdout_indices = np.asarray(fold.holdout_indices, dtype=np.int64)
    test_indices = _limit_indices(np.arange(len(raw_data.test_frame), dtype=np.int64), test_limit)

    aux_scaler = ArrayStandardizer.fit(raw_data.labeled_aux_raw[train_indices])
    target_scaler = ArrayStandardizer.fit(raw_data.labeled_targets_raw[train_indices])
    spectral_scaler = SpectralStandardizer.fit(
        raw_data.labeled_sample_spectra[train_indices],
        fixed_channels=raw_data.fixed_channels,
    )

    def transform_labeled(indices: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        aux_scaled = aux_scaler.transform(raw_data.labeled_aux_raw[indices])
        spectra_scaled = spectral_scaler.transform(raw_data.labeled_sample_spectra[indices])
        raw_targets = raw_data.labeled_targets_raw[indices]
        targets_scaled = target_scaler.transform(raw_targets)
        return aux_scaled, spectra_scaled, targets_scaled, raw_targets

    train_aux, train_spectra, train_targets_scaled, train_targets_raw = transform_labeled(train_indices)
    val_aux, val_spectra, val_targets_scaled, val_targets_raw = transform_labeled(val_indices)
    holdout_aux, holdout_spectra, holdout_targets_scaled, holdout_targets_raw = transform_labeled(holdout_indices)
    test_aux = aux_scaler.transform(raw_data.test_aux_raw[test_indices])
    test_spectra = spectral_scaler.transform(raw_data.test_sample_spectra[test_indices])

    split_manifest = {
        "mode": "cross_validation",
        "fold_index": int(fold.fold_index),
        "fold_name": f"fold_{fold.fold_index + 1:02d}",
        "rows_total": int(len(raw_data.labeled_frame)),
        "testdata_rows_total": int(len(raw_data.test_frame)),
        "train_rows": int(len(train_indices)),
        "val_rows": int(len(val_indices)),
        "holdout_rows": int(len(holdout_indices)),
        "testdata_rows": int(len(test_indices)),
        "wavelength_bins": int(len(raw_data.wavelength_um)),
        "wavelength_min_um": float(raw_data.wavelength_um.min()),
        "wavelength_max_um": float(raw_data.wavelength_um.max()),
        "raw_spectrum_shape": [52, len(RAW_SPECTRAL_CHANNELS)],
        "model_spectrum_shape": [len(MODEL_SPECTRAL_CHANNELS), 52],
        "aux_columns": AUX_COLUMNS,
        "log10_aux_columns": LOG10_AUX_COLUMNS,
        "target_columns": TARGET_COLUMNS,
        "raw_spectral_channels": RAW_SPECTRAL_CHANNELS,
        "sample_spectral_channels": SAMPLE_SPECTRAL_CHANNELS,
        "fixed_spectral_channels": FIXED_SPECTRAL_CHANNELS,
        "model_spectral_channels": MODEL_SPECTRAL_CHANNELS,
        "sample_spectral_normalization": {
            "mode": "divide_by_sample_mean",
            "reference_channel": SAMPLE_SPECTRAL_CHANNELS[0],
            "applied_channels": SAMPLE_SPECTRAL_CHANNELS,
        },
        "presence_threshold_log10_vmr": PRESENCE_THRESHOLD_LOG10_VMR,
        "split_seed": int(seed),
        "outer_splitter": fold.outer_splitter,
        "inner_splitter": fold.inner_splitter,
    }
    prepared_manifest = {
        "mode": "cross_validation",
        "data_root": str(raw_data.data_root),
        "output_dir": str(Path(output_dir).expanduser().resolve()),
        "seed": int(seed),
        "fold_index": int(fold.fold_index),
        "cache_hit": False,
    }

    return PreparedData(
        train=_make_labeled_split(
            planet_ids=raw_data.labeled_frame.iloc[train_indices]["planet_ID"].to_numpy(dtype="U32"),
            aux_values=train_aux,
            spectra_values=train_spectra,
            targets_scaled=train_targets_scaled,
            raw_targets=train_targets_raw,
        ),
        val=_make_labeled_split(
            planet_ids=raw_data.labeled_frame.iloc[val_indices]["planet_ID"].to_numpy(dtype="U32"),
            aux_values=val_aux,
            spectra_values=val_spectra,
            targets_scaled=val_targets_scaled,
            raw_targets=val_targets_raw,
        ),
        holdout=_make_labeled_split(
            planet_ids=raw_data.labeled_frame.iloc[holdout_indices]["planet_ID"].to_numpy(dtype="U32"),
            aux_values=holdout_aux,
            spectra_values=holdout_spectra,
            targets_scaled=holdout_targets_scaled,
            raw_targets=holdout_targets_raw,
        ),
        testdata=_make_inference_split(
            planet_ids=raw_data.test_frame.iloc[test_indices]["planet_ID"].to_numpy(dtype="U32"),
            aux_values=test_aux,
            spectra_values=test_spectra,
        ),
        aux_scaler=aux_scaler,
        target_scaler=target_scaler,
        spectral_scaler=spectral_scaler,
        wavelength_um=raw_data.wavelength_um.astype(np.float32),
        split_manifest=split_manifest,
        prepared_manifest=prepared_manifest,
    )


def _build_labeled_prediction_frame(fold_number: int, planet_ids: np.ndarray, metrics: dict[str, Any]) -> pd.DataFrame:
    frame = pd.DataFrame({"fold": int(fold_number), "planet_ID": planet_ids.tolist()})
    for index, target_name in enumerate(TARGET_COLUMNS):
        frame[f"true_{target_name}"] = metrics["true_orig"][:, index]
        frame[f"pred_{target_name}"] = metrics["pred_orig"][:, index]
    return frame


def _build_test_prediction_frame(
    fold_number: int,
    planet_ids: np.ndarray,
    predictions: np.ndarray,
) -> pd.DataFrame:
    frame = pd.DataFrame({"fold": int(fold_number), "planet_ID": planet_ids.tolist()})
    for index, target_name in enumerate(TARGET_COLUMNS):
        frame[target_name] = predictions[:, index]
    return frame


def _aggregate_test_predictions(frames: list[pd.DataFrame]) -> pd.DataFrame:
    combined = pd.concat(frames, ignore_index=True)
    result = combined[["planet_ID"]].drop_duplicates().sort_values("planet_ID").reset_index(drop=True)
    for target_name in TARGET_COLUMNS:
        stats = (
            combined.groupby("planet_ID", sort=True)[target_name]
            .agg(["mean", "std"])
            .reset_index()
            .rename(columns={"mean": target_name, "std": f"{target_name}_std"})
        )
        result = result.merge(stats, on="planet_ID", how="left", validate="one_to_one")
        result[f"{target_name}_std"] = result[f"{target_name}_std"].fillna(0.0)
    return result


def run_cross_validation_experiment(
    base_config=None,
    cv_config: Optional[CrossValidationConfig] = None,
) -> dict[str, Any]:
    import torch

    from .training import (
        TrainingConfig,
        build_loss_fn,
        configure_runtime,
        evaluate_labeled_split,
        metrics_payload,
        move_prepared_data_to_device,
        predict_inference_split,
        resolve_training_device,
        save_inference_predictions,
        save_json,
        save_labeled_predictions,
        set_runtime_seed,
        train_model,
    )

    cfg = base_config or TrainingConfig()
    cv_cfg = cv_config or CrossValidationConfig()
    cfg.project_root = str(cfg.resolved_project_root())

    configure_runtime()
    device = resolve_training_device(cfg)
    output_root = cfg.resolved_output_dir()
    output_root.mkdir(parents=True, exist_ok=True)

    save_json(output_root / "cross_validation_config.json", cv_cfg.to_json_dict())
    save_json(output_root / "base_training_config.json", cfg.to_json_dict())

    raw_data = load_raw_ariel_data(cfg.resolved_data_root())
    folds = build_cross_validation_folds(
        raw_data.labeled_targets_raw,
        num_folds=cv_cfg.num_folds,
        seed=cfg.seed,
        val_fraction=cv_cfg.val_fraction,
        train_limit=cfg.train_limit,
        val_limit=cfg.val_limit,
        holdout_limit=cfg.holdout_limit,
    )

    fold_summaries: list[dict[str, Any]] = []
    oof_frames: list[pd.DataFrame] = []
    test_prediction_frames: list[pd.DataFrame] = []

    for fold in folds:
        fold_number = fold.fold_index + 1
        fold_output_dir = output_root / f"fold_{fold_number:02d}"
        fold_output_dir.mkdir(parents=True, exist_ok=True)
        fold_seed = cfg.seed + fold.fold_index
        fold_config = replace(
            cfg,
            output_dir=str(fold_output_dir),
            prepared_cache_dir=None,
            seed=fold_seed,
        )

        print(f"Cross-validation fold {fold_number}/{cv_cfg.num_folds}", flush=True)
        print(
            f"Fold rows | train={len(fold.train_indices)} | val={len(fold.val_indices)} | "
            f"holdout={len(fold.holdout_indices)}",
            flush=True,
        )
        save_json(fold_output_dir / "config.json", fold_config.to_json_dict())

        set_runtime_seed(fold_seed)
        data = prepare_fold_data(
            raw_data=raw_data,
            fold=fold,
            output_dir=fold_output_dir,
            seed=fold_seed,
            test_limit=fold_config.test_limit,
        )
        save_json(fold_output_dir / "split_manifest.json", data.split_manifest)
        save_json(fold_output_dir / "prepared_manifest.json", data.prepared_manifest)
        save_json(
            fold_output_dir / "scalers.json",
            {
                "aux_scaler": data.aux_scaler.state_dict(),
                "target_scaler": data.target_scaler.state_dict(),
                "spectral_scaler": data.spectral_scaler.state_dict(),
            },
        )

        data = move_prepared_data_to_device(data, device)
        training = train_model(fold_config, data, device, output_dir=fold_output_dir)
        model = training["model"]
        loss_fn = build_loss_fn(fold_config)

        validation_metrics = evaluate_labeled_split(model, data.val, data.target_scaler, fold_config.eval_batch_size, loss_fn)
        holdout_metrics = evaluate_labeled_split(
            model,
            data.holdout,
            data.target_scaler,
            fold_config.eval_batch_size,
            loss_fn,
        )
        test_predictions = predict_inference_split(model, data.testdata, data.target_scaler, fold_config.eval_batch_size)

        save_json(fold_output_dir / "validation_metrics.json", metrics_payload(validation_metrics))
        save_json(fold_output_dir / "holdout_metrics.json", metrics_payload(holdout_metrics))
        save_labeled_predictions(fold_output_dir / "validation_predictions.csv", data.val, validation_metrics)
        save_labeled_predictions(fold_output_dir / "holdout_predictions.csv", data.holdout, holdout_metrics)
        save_inference_predictions(fold_output_dir / "testdata_predictions.csv", data.testdata, test_predictions)

        fold_summary = {
            "fold": int(fold_number),
            "best_epoch": int(training["best_epoch"]),
            "best_val_rmse_mean": float(training["best_val_rmse"]),
            "validation_rmse_mean": float(validation_metrics["rmse_mean"]),
            "validation_mae_mean": float(validation_metrics["mae_mean"]),
            "holdout_rmse_mean": float(holdout_metrics["rmse_mean"]),
            "holdout_mae_mean": float(holdout_metrics["mae_mean"]),
            "testdata_rows": int(data.testdata.rows),
            "output_dir": str(fold_output_dir),
            "dataset": data.split_manifest,
        }
        save_json(fold_output_dir / "run_summary.json", fold_summary)
        fold_summaries.append(fold_summary)
        oof_frames.append(_build_labeled_prediction_frame(fold_number, data.holdout.planet_ids, holdout_metrics))
        test_prediction_frames.append(_build_test_prediction_frame(fold_number, data.testdata.planet_ids, test_predictions))

        del model, training, data
        if device.type == "cuda":
            torch.cuda.empty_cache()

    fold_summary_frame = pd.DataFrame(fold_summaries)
    fold_summary_frame.to_csv(output_root / "fold_summaries.csv", index=False)

    oof_frame = pd.concat(oof_frames, ignore_index=True).sort_values(["planet_ID", "fold"]).reset_index(drop=True)
    oof_frame.to_csv(output_root / "oof_predictions.csv", index=False)
    oof_true = oof_frame[[f"true_{target_name}" for target_name in TARGET_COLUMNS]].to_numpy(dtype=np.float32)
    oof_pred = oof_frame[[f"pred_{target_name}" for target_name in TARGET_COLUMNS]].to_numpy(dtype=np.float32)
    oof_metrics = compute_regression_metrics(oof_true, oof_pred)
    save_json(output_root / "oof_metrics.json", oof_metrics)

    aggregated_test_predictions = _aggregate_test_predictions(test_prediction_frames)
    aggregated_test_predictions.to_csv(output_root / "testdata_predictions_ensemble.csv", index=False)

    summary = {
        "num_folds": int(cv_cfg.num_folds),
        "val_fraction": float(cv_cfg.val_fraction),
        "output_dir": str(output_root),
        "folds": fold_summaries,
        "aggregate": {
            "best_val_rmse_mean_avg": float(fold_summary_frame["best_val_rmse_mean"].mean()),
            "best_val_rmse_mean_std": float(fold_summary_frame["best_val_rmse_mean"].std(ddof=0)),
            "holdout_rmse_mean_avg": float(fold_summary_frame["holdout_rmse_mean"].mean()),
            "holdout_rmse_mean_std": float(fold_summary_frame["holdout_rmse_mean"].std(ddof=0)),
            "holdout_mae_mean_avg": float(fold_summary_frame["holdout_mae_mean"].mean()),
            "holdout_mae_mean_std": float(fold_summary_frame["holdout_mae_mean"].std(ddof=0)),
            "best_epoch_avg": float(fold_summary_frame["best_epoch"].mean()),
            "best_epoch_std": float(fold_summary_frame["best_epoch"].std(ddof=0)),
        },
        "oof_metrics": oof_metrics,
    }
    save_json(output_root / "cross_validation_summary.json", summary)
    return summary
