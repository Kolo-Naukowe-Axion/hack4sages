"""Preprocessing helpers for the Ariel winner-family tracedata rerun package."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .constants import G_NEWTON, RJUP_M, SPECTRAL_LENGTH, VERTICAL_LINE_HIGH, VERTICAL_LINE_LOW


def _safe_scale(scale: np.ndarray) -> np.ndarray:
    scale = np.asarray(scale, dtype=np.float32)
    scale[~np.isfinite(scale)] = 1.0
    scale[np.abs(scale) < 1e-6] = 1.0
    return scale


def compute_radius_features_numpy(aux: np.ndarray, spectra: np.ndarray) -> np.ndarray:
    star_radius = aux[:, 2:3]
    planet_mass = aux[:, 4:5]
    surface_gravity = aux[:, 7:8]
    mean_spectrum = np.clip(np.mean(spectra, axis=1, keepdims=True), 1e-12, None)
    radius_from_spectrum = star_radius / RJUP_M * np.sqrt(mean_spectrum)
    radius_from_gravity = np.sqrt(np.clip(planet_mass / surface_gravity * G_NEWTON, 1e-18, None)) / RJUP_M
    in_vertical_line = (
        (radius_from_gravity > VERTICAL_LINE_LOW) & (radius_from_gravity < VERTICAL_LINE_HIGH)
    ).astype(np.float32)
    radius_combo = radius_from_gravity * (1.0 - in_vertical_line) + radius_from_spectrum * in_vertical_line
    return np.concatenate(
        [aux.astype(np.float32), radius_from_spectrum, radius_from_gravity, in_vertical_line, radius_combo],
        axis=1,
    ).astype(np.float32)


@dataclass
class PreparedScalers:
    aux_min: np.ndarray
    aux_scale: np.ndarray
    noise_min: np.ndarray
    noise_scale: np.ndarray
    spectrum_stats_min: np.ndarray
    spectrum_stats_scale: np.ndarray
    target_min: np.ndarray
    target_scale: np.ndarray

    def save(self, path) -> None:
        np.savez_compressed(
            path,
            aux_min=self.aux_min,
            aux_scale=self.aux_scale,
            noise_min=self.noise_min,
            noise_scale=self.noise_scale,
            spectrum_stats_min=self.spectrum_stats_min,
            spectrum_stats_scale=self.spectrum_stats_scale,
            target_min=self.target_min,
            target_scale=self.target_scale,
        )

    @classmethod
    def load(cls, path) -> "PreparedScalers":
        payload = np.load(path)
        return cls(
            aux_min=payload["aux_min"].astype(np.float32),
            aux_scale=payload["aux_scale"].astype(np.float32),
            noise_min=payload["noise_min"].astype(np.float32),
            noise_scale=payload["noise_scale"].astype(np.float32),
            spectrum_stats_min=payload["spectrum_stats_min"].astype(np.float32),
            spectrum_stats_scale=payload["spectrum_stats_scale"].astype(np.float32),
            target_min=payload["target_min"].astype(np.float32),
            target_scale=payload["target_scale"].astype(np.float32),
        )


def fit_scalers(
    train_aux: np.ndarray,
    train_spectra: np.ndarray,
    train_noise: np.ndarray,
    train_trace_targets: np.ndarray,
    train_trace_weights: np.ndarray,
) -> tuple[PreparedScalers, np.ndarray]:
    aux_features = compute_radius_features_numpy(train_aux, train_spectra)
    radius_reference = aux_features[:, -1].astype(np.float32)

    spectrum_mean = np.mean(train_spectra, axis=1, keepdims=True).astype(np.float32)
    spectrum_std = np.std(train_spectra, axis=1, keepdims=True).astype(np.float32)
    spectrum_stats = np.concatenate([spectrum_mean, np.clip(spectrum_std, 1.0e-6, None)], axis=1)

    adjusted_targets = train_trace_targets.astype(np.float32).copy()
    adjusted_targets[..., 0] -= radius_reference[:, None]
    valid_targets = adjusted_targets[train_trace_weights > 0]
    if valid_targets.size == 0:
        raise RuntimeError("Training tracedata contains no positive-weight target samples.")

    scalers = PreparedScalers(
        aux_min=aux_features.min(axis=0).astype(np.float32),
        aux_scale=_safe_scale(aux_features.max(axis=0) - aux_features.min(axis=0)),
        noise_min=train_noise.min(axis=0).astype(np.float32),
        noise_scale=_safe_scale(train_noise.max(axis=0) - train_noise.min(axis=0)),
        spectrum_stats_min=spectrum_stats.min(axis=0).astype(np.float32),
        spectrum_stats_scale=_safe_scale(spectrum_stats.max(axis=0) - spectrum_stats.min(axis=0)),
        target_min=valid_targets.min(axis=0).astype(np.float32),
        target_scale=_safe_scale(valid_targets.max(axis=0) - valid_targets.min(axis=0)),
    )
    return scalers, radius_reference


def build_context_numpy(aux: np.ndarray, spectra: np.ndarray, noise: np.ndarray, scalers: PreparedScalers) -> tuple[np.ndarray, np.ndarray]:
    aux_features = compute_radius_features_numpy(aux, spectra)
    radius_reference = aux_features[:, -1].astype(np.float32)

    spectrum_mean = np.mean(spectra, axis=1, keepdims=True).astype(np.float32)
    spectrum_std = np.std(spectra, axis=1, keepdims=True).astype(np.float32)
    spectrum_std = np.clip(spectrum_std, 1.0e-6, None)
    normalized_spectra = ((spectra - spectrum_mean) / spectrum_std).astype(np.float32)

    aux_scaled = ((aux_features - scalers.aux_min) / scalers.aux_scale).astype(np.float32)
    noise_scaled = ((noise - scalers.noise_min) / scalers.noise_scale).astype(np.float32)
    stats_scaled = ((np.concatenate([spectrum_mean, spectrum_std], axis=1) - scalers.spectrum_stats_min) / scalers.spectrum_stats_scale).astype(
        np.float32
    )

    context = np.concatenate([aux_scaled, stats_scaled, normalized_spectra, noise_scaled], axis=1).astype(np.float32)
    expected_dim = aux_scaled.shape[1] + 2 + SPECTRAL_LENGTH + SPECTRAL_LENGTH
    if context.shape[1] != expected_dim:
        raise RuntimeError(f"Unexpected context dimension {context.shape[1]} != {expected_dim}.")
    return context, radius_reference


def transform_targets_numpy(targets: np.ndarray, radius_reference: np.ndarray, scalers: PreparedScalers) -> np.ndarray:
    adjusted = np.asarray(targets, dtype=np.float32).copy()
    reshape = (radius_reference.shape[0],) + (1,) * (adjusted.ndim - 2) + (1,)
    adjusted[..., 0:1] -= radius_reference.reshape(reshape)
    return ((adjusted - scalers.target_min) / scalers.target_scale).astype(np.float32)


def inverse_transform_targets_numpy(targets: np.ndarray, radius_reference: np.ndarray, scalers: PreparedScalers) -> np.ndarray:
    restored = np.asarray(targets, dtype=np.float32) * scalers.target_scale + scalers.target_min
    reshape = (radius_reference.shape[0],) + (1,) * (restored.ndim - 2) + (1,)
    restored[..., 0:1] += radius_reference.reshape(reshape)
    return restored.astype(np.float32)


def transform_targets_torch(targets: torch.Tensor, radius_reference: torch.Tensor, scalers: PreparedScalers) -> torch.Tensor:
    target_min = torch.as_tensor(scalers.target_min, device=targets.device, dtype=targets.dtype)
    target_scale = torch.as_tensor(scalers.target_scale, device=targets.device, dtype=targets.dtype)
    reshape = (radius_reference.shape[0],) + (1,) * (targets.ndim - 2) + (1,)
    adjusted = targets.clone()
    adjusted[..., 0:1] -= radius_reference.reshape(reshape).to(device=targets.device, dtype=targets.dtype)
    return (adjusted - target_min) / target_scale


def inverse_transform_targets_torch(targets: torch.Tensor, radius_reference: torch.Tensor, scalers: PreparedScalers) -> torch.Tensor:
    target_min = torch.as_tensor(scalers.target_min, device=targets.device, dtype=targets.dtype)
    target_scale = torch.as_tensor(scalers.target_scale, device=targets.device, dtype=targets.dtype)
    reshape = (radius_reference.shape[0],) + (1,) * (targets.ndim - 2) + (1,)
    restored = targets * target_scale + target_min
    restored[..., 0:1] += radius_reference.reshape(reshape).to(device=targets.device, dtype=targets.dtype)
    return restored

