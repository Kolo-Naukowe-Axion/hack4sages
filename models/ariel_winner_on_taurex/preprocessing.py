"""Preprocessing helpers for the winner-style independent NSF model."""

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


def compute_radius_features_torch(aux: torch.Tensor, spectra: torch.Tensor) -> torch.Tensor:
    star_radius = aux[:, 2:3]
    planet_mass = aux[:, 4:5]
    surface_gravity = aux[:, 7:8]
    mean_spectrum = torch.clamp(spectra.mean(dim=1, keepdim=True), min=1e-12)
    radius_from_spectrum = star_radius / RJUP_M * torch.sqrt(mean_spectrum)
    radius_from_gravity = torch.sqrt(torch.clamp(planet_mass / surface_gravity * G_NEWTON, min=1e-18)) / RJUP_M
    in_vertical_line = (
        (radius_from_gravity > VERTICAL_LINE_LOW) & (radius_from_gravity < VERTICAL_LINE_HIGH)
    ).to(aux.dtype)
    radius_combo = radius_from_gravity * (1.0 - in_vertical_line) + radius_from_spectrum * in_vertical_line
    return torch.cat([aux, radius_from_spectrum, radius_from_gravity, in_vertical_line, radius_combo], dim=1)


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
        data = np.load(path)
        return cls(
            aux_min=data["aux_min"].astype(np.float32),
            aux_scale=data["aux_scale"].astype(np.float32),
            noise_min=data["noise_min"].astype(np.float32),
            noise_scale=data["noise_scale"].astype(np.float32),
            spectrum_stats_min=data["spectrum_stats_min"].astype(np.float32),
            spectrum_stats_scale=data["spectrum_stats_scale"].astype(np.float32),
            target_min=data["target_min"].astype(np.float32),
            target_scale=data["target_scale"].astype(np.float32),
        )


@dataclass
class RuntimeScalers:
    aux_min: torch.Tensor
    aux_scale: torch.Tensor
    noise_min: torch.Tensor
    noise_scale: torch.Tensor
    spectrum_stats_min: torch.Tensor
    spectrum_stats_scale: torch.Tensor


def scalers_to_device(scalers: PreparedScalers, device: torch.device, dtype: torch.dtype = torch.float32) -> RuntimeScalers:
    return RuntimeScalers(
        aux_min=torch.as_tensor(scalers.aux_min, device=device, dtype=dtype),
        aux_scale=torch.as_tensor(scalers.aux_scale, device=device, dtype=dtype),
        noise_min=torch.as_tensor(scalers.noise_min, device=device, dtype=dtype),
        noise_scale=torch.as_tensor(scalers.noise_scale, device=device, dtype=dtype),
        spectrum_stats_min=torch.as_tensor(scalers.spectrum_stats_min, device=device, dtype=dtype),
        spectrum_stats_scale=torch.as_tensor(scalers.spectrum_stats_scale, device=device, dtype=dtype),
    )


def fit_scalers(train_aux: np.ndarray, train_spectra: np.ndarray, train_noise: np.ndarray, train_targets: np.ndarray) -> PreparedScalers:
    aux_features = compute_radius_features_numpy(train_aux, train_spectra)
    spectrum_mean = np.mean(train_spectra, axis=1, keepdims=True).astype(np.float32)
    spectrum_std = np.std(train_spectra, axis=1, keepdims=True).astype(np.float32)
    spectrum_stats = np.concatenate([spectrum_mean, spectrum_std], axis=1)
    return PreparedScalers(
        aux_min=aux_features.min(axis=0).astype(np.float32),
        aux_scale=_safe_scale(aux_features.max(axis=0) - aux_features.min(axis=0)),
        noise_min=train_noise.min(axis=0).astype(np.float32),
        noise_scale=_safe_scale(train_noise.max(axis=0) - train_noise.min(axis=0)),
        spectrum_stats_min=spectrum_stats.min(axis=0).astype(np.float32),
        spectrum_stats_scale=_safe_scale(spectrum_stats.max(axis=0) - spectrum_stats.min(axis=0)),
        target_min=train_targets.min(axis=0).astype(np.float32),
        target_scale=_safe_scale(train_targets.max(axis=0) - train_targets.min(axis=0)),
    )


def transform_targets(targets: np.ndarray, scalers: PreparedScalers) -> np.ndarray:
    return ((targets - scalers.target_min) / scalers.target_scale).astype(np.float32)


def inverse_transform_targets(targets: np.ndarray, scalers: PreparedScalers) -> np.ndarray:
    return (targets * scalers.target_scale + scalers.target_min).astype(np.float32)


def build_context(
    aux: torch.Tensor,
    spectra: torch.Tensor,
    noise: torch.Tensor,
    scalers: PreparedScalers | RuntimeScalers,
    *,
    sample_noise: bool,
    noise_generator: torch.Generator | None = None,
) -> torch.Tensor:
    dtype = spectra.dtype
    device = spectra.device

    if sample_noise:
        sampled_spectra = torch.normal(mean=spectra, std=noise, generator=noise_generator)
    else:
        sampled_spectra = spectra

    spectrum_mean = sampled_spectra.mean(dim=1, keepdim=True)
    spectrum_std = sampled_spectra.std(dim=1, keepdim=True)
    spectrum_std = torch.clamp(spectrum_std, min=1e-6)
    normalized_spectra = (sampled_spectra - spectrum_mean) / spectrum_std

    aux_engineered = compute_radius_features_torch(aux, sampled_spectra)

    if isinstance(scalers, RuntimeScalers):
        aux_min = scalers.aux_min
        aux_scale = scalers.aux_scale
        noise_min = scalers.noise_min
        noise_scale = scalers.noise_scale
        stats_min = scalers.spectrum_stats_min
        stats_scale = scalers.spectrum_stats_scale
    else:
        aux_min = torch.as_tensor(scalers.aux_min, device=device, dtype=dtype)
        aux_scale = torch.as_tensor(scalers.aux_scale, device=device, dtype=dtype)
        noise_min = torch.as_tensor(scalers.noise_min, device=device, dtype=dtype)
        noise_scale = torch.as_tensor(scalers.noise_scale, device=device, dtype=dtype)
        stats_min = torch.as_tensor(scalers.spectrum_stats_min, device=device, dtype=dtype)
        stats_scale = torch.as_tensor(scalers.spectrum_stats_scale, device=device, dtype=dtype)

    aux_scaled = (aux_engineered - aux_min) / aux_scale
    noise_scaled = (noise - noise_min) / noise_scale
    stats = torch.cat([spectrum_mean, spectrum_std], dim=1)
    stats_scaled = (stats - stats_min) / stats_scale

    context = torch.cat([aux_scaled, stats_scaled, normalized_spectra, noise_scaled], dim=1)
    expected_dim = aux_scaled.shape[1] + 2 + SPECTRAL_LENGTH + SPECTRAL_LENGTH
    if context.shape[1] != expected_dim:
        raise RuntimeError(f"Unexpected context dimension {context.shape[1]} != {expected_dim}.")
    return context
