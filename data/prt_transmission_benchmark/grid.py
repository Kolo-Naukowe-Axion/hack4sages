"""Benchmark grid and rebinning helpers."""

from __future__ import annotations

import numpy as np


def make_log_resolving_power_grid(
    wavelength_min_um: float,
    wavelength_max_um: float,
    resolving_power: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return geometric-mean centers and edges for a constant-R wavelength grid."""

    if wavelength_min_um <= 0.0 or wavelength_max_um <= wavelength_min_um:
        raise ValueError("Invalid wavelength bounds.")
    if resolving_power <= 1.0:
        raise ValueError("Resolving power must be > 1.")

    ratio = float(np.exp(1.0 / resolving_power))
    edges = [float(wavelength_min_um)]
    while edges[-1] < wavelength_max_um:
        next_edge = edges[-1] * ratio
        if next_edge <= edges[-1]:
            raise RuntimeError("Grid construction stalled.")
        edges.append(min(next_edge, wavelength_max_um))
        if edges[-1] == wavelength_max_um:
            break

    edges_arr = np.asarray(edges, dtype=np.float64)
    centers_arr = np.sqrt(edges_arr[:-1] * edges_arr[1:])
    return centers_arr, edges_arr


def bin_edges_from_centers(centers: np.ndarray) -> np.ndarray:
    """Infer bin edges from strictly increasing bin centers."""

    values = np.asarray(centers, dtype=np.float64)
    if values.ndim != 1 or len(values) < 2:
        raise ValueError("At least two centers are required.")
    if not np.all(np.diff(values) > 0.0):
        raise ValueError("Centers must be strictly increasing.")

    edges = np.empty(len(values) + 1, dtype=np.float64)
    edges[1:-1] = np.sqrt(values[:-1] * values[1:])
    edges[0] = values[0] * values[0] / edges[1]
    edges[-1] = values[-1] * values[-1] / edges[-2]
    return edges


def build_rebin_matrix(source_centers_um: np.ndarray, target_edges_um: np.ndarray) -> np.ndarray:
    """Build a flux-conserving overlap matrix from source bins to target bins."""

    source_centers = np.asarray(source_centers_um, dtype=np.float64)
    target_edges = np.asarray(target_edges_um, dtype=np.float64)
    source_edges = bin_edges_from_centers(source_centers)

    matrix = np.zeros((len(target_edges) - 1, len(source_centers)), dtype=np.float64)
    src_index = 0
    for tgt_index in range(len(target_edges) - 1):
        tgt_lo = target_edges[tgt_index]
        tgt_hi = target_edges[tgt_index + 1]
        while src_index + 1 < len(source_edges) and source_edges[src_index + 1] <= tgt_lo:
            src_index += 1
        scan_index = src_index
        total = 0.0
        while scan_index < len(source_centers) and source_edges[scan_index] < tgt_hi:
            overlap_lo = max(tgt_lo, source_edges[scan_index])
            overlap_hi = min(tgt_hi, source_edges[scan_index + 1])
            overlap = max(overlap_hi - overlap_lo, 0.0)
            if overlap > 0.0:
                matrix[tgt_index, scan_index] = overlap
                total += overlap
            scan_index += 1
        if total <= 0.0:
            raise RuntimeError("Target bin falls outside the source grid.")
        matrix[tgt_index, :] /= total
    return matrix

