"""Shared wavelength-grid and rebinning helpers."""

from __future__ import annotations

import numpy as np


def make_constant_resolution_grid(wavelength_min_um: float, wavelength_max_um: float, resolution: float) -> tuple[np.ndarray, np.ndarray]:
    """Return geometric-bin centers and edges for a constant-resolution grid."""

    if wavelength_min_um <= 0.0:
        raise ValueError("wavelength_min_um must be positive.")
    if wavelength_max_um <= wavelength_min_um:
        raise ValueError("wavelength_max_um must exceed wavelength_min_um.")
    if resolution <= 0.0:
        raise ValueError("resolution must be positive.")

    edges = [float(wavelength_min_um)]
    growth = 1.0 + (1.0 / float(resolution))
    while edges[-1] < float(wavelength_max_um):
        edges.append(edges[-1] * growth)
    edge_array = np.asarray(edges, dtype=np.float64)
    centers = np.sqrt(edge_array[:-1] * edge_array[1:])
    return centers, edge_array


def infer_bin_edges_from_centers(centers: np.ndarray) -> np.ndarray:
    """Infer monotonic bin edges from monotonic positive bin centers."""

    center_array = np.asarray(centers, dtype=np.float64)
    if center_array.ndim != 1 or len(center_array) < 2:
        raise ValueError("centers must be a 1D array with at least two elements.")
    if np.any(center_array <= 0.0):
        raise ValueError("centers must be strictly positive.")
    if not np.all(np.diff(center_array) > 0.0):
        raise ValueError("centers must be strictly increasing.")

    edges = np.empty(len(center_array) + 1, dtype=np.float64)
    interior = np.sqrt(center_array[:-1] * center_array[1:])
    edges[1:-1] = interior
    edges[0] = center_array[0] * center_array[0] / interior[0]
    edges[-1] = center_array[-1] * center_array[-1] / interior[-1]
    return edges


def build_rebin_matrix(source_centers_um: np.ndarray, target_edges_um: np.ndarray) -> np.ndarray:
    """Build a piecewise-constant wavelength-overlap matrix."""

    source_centers = np.asarray(source_centers_um, dtype=np.float64)
    target_edges = np.asarray(target_edges_um, dtype=np.float64)
    if source_centers.ndim != 1 or target_edges.ndim != 1:
        raise ValueError("source_centers_um and target_edges_um must be 1D arrays.")
    if len(target_edges) < 2:
        raise ValueError("target_edges_um must contain at least two edges.")
    if not np.all(np.diff(target_edges) > 0.0):
        raise ValueError("target_edges_um must be strictly increasing.")

    source_edges = infer_bin_edges_from_centers(source_centers)
    target_widths = target_edges[1:] - target_edges[:-1]
    matrix = np.zeros((len(target_widths), len(source_centers)), dtype=np.float64)

    source_idx = 0
    for target_idx, (target_left, target_right) in enumerate(zip(target_edges[:-1], target_edges[1:])):
        while source_idx < len(source_centers) and source_edges[source_idx + 1] <= target_left:
            source_idx += 1
        scan_idx = source_idx
        while scan_idx < len(source_centers) and source_edges[scan_idx] < target_right:
            overlap_left = max(target_left, source_edges[scan_idx])
            overlap_right = min(target_right, source_edges[scan_idx + 1])
            if overlap_right > overlap_left:
                matrix[target_idx, scan_idx] = (overlap_right - overlap_left) / target_widths[target_idx]
            scan_idx += 1

        row_sum = matrix[target_idx].sum()
        if row_sum <= 0.0:
            raise ValueError("Target bin is not covered by the source grid.")
        matrix[target_idx] /= row_sum

    return matrix


def rebin_spectrum(source_centers_um: np.ndarray, source_values: np.ndarray, target_edges_um: np.ndarray) -> np.ndarray:
    """Rebin a 1D spectrum from an arbitrary source grid onto target edges."""

    matrix = build_rebin_matrix(source_centers_um, target_edges_um)
    values = np.asarray(source_values, dtype=np.float64)
    if values.shape != (len(source_centers_um),):
        raise ValueError("source_values must have the same length as source_centers_um.")
    return matrix.dot(values)
