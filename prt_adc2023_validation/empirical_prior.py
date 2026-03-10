"""Empirical conditional prior used to generate ADC-compatible auxiliary metadata."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np
from scipy.spatial import cKDTree


@dataclass
class EmpiricalPriorSample:
    """One sampled auxiliary row and its provenance."""

    row_index: int
    planet_id: str
    rank: int
    distance: float
    aux_values: Dict[str, float]


class EmpiricalConditionalPrior:
    """Nearest-neighbor empirical prior fit from the local ADC2023 training data."""

    def __init__(self, bundle_path: Path):
        bundle = np.load(bundle_path, allow_pickle=False)
        self.planet_id = bundle["planet_id"]
        self.feature_matrix_z = bundle["feature_matrix_z"]
        self.feature_center = bundle["feature_center"]
        self.feature_scale = bundle["feature_scale"]
        self.feature_min = bundle["feature_min"]
        self.feature_max = bundle["feature_max"]
        self.empirical_feature_columns = bundle["empirical_feature_columns"].tolist()
        self.empirical_aux_columns = bundle["empirical_aux_columns"].tolist()
        self.aux_values = bundle["aux_values"]
        self.canonical_wlgrid = bundle["canonical_wlgrid"]
        self.canonical_instrument_width = bundle["canonical_instrument_width"]
        self.tree = cKDTree(self.feature_matrix_z)

    def _query_vector(
        self,
        planet_mass_kg: float,
        planet_surface_gravity: float,
        planet_radius_r_jup: float,
        t_connect_k: float,
    ) -> np.ndarray:
        raw = np.array(
            [
                float(planet_mass_kg),
                float(planet_surface_gravity),
                float(planet_radius_r_jup),
                float(t_connect_k),
            ],
            dtype=np.float64,
        )
        raw_log = np.log10(raw)
        clipped = np.clip(raw_log, self.feature_min, self.feature_max)
        return (clipped - self.feature_center) / self.feature_scale

    def sample(
        self,
        *,
        planet_mass_kg: float,
        planet_surface_gravity: float,
        planet_radius_r_jup: float,
        t_connect_k: float,
        neighbor_count: int,
        rng: np.random.Generator,
    ) -> EmpiricalPriorSample:
        """Sample one auxiliary row from the nearest empirical neighborhood."""

        query = self._query_vector(
            planet_mass_kg=planet_mass_kg,
            planet_surface_gravity=planet_surface_gravity,
            planet_radius_r_jup=planet_radius_r_jup,
            t_connect_k=t_connect_k,
        )
        k = min(int(neighbor_count), len(self.planet_id))
        distances, indices = self.tree.query(query, k=k)

        if np.isscalar(indices):
            indices = np.array([int(indices)], dtype=np.int64)
            distances = np.array([float(distances)], dtype=np.float64)
        else:
            indices = np.asarray(indices, dtype=np.int64)
            distances = np.asarray(distances, dtype=np.float64)

        rank = int(rng.integers(0, len(indices)))
        row_index = int(indices[rank])
        aux_values = {
            column: float(self.aux_values[row_index, idx])
            for idx, column in enumerate(self.empirical_aux_columns)
        }

        return EmpiricalPriorSample(
            row_index=row_index,
            planet_id=str(self.planet_id[row_index]),
            rank=rank,
            distance=float(distances[rank]),
            aux_values=aux_values,
        )

