"""Loading and parsing for the ExoBiome data layer

This is the only module with side effects (reading HDF5 / CSV / uploads)
Keeping I/O isolated here lets the rest of the data layer
(``preprocessing``, ``pipeline``, ``comparison``) stay pure.

The real ADC2023 data lives under ``<data_root>/{TrainingData,TestData}/``:
    <data_root>/TrainingData/SpectralData.hdf5
    <data_root>/TrainingData/AuxillaryTable.csv
    <data_root>/TrainingData/Ground Truth Package/FM_Parameter_Table.csv
    <data_root>/TestData/...(no ground truth)

We reuse the channel/column conventions from
``models/sbi_ariel_adc2023/constants.py``.
"""

from __future__ import annotations

import io
import os
from functools import lru_cache
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd

from .types import (
    AUX_COLS,
    GASES,
    N_BINS,
    AuxFeatures,
    CuratedPlanet,
    DataError,
    GroundTruth,
    PlanetRecord,
    RawSpectrum,
)

# conventions from models/sbi_ariel_adc2023/constants.py
HDF5_GROUP_PREFIX = "Planet_"
SPECTRUM_DS = "instrument_spectrum"
NOISE_DS = "instrument_noise"
WIDTH_DS = "instrument_width"
WAVELENGTH_DS = "instrument_wlgrid"

# split name -> directory on disk
_SPLIT_DIRS = {"train": "TrainingData", "test": "TestData"}

# A small, hand-picked set of planets with diverse chemistry, used for the app's
# dropdown. All live in the held-out validation split (the model never trained on
# them) yet carry ground truth, since that split is a slice of TrainingData.
CURATED_IDS: tuple[str, ...] = (
    "train37",     # H2O-rich
    "train13860",  # H2O-rich
    "train47",     # H2O + CO2 present
    "train28",     # CH4-rich
    "train3244",   # CH4-rich
    "train34274",  # CH4-rich
    "train12",     # CO-rich
    "train5975",   # CO-rich
)

def _resolve_data_root(data_root: str | Path | None = None) -> Path:
    """Find the ADC2023 dataset directory.
    Order: explicit argument -> ``$EXOBIOME_DATA`` -> walk up from this file /
    the cwd looking for ``data/ariel-ml-dataset/TrainingData``
    """
    candidates: list[Path] = []
    if data_root is not None:
        candidates.append(Path(data_root))
    env = os.environ.get("EXOBIOME_DATA")
    if env:
        candidates.append(Path(env))
    for start in (Path.cwd(), Path(__file__).resolve()):
        for parent in (start, *start.parents):
            candidates.append(parent / "data" / "ariel-ml-dataset")

    for cand in candidates:
        cand = cand.expanduser() # to make ~/data -> C:/Users/mery_christmas/data
        if (cand / "TrainingData").is_dir():
            return cand.resolve()
    raise DataError(
        "Could not locate the ADC2023 dataset (data/ariel-ml-dataset). "
        "Pass data_root explicitly or set $EXOBIOME_DATA."
    )

@lru_cache(maxsize=8)
def _aux_table(path: str) -> pd.DataFrame:
    return pd.read_csv(path).set_index("planet_ID")


@lru_cache(maxsize=8)
def _truth_table(path: str) -> pd.DataFrame:
    return pd.read_csv(path).set_index("planet_ID")


# --- low-level reads --------------------------------------------------------


def _read_group(group: "h5py.Group", planet_id: str) -> RawSpectrum:
    """Build a RawSpectrum from one HDF5 group"""

    def channel(name: str) -> np.ndarray:
        arr = np.asarray(group[name][:], dtype=np.float32)
        if arr.shape != (N_BINS,):
            raise DataError(f"{name} for {planet_id} has shape {arr.shape}, expected ({N_BINS},).")
        return arr

    return RawSpectrum(
        planet_id=planet_id,
        flux=channel(SPECTRUM_DS),
        noise=channel(NOISE_DS),
        width=channel(WIDTH_DS),
        wavelength=channel(WAVELENGTH_DS),
    )


def _read_aux(root: Path, split: str, planet_id: str) -> AuxFeatures:
    path = root / _SPLIT_DIRS[split] / "AuxillaryTable.csv"
    table = _aux_table(str(path))
    if planet_id not in table.index:
        raise DataError(f"No auxiliary row for '{planet_id}' in {path.name}.")
    values = table.loc[planet_id, list(AUX_COLS)].to_numpy(dtype=np.float32)
    return AuxFeatures(planet_id=planet_id, values=values)


def _read_truth(root: Path, split: str, planet_id: str) -> GroundTruth | None:
    path = root / _SPLIT_DIRS[split] / "Ground Truth Package" / "FM_Parameter_Table.csv"
    if not path.exists():
        return None  # e.g. the public TestData split has no ground truth
    table = _truth_table(str(path))
    if planet_id not in table.index:
        return None
    row = table.loc[planet_id, list(GASES)]
    return GroundTruth(log_vmr={gas: float(row[gas]) for gas in GASES})


# --- public API -------------------------------------------------------------


def load_by_id(planet_id: str, data_root: str | Path | None = None) -> PlanetRecord:
    """Load one planet (spectrum + aux + truth-if-available) by its id.

    The bridge to the real ADC2023 data: opens each split's HDF5 at most once
    and reads the matching ``Planet_<id>`` group in the same handle (no redundant
    open), instead of loading the whole dataset.
    """
    root = _resolve_data_root(data_root)
    group_name = f"{HDF5_GROUP_PREFIX}{planet_id}"
    import h5py

    for split, subdir in _SPLIT_DIRS.items():
        hdf5 = root / subdir / "SpectralData.hdf5"
        if not hdf5.exists():
            continue
        with h5py.File(hdf5, "r") as handle:
            if group_name not in handle:
                continue
            spectrum = _read_group(handle[group_name], planet_id)
        aux = _read_aux(root, split, planet_id)
        truth = _read_truth(root, split, planet_id)
        return PlanetRecord(spectrum=spectrum, aux=aux, truth=truth)
    raise DataError(f"Planet '{planet_id}' not found in any split under {root}.")


def list_curated_planets(data_root: str | Path | None = None) -> list[CuratedPlanet]:
    """Curated dropdown: real planets with diverse chemistry, labelled by the
    dominant gas. Reads only the (cached) ground-truth table - no HDF5 is opened,
    since the label needs truth, not the spectrum. The curated ids live in the
    held-out validation split, stored under TrainingData (which carries truth)."""
    root = _resolve_data_root(data_root)
    truth_path = root / "TrainingData" / "Ground Truth Package" / "FM_Parameter_Table.csv"
    if not truth_path.exists():
        return []
    table = _truth_table(str(truth_path))
    out: list[CuratedPlanet] = []
    for planet_id in CURATED_IDS:
        if planet_id not in table.index:
            continue
        row = table.loc[planet_id, list(GASES)]
        dominant = max(GASES, key=lambda gas: float(row[gas]))
        gas = dominant.replace("log_", "")
        out.append(CuratedPlanet(planet_id=planet_id, dominant_gas=gas, label=f"{planet_id} - {gas}-rich"))
    return out


def iter_spectra(
    data_root: str | Path | None = None, split: str = "train", limit: int | None = None
) -> Iterator[RawSpectrum]:
    """Lazily yield spectra one planet at a time (generator → no need to hold
    all 40k+ spectra in memory). Demonstrates lazy evaluation for report 2.3."""
    if split not in _SPLIT_DIRS:
        raise DataError(f"Unknown split '{split}', expected one of {tuple(_SPLIT_DIRS)}.")
    root = _resolve_data_root(data_root)
    hdf5 = root / _SPLIT_DIRS[split] / "SpectralData.hdf5"
    import h5py

    with h5py.File(hdf5, "r") as handle:
        for count, group_name in enumerate(handle):
            if limit is not None and count >= limit:
                break
            planet_id = group_name[len(HDF5_GROUP_PREFIX):]
            yield _read_group(handle[group_name], planet_id)


# --- upload / export (self-contained one-row CSV) ---------------------------

_FLUX_COLS = [f"flux_{i}" for i in range(N_BINS)]
_NOISE_COLS = [f"noise_{i}" for i in range(N_BINS)]
_WL_COLS = [f"wl_{i}" for i in range(N_BINS)]
_TRUE_COLS = [f"true_{gas}" for gas in GASES]


def record_to_csv(record: PlanetRecord) -> str:
    """Serialize a PlanetRecord to a one-row CSV string (pure).

    This is the format the app exports as a template and accepts on upload, so a
    user always has a valid example to edit instead of inventing numbers.
    """
    row: dict[str, object] = {"planet_id": record.planet_id}
    for col, value in zip(AUX_COLS, record.aux.values):
        row[col] = float(value)
    row.update({c: float(v) for c, v in zip(_FLUX_COLS, record.spectrum.flux)})
    row.update({c: float(v) for c, v in zip(_NOISE_COLS, record.spectrum.noise)})
    row.update({c: float(v) for c, v in zip(_WL_COLS, record.spectrum.wavelength)})
    if record.truth is not None:
        row.update({f"true_{gas}": record.truth.log_vmr[gas] for gas in GASES})
    return pd.DataFrame([row]).to_csv(index=False)


def export_record(planet_id: str, data_root: str | Path | None = None) -> str:
    """Load a real planet and return it as an upload-format CSV template."""
    return record_to_csv(load_by_id(planet_id, data_root))


def parse_upload(source: str | Path | io.IOBase) -> PlanetRecord:
    """Parse a user-uploaded one-row CSV into a PlanetRecord.

    Validates that the spectrum (52), noise (52) and all 8 aux features are
    present - a bare spectrum is rejected, because the model needs aux too.
    Ground-truth columns are optional.
    """
    frame = pd.read_csv(source)
    if len(frame) != 1:
        raise DataError(f"Upload must contain exactly one planet row, got {len(frame)}.")
    row = frame.iloc[0]

    missing_aux = [c for c in AUX_COLS if c not in frame.columns]
    if missing_aux:
        raise DataError(
            "Upload is missing required auxiliary columns: "
            f"{missing_aux}. A spectrum alone is not enough - the model needs "
            "the 8 stellar/planetary features."
        )

    def vector(cols: list[str], what: str) -> np.ndarray:
        missing = [c for c in cols if c not in frame.columns]
        if missing:
            raise DataError(f"Upload is missing {what} columns (need {len(cols)}, e.g. {missing[:3]}).")
        return row[cols].to_numpy(dtype=np.float32)

    planet_id = str(row["planet_id"]) if "planet_id" in frame.columns else "upload"
    spectrum = RawSpectrum(
        planet_id=planet_id,
        flux=vector(_FLUX_COLS, "flux"),
        noise=vector(_NOISE_COLS, "noise"),
        width=np.zeros(N_BINS, dtype=np.float32),  # not used by the model (fixed channel)
        wavelength=vector(_WL_COLS, "wavelength (wl)") if _WL_COLS[0] in frame.columns
        else np.zeros(N_BINS, dtype=np.float32),
    )
    aux = AuxFeatures(planet_id=planet_id, values=row[list(AUX_COLS)].to_numpy(dtype=np.float32))

    truth = None
    if all(c in frame.columns for c in _TRUE_COLS):
        truth = GroundTruth(log_vmr={gas: float(row[f"true_{gas}"]) for gas in GASES})
    return PlanetRecord(spectrum=spectrum, aux=aux, truth=truth)
