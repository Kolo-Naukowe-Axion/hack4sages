"""Generation entrypoint for the cross-generator biosignature dataset."""

from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .constants import (
    DATASET_NAME,
    DATASET_PATHS,
    MASTER_SEED,
    POSEIDON_GENERATOR_KEY,
    POSEIDON_NATIVE_RESOLUTION,
    TARGET_RESOLUTION,
    TARGET_WAVELENGTH_MAX_UM,
    TARGET_WAVELENGTH_MIN_UM,
    TAUREX_GENERATOR_KEY,
)
from .dataset_io import read_latents_parquet, write_labels_parquet, write_latents_parquet, write_spectra_h5
from .grid import build_rebin_matrix, make_constant_resolution_grid, rebin_spectrum
from .latents import build_latents, final_labels_frame, validate_latent_frame
from .utils import atomic_write_json, chunked, derived_seed, ensure_directory


@dataclass(frozen=True)
class GenerationConfig:
    output_root: Path
    mode: str = "all"
    seed: int = MASTER_SEED
    shard_size: int = 256
    workers: int = max(1, (os.cpu_count() or 1))
    force_latents: bool = False
    force_shards: bool = False
    force_assemble: bool = False


def _get_backend(generator: str) -> Any:
    if generator == TAUREX_GENERATOR_KEY:
        from .taurex_backend import TauRExBackend

        return TauRExBackend()
    if generator == POSEIDON_GENERATOR_KEY:
        from .poseidon_backend import PoseidonBackend

        return PoseidonBackend()
    raise ValueError(f"Unsupported generator {generator!r}.")


def _generator_shard_rows(latents: pd.DataFrame, generator: str) -> list[dict[str, Any]]:
    subset = latents.loc[latents["generator"] == generator].sort_values("sample_index").reset_index(drop=True)
    return subset.to_dict(orient="records")


def _shard_path(output_root: Path, generator: str, start_sample_index: int, end_sample_index: int) -> Path:
    shard_dir = ensure_directory(output_root / DATASET_PATHS.shards_dir / generator)
    return shard_dir / f"{generator}_{start_sample_index:06d}_{end_sample_index:06d}.npz"


def _save_shard(path: Path, payload: dict[str, np.ndarray]) -> None:
    temp_path = path.with_suffix(".tmp.npz")
    np.savez_compressed(temp_path, **payload)
    temp_path.replace(path)


def _render_shard(task: dict[str, Any]) -> dict[str, Any]:
    generator = str(task["generator"])
    output_path = Path(task["output_path"])
    rows = task["rows"]
    master_seed = int(task["seed"])
    target_centers, target_edges = make_constant_resolution_grid(
        TARGET_WAVELENGTH_MIN_UM,
        TARGET_WAVELENGTH_MAX_UM,
        TARGET_RESOLUTION,
    )

    backend = _get_backend(generator)
    fixed_native_matrix = None
    native_wavelength = getattr(backend, "native_wavelength_um", None)
    if native_wavelength is not None:
        fixed_native_matrix = build_rebin_matrix(np.asarray(native_wavelength, dtype=np.float64), target_edges)

    noiseless = np.empty((len(rows), len(target_centers)), dtype=np.float32)
    noisy = np.empty_like(noiseless)
    sigma_ppm = np.empty(len(rows), dtype=np.float32)
    sample_ids: list[str] = []
    generators: list[str] = []
    splits: list[str] = []

    for row_idx, row in enumerate(rows):
        native_wavelength_um, native_depth = backend.render_native(row)
        if fixed_native_matrix is None:
            rebinned = rebin_spectrum(native_wavelength_um, native_depth, target_edges)
        else:
            rebinned = fixed_native_matrix.dot(np.asarray(native_depth, dtype=np.float64))

        local_sigma_ppm = float(row["sigma_ppm"])
        noise_rng = np.random.default_rng(derived_seed(master_seed, row["sample_id"], "noise"))
        noise = noise_rng.normal(loc=0.0, scale=local_sigma_ppm * 1.0e-6, size=len(target_centers))
        rebinned_noisy = np.clip(rebinned + noise, 1.0e-12, None)

        if not np.all(np.isfinite(rebinned)) or not np.all(np.isfinite(rebinned_noisy)):
            raise RuntimeError(f"Non-finite spectrum encountered for {row['sample_id']}.")

        noiseless[row_idx] = rebinned.astype(np.float32)
        noisy[row_idx] = rebinned_noisy.astype(np.float32)
        sigma_ppm[row_idx] = local_sigma_ppm
        sample_ids.append(str(row["sample_id"]))
        generators.append(str(row["generator"]))
        splits.append(str(row["split"]))

    _save_shard(
        output_path,
        {
            "sample_id": np.asarray(sample_ids, dtype="U32"),
            "generator": np.asarray(generators, dtype="U16"),
            "split": np.asarray(splits, dtype="U16"),
            "transit_depth_noiseless": noiseless,
            "transit_depth_noisy": noisy,
            "sigma_ppm": sigma_ppm,
        },
    )
    return {
        "generator": generator,
        "row_count": len(rows),
        "output_path": str(output_path),
        "start_sample_index": int(rows[0]["sample_index"]),
        "end_sample_index": int(rows[-1]["sample_index"]),
        "software_versions": backend.software_versions(),
    }


def ensure_latents(output_root: Path, seed: int, force: bool = False) -> pd.DataFrame:
    """Create or load the canonical latent table."""

    ensure_directory(output_root)
    latents_path = output_root / DATASET_PATHS.latents_parquet
    if latents_path.exists() and not force:
        latents = read_latents_parquet(output_root)
    else:
        latents = build_latents(master_seed=seed)
        validate_latent_frame(latents)
        write_latents_parquet(output_root, latents)
    return latents


def _run_tasks(
    tasks: list[dict[str, Any]],
    workers: int,
    *,
    total_rows: int,
    completed_rows_initial: int = 0,
    progress_interval_rows: int = 1_000,
) -> list[dict[str, Any]]:
    if len(tasks) == 0:
        return []
    completed_rows = int(completed_rows_initial)
    next_progress_row = (
        ((completed_rows // progress_interval_rows) + 1) * progress_interval_rows if progress_interval_rows > 0 else total_rows + 1
    )
    if workers <= 1:
        results: list[dict[str, Any]] = []
        for completed_count, task in enumerate(tasks, start=1):
            result = _render_shard(task)
            completed_rows += int(result["row_count"])
            print(
                f"[{result['generator']}] completed shard {completed_count}/{len(tasks)} "
                f"shard_rows={result['row_count']} completed_rows={completed_rows}/{total_rows} "
                f"sample_index={result['start_sample_index']}-{result['end_sample_index']}",
                flush=True,
            )
            while progress_interval_rows > 0 and completed_rows >= next_progress_row:
                print(
                    f"[{result['generator']}] progress completed_rows={completed_rows}/{total_rows}",
                    flush=True,
                )
                next_progress_row += progress_interval_rows
            results.append(result)
        return results
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_render_shard, task) for task in tasks]
        results = []
        for completed_count, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            completed_rows += int(result["row_count"])
            print(
                f"[{result['generator']}] completed shard {completed_count}/{len(tasks)} "
                f"shard_rows={result['row_count']} completed_rows={completed_rows}/{total_rows} "
                f"sample_index={result['start_sample_index']}-{result['end_sample_index']}",
                flush=True,
            )
            while progress_interval_rows > 0 and completed_rows >= next_progress_row:
                print(
                    f"[{result['generator']}] progress completed_rows={completed_rows}/{total_rows}",
                    flush=True,
                )
                next_progress_row += progress_interval_rows
            results.append(result)
        return results


def generate_generator_shards(
    output_root: Path,
    latents: pd.DataFrame,
    generator: str,
    seed: int,
    shard_size: int,
    workers: int,
    force: bool = False,
) -> dict[str, Any]:
    """Render one generator into resumable shard files."""

    rows = _generator_shard_rows(latents, generator)
    tasks: list[dict[str, Any]] = []
    for chunk in chunked(rows, shard_size):
        start_sample_index = int(chunk[0]["sample_index"])
        end_sample_index = int(chunk[-1]["sample_index"])
        output_path = _shard_path(output_root, generator, start_sample_index, end_sample_index)
        if output_path.exists() and not force:
            continue
        tasks.append(
            {
                "generator": generator,
                "output_path": str(output_path),
                "rows": chunk,
                "seed": seed,
            }
        )

    total_shards = (len(rows) + shard_size - 1) // shard_size if rows else 0
    existing_shards = total_shards - len(tasks)
    pending_rows = sum(len(task["rows"]) for task in tasks)
    existing_rows = len(rows) - pending_rows
    print(
        f"[{generator}] rows={len(rows)} workers={workers} shard_size={shard_size} "
        f"total_shards={total_shards} existing_shards={existing_shards} pending_shards={len(tasks)} "
        f"existing_rows={existing_rows} pending_rows={pending_rows}",
        flush=True,
    )
    results = _run_tasks(tasks, workers=workers, total_rows=len(rows), completed_rows_initial=existing_rows)
    meta_path = ensure_directory(output_root / DATASET_PATHS.metadata_dir) / f"{generator}_generation.json"
    if results:
        software_versions = results[0]["software_versions"]
    else:
        software_versions = _get_backend(generator).software_versions()
    metadata = {
        "generator": generator,
        "data_set": DATASET_NAME,
        "row_count": sum(1 for _ in rows),
        "shard_count": len(list((output_root / DATASET_PATHS.shards_dir / generator).glob("*.npz"))),
        "shard_size": shard_size,
        "software_versions": software_versions,
        "target_resolution": TARGET_RESOLUTION,
        "target_wavelength_min_um": TARGET_WAVELENGTH_MIN_UM,
        "target_wavelength_max_um": TARGET_WAVELENGTH_MAX_UM,
        "poseidon_native_resolution": POSEIDON_NATIVE_RESOLUTION if generator == POSEIDON_GENERATOR_KEY else None,
    }
    atomic_write_json(meta_path, metadata)
    return metadata


def assemble_dataset(output_root: Path, latents: pd.DataFrame, force: bool = False) -> dict[str, Any]:
    """Assemble the final public dataset from latent and shard artifacts."""

    spectra_path = output_root / DATASET_PATHS.spectra_h5
    labels_path = output_root / DATASET_PATHS.labels_parquet
    manifest_path = output_root / DATASET_PATHS.manifest_json
    if spectra_path.exists() and labels_path.exists() and manifest_path.exists() and not force:
        return {"spectra_h5": str(spectra_path), "labels_parquet": str(labels_path), "manifest_json": str(manifest_path)}

    target_centers, target_edges = make_constant_resolution_grid(
        TARGET_WAVELENGTH_MIN_UM,
        TARGET_WAVELENGTH_MAX_UM,
        TARGET_RESOLUTION,
    )
    labels = final_labels_frame(latents).sort_values(["generator", "sample_id"]).reset_index(drop=True)
    ordered_latents = latents.sort_values(["generator", "sample_id"]).reset_index(drop=True)
    sample_to_index = {sample_id: idx for idx, sample_id in enumerate(labels["sample_id"].tolist())}

    noiseless = np.empty((len(labels), len(target_centers)), dtype=np.float32)
    noisy = np.empty_like(noiseless)
    sigma_ppm = ordered_latents["sigma_ppm"].to_numpy(dtype=np.float32)
    seen = np.zeros(len(labels), dtype=bool)

    available_generators = tuple(sorted(labels["generator"].astype(str).unique().tolist()))
    for generator in available_generators:
        shard_dir = output_root / DATASET_PATHS.shards_dir / generator
        if not shard_dir.exists():
            raise FileNotFoundError(str(shard_dir))
        for shard_path in sorted(shard_dir.glob("*.npz")):
            payload = np.load(shard_path)
            sample_ids = payload["sample_id"].astype("U32")
            for shard_row_idx, sample_id in enumerate(sample_ids):
                target_row_idx = sample_to_index[str(sample_id)]
                noiseless[target_row_idx] = payload["transit_depth_noiseless"][shard_row_idx]
                noisy[target_row_idx] = payload["transit_depth_noisy"][shard_row_idx]
                seen[target_row_idx] = True

    if not np.all(seen):
        missing = labels.loc[~seen, "sample_id"].tolist()
        raise RuntimeError(f"Missing generated spectra for {len(missing)} samples. First few: {missing[:5]}")

    write_labels_parquet(output_root, ordered_latents)
    write_spectra_h5(output_root, labels, target_centers, noiseless, noisy, sigma_ppm)

    generator_summary: dict[str, Any] = {}
    for generator in available_generators:
        subset = labels.loc[labels["generator"] == generator].reset_index(drop=True)
        generator_summary[generator] = {
            "row_count": int(len(subset)),
            "split_counts": {str(key): int(value) for key, value in subset["split"].value_counts().sort_index().to_dict().items()},
            "target_means": {
                column: float(subset[column].mean())
                for column in labels.columns
                if column.startswith("log10_vmr_")
            },
            "target_stds": {
                column: float(subset[column].std(ddof=0))
                for column in labels.columns
                if column.startswith("log10_vmr_")
            },
            "prevalence": {
                column: float(subset[column].mean())
                for column in labels.columns
                if column.startswith("present_")
            },
        }

    if {TAUREX_GENERATOR_KEY, POSEIDON_GENERATOR_KEY}.issubset(set(available_generators)):
        comparison = {
            "delta_mean_log10_vmr": {
                column: float(generator_summary[POSEIDON_GENERATOR_KEY]["target_means"][column] - generator_summary[TAUREX_GENERATOR_KEY]["target_means"][column])
                for column in generator_summary[TAUREX_GENERATOR_KEY]["target_means"]
            },
            "delta_std_log10_vmr": {
                column: float(generator_summary[POSEIDON_GENERATOR_KEY]["target_stds"][column] - generator_summary[TAUREX_GENERATOR_KEY]["target_stds"][column])
                for column in generator_summary[TAUREX_GENERATOR_KEY]["target_stds"]
            },
            "delta_prevalence": {
                column: float(generator_summary[POSEIDON_GENERATOR_KEY]["prevalence"][column] - generator_summary[TAUREX_GENERATOR_KEY]["prevalence"][column])
                for column in generator_summary[TAUREX_GENERATOR_KEY]["prevalence"]
            },
        }
    else:
        comparison = {}

    software_versions: dict[str, Any] = {}
    for generator in available_generators:
        meta_path = output_root / DATASET_PATHS.metadata_dir / f"{generator}_generation.json"
        if meta_path.exists():
            software_versions[generator] = pd.read_json(meta_path, typ="series").to_dict().get("software_versions", {})

    manifest = {
        "dataset_name": DATASET_NAME,
        "seed": int(MASTER_SEED),
        "row_count": int(len(labels)),
        "generator_counts": {generator: int((labels["generator"] == generator).sum()) for generator in available_generators},
        "wavelength_grid": {
            "type": "constant_resolution",
            "resolution": TARGET_RESOLUTION,
            "min_um": TARGET_WAVELENGTH_MIN_UM,
            "max_um_requested": TARGET_WAVELENGTH_MAX_UM,
            "max_um_actual_edge": float(target_edges[-1]),
            "bin_count": int(len(target_centers)),
        },
        "noise_model": {
            "type": "iid_gaussian_white",
            "sigma_ppm_min": float(ordered_latents["sigma_ppm"].min()),
            "sigma_ppm_max": float(ordered_latents["sigma_ppm"].max()),
            "per_sample_scalar": True,
        },
        "software_versions": software_versions,
        "generator_summary": generator_summary,
        "comparison_summary": comparison,
    }
    atomic_write_json(output_root / DATASET_PATHS.manifest_json, manifest)
    return {"spectra_h5": str(spectra_path), "labels_parquet": str(labels_path), "manifest_json": str(manifest_path)}


def run_generation(config: GenerationConfig) -> dict[str, Any]:
    """Run the requested generation mode."""

    latents = ensure_latents(config.output_root, seed=config.seed, force=config.force_latents)
    results: dict[str, Any] = {"latents": str(config.output_root / DATASET_PATHS.latents_parquet)}

    if config.mode in ("all", TAUREX_GENERATOR_KEY):
        results[TAUREX_GENERATOR_KEY] = generate_generator_shards(
            output_root=config.output_root,
            latents=latents,
            generator=TAUREX_GENERATOR_KEY,
            seed=config.seed,
            shard_size=config.shard_size,
            workers=config.workers,
            force=config.force_shards,
        )
    if config.mode in ("all", POSEIDON_GENERATOR_KEY):
        results[POSEIDON_GENERATOR_KEY] = generate_generator_shards(
            output_root=config.output_root,
            latents=latents,
            generator=POSEIDON_GENERATOR_KEY,
            seed=config.seed,
            shard_size=config.shard_size,
            workers=config.workers,
            force=config.force_shards,
        )
    if config.mode in ("all", "assemble"):
        results["assembly"] = assemble_dataset(config.output_root, latents, force=config.force_assemble)
    return results


def parse_args() -> GenerationConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--mode", choices=("all", "tau", "poseidon", "assemble"), default="all")
    parser.add_argument("--seed", type=int, default=MASTER_SEED)
    parser.add_argument("--shard-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 1)))
    parser.add_argument("--force-latents", action="store_true")
    parser.add_argument("--force-shards", action="store_true")
    parser.add_argument("--force-assemble", action="store_true")
    args = parser.parse_args()
    return GenerationConfig(
        output_root=args.output_root,
        mode=args.mode,
        seed=args.seed,
        shard_size=args.shard_size,
        workers=args.workers,
        force_latents=args.force_latents,
        force_shards=args.force_shards,
        force_assemble=args.force_assemble,
    )


def main() -> None:
    config = parse_args()
    summary = run_generation(config)
    print("Cross-generator dataset generation summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
