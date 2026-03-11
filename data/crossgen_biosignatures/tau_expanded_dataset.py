"""Build a full expanded cross-generator dataset with additional TauREx rows."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .constants import DATASET_PATHS, MASTER_SEED, POSEIDON_GENERATOR_KEY, TAUREX_GENERATOR_KEY
from .dataset_io import read_latents_parquet, write_latents_parquet
from .generate_dataset import assemble_dataset, generate_generator_shards
from .latents import build_tau_extension_latents, validate_latent_frame
from .utils import atomic_write_json, atomic_write_text, ensure_directory
from .validate_dataset import validate_dataset


@dataclass(frozen=True)
class TauExpandedDatasetConfig:
    """Configuration for building a full expanded dataset bundle."""

    base_output_root: Path
    output_root: Path
    new_tau_count: int
    seed: int = MASTER_SEED
    workers: int = max(1, (os.cpu_count() or 1))
    shard_size: int = 256
    force_copy: bool = False
    force_latents: bool = False
    force_shards: bool = False
    force_assemble: bool = False


def _copy_file(src: Path, dst: Path, *, force: bool) -> None:
    if dst.exists() and not force:
        return
    ensure_directory(dst.parent)
    shutil.copy2(src, dst)


def stage_base_artifacts(base_output_root: Path, output_root: Path, *, force: bool = False) -> dict[str, int]:
    """Copy the canonical base dataset shards needed for the expanded bundle."""

    copied_counts: dict[str, int] = {}
    for generator in (TAUREX_GENERATOR_KEY, POSEIDON_GENERATOR_KEY):
        src_dir = base_output_root / DATASET_PATHS.shards_dir / generator
        if not src_dir.exists():
            raise FileNotFoundError(str(src_dir))
        dst_dir = ensure_directory(output_root / DATASET_PATHS.shards_dir / generator)
        copied = 0
        for src_path in sorted(src_dir.glob("*.npz")):
            dst_path = dst_dir / src_path.name
            if not dst_path.exists() or force:
                shutil.copy2(src_path, dst_path)
                copied += 1
        copied_counts[generator] = copied

    meta_dir = ensure_directory(output_root / DATASET_PATHS.metadata_dir)
    poseidon_meta = base_output_root / DATASET_PATHS.metadata_dir / f"{POSEIDON_GENERATOR_KEY}_generation.json"
    if poseidon_meta.exists():
        _copy_file(poseidon_meta, meta_dir / poseidon_meta.name, force=force)
    return copied_counts


def ensure_expanded_latents(
    base_output_root: Path,
    output_root: Path,
    *,
    new_tau_count: int,
    seed: int,
    force: bool = False,
) -> pd.DataFrame:
    """Create or load the combined latent table for the expanded dataset."""

    ensure_directory(output_root)
    latents_path = output_root / DATASET_PATHS.latents_parquet
    if latents_path.exists() and not force:
        latents = read_latents_parquet(output_root)
    else:
        base_latents = read_latents_parquet(base_output_root)
        base_tau = base_latents.loc[base_latents["generator"] == TAUREX_GENERATOR_KEY].reset_index(drop=True)
        base_poseidon = base_latents.loc[base_latents["generator"] == POSEIDON_GENERATOR_KEY].reset_index(drop=True)
        start_ordinal = int(base_tau["sample_index"].max())
        new_tau_latents = build_tau_extension_latents(
            count=new_tau_count,
            master_seed=seed,
            start_ordinal=start_ordinal,
        )
        latents = pd.concat([base_tau, new_tau_latents, base_poseidon], ignore_index=True)
        latents.loc[:, "row_index"] = range(len(latents))
        validate_latent_frame(
            latents,
            expected_counts={
                TAUREX_GENERATOR_KEY: int(len(base_tau) + len(new_tau_latents)),
                POSEIDON_GENERATOR_KEY: int(len(base_poseidon)),
            },
            required_generators=(TAUREX_GENERATOR_KEY, POSEIDON_GENERATOR_KEY),
        )
        write_latents_parquet(output_root, latents)
    tau_count = int((latents["generator"] == TAUREX_GENERATOR_KEY).sum())
    poseidon_count = int((latents["generator"] == POSEIDON_GENERATOR_KEY).sum())
    validate_latent_frame(
        latents,
        expected_counts={
            TAUREX_GENERATOR_KEY: tau_count,
            POSEIDON_GENERATOR_KEY: poseidon_count,
        },
        required_generators=(TAUREX_GENERATOR_KEY, POSEIDON_GENERATOR_KEY),
    )
    return latents


def annotate_expanded_manifest(
    output_root: Path,
    *,
    base_output_root: Path,
    base_tau_count: int,
    new_tau_count: int,
    poseidon_count: int,
) -> dict[str, Any]:
    """Record how the expanded dataset relates to the canonical base release."""

    manifest_path = output_root / DATASET_PATHS.manifest_json
    manifest = json.loads(manifest_path.read_text())
    manifest.update(
        {
            "dataset_name": "crossgen_biosignatures_expanded",
            "dataset_variant": "tau_expanded_full_bundle",
            "base_dataset_root": str(base_output_root),
            "expansion_summary": {
                "base_tau_row_count": int(base_tau_count),
                "new_tau_row_count": int(new_tau_count),
                "total_tau_row_count": int(base_tau_count + new_tau_count),
                "poseidon_row_count": int(poseidon_count),
                "new_tau_sample_id_start": f"tau_{base_tau_count + 1:06d}",
                "new_tau_sample_id_end": f"tau_{base_tau_count + new_tau_count:06d}",
            },
        }
    )
    atomic_write_json(manifest_path, manifest)
    return manifest


def write_expanded_readme(
    output_root: Path,
    *,
    base_tau_count: int,
    new_tau_count: int,
    poseidon_count: int,
) -> Path:
    """Write a dataset-local README for the expanded bundle."""

    readme = f"""# Expanded Cross-Generator Dataset

This directory contains the canonical cross-generator dataset expanded with additional `TauREx` rows.

## Contents

- Existing TauREx rows retained from the original release: `{base_tau_count}`
- Newly generated TauREx rows: `{new_tau_count}`
- Total TauREx rows: `{base_tau_count + new_tau_count}`
- Existing POSEIDON test rows retained unchanged: `{poseidon_count}`

## Compatibility

- Public file contract matches the original release:
  - `labels.parquet`
  - `spectra.h5`
  - `manifest.json`
- Shared wavelength grid is unchanged: `0.6-5.2 um`, constant `R=100`, `218` bins.
- New TauREx sample identifiers continue the original sequence:
  - start: `tau_{base_tau_count + 1:06d}`
  - end: `tau_{base_tau_count + new_tau_count:06d}`

## Notes

- The original POSEIDON rows are copied forward unchanged so the test set remains compatible with models trained on the expanded TauREx training pool.
- The expanded bundle is stored as a standalone dataset directory so it can be versioned independently of the original release.
"""
    readme_path = output_root / "README.md"
    atomic_write_text(readme_path, readme)
    return readme_path


def run_tau_expanded_dataset(config: TauExpandedDatasetConfig) -> dict[str, Any]:
    """Build a standalone expanded dataset containing base + new TauREx rows."""

    copy_summary = stage_base_artifacts(config.base_output_root, config.output_root, force=config.force_copy)
    latents = ensure_expanded_latents(
        config.base_output_root,
        config.output_root,
        new_tau_count=config.new_tau_count,
        seed=config.seed,
        force=config.force_latents,
    )
    tau_count = int((latents["generator"] == TAUREX_GENERATOR_KEY).sum())
    poseidon_count = int((latents["generator"] == POSEIDON_GENERATOR_KEY).sum())
    base_tau_count = tau_count - int(config.new_tau_count)
    new_tau_latents = latents.loc[
        (latents["generator"] == TAUREX_GENERATOR_KEY) & (latents["sample_index"] > base_tau_count)
    ].reset_index(drop=True)

    results: dict[str, Any] = {
        "copied_shards": copy_summary,
        "latents": str(config.output_root / DATASET_PATHS.latents_parquet),
    }
    results[TAUREX_GENERATOR_KEY] = generate_generator_shards(
        output_root=config.output_root,
        latents=new_tau_latents,
        generator=TAUREX_GENERATOR_KEY,
        seed=config.seed,
        shard_size=config.shard_size,
        workers=config.workers,
        force=config.force_shards,
    )
    results["assembly"] = assemble_dataset(config.output_root, latents, force=config.force_assemble)
    results["manifest"] = annotate_expanded_manifest(
        config.output_root,
        base_output_root=config.base_output_root,
        base_tau_count=base_tau_count,
        new_tau_count=config.new_tau_count,
        poseidon_count=poseidon_count,
    )
    results["readme"] = str(
        write_expanded_readme(
            config.output_root,
            base_tau_count=base_tau_count,
            new_tau_count=config.new_tau_count,
            poseidon_count=poseidon_count,
        )
    )
    results["validation"] = validate_dataset(
        config.output_root,
        expected_counts={
            TAUREX_GENERATOR_KEY: tau_count,
            POSEIDON_GENERATOR_KEY: poseidon_count,
        },
        required_generators=(TAUREX_GENERATOR_KEY, POSEIDON_GENERATOR_KEY),
    )
    return results


def parse_args() -> TauExpandedDatasetConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-output-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--new-tau-count", type=int, required=True)
    parser.add_argument("--seed", type=int, default=MASTER_SEED)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 1)))
    parser.add_argument("--shard-size", type=int, default=256)
    parser.add_argument("--force-copy", action="store_true")
    parser.add_argument("--force-latents", action="store_true")
    parser.add_argument("--force-shards", action="store_true")
    parser.add_argument("--force-assemble", action="store_true")
    args = parser.parse_args()
    return TauExpandedDatasetConfig(
        base_output_root=args.base_output_root,
        output_root=args.output_root,
        new_tau_count=args.new_tau_count,
        seed=args.seed,
        workers=args.workers,
        shard_size=args.shard_size,
        force_copy=args.force_copy,
        force_latents=args.force_latents,
        force_shards=args.force_shards,
        force_assemble=args.force_assemble,
    )


def main() -> None:
    config = parse_args()
    summary = run_tau_expanded_dataset(config)
    print("Expanded cross-generator dataset summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
