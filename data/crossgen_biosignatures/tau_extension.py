"""TauREx-only extension generation for the cross-generator biosignature schema."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .constants import DATASET_PATHS, MASTER_SEED, TAUREX_GENERATOR_KEY, TAUREX_SAMPLE_COUNT
from .dataset_io import read_latents_parquet, write_latents_parquet
from .generate_dataset import assemble_dataset, generate_generator_shards
from .latents import build_tau_extension_latents, validate_latent_frame
from .utils import atomic_write_json, atomic_write_text, ensure_directory
from .validate_dataset import validate_dataset


@dataclass(frozen=True)
class TauExtensionConfig:
    """Configuration for an append-compatible TauREx-only extension run."""

    output_root: Path
    count: int
    start_ordinal: int = TAUREX_SAMPLE_COUNT
    seed: int = MASTER_SEED
    shard_size: int = 512
    workers: int = max(1, (os.cpu_count() or 1))
    force_latents: bool = False
    force_shards: bool = False
    force_assemble: bool = False


def ensure_tau_extension_latents(
    output_root: Path,
    *,
    count: int,
    start_ordinal: int,
    seed: int,
    force: bool = False,
) -> pd.DataFrame:
    """Create or load the TauREx-only extension latent table."""

    ensure_directory(output_root)
    latents_path = output_root / DATASET_PATHS.latents_parquet
    if latents_path.exists() and not force:
        latents = read_latents_parquet(output_root)
    else:
        latents = build_tau_extension_latents(count=count, master_seed=seed, start_ordinal=start_ordinal)
        validate_latent_frame(
            latents,
            expected_counts={TAUREX_GENERATOR_KEY: count},
            required_generators=(TAUREX_GENERATOR_KEY,),
        )
        write_latents_parquet(output_root, latents)

    validate_latent_frame(
        latents,
        expected_counts={TAUREX_GENERATOR_KEY: count},
        required_generators=(TAUREX_GENERATOR_KEY,),
    )
    sample_indices = latents["sample_index"].to_numpy(dtype=np.int64)
    expected_start = int(start_ordinal) + 1
    expected_end = int(start_ordinal) + int(count)
    if int(sample_indices.min()) != expected_start or int(sample_indices.max()) != expected_end:
        raise AssertionError(
            f"Tau extension sample_index range is inconsistent: observed {sample_indices.min()}-{sample_indices.max()}, "
            f"expected {expected_start}-{expected_end}."
        )
    return latents


def annotate_tau_extension_manifest(output_root: Path, *, count: int, start_ordinal: int) -> dict[str, Any]:
    """Mark the assembled manifest as an append-compatible TauREx extension."""

    manifest_path = output_root / DATASET_PATHS.manifest_json
    manifest = json.loads(manifest_path.read_text())
    manifest.update(
        {
            "dataset_name": "crossgen_biosignatures_tau_extension",
            "dataset_variant": "tau_extension",
            "schema_compatible_with": "crossgen_biosignatures",
            "append_compatible_with": {
                "generator": TAUREX_GENERATOR_KEY,
                "base_dataset_name": "crossgen_biosignatures",
                "base_tau_row_count": int(start_ordinal),
                "new_tau_row_count": int(count),
                "sample_id_start": f"tau_{start_ordinal + 1:06d}",
                "sample_id_end": f"tau_{start_ordinal + count:06d}",
            },
        }
    )
    atomic_write_json(manifest_path, manifest)
    return manifest


def write_tau_extension_readme(output_root: Path, *, count: int, start_ordinal: int) -> Path:
    """Describe the extension bundle and its relationship to the base dataset."""

    train_count = count - int(round(count * (4_142 / 41_423)))
    val_count = count - train_count
    readme = f"""# Cross-Generator TauREx Extension

This directory contains an append-compatible `TauREx`-only extension for the canonical `crossgen_biosignatures` dataset.

## Compatibility

- Public table schema matches the original dataset exactly:
  - `labels.parquet`
  - `spectra.h5`
  - `manifest.json`
- Shared wavelength grid is unchanged: `0.6-5.2 um`, constant `R=100`, `218` bins.
- All rows are `generator == "tau"`.
- Sample identifiers continue the original TauREx ordinal range:
  - start: `tau_{start_ordinal + 1:06d}`
  - end: `tau_{start_ordinal + count:06d}`

## Counts

- New TauREx rows: `{count}`
- Train rows: `{train_count}`
- Val rows: `{val_count}`
- Existing base TauREx rows before this extension: `{start_ordinal}`

## Relationship To The Base Dataset

This bundle is meant to extend the TauREx side of the original cross-generator release without changing the POSEIDON test set.
Keep the original dataset in its own directory and treat this folder as a separately versioned TauREx expansion.
"""
    readme_path = output_root / "README.md"
    atomic_write_text(readme_path, readme)
    return readme_path


def run_tau_extension(config: TauExtensionConfig) -> dict[str, Any]:
    """Run a TauREx-only extension generation and validation pass."""

    latents = ensure_tau_extension_latents(
        config.output_root,
        count=config.count,
        start_ordinal=config.start_ordinal,
        seed=config.seed,
        force=config.force_latents,
    )
    results: dict[str, Any] = {"latents": str(config.output_root / DATASET_PATHS.latents_parquet)}
    results[TAUREX_GENERATOR_KEY] = generate_generator_shards(
        output_root=config.output_root,
        latents=latents,
        generator=TAUREX_GENERATOR_KEY,
        seed=config.seed,
        shard_size=config.shard_size,
        workers=config.workers,
        force=config.force_shards,
    )
    results["assembly"] = assemble_dataset(config.output_root, latents, force=config.force_assemble)
    results["manifest"] = annotate_tau_extension_manifest(
        config.output_root,
        count=config.count,
        start_ordinal=config.start_ordinal,
    )
    results["readme"] = str(
        write_tau_extension_readme(
            config.output_root,
            count=config.count,
            start_ordinal=config.start_ordinal,
        )
    )
    results["validation"] = validate_dataset(
        config.output_root,
        expected_counts={TAUREX_GENERATOR_KEY: config.count},
        required_generators=(TAUREX_GENERATOR_KEY,),
    )
    return results


def parse_args() -> TauExtensionConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--start-ordinal", type=int, default=TAUREX_SAMPLE_COUNT)
    parser.add_argument("--seed", type=int, default=MASTER_SEED)
    parser.add_argument("--shard-size", type=int, default=512)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 1)))
    parser.add_argument("--force-latents", action="store_true")
    parser.add_argument("--force-shards", action="store_true")
    parser.add_argument("--force-assemble", action="store_true")
    args = parser.parse_args()
    return TauExtensionConfig(
        output_root=args.output_root,
        count=args.count,
        start_ordinal=args.start_ordinal,
        seed=args.seed,
        shard_size=args.shard_size,
        workers=args.workers,
        force_latents=args.force_latents,
        force_shards=args.force_shards,
        force_assemble=args.force_assemble,
    )


def main() -> None:
    config = parse_args()
    summary = run_tau_extension(config)
    print("TauREx extension generation summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
