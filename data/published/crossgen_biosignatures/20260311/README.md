# Cross-Generator Biosignature Dataset

Published artifact set for the completed `crossgen_biosignatures` run generated on `2026-03-11`.

## Summary

- TauREx rows: `41,423`
- POSEIDON rows: `685`
- Total rows: `42,108`
- Feature shape: `(42108, 218)`
- Target biosignatures: `H2O`, `CO2`, `CO`, `CH4`, `NH3`

## Canonical Files

- `spectra.h5`: assembled feature tensors
- `labels.parquet`: assembled public label table
- `manifest.json`: dataset summary and validation metadata
- `latents.parquet`: latent parameter table used for generation
- `baseline_smoke.json`: baseline train/val/test metrics
- `baseline_poseidon_predictions.csv`: baseline predictions on the POSEIDON test split
- `meta/`: per-generator generation metadata

## Split Contract

- TauREx train: `37,281`
- TauREx val: `4,142`
- POSEIDON test: `685`

## Notes

- This published copy intentionally excludes shard files and scratch generation outputs.
- The implementation that generated this dataset lives on `main` under `data/crossgen_biosignatures/` and `data/scripts/`.
