# ExoBiome

Quantum biosignature detection in exoplanet atmospheres using QELM on real quantum hardware.

Built by **Axion** for HACK-4-SAGES 2026 (ETH Zurich COPL).

## Structure

| Branch | What |
|--------|------|
| `main` | Project root, EDA, datasets, validation-set generator |
| `feat/web` | Web frontend (two design variants) |

## Quick Start

- **EDA**: see [`eda/README.md`](eda/README.md)
- **Validation Set Method**: see [`VALIDATION_SET_METHOD.md`](VALIDATION_SET_METHOD.md)
- **Web App**: switch to `feat/web` branch for setup instructions

## Datasets

- `ariel-ml-dataset/`: canonical ADC2023-format challenge dataset, committed with Git LFS.
- `petitradtrans-adc2023-validation/`: generated pRT-based ADC2023 validation dataset, committed with Git LFS.

## Validation Set Tooling

- `prt_adc2023_validation/`: generator, physics helpers, empirical prior, and validator.
- `reference_data/adc2023_reference_bundle.npz`: compact empirical reference bundle built from the local ADC dataset.
- `scripts/build_reference_bundle.py`: rebuild the compact empirical prior bundle.
- `scripts/rebin_prt_opacities.py`: rebin official pRT correlated-k opacities to `R=400`.
- `scripts/generate_validation_set.py`: generate shard outputs and assemble the final dataset.
- `scripts/validate_validation_set.py`: validate shapes, physical consistency, noise, and checksums.
- `scripts/check_generation_status.py`: inspect or watch generation progress from `work/progress.json`.
- `scripts/run_local_generation.sh`: launch a local background generation run.

## Scientific Sources

- `petitRADTRANS==2.6.7`
- Gebhard et al. 2025: `https://arxiv.org/html/2410.21477v1`
- Vasist et al. 2023: `https://arxiv.org/abs/2301.06575`
- Official ADC2023 baseline binning: `https://github.com/ucl-exoplanets/ADC2023-baseline`
- sbi-ear input-data bundle and rebin pattern: `https://github.com/francois-rozet/sbi-ear`

## Research

- [Vetrano et al. 2025 — QELM for atmospheric retrieval](https://arxiv.org/abs/2509.03617)
- [Cardenas et al. 2025 — MultiREx dataset](https://doi.org/10.1093/mnras/stae2948)
