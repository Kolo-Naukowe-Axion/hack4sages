# Curated Datasets

This branch is the canonical home for large datasets and EDA assets that are intentionally kept off `main`.

## Index

| Name | Type | Canonical Path | Approx. Size | Notes |
| --- | --- | --- | ---:| --- |
| ARIEL ML dataset | source | `data/ariel-ml-dataset/` | 1.9 GB | Canonical ADC2023-format challenge dataset |
| pRT ADC2023 validation | validation | `data/petitradtrans-adc2023-validation/` | 330 MB | Generated validation dataset in ADC2023 format |
| ADC reference bundle | reference | `data/reference_data/adc2023_reference_bundle.npz` | 3.1 MB | Compact empirical bundle used by validation workflows |
| EDA notebook | EDA | `data/eda/` | 1.6 MB | Notebook, README, and requirements for exploratory analysis |
| Cross-generator biosignatures 20260311 | generated | `data/published/crossgen_biosignatures/20260311/` | 153 MB | Final TauREx-train / POSEIDON-test dataset |

## Notes

- Binary dataset payloads on this branch are tracked with Git LFS.
- Text metadata such as `README.md`, `manifest.json`, and this index stay in normal Git.
- Scratch outputs and temporary generation runs are intentionally excluded from this branch.
