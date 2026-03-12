# Data Explanation: `crossgen_biosignatures_20260311`

## Dataset at a glance

- Total samples: `42,108`
- Generators:
  - `tau`: `41,423` rows
  - `poseidon`: `685` rows
- Splits:
  - `train`: `37,281` rows
  - `val`: `4,142` rows
  - `test`: `685` rows
- Target biosignatures tracked throughout the dataset:
  - `H2O`
  - `CO2`
  - `CO`
  - `CH4`
  - `NH3`
- Spectral grid: `218` wavelength bins from `0.6` to about `5.25 um`
- Noise model: per-sample iid Gaussian white noise with `sigma_ppm` in about `[20.0005, 99.9999]`

## Important join/alignment note

- `labels.parquet` and `spectra.h5` use the same sample ordering.
- `latents.parquet` contains the same `42,108` unique `sample_id` values, but in a different row order.
- Use `sample_id` to join files. Do not assume row `N` in every file refers to the same sample.

## Files and what is in them

| File | Size | What it contains | Quantity |
| --- | ---: | --- | ---: |
| `README.md` | 1.0 KB | Human-readable dataset summary | 1 file |
| `manifest.json` | 3.1 KB | Dataset-wide counts, prevalence, generator summaries, wavelength/noise metadata | 1 JSON object |
| `spectra.h5` | 66 MB | Core spectral arrays and indexing fields | 42,108 samples |
| `labels.parquet` | 5.1 MB | Public label table: metadata, continuous targets, binary presence labels | 42,108 rows, 20 columns |
| `latents.parquet` | 6.0 MB | Generation latent table: everything in labels plus generation-only fields | 42,108 rows, 23 columns |
| `baseline_smoke.json` | 1.1 KB | Baseline model metrics over train/val/test contract | 1 JSON object |
| `baseline_poseidon_predictions.csv` | 74 KB | Baseline predictions for the POSEIDON test set | 685 predictions + header |
| `meta/poseidon_generation.json` | 325 B | POSEIDON generation provenance | 1 JSON object |
| `meta/tau_generation.json` | 320 B | TauREx generation provenance | 1 JSON object |

## `spectra.h5`

This is the main feature store.

| Dataset inside HDF5 | Shape | Type | Meaning |
| --- | ---: | --- | --- |
| `transit_depth_noiseless` | `(42108, 218)` | `float32` | Clean transmission spectrum per sample |
| `transit_depth_noisy` | `(42108, 218)` | `float32` | Noisy transmission spectrum per sample |
| `wavelength_um` | `(218,)` | `float64` | Shared wavelength grid |
| `sigma_ppm` | `(42108,)` | `float32` | Per-sample scalar noise level |
| `sample_id` | `(42108,)` | bytes | Sample identifier |
| `generator` | `(42108,)` | bytes | `poseidon` or `tau` |
| `split` | `(42108,)` | bytes | `train`, `val`, or `test` |

Interpretation:

- There are `218` spectral features per sample.
- The baseline metadata reports `feature_dim = 219`, which matches `218` spectral bins plus the scalar `sigma_ppm`.

## `labels.parquet`

This is the compact supervised-learning label table. It has `42,108` rows and `20` columns:

| Column group | Columns |
| --- | --- |
| IDs / partitioning | `sample_id`, `generator`, `split` |
| Planet/system metadata | `planet_radius_rjup`, `log_g_cgs`, `temperature_k`, `star_radius_rsun`, `trace_vmr_total`, `vmr_h2`, `vmr_he` |
| Continuous biosignature targets | `log10_vmr_h2o`, `log10_vmr_co2`, `log10_vmr_co`, `log10_vmr_ch4`, `log10_vmr_nh3` |
| Binary presence labels | `present_h2o`, `present_co2`, `present_co`, `present_ch4`, `present_nh3` |

### Label quantities in `labels.parquet`

| Label column | `1` count | `0` count | Positive rate |
| --- | ---: | ---: | ---: |
| `present_h2o` | `25,391` | `16,717` | `60.30%` |
| `present_co2` | `25,414` | `16,694` | `60.36%` |
| `present_co` | `25,231` | `16,877` | `59.92%` |
| `present_ch4` | `25,294` | `16,814` | `60.07%` |
| `present_nh3` | `25,111` | `16,997` | `59.63%` |

### Split and generator quantities in `labels.parquet`

| Grouping | Count |
| --- | ---: |
| `poseidon / test` | `685` |
| `tau / train` | `37,281` |
| `tau / val` | `4,142` |

### Per-generator label counts

| Label | Tau `1` | Tau `0` | POSEIDON `1` | POSEIDON `0` |
| --- | ---: | ---: | ---: | ---: |
| `present_h2o` | `24,996` | `16,427` | `395` | `290` |
| `present_co2` | `25,006` | `16,417` | `408` | `277` |
| `present_co` | `24,804` | `16,619` | `427` | `258` |
| `present_ch4` | `24,877` | `16,546` | `417` | `268` |
| `present_nh3` | `24,696` | `16,727` | `415` | `270` |

## `latents.parquet`

This is the fuller generation table. It has the same `42,108` samples, but includes generation-only columns not present in `labels.parquet`.

Extra columns relative to `labels.parquet`:

- `row_index`
- `sample_index`
- `sigma_ppm`

So `latents.parquet` is useful when you need:

- the generation row/sample indexing,
- the explicit scalar noise level without opening `spectra.h5`,
- a single tabular source for labels plus latent/generation parameters.

## `manifest.json`

This file is the dataset summary and validation manifest. It contains:

- dataset-level row counts
- per-generator counts
- per-generator split counts
- per-generator prevalence of each `present_*` label
- mean and standard deviation for each `log10_vmr_*` target
- software versions:
  - TauREx: `3.2.4`
  - POSEIDON: `1.3.2`
- wavelength grid metadata:
  - minimum: `0.6 um`
  - requested maximum: `5.2 um`
  - actual edge maximum: `5.250621771326985 um`
  - resolution: `100`
  - bin count: `218`
- noise model metadata

Use this file when you need the dataset summary without scanning the Parquet/HDF5 artifacts.

## `baseline_smoke.json`

This is a baseline model sanity-check result file.

Contents:

- `feature_dim`: `219`
- `target_columns`: 5 continuous regression targets
- `train_rows`: `37,281`
- `val_rows`: `4,142`
- `test_rows`: `685`
- validation RMSE for each `log10_vmr_*` target
- test RMSE for each `log10_vmr_*` target
- validation binary accuracy for each `present_*` target
- test binary accuracy for each `present_*` target

This file does not store raw examples; it stores aggregate baseline metrics only.

## `baseline_poseidon_predictions.csv`

This file contains the baseline model predictions for the `685` POSEIDON test samples.

- Rows: `685` data rows (`686` lines including header)
- Columns: `6`
  - `sample_id`
  - `pred_log10_vmr_h2o`
  - `pred_log10_vmr_co2`
  - `pred_log10_vmr_co`
  - `pred_log10_vmr_ch4`
  - `pred_log10_vmr_nh3`

This is prediction output only. It does not include the true labels alongside the predictions.

## `meta/poseidon_generation.json`

Generation provenance for the POSEIDON subset:

- generator: `poseidon`
- rows: `685`
- shard count: `1`
- shard size: `685`
- native resolution: `1000`
- target resolution: `100`
- wavelength target range: `0.6` to `5.2 um`

## `meta/tau_generation.json`

Generation provenance for the TauREx subset:

- generator: `tau`
- rows: `41,423`
- shard count: `162`
- shard size: `256`
- target resolution: `100`
- wavelength target range: `0.6` to `5.2 um`

## Practical usage summary

- Use `spectra.h5` for model inputs.
- Use `labels.parquet` for supervised targets and public labels.
- Use `latents.parquet` when you also need generation-only metadata like `sigma_ppm`, `row_index`, or `sample_index`.
- Use `manifest.json` for quick summary stats and provenance.
- Use `baseline_*` files only as reference model outputs/metrics, not as source training data.
