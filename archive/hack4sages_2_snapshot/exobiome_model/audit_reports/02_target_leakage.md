# Audit Report 02: Target Leakage -- Features Containing Target Information

**Auditor**: legacy_model Opus 4.6 (automated scientific rigor audit)
**Date**: 2026-03-12
**Scope**: All input features used by the hybrid quantum-classical model, their relationship to target variables, and whether any feature encodes or proxies target information.

**Files reviewed**:
- `/Users/michalszczesny/projects/hack4sages/exobiome_model/crossgen_hybrid_training.py` (987 lines, every line read)
- `/Users/michalszczesny/projects/hack4sages/exobiome_model/data/labels.parquet` (42,108 rows, 20 columns, schema + statistics)
- `/Users/michalszczesny/projects/hack4sages/exobiome_model/data/spectra.h5` (7 datasets, schema + statistics)
- `/Users/michalszczesny/projects/hack4sages/exobiome_model/data/manifest.json` (dataset metadata)
- `/Users/michalszczesny/projects/hack4sages/exobiome_model/data/meta/tau_generation.json`
- `/Users/michalszczesny/projects/hack4sages/exobiome_model/data/meta/poseidon_generation.json`
- `/Users/michalszczesny/projects/hack4sages/exobiome_model/data/latents.parquet` (42,108 rows, 23 columns, schema inspected)
- `/Users/michalszczesny/projects/hack4sages/exobiome_model/data/baseline_smoke.json`
- `/Users/michalszczesny/projects/hack4sages/exobiome_model/outputs/model_crossgen_rebinned/config.json`
- `/Users/michalszczesny/projects/hack4sages/baseline/crossgen-baseline.ipynb` (full read)
- `/Users/michalszczesny/projects/hack4sages/data/dataset_strategy.ipynb` (full read)
- `/Users/michalszczesny/projects/hack4sages/eda/dataset_strategy.ipynb` (full read)
- Raw data shards (`data/shards/tau/tau_000001_000256.npz`, schema inspected)

---

## Executive Summary

The pipeline has **no direct target leakage** in the classical sense (no feature is mathematically derived from or correlated with the targets). All auxiliary features and targets are independently sampled in the forward model, confirmed by near-zero Pearson correlations (|r| < 0.01) and near-zero mutual information (MI < 0.005 nats).

However, there is a **significant methodological concern** (rated MEDIUM): two auxiliary features used as model inputs (`planet_radius_rjup` and `temperature_k`) are treated as **retrieval targets** in the ADC2023 benchmark framework that this dataset originates from. The baseline model in this very project predicts them as outputs. Using them as known inputs to the quantum model assumes oracle access to quantities that would themselves need to be estimated from the spectrum in a real retrieval pipeline. This does not constitute target leakage in the statistical sense, but it is a **deployment leakage** issue that inflates apparent model capability relative to a realistic inference scenario.

One additional positive finding: `latents.parquet` exists in the data directory but is never loaded or referenced by the training code.

---

## Complete Feature Inventory

### Auxiliary Features (5 scalar inputs)

Defined at `crossgen_hybrid_training.py`, lines 27-33:
```python
SAFE_AUX_FEATURE_COLS = [
    "planet_radius_rjup",
    "log_g_cgs",
    "temperature_k",
    "star_radius_rsun",
    "log10_sigma_ppm",
]
```

### Spectral Features (44 wavelength bins)

Transit depth spectra rebinned from 218 to 44 bins via `spectres` (`crossgen_hybrid_training.py`, lines 56-59, 264), then per-sample mean-normalized (lines 278-280) and globally standardized (line 331).

### Target Variables (5 outputs)

Defined at `crossgen_hybrid_training.py`, lines 35-41:
```python
TARGET_COLS = [
    "log10_vmr_h2o",
    "log10_vmr_co2",
    "log10_vmr_co",
    "log10_vmr_ch4",
    "log10_vmr_nh3",
]
```

### Columns in `labels.parquet` NOT used as features

The following columns exist in `labels.parquet` but are correctly excluded from model input: `sample_id`, `generator`, `split`, `trace_vmr_total`, `vmr_h2`, `vmr_he`, `present_h2o`, `present_co2`, `present_co`, `present_ch4`, `present_nh3`.

---

## Finding 1: `planet_radius_rjup` and `temperature_k` -- Retrieval Targets Used as Inputs

### Severity: MEDIUM

**The issue**: In the ADC2023 framework (from which this dataset originates), the forward model task is to **recover all simulation parameters from the spectrum**. The official target parameter set is 7 values: `planet_radius`, `planet_temp`, and 5 gas VMRs. This is confirmed in the dataset strategy notebook (`data/dataset_strategy.ipynb`, cell 6):

```python
ALL_TARGET_COLS = ['planet_radius', 'planet_temp'] + GAS_COLS
```

The baseline model in this project (`baseline/crossgen-baseline.ipynb`) correctly treats all 7 as targets, predicting `planet_radius_rjup` and `temperature_k` from the spectrum alongside the VMRs:

```
Test RMSE (POSEIDON, cross-generator):
  planet_radius_rjup: 0.0997
  temperature_k: 515.7788
```

The quantum model, however, uses `planet_radius_rjup` and `temperature_k` as **known auxiliary inputs** and only predicts the 5 gas VMRs.

**Why this matters**: In a real observational scenario:
1. **Planet radius** is derived from the transit depth (the spectrum itself). The transit depth ~ (Rp/Rs)^2, so knowing Rp provides direct information about the baseline spectrum shape. While not correlated with VMRs in this independently-sampled synthetic dataset, in a real deployment you would not know Rp a priori -- you would estimate it from the same spectrum the model uses.
2. **Temperature** in this dataset is the isothermal atmospheric temperature (confirmed in `dataset_strategy.ipynb`: "Temperature: `planet_temp` (K, isothermal)"). In a real exoplanet, the atmospheric temperature profile is unknown and must be retrieved. In chemical equilibrium scenarios, temperature directly determines molecular abundances, creating a physical link to the targets.

**Why it is NOT classical target leakage**: In this specific synthetic dataset, `planet_radius_rjup`, `temperature_k`, and all 5 VMRs are sampled independently from uniform distributions. Empirical verification confirms this:

| Feature pair | Pearson r | Mutual Information |
|---|---|---|
| planet_radius_rjup vs all VMRs | |r| < 0.005 | MI < 0.005 |
| temperature_k vs all VMRs | |r| < 0.004 | MI < 0.005 |

These values are indistinguishable from sampling noise at N=42,108.

**Why it IS a methodological concern**: The model's architecture assumes access to information that would be unavailable at deployment time. This means:
- The model cannot be directly deployed on real JWST observations without a separate radius/temperature estimation step.
- Fair comparison with the baseline (which predicts radius and temperature) requires accounting for the additional information provided to the quantum model.
- A reviewer would rightly ask: "If you already know Rp and T, how much of the model's performance comes from the spectrum versus from these auxiliary features?"

**File**: `crossgen_hybrid_training.py`, lines 27-33 (feature definition), lines 277 (feature extraction), lines 325 (scaler fitting on features)

**Recommendation**: Run an ablation experiment removing `planet_radius_rjup` and `temperature_k` from `SAFE_AUX_FEATURE_COLS` to measure the quantum model's performance using only `log_g_cgs`, `star_radius_rsun`, `log10_sigma_ppm`, and the spectrum. This quantifies the model's reliance on these oracle features.

---

## Finding 2: `temperature_k` Is Isothermal Atmospheric Temperature

### Severity: LOW (clarification, not leakage)

The `temperature_k` column represents the isothermal atmospheric temperature used as a **forward model input parameter** in both TauREx and POSEIDON. This is confirmed by the dataset_strategy notebook:

> "Temperature: `planet_temp` (K, isothermal)"

This is NOT an equilibrium temperature derived from stellar/orbital parameters (which would be independently measurable). It is the actual atmospheric temperature profile parameter that controls the pressure-temperature structure in the forward model.

**In this dataset**: Because the forward models use **free retrieval** (independently sampled VMRs, not chemical equilibrium), temperature and VMRs are statistically independent. The temperature affects the spectrum shape through pressure broadening and scale height, but does not determine the gas abundances.

**In reality**: For planets in chemical equilibrium, atmospheric temperature directly determines molecular abundances (e.g., CH4/CO ratio is a strong function of temperature via thermochemistry). The model could potentially learn to exploit temperature as a proxy for composition ratios if trained on equilibrium-chemistry data.

**Evidence of independence in this dataset**:
- `temperature_k` distribution: uniform over [500, 1800] K (verified by histogram).
- Correlation with all targets: |r| < 0.004.
- All inter-feature correlations: |r| < 0.009.

**Severity**: LOW. No leakage in this dataset. But the paper should explicitly state that `temperature_k` is the isothermal forward-model temperature, not a derived quantity.

---

## Finding 3: `planet_radius_rjup` Is the Forward-Model Input Radius

### Severity: LOW (clarification, not leakage)

The `planet_radius_rjup` column is the planet radius parameter passed to TauREx/POSEIDON as an input to the forward model. It is NOT a fitted radius from spectral retrieval. Evidence:

1. Distribution is uniform over [0.70, 1.50] Rjup (verified by histogram), consistent with independent parameter sampling.
2. Zero correlation with all target VMRs (|r| < 0.005).
3. Identical distribution between TauREx (train/val) and Poseidon (test) generators.

**Physical note**: The transit depth encodes Rp/Rs, so providing Rp as an auxiliary input alongside the spectrum is physically redundant. The spectral encoder already has access to transit depth values that encode Rp. In practice, the model receives Rp both through the aux encoder and implicitly through the spectral encoder, but since Rp is independent of VMRs in this dataset, this redundancy does not create leakage.

**Severity**: LOW. No leakage. Document the provenance of this parameter in the paper.

---

## Finding 4: `log10_sigma_ppm` Does Not Encode Atmospheric Composition

### Severity: NONE (clean)

The `log10_sigma_ppm` feature is the log10 of the per-sample noise level in parts per million. It is derived at runtime from `sigma_ppm` stored in `spectra.h5` (`crossgen_hybrid_training.py`, line 268):

```python
labels["log10_sigma_ppm"] = np.log10(np.clip(sigma_ppm, 1e-10, None))
```

**Verification**:
- `sigma_ppm` is a scalar per sample (shape `(42108,)`), uniformly distributed over [20, 100] ppm.
- Correlation with all targets: |r| < 0.01 (maximum is 0.0094 for NH3).
- The noise model is documented in `manifest.json` as `"type": "iid_gaussian_white"` with `"per_sample_scalar": true`, confirming it is drawn independently of atmospheric parameters.
- The noise is applied to spectra as: `transit_depth_noisy = transit_depth_noiseless + N(0, sigma_ppm * 1e-6)`.

`sigma_ppm` is an observation characteristic (instrument noise level), not an atmospheric property. It is physically appropriate as a model input -- it tells the model how much to trust the spectrum.

**Severity**: NONE.

---

## Finding 5: `latents.parquet` Is Not Used

### Severity: NONE (clean)

A file named `latents.parquet` exists in the data directory (42,108 rows, 23 columns). Its schema contains the same columns as `labels.parquet` plus `row_index` and `sample_index`. Despite its name suggesting encoder bottleneck representations, it appears to be a metadata/bookkeeping file from the dataset generation process.

**Verification that it is never loaded**:
1. `crossgen_hybrid_training.py` loads only `labels.parquet` and `spectra.h5` (lines 257-262).
2. The string "latents" does not appear anywhere in any `.py` file in the exobiome_model directory.
3. The design document (`docs/plans/2026-03-11-prepare-training-design.md`, line 56) explicitly states: "No use of latents.parquet or shards/"

**Severity**: NONE. The file is inert. Consider removing it from the data directory to avoid confusion.

---

## Finding 6: `log_g_cgs` and `star_radius_rsun` Are Physically Appropriate Inputs

### Severity: NONE (clean)

**`log_g_cgs`** (surface gravity): This is a fundamental planetary property that determines atmospheric scale height. It is independently measurable from radial velocity observations and transit timing, making it a legitimate known input at inference time. Distribution: uniform over [2.80, 3.70], zero correlation with targets.

**`star_radius_rsun`** (stellar radius): This is a property of the host star, independently measurable from photometry/spectroscopy. It scales the transit depth (Rp/Rs)^2 and is a legitimate known input at inference time. Distribution: uniform over [0.20, 1.30], zero correlation with targets.

Both features are used only in the `AuxEncoder` module and processed through the standard pipeline (standardization on training data, then neural network encoding).

**Severity**: NONE.

---

## Finding 7: Spectral Data Has No Embedded Target Information

### Severity: NONE (clean)

The spectral input undergoes the following processing chain:

1. **Load**: `transit_depth_noisy` from `spectra.h5` (line 261) -- NOT `transit_depth_noiseless`.
2. **Rebin**: 218 bins to 44 bins via `spectres` (line 264) -- purely wavelength-axis operation.
3. **Per-sample normalize**: Divide by per-sample mean transit depth (lines 278-280) -- no cross-sample information.
4. **Global standardize**: Subtract training-set channel means, divide by training-set channel stds (lines 327, 331) -- standard z-scoring.

None of these operations inject target information into the spectral features.

The `transit_depth_noiseless` array also exists in `spectra.h5` but is never loaded. If it were used instead of the noisy version, it would provide a cleaner signal but would still not constitute target leakage (the noiseless spectrum encodes atmospheric physics, which is what the model is supposed to learn from).

**Severity**: NONE.

---

## Finding 8: No Feature Engineering Uses Target Values

### Severity: NONE (clean)

Full audit of all feature construction:

| Feature | Construction | Uses targets? |
|---|---|---|
| `planet_radius_rjup` | Direct from `labels.parquet` | No |
| `log_g_cgs` | Direct from `labels.parquet` | No |
| `temperature_k` | Direct from `labels.parquet` | No |
| `star_radius_rsun` | Direct from `labels.parquet` | No |
| `log10_sigma_ppm` | `log10(sigma_ppm)` from `spectra.h5` | No |
| Spectral features | `transit_depth_noisy` from `spectra.h5`, rebinned, normalized | No |

No target values are ever used in feature construction. The targets are loaded separately (line 316):
```python
targets_raw = labels[TARGET_COLS].to_numpy(dtype=np.float32, copy=True)
```

And are only used for: (a) computing target scaler statistics on training data, (b) computing loss during training, (c) computing evaluation metrics.

**Severity**: NONE.

---

## Finding 9: Columns Not Used That Could Have Been Leaky

### Severity: NONE (correct exclusion)

The `labels.parquet` file contains several columns that would constitute target leakage if used as features:

| Column | Why it would leak | Used? |
|---|---|---|
| `trace_vmr_total` | Sum of all trace gas VMRs -- directly derived from targets | NOT used |
| `vmr_h2` | H2 fill gas fraction = 0.85 - trace_vmr_total -- inversely related to targets | NOT used |
| `vmr_he` | He fill gas fraction = 0.15 - trace_vmr_total * He_ratio -- inversely related | NOT used |
| `present_h2o` ... `present_nh3` | Binary indicator of whether gas is "present" -- directly encodes target | NOT used |

Verified via grep: none of these column names appear in the training code (only in `TARGET_COLS` definitions for the VMR targets).

The `trace_vmr_total` column has Pearson correlations of r=0.22 with all 5 targets. If it had been included as an auxiliary feature, it would have provided meaningful signal about gas abundances. Its exclusion is correct.

**Severity**: NONE (correctly excluded).

---

## Summary Table

| # | Finding | Leakage? | Severity |
|---|---|---|---|
| 1 | `planet_radius_rjup` and `temperature_k` are retrieval targets in ADC2023, used as inputs here | No statistical leakage; deployment concern | **MEDIUM** |
| 2 | `temperature_k` is isothermal atmospheric temperature, not equilibrium T | No leakage (clarification needed) | LOW |
| 3 | `planet_radius_rjup` is forward-model input, not fitted radius | No leakage (clarification needed) | LOW |
| 4 | `log10_sigma_ppm` is independent of composition | Clean | NONE |
| 5 | `latents.parquet` exists but is never loaded | Clean | NONE |
| 6 | `log_g_cgs` and `star_radius_rsun` are legitimate inputs | Clean | NONE |
| 7 | Spectral data processing is target-free | Clean | NONE |
| 8 | No feature engineering uses target values | Clean | NONE |
| 9 | Potentially leaky columns correctly excluded | Clean | NONE |

---

## Verdict: PASS (conditional)

The pipeline contains **no statistical target leakage**. All input features are verified to be statistically independent of the target variables in this dataset (|r| < 0.01, MI < 0.005). No feature engineering step uses target values. Potentially dangerous columns (`trace_vmr_total`, `vmr_h2`, `vmr_he`, `present_*`) are correctly excluded. The `latents.parquet` file is never loaded.

The **conditional** qualification is due to Finding 1: the model assumes oracle knowledge of `planet_radius_rjup` and `temperature_k`, which are themselves retrieval targets in the ADC2023 framework and would need to be estimated from the spectrum in a real deployment. This does not invalidate the model's results but must be:

1. **Documented** clearly in any paper or presentation.
2. **Ablated** to measure performance without these features.
3. **Addressed** in the deployment pipeline (e.g., two-stage retrieval: first estimate Rp and T, then predict VMRs).
