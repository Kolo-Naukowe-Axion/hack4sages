# Audit Report 03: Cross-Generator Validity -- TauREx vs POSEIDON

**Auditor**: Scientific Rigor Audit
**Date**: 2026-03-12
**Model**: Hybrid Quantum-Classical (QELM) Biosignature Detector
**Scope**: Cross-generator generalization -- does a model trained on TauREx spectra generalize to POSEIDON spectra?

---

## Executive Summary

The model is trained exclusively on TauREx-generated spectra (37,281 train + 4,142 val) and tested exclusively on POSEIDON-generated spectra (685 test). On the POSEIDON test set, the model performs **worse than a naive mean predictor** on all five targets, with Pearson correlations indistinguishable from zero. The model has learned TauREx-specific artifacts rather than transferable atmospheric physics. Any generalization claim based on these results is invalid.

**Verdict: FAIL**

---

## 1. Dataset Structure and Split Design

### Finding 1.1: Perfectly Confounded Generator-Split Assignment [CRITICAL]

**Files**:
- `/Users/michalszczesny/projects/hack4sages/exobiome_model/crossgen_hybrid_training.py` lines 318-320
- `/Users/michalszczesny/projects/hack4sages/datasets/crossgen_biosignatures_20260311/manifest.json`

The split assignment is 100% determined by the generator:

| Generator | train | val | test | Total |
|-----------|-------|-----|------|-------|
| TauREx    | 37,281 | 4,142 | 0 | 41,423 |
| POSEIDON  | 0 | 0 | 685 | 685 |
| **Total** | **37,281** | **4,142** | **685** | **42,108** |

The code in `prepare_data()` reads splits directly from the parquet labels:

```python
inner_train_indices = np.where(labels["split"].values == "train")[0]
inner_val_indices = np.where(labels["split"].values == "val")[0]
test_indices = np.where(labels["split"].values == "test")[0]
```

There is no mixing, no stratification, and no cross-generator contamination. Train and val contain only TauREx samples; test contains only POSEIDON samples. This means that the validation loss tracks in-distribution (TauREx-to-TauREx) performance and provides **zero signal** about cross-generator generalization. Early stopping (patience=6) and model selection are optimized for TauREx performance only.

### Finding 1.2: Extreme Size Imbalance [MEDIUM]

POSEIDON contributes only 685 samples (1.6% of the dataset) vs 41,423 for TauREx. The test set is 60x smaller than the training set. While this is a constraint of the dataset design, it means:

- Test RMSE has high variance (685 samples give only ~137 samples per target for statistical power)
- The model has zero POSEIDON examples to learn from during training

---

## 2. Cross-Generator Performance Gap

### Finding 2.1: Model Performs Worse Than a Constant Predictor [CRITICAL]

**Files**:
- `/Users/michalszczesny/projects/hack4sages/exobiome_model/outputs/model_crossgen_rebinned/inner_val_metrics.json`
- `/Users/michalszczesny/projects/hack4sages/exobiome_model/outputs/model_crossgen_rebinned/test_metrics.json`
- `/Users/michalszczesny/projects/hack4sages/exobiome_model/outputs/model_crossgen_rebinned/run_summary.json`

A naive baseline that predicts the training-set mean for every test sample yields a lower RMSE than the trained model on **all five targets**:

| Target | Model RMSE (test) | Naive Mean RMSE | Val RMSE (TauREx) | Test/Val Ratio |
|--------|------------------:|----------------:|-------------------:|---------------:|
| H2O    | 3.131 | 2.874 | 1.662 | 1.88x |
| CO2    | **4.726** | 2.966 | 1.158 | **4.08x** |
| CO     | 3.283 | 2.846 | 2.182 | 1.50x |
| CH4    | 3.227 | 2.942 | 1.219 | 2.65x |
| NH3    | 2.948 | 2.842 | 1.503 | 1.96x |
| **Mean** | **3.463** | **2.894** | **1.545** | **2.24x** |

The model is not just failing to generalize -- it is actively **worse** than guessing the population mean. This means the TauREx-specific patterns it learned are anti-correlated with POSEIDON behavior. CO2 is the most extreme case (4.08x degradation), likely because its spectral features differ most between the two radiative transfer codes.

### Finding 2.2: Near-Zero Predictive Skill on Test Set [CRITICAL]

**File**: `/Users/michalszczesny/projects/hack4sages/exobiome_model/outputs/model_crossgen_rebinned/test_predictions.csv`

Pearson correlation between true and predicted values on the POSEIDON test set:

| Target | Pearson r | p-value |
|--------|----------:|--------:|
| H2O    | 0.0174 | 0.650 |
| CO2    | -0.0226 | 0.554 |
| CO     | 0.0367 | 0.338 |
| CH4    | 0.0191 | 0.618 |
| NH3    | -0.0030 | 0.937 |

All correlations are indistinguishable from zero. The model has **no predictive skill** on POSEIDON data. It is outputting values that are statistically independent of the true concentrations.

### Finding 2.3: Severe Prediction Collapse [HIGH]

**File**: `/Users/michalszczesny/projects/hack4sages/exobiome_model/outputs/model_crossgen_rebinned/test_predictions.csv`

The predicted standard deviation is drastically compressed compared to the true distribution:

| Target | Pred std | True std | Ratio (pred/true) |
|--------|--------:|--------:|---------:|
| H2O    | 1.179 | 2.874 | 0.41 |
| CO2    | 0.713 | 2.967 | **0.24** |
| CO     | 1.034 | 2.845 | 0.36 |
| CH4    | 0.972 | 2.943 | 0.33 |
| NH3    | 0.778 | 2.844 | 0.27 |

The model outputs a narrow band of predictions (especially for CO2, which is almost constant around -10.7). This is characteristic of a model that has memorized the "safe zone" of its training distribution and collapses to a learned TauREx-mean when confronted with out-of-distribution inputs.

### Finding 2.4: Systematic Bias in CO2 [CRITICAL]

CO2 predictions are catastrophically biased. The model predicts values almost exclusively in the [-12, -10] range:

- 598 of 685 predictions (87%) fall in [-12, -10)
- True values are uniformly distributed across [-12, -2]
- Mean prediction error: -3.60 (massive systematic underestimate)

The model has learned that CO2 concentrations in TauREx training data tend to map to certain spectral features, but POSEIDON encodes CO2 absorption differently. The model interprets POSEIDON spectra as consistently indicating very low CO2.

---

## 3. Physics and Opacity Database Differences

### Finding 3.1: Different Radiative Transfer Codes with Distinct Assumptions [HIGH]

**File**: `/Users/michalszczesny/projects/hack4sages/datasets/crossgen_biosignatures_20260311/manifest.json`

The two generators use fundamentally different software:

| Property | TauREx 3.2.4 | POSEIDON 1.3.2 |
|----------|-------------|----------------|
| RT solver | Custom C-extension, transmission geometry | MultiNest/PyMultiNest-backed, transmission geometry |
| Opacity sources | ExoMol/HITRAN cross-sections (Zenodo) | POSEIDON built-in opacity database |
| Line broadening | Voigt profiles with specific wing cutoffs | Potentially different pressure broadening treatment |
| CIA | H2-H2, H2-He from HITRAN | Independent implementation |
| Native resolution | R=100 (target) | R=1000 (downsampled to R=100) |

The manifest confirms POSEIDON generates at native R=1000 then downsamples to R=100, while TauREx generates directly at R=100. This resolution mismatch before rebinning means the two generators produce systematically different spectral shapes even for identical atmospheric compositions, because spectral binning is a lossy operation that depends on the input resolution.

### Finding 3.2: No Documentation of Opacity Database Alignment [HIGH]

There is no evidence in the codebase or manifest that TauREx and POSEIDON use the same:
- Molecular line lists (ExoMol versions can differ)
- Pressure-temperature grid interpolation
- Continuum absorption handling
- Rayleigh scattering cross-sections

These differences produce systematic offsets in transit depth that the model cannot disentangle from atmospheric composition signals.

---

## 4. Wavelength Grid Compatibility

### Finding 4.1: Shared Target Grid, Different Source Resolutions [MEDIUM]

**Files**:
- `/Users/michalszczesny/projects/hack4sages/exobiome_model/crossgen_hybrid_training.py` lines 43-53, 56-59
- `/Users/michalszczesny/projects/hack4sages/datasets/crossgen_biosignatures_20260311/meta/tau_generation.json`
- `/Users/michalszczesny/projects/hack4sages/datasets/crossgen_biosignatures_20260311/meta/poseidon_generation.json`

Both generators output spectra on the same 218-bin wavelength grid (R=100, 0.6-5.2 um) which is then rebinned to a 44-bin Ariel grid (0.95-4.91 um) via SpectRes:

```python
def rebin_spectra(old_wavelengths, spectra, new_wavelengths=ARIEL_WAVELENGTH_GRID):
    rebinned = spectres.spectres(new_wavelengths, old_wavelengths, spectra, verbose=False)
    return new_wavelengths.astype(np.float32), rebinned.astype(np.float32)
```

The wavelength grids are formally compatible. However, as noted in Finding 3.1, POSEIDON generates at R=1000 then bins to R=100, while TauREx generates directly at R=100. This means the R=100 spectra themselves encode different effective line-spread functions, creating a subtle but systematic spectral signature that the model can exploit to distinguish generators without learning atmospheric physics.

---

## 5. Domain Adaptation and Distribution Shift Handling

### Finding 5.1: Zero Domain Adaptation Techniques [HIGH]

**File**: `/Users/michalszczesny/projects/hack4sages/exobiome_model/crossgen_hybrid_training.py` (entire file)

A thorough search of the training code reveals **no domain adaptation techniques** of any kind:

- No adversarial domain adaptation (DANN, CORAL, MMD)
- No data augmentation to simulate inter-generator variation
- No test-time adaptation or normalization adjustment
- No generator-aware conditioning or multi-task learning
- No feature alignment or distribution matching losses
- The `generator` column is read from labels but never used in training

The scalers (`ArrayStandardizer`, `SpectralStandardizer`) at lines 148-188 are fit exclusively on TauREx training data:

```python
aux_scaler = ArrayStandardizer.fit(aux_raw[inner_train_indices])
target_scaler = ArrayStandardizer.fit(targets_raw[inner_train_indices])
spectral_scaler = SpectralStandardizer.fit(spectra_raw[inner_train_indices, 0, :])
```

POSEIDON test data is then normalized using TauREx statistics. While the marginal distributions of aux features and targets are similar between generators, the spectral normalization may amplify systematic differences.

### Finding 5.2: Per-Sample Spectral Normalization May Be Insufficient [MEDIUM]

**File**: `/Users/michalszczesny/projects/hack4sages/exobiome_model/crossgen_hybrid_training.py` lines 278-280

```python
per_sample_mean = noisy_spectra.mean(axis=1, keepdims=True)
per_sample_mean = np.where(per_sample_mean == 0, 1.0, per_sample_mean)
spectra = (noisy_spectra / per_sample_mean)[:, None, :].astype(np.float32)
```

This per-sample mean-division attempts to remove the absolute transit-depth scale (which depends on planet/star radius ratio). However, the generator-specific spectral shape differences -- which are the actual source of the distribution shift -- survive this normalization, since both generators produce spectra with similar mean transit depths (TauREx: 0.0547, POSEIDON: 0.0527) but potentially different wavelength-dependent patterns.

---

## 6. Evidence of Memorization vs Physics Learning

### Finding 6.1: Training Converges Well on TauREx [LOW]

**File**: `/Users/michalszczesny/projects/hack4sages/exobiome_model/outputs/model_crossgen_rebinned/history.csv`

The model trains for all 30 epochs without early stopping (best at epoch 29), achieving:
- Train loss: 0.305 (epoch 30)
- Val loss: 0.300 (epoch 29, best)
- Val RMSE mean: 1.538

The train/val gap is small (0.005 loss), indicating no severe overfitting within the TauREx distribution. However, the val RMSE of ~1.5 orders of magnitude is already mediocre for VMR retrieval (usable but not precise).

### Finding 6.2: Val Performance Improves While Cross-Generator Transfer Does Not [HIGH]

The learning rate never decayed (stayed at 2e-3 classical, 6e-4 quantum for all 30 epochs -- scheduler patience=5 was never triggered). The model kept improving on TauREx validation throughout training. Combined with the catastrophic test performance, this is a textbook signature of overfitting to generator-specific features: the model finds increasingly precise TauREx-specific correlations that are entirely meaningless on POSEIDON data.

---

## 7. Summary of Findings

| ID | Finding | Severity |
|----|---------|----------|
| 1.1 | Train/val = 100% TauREx, test = 100% POSEIDON. Generator perfectly confounds split. | CRITICAL |
| 2.1 | Model is worse than predicting the training-set mean on all 5 targets | CRITICAL |
| 2.2 | Pearson r ~ 0 on test: zero predictive skill | CRITICAL |
| 2.4 | CO2 predictions systematically biased (-3.6 mean error) | CRITICAL |
| 2.3 | Prediction std is 24-41% of true std (severe collapse) | HIGH |
| 3.1 | TauREx (R=100 native) vs POSEIDON (R=1000 downsampled) create systematic spectral differences | HIGH |
| 3.2 | No evidence of opacity database alignment between generators | HIGH |
| 5.1 | Zero domain adaptation techniques implemented | HIGH |
| 6.2 | Continued TauREx improvement with no cross-generator transfer = generator-specific memorization | HIGH |
| 1.2 | POSEIDON is only 1.6% of total data; no training signal from this generator | MEDIUM |
| 4.1 | Shared wavelength grid but different effective line-spread functions | MEDIUM |
| 5.2 | Per-sample normalization does not address spectral shape differences | MEDIUM |
| 6.1 | Small train/val gap within TauREx shows model fits well in-distribution | LOW |

---

## 8. Implications for Claims

1. **"Cross-generator generalization"**: The model demonstrably does not generalize across generators. Any claim of cross-generator validity is refuted by the data.

2. **"Learning atmospheric physics"**: If the model learned real physics (absorption features, molecular signatures), those physics would transfer between codes that solve the same radiative transfer equations. The zero correlation proves the model has learned TauREx-specific numerical artifacts instead.

3. **"Quantum advantage"**: The quantum block's contribution is irrelevant when the classical components upstream produce generator-biased representations. The quantum circuit processes TauREx-specific latent features and cannot recover from the upstream failure.

4. **"Biosignature detection"**: The model cannot detect biosignatures in spectra from an independent radiative transfer code. Any deployment on real observational data (which resembles neither generator) would produce meaningless predictions.

---

## 9. Recommended Mitigations

1. **Mixed-generator training**: Include POSEIDON samples in train/val (e.g., 80/10/10 stratified split within each generator).
2. **Domain adversarial training**: Add a generator-discriminator head with gradient reversal to force generator-invariant representations.
3. **Multi-generator augmentation**: Generate training data with at least 3 codes (TauREx, POSEIDON, petitRADTRANS) to force learning of shared physics.
4. **Opacity harmonization**: Ensure all generators use the same line lists and continuum opacities, or explicitly model the residual as a nuisance parameter.
5. **Spectral difference analysis**: Compute and visualize the per-wavelength systematic offset between generators for identical atmospheric compositions. If a constant offset exists, a simple affine correction may partially close the gap.
6. **Test on held-out TauREx samples**: Report both in-distribution and cross-generator metrics to separate model capacity from transfer failure.

---

## Verdict: FAIL

The cross-generator evaluation reveals a complete failure of transfer learning. The model trained on TauREx data performs worse than a constant predictor on POSEIDON data (mean RMSE 3.46 vs 2.89 for naive baseline), with zero predictive correlation (r < 0.04 for all targets). This is not a marginal degradation but a fundamental breakdown: the model has memorized TauREx-specific spectral artifacts rather than learning transferable atmospheric physics. No domain adaptation techniques are employed. The current results cannot support any claim of generalization to independent radiative transfer codes, let alone real observational data.
