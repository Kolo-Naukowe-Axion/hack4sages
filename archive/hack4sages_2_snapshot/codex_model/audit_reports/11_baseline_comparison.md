# Audit Report 11: Baseline Comparison Fairness

**Auditor:** Scientific Rigor Audit Agent
**Date:** 2026-03-12
**Scope:** Fairness of quantum-hybrid model vs. classical baseline comparison
**Verdict:** FAIL

---

## Executive Summary

The comparison between the quantum-hybrid model and the classical baseline contains multiple critical fairness issues. The most damning finding is that the baseline predictions stored in `data/baseline_poseidon_predictions.csv` appear to come from a trivial mean-predictor that outputs near-constant values (~-7.0 for all gases, all samples), not from a properly trained neural network. The crossgen-baseline notebook (`crossgen-baseline.ipynb`) does run a real CNN on the same data but predicts 7 targets (including planet_radius and temperature) vs. the quantum model's 5, uses different auxiliary features, different standardization, no learning rate scheduling, no early stopping, and a smaller batch size. Multiple confounds render the comparison scientifically unsound.

---

## Finding 1: `baseline_poseidon_predictions.csv` is a Mean-Predictor, Not a Real Baseline

**Severity: CRITICAL**

The file `/Users/michalszczesny/projects/hack4sages/codex_model/data/baseline_poseidon_predictions.csv` contains predictions that are essentially constant across all 685 test samples. Every prediction for every gas clusters tightly around -7.0:

```
poseidon_000001,-7.003,-7.023,-7.063,-6.937,-6.965
poseidon_000002,-7.012,-6.943,-6.975,-7.057,-6.899
poseidon_000003,-6.954,-6.986,-6.995,-7.003,-7.034
...
```

The training set target means from the crossgen-baseline notebook are:
```
log10_vmr_h2o: -6.981, log10_vmr_co2: -6.980, log10_vmr_co: -6.989,
log10_vmr_ch4: -7.009, log10_vmr_nh3: -7.013
```

These "predictions" are nearly identical to the training set means for every single sample, with only minor noise. This is the output of a model that has learned nothing and simply predicts the mean -- or it is literally a mean-predictor with small perturbations. This file is sourced from the dataset generation pipeline (`baseline/crossgenn/`) and labeled a "baseline smoke test," meaning it was generated as a sanity-check stub, not as a competitive baseline.

The `baseline_smoke.json` from the same dataset package reports its own test RMSE values (H2O: 2.87, CO2: 2.97, CO: 2.85, CH4: 2.95, NH3: 2.84) which are consistent with a mean-predictor on uniformly distributed log10 VMR values spanning ~10 decades. If this file is used anywhere as a "baseline" comparator, the comparison is fraudulent.

**Files:**
- `/Users/michalszczesny/projects/hack4sages/codex_model/data/baseline_poseidon_predictions.csv`
- `/Users/michalszczesny/projects/hack4sages/baseline/data/crossgen/baseline_smoke.json`

---

## Finding 2: Two Completely Different "Baselines" Exist, Creating Confusion

**Severity: HIGH**

The project contains two distinct baselines that could be referenced:

| Attribute | `baseline_smoke.json` (mean-predictor) | `crossgen-baseline.ipynb` (CNN) |
|---|---|---|
| Source | Dataset generation pipeline | Dedicated notebook |
| Architecture | Mean/trivial predictor | MC Dropout CNN (834,303 params) |
| Test RMSE H2O | 2.87 | 3.13 |
| Test RMSE CO2 | 2.97 | 3.69 |
| Test RMSE CO | 2.85 | 3.26 |
| Test RMSE CH4 | 2.95 | 3.35 |
| Test RMSE NH3 | 2.84 | 3.17 |

Paradoxically, the "mean-predictor" baseline_smoke values are actually better than the trained CNN on most gases. This is because the CNN in `crossgen-baseline.ipynb` suffers from cross-generator generalization failure (trained on TauREx, tested on POSEIDON) and its predictions are driven further from the mean in wrong directions. This makes interpretation of any "improvement" over either baseline extremely problematic.

The quantum model test RMSE values are:
```
H2O: 3.13, CO2: 4.73, CO: 3.28, CH4: 3.23, NH3: 2.95
```

The quantum model is *worse* than the mean-predictor baseline on 4 of 5 gases and worse than the CNN baseline on CO2 (4.73 vs 3.69). It only beats the CNN on CH4 (3.23 vs 3.35) and NH3 (2.95 vs 3.17). There is no clear advantage.

**Files:**
- `/Users/michalszczesny/projects/hack4sages/codex_model/outputs/model_crossgen_rebinned/metrics_summary.csv`
- `/Users/michalszczesny/projects/hack4sages/baseline/data/crossgen/baseline_smoke.json`

---

## Finding 3: Spectral Resolution Mismatch Between Quantum Model and Baseline

**Severity: HIGH**

The quantum model rebins 218-channel spectra down to 44 bins on the Ariel wavelength grid before processing:

```python
# crossgen_hybrid_training.py, line 56-59
def rebin_spectra(old_wavelengths, spectra, new_wavelengths=ARIEL_WAVELENGTH_GRID):
    rebinned = spectres.spectres(new_wavelengths, old_wavelengths, spectra, verbose=False)
    return new_wavelengths.astype(np.float32), rebinned.astype(np.float32)
```

The ARIEL_WAVELENGTH_GRID has 44 bins (lines 43-53).

The crossgen-baseline notebook uses the full 218 channels:
```python
# crossgen-baseline.ipynb, cell "split"
wl_channels = len(wavelength)  # 218
```

This means the baseline CNN sees ~5x more spectral information than the quantum model. If the quantum model performs comparably despite this handicap, that is actually noteworthy -- but this difference is never disclosed or discussed. Without disclosure, a reviewer cannot assess whether performance differences come from architecture or from input data differences.

**Files:**
- `/Users/michalszczesny/projects/hack4sages/codex_model/crossgen_hybrid_training.py` (lines 43-59, 264-266)
- `/Users/michalszczesny/projects/hack4sages/baseline/crossgen-baseline.ipynb` (cell "split")

---

## Finding 4: Different Target Sets Between Baseline and Quantum Model

**Severity: HIGH**

The crossgen-baseline predicts 7 targets:
```python
# crossgen-baseline.ipynb, cell "split"
target_labels = ['log10_vmr_h2o', 'log10_vmr_co2', 'log10_vmr_co', 'log10_vmr_ch4', 'log10_vmr_nh3']
aux_labels = ['planet_radius_rjup', 'temperature_k']
all_target_labels = aux_labels + target_labels  # 7 total
```

The quantum model predicts 5 targets (only gases):
```python
# crossgen_hybrid_training.py, lines 35-41
TARGET_COLS = [
    "log10_vmr_h2o", "log10_vmr_co2", "log10_vmr_co",
    "log10_vmr_ch4", "log10_vmr_nh3",
]
```

Predicting additional targets (planet_radius, temperature) alongside gases forces the baseline's shared CNN layers to learn a broader representation. This multi-task burden can hurt gas prediction performance. The baseline is effectively handicapped by predicting 2 additional physical quantities that the quantum model does not attempt. A fair comparison requires both models to predict the same target set.

---

## Finding 5: Different Auxiliary Feature Sets

**Severity: MEDIUM**

The quantum model uses 5 auxiliary features:
```python
# crossgen_hybrid_training.py, lines 27-33
SAFE_AUX_FEATURE_COLS = [
    "planet_radius_rjup", "log_g_cgs", "temperature_k",
    "star_radius_rsun", "log10_sigma_ppm",
]
```

The crossgen-baseline uses only 1 auxiliary feature:
```python
# crossgen-baseline.ipynb, cell "split"
# Only star_radius_rsun is used as auxiliary input
train_Rs = Rs[train_mask]  # shape (N, 1)
```

The quantum model has access to planet radius, surface gravity, temperature, star radius, AND noise level as input features. The baseline only has star radius. The quantum model's additional features (especially planet temperature and log_g) are highly informative for constraining atmospheric retrievals. This is a major information advantage that makes any performance comparison unfair in the quantum model's favor.

---

## Finding 6: Different Preprocessing Pipelines

**Severity: MEDIUM**

| Aspect | Quantum Model | Crossgen Baseline |
|---|---|---|
| Spectral normalization | Per-sample mean-division, then per-channel z-score | Global scalar mean/std z-score |
| Auxiliary scaling | Per-feature z-score (fit on train) | Per-feature z-score (fit on train) |
| Target scaling | Per-target z-score (fit on train) | Per-target z-score (fit on train+val+test*) |
| Spectra shape | (N, 1, 44) 1D convolution | (N, 218) reshaped to (N, 218, 1) |

*Note: The baseline standardizes val targets using training statistics, which is correct for the val inverse transform. However, the baseline's spectra standardization uses `global_mean = np.mean(train_spectra)` -- a single scalar mean across all training spectra -- while the quantum model first divides each spectrum by its own per-sample mean and then applies per-wavelength-bin z-scoring. The quantum model's preprocessing is substantially more sophisticated and provides better normalization across the wide dynamic range of transit depths.

**Files:**
- `/Users/michalszczesny/projects/hack4sages/codex_model/crossgen_hybrid_training.py` (lines 273-281, 325-327)
- `/Users/michalszczesny/projects/hack4sages/baseline/crossgen-baseline.ipynb` (cell "standardise")

---

## Finding 7: No Early Stopping or Learning Rate Scheduling in Baseline

**Severity: MEDIUM**

The quantum model uses:
- Early stopping with patience=6 (`crossgen_hybrid_training.py`, line 99)
- ReduceLROnPlateau with patience=5, factor=0.5 (lines 698-703)
- Gradient clipping at 5.0 for classical params, 1.0 for quantum params (lines 749-750)
- Weight decay 1e-4 (line 104)

The baseline uses:
- Fixed 30 epochs, no early stopping
- Fixed learning rate (1e-3), no scheduling
- No gradient clipping
- No weight decay

The baseline's val_loss was still decreasing at epoch 30 (0.2397), suggesting it was undertrained. The lowest val_loss was 0.2317 at epoch 22, meaning the final model used was not the best checkpoint. A properly tuned baseline with early stopping, learning rate scheduling, and best-checkpoint selection would likely perform better. This under-tuning biases the comparison in the quantum model's favor.

**Files:**
- `/Users/michalszczesny/projects/hack4sages/codex_model/crossgen_hybrid_training.py` (lines 98-100, 692-703)
- `/Users/michalszczesny/projects/hack4sages/baseline/crossgen-baseline.ipynb` (cell "model", cell "train")

---

## Finding 8: No Data Augmentation in Quantum Model vs. 5x Augmentation in Baseline

**Severity: MEDIUM**

The crossgen-baseline augments training data 5x with Gaussian noise:
```python
# crossgen-baseline.ipynb, cell "augment"
repeat = 5
noise_profile = np.random.normal(0, train_noise, size=(repeat, train_spectra.shape[0], wl_channels))
aug_spectra = (train_spectra[None, ...] + noise_profile).reshape(-1, wl_channels)
# Result: 186,405 training samples
```

The quantum model uses no data augmentation; it trains on 37,281 samples directly. Despite this, the quantum model's training pipeline includes more advanced regularization (dropout, weight decay, gradient clipping, learning rate scheduling) which may compensate. However, the difference in effective training set size (186k vs 37k) makes the comparison difficult to interpret: the baseline sees more examples per epoch but potentially lower-quality ones.

---

## Finding 9: Parameter Count Comparison Is Misleading

**Severity: MEDIUM**

The baseline CNN has 834,303 trainable parameters.

The quantum model parameter count can be estimated:
- AuxEncoder: 5*64 + 64 + 64*64 + 64 + 64*32 + 32 = 6,688
- SpectralEncoder: 1*32*7 + 32 + 32*64*5 + 64 + 64*64*3 + 64 + 64*32 + 32 = ~16,000
- FusionEncoder: (32+32)*48 + 48 + 48*12 + 12 = ~3,700
- QuantumBlock: 3 * 12 * 1 = 36 quantum parameters (for depth=2, which gives num_blocks=1)
- AtmosphereHead: (12+12+32+32)*96 + 96 + 96*96 + 96 + 96*5 + 5 + 12*5 + 5 = ~18,000

Rough total: ~44,000 classical + 36 quantum parameters = ~44,000 total.

The quantum model has approximately **19x fewer parameters** than the baseline. While this could be presented as an efficiency advantage, it also means the quantum model has far less capacity. A fair comparison would include a classical model of similar size to the quantum model, without the quantum circuit, to isolate the quantum contribution.

---

## Finding 10: MC Dropout Uncertainty vs. No Uncertainty -- Apples-to-Oranges

**Severity: MEDIUM**

The crossgen-baseline uses MC Dropout (100 forward passes) to produce uncertainty estimates:
```python
# crossgen-baseline.ipynb, cell "mc-test"
N_mc_test = 100
for i in tqdm(range(N_mc_test)):
    y_test_dist[i] = model([std_test_spectra, std_test_Rs], training=True)
y_test_mean = y_test_pred.mean(axis=0)  # ensemble mean used for RMSE
```

The quantum model produces point estimates only. Reporting RMSE from the MC Dropout mean (an ensemble of 100 stochastic forward passes) vs. a single-pass point estimate from the quantum model is not comparing the same thing. The MC Dropout ensemble mean generally performs better than any single pass, giving the baseline an advantage on this axis. However, this also means the baseline is doing 100x more compute at inference time, which is never accounted for.

---

## Finding 11: The ADC2023-baseline.ipynb Uses Completely Different Data

**Severity: HIGH**

The ADC2023 baseline notebook (`/Users/michalszczesny/projects/hack4sages/baseline/ADC2023-baseline.ipynb`) operates on a completely different dataset:
- 52 wavelength channels (vs. 218/44)
- 41,423 total samples with different HDF5 format (per-planet entries)
- Only first 5,000 samples used for training (line: `N = 5000`)
- Random 80/20 train/val split (not pre-defined)
- Different test set (ADC2023 leaderboard test, not POSEIDON cross-generator)

If any results from this notebook are compared with the quantum model, the comparison is scientifically invalid -- the datasets, splits, and evaluation protocols are entirely different.

**Files:**
- `/Users/michalszczesny/projects/hack4sages/baseline/ADC2023-baseline.ipynb` (cells "9946dccd", "41e90f7e")
- `/Users/michalszczesny/projects/hack4sages/baseline/run_baseline.py` (line 32)

---

## Finding 12: No Statistical Significance Testing

**Severity: MEDIUM**

Neither the baseline nor the quantum model reports confidence intervals, standard errors, or any statistical significance tests on the RMSE metrics. With only 685 test samples, the standard error of RMSE could be substantial. Without repeated runs or bootstrap confidence intervals, it is impossible to determine whether observed differences are real or within noise.

---

## Finding 13: No Training Compute Control

**Severity: LOW**

The baseline trains for 30 epochs on 186,405 augmented samples at batch_size=32, taking approximately 30 * 42s = ~21 minutes. The quantum model trains for up to 30 epochs (stopped at epoch 29) on 37,281 samples at batch_size=256, taking approximately 29 * 200s = ~97 minutes. The quantum model uses approximately 5x more wall-clock time despite training on 5x fewer samples, due to the quantum circuit simulation overhead. Training compute is not controlled or reported.

---

## Summary Table

| Finding | Severity | Impact Direction |
|---|---|---|
| 1. baseline_poseidon_predictions.csv is a mean-predictor | CRITICAL | Inflates quantum model advantage if used |
| 2. Two conflicting baselines create confusion | HIGH | Enables cherry-picking |
| 3. Spectral resolution mismatch (44 vs 218 bins) | HIGH | Disadvantages quantum model |
| 4. Different target sets (5 vs 7) | HIGH | Disadvantages baseline |
| 5. Different auxiliary features (5 vs 1) | MEDIUM | Advantages quantum model |
| 6. Different preprocessing pipelines | MEDIUM | Advantages quantum model |
| 7. No early stopping/LR scheduling in baseline | MEDIUM | Disadvantages baseline |
| 8. Augmentation asymmetry (0x vs 5x) | MEDIUM | Mixed |
| 9. Parameter count never compared or controlled | MEDIUM | Complicates interpretation |
| 10. MC Dropout ensemble vs. point estimate | MEDIUM | Mixed |
| 11. ADC2023 notebook is completely different data | HIGH | Invalid if compared |
| 12. No statistical significance testing | MEDIUM | All claims unsupported |
| 13. No training compute control | LOW | Uncontrolled confound |

---

## Required Remediation

1. **Create a truly matched classical baseline**: Same 44-bin spectral input, same 5 auxiliary features, same 5 target gases, same preprocessing pipeline, same optimizer/scheduler/early-stopping configuration. The only difference should be whether the quantum circuit is present.
2. **Add ablation study**: Train the quantum model architecture with the quantum block replaced by a classical MLP of identical input/output dimensions. This isolates the quantum contribution.
3. **Remove or clearly label `baseline_poseidon_predictions.csv`**: This is a mean-predictor stub from the dataset generation pipeline, not a trained model. Any comparison against it is misleading.
4. **Report confidence intervals**: Use bootstrap resampling over the 685 test samples to report 95% CIs on all RMSE values.
5. **Report parameter counts and training FLOPs** for both models side-by-side.
6. **Do not claim quantum advantage** unless a matched classical model of identical capacity, trained with identical protocol, is significantly outperformed.

---

## Verdict: FAIL

The baseline comparison is not scientifically valid in its current form. Multiple uncontrolled confounds (different spectral resolution, different features, different targets, different preprocessing, different training protocols) prevent any meaningful conclusion about the quantum model's value. The presence of a mean-predictor file labeled as "baseline predictions" in the quantum model's data directory is especially concerning. No claims of improvement or quantum advantage can be supported by the current experimental setup.
