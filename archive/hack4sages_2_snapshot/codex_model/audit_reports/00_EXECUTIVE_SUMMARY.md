# Executive Summary — Scientific Rigor Audit

**Model**: Hybrid Quantum-Classical Neural Network for Exoplanet Biosignature Detection
**Date**: 2026-03-12
**Auditors**: 16 parallel sub-agents, comprehensive code + output review

---

## Overall Verdict: FAIL — Not Paper-Ready

**7 of 16 audits FAILED, 5 CONDITIONAL PASS, 4 PASS.**
The model has correct data handling and preprocessing, but suffers from a fundamental generalization failure that renders its scientific claims unsupported. Two code bugs compound the problem.

---

## CRITICAL Findings (Paper-Invalidating)

### C1. Model Performs Worse Than a Mean Predictor (Reports: 3, 4, 10, 14, 16)
All 5 targets have **negative R-squared** on the POSEIDON test set. A trivial model outputting the training-set mean for every sample outperforms this model on every target. CO2 is worst: 4.73 RMSE vs 2.97 for the mean baseline. Pearson correlations are all |r| < 0.04 (p > 0.33). **The model's predictions are statistically indistinguishable from noise.**

### C2. Complete Cross-Generator Failure (Reports: 3, 10, 16)
The model was trained on TauREx spectra and tested on POSEIDON spectra. The val→test RMSE degradation is **2.24x** (1.54 → 3.46). CO2 degrades 4.08x. Predictions collapse to a narrow band (13-41% of true IQR), with CO2 showing 87% of predictions clustered in [-12, -10] while true values span [-12, -2]. The model learned TauREx simulator artifacts, not atmospheric physics.

### C3. CPU State Dict Aliasing Bug (Report: 4)
**File**: `crossgen_hybrid_training.py:803`
On CPU, `value.detach().cpu()` returns the **same tensor** (no-op), so `best_state` is silently overwritten as training continues past the best epoch. The saved "best" model actually contains **last-epoch weights**. Verified: `inner_val_metrics.json` matches epoch 30 (last), not epoch 29 (best).
**Fix**: Replace `.detach().cpu()` with `.detach().clone()`.

### C4. Checkpoint Resume Crashes with KeyError (Report: 13)
**File**: `crossgen_hybrid_training.py:716-717, 808, 911-921`
Mid-training checkpoint saves keys `{epoch, val_loss, model_state_dict}`, but `run_training_experiment` overwrites the same file with different keys `{best_epoch, best_val_loss, ...}`. Resume reads `ckpt["epoch"]` which doesn't exist in the overwritten file → crash.

### C5. Baseline Comparison Is Invalid (Report: 11)
`data/baseline_poseidon_predictions.csv` contains near-constant predictions (~-7.0 for all targets), matching training-set means. This is a **mean-predictor stub**, not a real trained model. Additionally, the baseline and quantum model use different spectral resolutions (218 vs 44 bins), different target sets (7 vs 5), different feature sets, and different preprocessing.

### C6. "Biosignature Detection" Claim Is Misleading (Report: 14)
The model performs VMR regression (atmospheric retrieval), not biosignature detection. There is no classification, no disequilibrium metrics, no abiotic/biotic discrimination. These are fundamentally different scientific claims.

### C7. No Evidence of Quantum Advantage (Reports: 6, 9, 14)
The quantum block has 36 parameters (0.07% of total ~50k). A classical layer could trivially match this capacity. The skip connection allows the network to bypass quantum computation entirely. No ablation study (quantum vs classical-only) exists. Simulated quantum circuits provide no computational advantage.

---

## HIGH Findings (Must Fix)

| ID | Finding | Reports |
|----|---------|---------|
| H1 | No confidence intervals reported (685 test samples → bootstrap 95% CI spans 0.2-0.4 dex) | 4 |
| H2 | CO2 systematic bias: 93% of MAE is pure bias (-3.60 dex shift) | 4 |
| H3 | No library version pinning — zero record of PyTorch/PennyLane/NumPy versions used | 5 |
| H4 | cuDNN benchmark=True without deterministic=True — GPU runs not reproducible | 5 |
| H5 | PennyLane RNG not seeded — quantum parameter initialization may vary | 5 |
| H6 | Checkpoint resume missing optimizer/scheduler state — warm restart corrupts training | 7, 13 |
| H7 | ReduceLROnPlateau never fired (scheduler_patience=5 ≈ early_stop_patience=6) | 7 |
| H8 | No data augmentation whatsoever (no noise injection, spectral dropout, mixup) | 10 |
| H9 | Insufficient regularization (dropout=0.05, weight_decay=1e-4, quantum wd=0) | 10 |
| H10 | Conflicting baselines — two different "baselines" exist enabling cherry-picking | 11 |
| H11 | Spectral resolution mismatch between baseline (218 bins) and quantum model (44 bins) | 11 |
| H12 | No NaN/Inf guards in training loop — single bad quantum gradient corrupts all weights | 7 |

---

## MEDIUM Findings (Should Fix)

| ID | Finding | Reports |
|----|---------|---------|
| M1 | Dead config parameter `internal_val_fraction` never used, misleads reviewers | 1, 13 |
| M2 | `planet_radius_rjup` and `temperature_k` are oracle inputs (not available in real deployment) | 2 |
| M3 | Per-sample spectral normalization removes absolute transit depth (physically meaningful) | 1, 8 |
| M4 | No NaN validation after SpectRes rebinning | 8, 12 |
| M5 | No inference deserialization code — scalers cannot be loaded from `scalers.json` | 8 |
| M6 | Validation loss averages across potentially unequal batch sizes | 13 |
| M7 | Model variant selection occurred after test RMSE was visible | 15 |
| M8 | Undocumented qubit count reduction (16→12) | 15 |
| M9 | `tanh * π` encoding compresses extreme fusion values | 6 |
| M10 | Residual skip connection may bypass quantum block entirely | 6, 9 |
| M11 | Missing GradScaler for float16 AMP on older GPUs | 12 |

---

## What Passed Clean

| Area | Status | Notes |
|------|--------|-------|
| Data Leakage (train/val/test) | **PASS** | Splits are pre-assigned, standardizers fit on train only |
| Target Leakage | **PASS** | No statistical leakage (all |r| < 0.01 with targets) |
| Preprocessing Correctness | **PASS** | Mathematically correct transforms, properly serialized |
| Numerical Stability | **PASS** | No critical numerical issues in current pipeline |
| Experimental Design | **PASS** | No hyperparameter search, single seed, no test snooping |
| RMSE Computation | **PASS** | Formula correct, matches independent recomputation |
| Wavelength Coverage | **PASS** | 0.95-4.91 μm covers key absorption bands for all 5 targets |

---

## Priority Remediation Plan

### Tier 1 — Must Fix Before Any Publication

1. **Fix the state dict aliasing bug** (C3): Change `.detach().cpu()` to `.detach().clone()` at line 803, then **retrain the model**.

2. **Address cross-generator failure** (C1, C2): Either:
   - (a) Include POSEIDON samples in training (mixed-generator split), or
   - (b) Add domain adaptation (adversarial training, gradient reversal), or
   - (c) Train on POSEIDON data separately, or
   - (d) Reframe the paper as TauREx-only retrieval (dropping cross-generator claims)

3. **Create a matched classical ablation** (C7): Same architecture with quantum block replaced by a classical MLP of equal dimensions. Same data, features, preprocessing, optimizer settings. This is the minimum requirement to discuss quantum contribution.

4. **Fix baseline comparison** (C5): Run the real CNN baseline on the identical 44-bin crossgen dataset with the same 5 targets and same splits.

5. **Correct scientific claims** (C6): Either add a biosignature classification head (disequilibrium metrics, biotic/abiotic discrimination) or rename to "atmospheric retrieval."

### Tier 2 — Must Fix for Scientific Rigor

6. Add confidence intervals (bootstrap) for all reported metrics.
7. Pin library versions and add `requirements.txt`.
8. Seed PennyLane RNG and enable deterministic CUDA.
9. Fix checkpoint resume (save/restore optimizer and scheduler state, use consistent keys).
10. Add NaN/Inf validation after data loading and rebinning.
11. Add data augmentation (noise injection at minimum).

### Tier 3 — Acknowledge as Limitations

- Oracle auxiliary features (temperature, radius) not available in real deployment
- Per-sample normalization removes absolute transit depth information
- 685-sample test set limits statistical power
- Simulated quantum circuit provides no computational advantage over classical equivalent

---

## Bottom Line

The pipeline engineering is solid — data splits are clean, preprocessing is correct, there's no data leakage, and the code is well-structured. The fundamental problem is **the model doesn't generalize across radiative transfer codes**, which is the central scientific claim. Combined with the state-dict aliasing bug (the "best" model isn't actually the best), the invalid baseline comparison, and the absence of any quantum advantage evidence, the paper is not ready for submission in its current form.

The path forward is clear: fix the bugs, retrain, add domain adaptation or reframe as single-generator retrieval, create a proper classical ablation, and honestly report limitations.
