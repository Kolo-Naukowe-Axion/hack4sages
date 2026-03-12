# Audit Report 04: Metric Validity and Statistical Rigor

**Auditor:** Scientific Rigor Audit
**Target:** `crossgen_hybrid_training.py` metric computation, reported results
**Date:** 2026-03-12
**Verdict:** FAIL

---

## Summary

The metric computation pipeline contains one critical bug (checkpoint aliasing on CPU), one high-severity concern (no confidence intervals or significance tests are reported), and several medium/low issues. The reported RMSE values are numerically correct for the model weights that were actually evaluated, but those weights are NOT from the best checkpoint as claimed -- they are from the last training epoch due to a memory aliasing bug in PyTorch on CPU. Additionally, the model performs worse than a trivial mean-prediction baseline on the cross-generator test set, which is not flagged anywhere in the outputs.

---

## Finding 1: Best-Checkpoint State Dict Aliasing on CPU

**Severity:** CRITICAL
**File:** `/Users/michalszczesny/projects/hack4sages/exobiome_model/crossgen_hybrid_training.py`
**Lines:** 803, 819-820

**Description:**

When training on CPU (`lightning.qubit` backend), the best checkpoint state dict is not actually a deep copy. The code at line 803:

```python
best_state = {name: value.detach().cpu() for name, value in model.state_dict().items()}
```

On CPU, `value.detach().cpu()` returns the **same tensor** (same underlying storage), not a copy. `.detach()` creates a view, and `.cpu()` is a no-op for tensors already on CPU. This was verified empirically:

```python
x = torch.tensor([1.0, 2.0])
y = x.detach().cpu()
assert x.data_ptr() == y.data_ptr()  # True -- same memory!
```

**Consequence:** When training continues past epoch 29 (the best epoch) to epoch 30, the optimizer modifies model parameters in-place, silently corrupting `best_state`. At line 820, `model.load_state_dict(best_state)` loads the **last epoch weights**, not the best.

**Evidence:** The `inner_val_metrics.json` reports RMSE values that match history epoch 30 (last), not epoch 29 (best):

| Target | Epoch 29 (best) | Epoch 30 (last) | inner_val_metrics.json |
|--------|-----------------|-----------------|----------------------|
| H2O | 1.6570641 | 1.6619719 | 1.6619719 |
| CO2 | 1.1559807 | 1.1580735 | 1.1580735 |
| CO | 2.1978977 | 2.1817324 | 2.1817324 |
| CH4 | 1.2251188 | 1.2188259 | 1.2188259 |
| NH3 | 1.4532806 | 1.5032080 | 1.5032080 |

Every value matches epoch 30 exactly, confirming the bug.

**Additional impact:** The `best_model.pt` file is saved twice -- once correctly during training (line 808) and once with corrupted weights after training (line 911). The second save overwrites the first, so the saved checkpoint also contains last-epoch weights, not best-epoch weights.

**Practical impact in this run:** Minimal, since best epoch (29) and last epoch (30) differ by only 1 epoch and val loss difference is small (0.2999 vs 0.3019). However, in runs with early stopping triggering much earlier, this bug would report metrics from a significantly worse model.

**Fix:** Replace `.detach().cpu()` with `.detach().clone()`:

```python
best_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
```

---

## Finding 2: No Confidence Intervals or Statistical Tests Reported

**Severity:** HIGH
**File:** `/Users/michalszczesny/projects/hack4sages/exobiome_model/crossgen_hybrid_training.py`
**Lines:** 640-676, 948-955

**Description:**

The test set has only 685 samples. Point estimates of RMSE are reported without any uncertainty quantification. Bootstrap analysis reveals non-trivial confidence intervals:

| Target | RMSE | 95% CI |
|--------|------|--------|
| H2O | 3.131 | [2.998, 3.262] |
| CO2 | 4.726 | [4.537, 4.907] |
| CO | 3.283 | [3.149, 3.416] |
| CH4 | 3.227 | [3.093, 3.361] |
| NH3 | 2.948 | [2.843, 3.052] |
| **Mean** | **3.463** | **[3.399, 3.525]** |

The CI width is approximately 0.2-0.4 dex per target, meaning reported differences smaller than this threshold are not statistically reliable.

**Recommendation:** Add bootstrap CIs to `test_metrics.json` and `metrics_summary.csv`.

---

## Finding 3: Model Performs Worse Than Mean-Prediction Baseline on Test Set

**Severity:** HIGH
**File:** `/Users/michalszczesny/projects/hack4sages/exobiome_model/outputs/model_crossgen_rebinned/test_metrics.json`

**Description:**

On the test set, the model is worse than simply predicting the training-set mean for every sample:

| Target | Model RMSE | Baseline RMSE (predict mean) | Ratio | R-squared |
|--------|-----------|------------------------------|-------|-----------|
| H2O | 3.131 | 2.874 | 1.09x | -0.188 |
| CO2 | 4.726 | 2.967 | 1.59x | -1.541 |
| CO | 3.283 | 2.845 | 1.15x | -0.334 |
| CH4 | 3.227 | 2.943 | 1.10x | -0.205 |
| NH3 | 2.948 | 2.844 | 1.04x | -0.077 |

All R-squared values are negative, meaning zero explanatory power. CO2 is catastrophically bad at 1.59x worse than baseline (R2 = -1.54).

This is NOT flagged anywhere in the outputs. The `run_summary.json` reports `test_rmse_mean: 3.463` without context about what constitutes a good or bad value.

**Root cause:** Domain shift between training data (TauREx generator) and test data (Poseidon generator). The val RMSE is ~1.5 (reasonable, ratio ~0.5x baseline), confirming the model learns the TauREx distribution but fails to generalize.

**Recommendation:** Report baseline comparison (mean predictor, per-target std) alongside model metrics. Flag negative R-squared values.

---

## Finding 4: Massive Systematic Bias on CO2 Predictions

**Severity:** HIGH
**File:** `/Users/michalszczesny/projects/hack4sages/exobiome_model/outputs/model_crossgen_rebinned/test_predictions.csv`

**Description:**

The prediction residuals (pred - true) show large systematic bias, especially for CO2:

| Target | Mean Error (bias) | MAE | |bias|/MAE |
|--------|------------------|-----|-----------|
| H2O | -0.530 | 2.611 | 20.3% |
| CO2 | **-3.598** | 3.857 | **93.3%** |
| CO | -1.359 | 2.761 | 49.2% |
| CH4 | -0.967 | 2.699 | 35.8% |
| NH3 | 0.020 | 2.526 | 0.8% |

For CO2, 93% of the MAE is pure bias (the model systematically predicts values ~3.6 dex too negative). This means the error is dominated by a constant offset, not noise.

RMSE does not decompose into bias and variance components in the reported metrics, obscuring this distinction. A model with high RMSE due to bias can be trivially improved by bias correction, while one with high RMSE due to variance cannot.

**Recommendation:** Report bias-variance decomposition: `RMSE^2 = bias^2 + variance`.

---

## Finding 5: Batch-Averaged Loss in evaluate_split

**Severity:** LOW (does not affect this run)
**File:** `/Users/michalszczesny/projects/hack4sages/exobiome_model/crossgen_hybrid_training.py`
**Lines:** 648-669

**Description:**

The evaluation loss is computed as `np.mean(losses)` where each element is the mean loss of a batch (line 660):

```python
losses.append(float(loss_fn(pred, targets).item()))
```

With `nn.MSELoss(reduction='mean')`, each batch loss is the mean over samples in that batch. Then `np.mean(losses)` averages these batch means. If the last batch is smaller, each sample in the last batch has more weight than samples in other batches.

In this run, `eval_batch_size=8192` exceeds all split sizes (train=37281 batches = 5, val=4142 = 1 batch, test=685 = 1 batch), so val and test are unaffected. But train evaluation during training (batch_losses averaging at line 774) IS affected, giving slightly incorrect train loss estimates.

**Fix:** Use a weighted mean: `total_loss / total_samples` instead of `mean(batch_losses)`.

---

## Finding 6: Equal Weighting of Targets is Reasonable

**Severity:** LOW (informational)
**File:** `/Users/michalszczesny/projects/hack4sages/exobiome_model/crossgen_hybrid_training.py`
**Line:** 675

**Description:**

The mean RMSE is an unweighted average of 5 per-target RMSEs. This is appropriate because:

1. All targets are on the same scale (log10 VMR)
2. All targets have similar ranges (~[-12, -2]) and standard deviations (~2.85-2.97)
3. There is no domain-specific reason to weight any gas species more than others

No action required.

---

## Finding 7: RMSE Formula is Correct

**Severity:** LOW (informational, passes check)
**File:** `/Users/michalszczesny/projects/hack4sages/exobiome_model/crossgen_hybrid_training.py`
**Lines:** 662-666

**Description:**

The RMSE computation is:

```python
pred_orig = target_scaler.inverse_transform(pred_scaled)
true_orig = target_scaler.inverse_transform(true_scaled)
rmse_orig = np.sqrt(np.mean((pred_orig - true_orig) ** 2, axis=0))
```

This correctly:
1. Inverse-transforms both predictions and targets from standardized to original scale
2. Computes per-target RMSE using `axis=0` (average over samples, one RMSE per target)
3. Uses population RMSE (divides by N, not N-1), which is standard in ML
4. Reports RMSE in the physically meaningful original scale (log10 VMR)

The inverse transform is algebraically correct: `inverse(transform(x)) = (((x - mean) / scale) * scale + mean) = x`.

Independently verified from `test_predictions.csv`: recomputed RMSE values match reported values to float32 precision (< 1e-6 difference due to CSV rounding).

---

## Finding 8: No Test-Set Information Leakage in Metrics

**Severity:** LOW (informational, passes check)
**File:** `/Users/michalszczesny/projects/hack4sages/exobiome_model/crossgen_hybrid_training.py`
**Lines:** 318-327

**Description:**

- Scalers (`aux_scaler`, `target_scaler`, `spectral_scaler`) are all fitted exclusively on `inner_train_indices` (lines 325-327)
- Early stopping and checkpoint selection use `inner_val_loss`, never test metrics
- Test evaluation happens only once after training (line 946)
- No hyperparameter tuning loop uses test metrics

No test leakage detected.

---

## Finding 9: Reported Metrics Are From Best Epoch (Correctly Labeled, Despite Bug)

**Severity:** MEDIUM
**File:** `/Users/michalszczesny/projects/hack4sages/exobiome_model/outputs/model_crossgen_rebinned/run_summary.json`

**Description:**

`run_summary.json` claims `best_epoch: 29`, but due to Finding 1, the actual evaluation used epoch 30 (last) weights. The report is technically misleading: it says the best checkpoint was used, but it was not.

However, the code's INTENT is correct -- it tries to load the best checkpoint (line 820). The bug is in the implementation, not the design.

---

## Finding 10: No Cherry-Picking Detected

**Severity:** LOW (informational, passes check)

**Description:**

- `metrics_summary.csv` reports all 5 targets, not just the best
- `test_metrics.json` and `inner_val_metrics.json` both report complete metrics
- `history.csv` contains all 30 epochs with per-target breakdown
- Both val and test metrics are saved; the test set is evaluated exactly once
- No evidence of selective reporting or multiple experimental runs with best-run selection

---

## Consolidated Findings Table

| # | Finding | Severity | Impact on Reported Results |
|---|---------|----------|---------------------------|
| 1 | CPU state dict aliasing bug | CRITICAL | Metrics are from last epoch, not best. Small impact this run. |
| 2 | No confidence intervals | HIGH | Cannot assess statistical reliability of reported RMSE. |
| 3 | Worse than mean baseline | HIGH | Model has negative R-squared on all test targets. Not reported. |
| 4 | Systematic bias in CO2 | HIGH | 93% of CO2 error is pure bias. Not decomposed. |
| 5 | Batch-averaged loss | LOW | Does not affect this run (single-batch eval). |
| 6 | Equal target weighting | LOW | Appropriate given homogeneous scales. |
| 7 | RMSE formula correct | LOW | Verified independently. |
| 8 | No test leakage | LOW | Scalers fit on train only. |
| 9 | Misleading best_epoch label | MEDIUM | Claims epoch 29, actually evaluates epoch 30 weights. |
| 10 | No cherry-picking | LOW | All metrics reported transparently. |

---

## Verdict: FAIL

The audit fails on three grounds:

1. **CRITICAL correctness bug:** The best-checkpoint restoration is broken on CPU due to PyTorch tensor aliasing. The reported "best model" metrics and saved checkpoint contain last-epoch weights. Fix: use `.detach().clone()` instead of `.detach().cpu()`.

2. **Missing statistical context:** No confidence intervals, no baseline comparison, no bias-variance decomposition. With only 685 test samples and RMSE CIs spanning ~0.2-0.4 dex, point estimates alone are insufficient for scientific claims.

3. **Unreported model failure:** All test-set R-squared values are negative (range: -0.08 to -1.54), meaning the model is strictly worse than predicting the mean. This must be prominently disclosed, not buried in raw RMSE numbers.
