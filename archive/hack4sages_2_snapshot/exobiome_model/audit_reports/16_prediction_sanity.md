# Audit Report #16: End-to-End Prediction Sanity Check

**Model**: Hybrid quantum-classical (QELM) for exoplanet biosignature detection
**Output dir**: `/Users/michalszczesny/projects/hack4sages/exobiome_model/outputs/model_crossgen_rebinned/`
**Test set**: 685 Poseidon-generated samples (cross-generator evaluation)
**Training set**: 37,281 TauREx samples | **Validation set**: 4,142 TauREx samples
**Best epoch**: 29/30 (best inner val loss = 0.2999)

---

## Finding 1: Model Performs Worse Than Predicting the Mean for All 5 Targets

**Severity: CRITICAL**

The model's test RMSE exceeds the trivial mean-prediction baseline for every target. A model that simply outputs the training set mean for all inputs would achieve lower error.

| Target | Model RMSE | Mean-Baseline RMSE | Skill Score | R-squared |
|--------|-----------|-------------------|-------------|-----------|
| H2O    | 3.131     | 2.872             | -0.090      | -0.188    |
| CO2    | 4.726     | 2.965             | -0.594      | -1.541    |
| CO     | 3.283     | 2.843             | -0.155      | -0.334    |
| CH4    | 3.227     | 2.940             | -0.098      | -0.205    |
| NH3    | 2.948     | 2.842             | -0.038      | -0.077    |

Negative R-squared across the board confirms the model is actively harmful compared to the constant-mean baseline. CO2 is catastrophically bad at R^2 = -1.54.

**Source**: Computed from `/Users/michalszczesny/projects/hack4sages/exobiome_model/outputs/model_crossgen_rebinned/test_predictions.csv` using `sklearn`-style R^2 = 1 - SS_res/SS_tot.

---

## Finding 2: Near-Zero Correlation Between Predictions and True Values

**Severity: CRITICAL**

The model's predictions have essentially zero correlation with ground truth, meaning it has learned nothing about the input-output relationship on the test set.

| Target | Pearson r  | Spearman rho | p-value (Spearman) |
|--------|-----------|-------------|-------------------|
| H2O    | 0.0174    | 0.0166      | 0.664             |
| CO2    | -0.0226   | -0.0116     | 0.762             |
| CO     | 0.0367    | 0.0216      | 0.572             |
| CH4    | 0.0191    | 0.0423      | 0.269             |
| NH3    | -0.0030   | -0.0202     | 0.597             |

No target achieves a p-value below 0.25. The model's predictions are statistically indistinguishable from random noise relative to the true values. The model is not using input features to inform predictions in any meaningful way on this test set.

---

## Finding 3: Severe Prediction Variance Collapse

**Severity: CRITICAL**

Predictions are compressed into a narrow band while true values span the full [-12, -2] range.

| Target | True Std | Pred Std | Std Ratio | True IQR | Pred IQR | IQR Ratio |
|--------|---------|---------|-----------|---------|---------|-----------|
| H2O    | 2.872   | 1.179   | 0.410     | 4.997   | 1.462   | 0.293     |
| CO2    | 2.965   | 0.712   | 0.240     | 5.226   | 0.819   | 0.157     |
| CO     | 2.843   | 1.034   | 0.364     | 4.734   | 1.331   | 0.281     |
| CH4    | 2.940   | 0.972   | 0.330     | 4.989   | 1.243   | 0.249     |
| NH3    | 2.842   | 0.777   | 0.274     | 4.945   | 0.654   | 0.132     |

The model retains only 13-29% of the true IQR. For CO2, 87.3% of all predictions fall in [-12, -10], while only 22.9% of true values are in that range. The model has effectively collapsed CO2 predictions to a near-constant "very low" value around -10.8.

The 90th-percentile prediction ranges confirm this:

| Target | 90% Pred Range | 90% True Range (approx) |
|--------|---------------|------------------------|
| H2O    | 3.99          | ~8.0                   |
| CO2    | 2.27          | ~8.0                   |
| CO     | 3.35          | ~8.0                   |
| CH4    | 3.16          | ~8.0                   |
| NH3    | 2.61          | ~8.0                   |

---

## Finding 4: Massive Systematic Bias for CO2 (3.6 log10 Units)

**Severity: CRITICAL**

| Target | Mean Bias (pred - true) | Median Bias |
|--------|------------------------|-------------|
| H2O    | -0.530                 | -0.487      |
| CO2    | **-3.598**             | **-3.624**  |
| CO     | -1.359                 | -1.541      |
| CH4    | -0.967                 | -0.922      |
| NH3    | +0.020                 | -0.088      |

CO2 predictions are systematically shifted 3.6 orders of magnitude too low. The scaled-space analysis reveals this originates from the model's internal predictions:

```
CO2 scaled space: pred_mean = -1.279 (should be ~0.0 if centered)
```

The model's raw output for CO2 is stuck at approximately -1.28 in standardized units, corresponding to a denormalized value near -10.66. This is 3.68 units below the training mean of -6.98. The model appears to have converged to a CO2-specific attractor far from the data center.

**Source**: Scaler parameters from `/Users/michalszczesny/projects/hack4sages/exobiome_model/outputs/model_crossgen_rebinned/scalers.json`:
- `target_scaler.mean[1]` (CO2) = -6.980
- `target_scaler.scale[1]` (CO2) = 2.881

---

## Finding 5: Asymmetric Bias Pattern -- Regression Toward Shifted Center

**Severity: HIGH**

The bias is strongly dependent on the true value magnitude, following a pattern of regression toward a biased center:

| Target | Bias (true < Q25) | Bias (Q25-Q75) | Bias (true > Q75) |
|--------|-------------------|----------------|-------------------|
| H2O    | +3.079            | -0.437         | -4.345            |
| CO2    | +0.204            | -3.548         | -7.521            |
| CO     | +2.384            | -1.513         | -4.814            |
| CH4    | +2.826            | -0.986         | -4.746            |
| NH3    | +3.729            | -0.000         | -3.670            |

When the true value is low (e.g., -10 to -12), the model over-predicts by 2-4 units. When the true value is high (e.g., -2 to -4), the model under-predicts by 4-8 units. This is the classic signature of a model that has collapsed to predicting a near-constant value around the mean (or a biased version of it) and cannot resolve the extremes of the distribution.

---

## Finding 6: Catastrophic Val-to-Test Generalization Gap (Domain Shift)

**Severity: CRITICAL**

| Target | Inner Val RMSE | Test RMSE | Degradation Factor |
|--------|---------------|-----------|-------------------|
| H2O    | 1.662         | 3.131     | 1.88x             |
| CO2    | 1.158         | 4.726     | **4.08x**         |
| CO     | 2.182         | 3.283     | 1.50x             |
| CH4    | 1.219         | 3.227     | 2.65x             |
| NH3    | 1.503         | 2.948     | 1.96x             |
| **Mean** | **1.545**   | **3.463** | **2.24x**         |

The inner validation set (TauREx samples) shows RMSE ~1.5, while the test set (Poseidon samples) shows RMSE ~3.5. This 2.24x degradation occurs despite the test target distributions being nearly identical to training:

| Target | Train Mean | Test True Mean | Difference |
|--------|-----------|---------------|------------|
| H2O    | -6.981    | -7.087        | -0.106     |
| CO2    | -6.980    | -7.066        | -0.086     |
| CO     | -6.989    | -6.854        | +0.135     |
| CH4    | -7.009    | -6.913        | +0.095     |
| NH3    | -7.013    | -6.964        | +0.048     |

The target distributions are well-aligned (differences < 0.14), so the generalization failure is not due to label shift. The model has overfit to TauREx-specific spectral features (or noise patterns) that do not transfer to Poseidon spectra. This is a **domain shift** problem at the input level.

**Source**: Scaler means from `scalers.json`, test true means computed from `test_predictions.csv`.

---

## Finding 7: Spurious Cross-Target Correlation in Predictions

**Severity: HIGH**

True target values are essentially uncorrelated (all |r| < 0.14), as expected for independently sampled log-uniform VMRs. However, predicted values show substantial positive correlations:

**Prediction cross-correlation matrix:**
|       | H2O   | CO2   | CO    | CH4   | NH3   |
|-------|-------|-------|-------|-------|-------|
| H2O   | 1.000 | 0.486 | 0.548 | 0.371 | 0.263 |
| CO2   | 0.486 | 1.000 | 0.463 | 0.286 | 0.307 |
| CO    | 0.548 | 0.463 | 1.000 | 0.533 | 0.388 |
| CH4   | 0.371 | 0.286 | 0.533 | 1.000 | 0.124 |
| NH3   | 0.263 | 0.307 | 0.388 | 0.124 | 1.000 |

**True cross-correlation matrix:**
|       | H2O    | CO2    | CO    | CH4   | NH3    |
|-------|--------|--------|-------|-------|--------|
| H2O   | 1.000  | -0.134 | 0.032 | 0.004 | -0.036 |
| CO2   | -0.134 | 1.000  | 0.042 | 0.013 | 0.049  |

All prediction cross-correlations are positive (0.12 to 0.55), while true values are near-zero. This means the model has learned a shared "activation pattern" driven by input features (likely spectral shape), but this pattern does not correspond to actual atmospheric composition. All five heads are responding to the same input signals in the same direction, rather than learning target-specific retrieval.

---

## Finding 8: Outlier Analysis -- Worst Errors Approach 10 Orders of Magnitude

**Severity: HIGH**

The worst individual predictions have absolute errors of 7-9.5 log10 units, meaning the model is off by a factor of 10^7 to 10^9.5 in VMR.

Top 3 worst per target:
| Target | Sample | True | Pred | |Error| |
|--------|--------|------|------|---------|
| CO2 | poseidon_000414 | -2.19 | -11.69 | **9.51** |
| CO2 | poseidon_000088 | -2.05 | -11.52 | **9.47** |
| CO2 | poseidon_000579 | -2.24 | -11.36 | **9.12** |
| CH4 | poseidon_000680 | -2.33 | -10.38 | 8.04 |
| H2O | poseidon_000123 | -2.45 | -10.01 | 7.56 |
| CO | poseidon_000559 | -2.08 | -9.85 | 7.77 |

All worst cases have **high true VMR** (near -2) but the model predicts **very low VMR** (near -10 to -12). The model cannot recover high-abundance cases at all. This is consistent with the variance collapse: predictions are anchored in the [-10, -8] region and simply cannot reach the [-4, -2] regime.

---

## Finding 9: Physically Valid Range -- No Impossible Predictions

**Severity: LOW (PASS)**

All 685 x 5 = 3,425 predictions fall within the physically plausible range for log10 VMR:
- No predictions > 0 (would imply VMR > 1, physically impossible)
- No predictions < -15 (would imply VMR < 10^-15, sub-trace)
- All predictions in [-11.9, -3.3], true values in [-12.0, -2.0]

The denormalization in `evaluate_split()` (line 664 of `crossgen_hybrid_training.py`) is mathematically correct: `pred_orig = target_scaler.inverse_transform(pred_scaled)` correctly applies `values * scale + mean`.

---

## Finding 10: RMSE Computation Verified -- Metrics Are Self-Consistent

**Severity: LOW (PASS)**

Independently recomputed RMSE from `test_predictions.csv` matches the reported values in `test_metrics.json` to within floating-point tolerance (<0.001):

| Target | Computed | Reported | Match |
|--------|---------|---------|-------|
| H2O    | 3.130655 | 3.130654 | YES  |
| CO2    | 4.726044 | 4.726046 | YES  |
| CO     | 3.283202 | 3.283202 | YES  |
| CH4    | 3.227484 | 3.227484 | YES  |
| NH3    | 2.948281 | 2.948282 | YES  |
| Mean   | 3.463133 | 3.463134 | YES  |

The RMSE computation in `evaluate_split()` at line 666 (`np.sqrt(np.mean((pred_orig - true_orig) ** 2, axis=0))`) is correct per-column RMSE, and `metrics_summary.csv` correctly records these values.

---

## Finding 11: Training Did Converge (on TauREx validation data)

**Severity: MEDIUM (context)**

The training history shows steady improvement over 30 epochs:
- Train loss: 0.829 (epoch 1) -> 0.301 (epoch 30)
- Inner val loss: 0.714 (epoch 1) -> 0.300 (epoch 29, best)
- Inner val RMSE mean: 2.418 (epoch 1) -> 1.538 (epoch 29, best)

No learning rate reduction was triggered (patience=5 was never reached since loss kept improving). The model legitimately learned TauREx-domain patterns. The problem is exclusively one of **cross-generator generalization**, not failure to train.

**Source**: `/Users/michalszczesny/projects/hack4sages/exobiome_model/outputs/model_crossgen_rebinned/history.csv`

---

## Root Cause Analysis

The model has fundamentally failed at cross-generator generalization. It achieved reasonable performance on the TauREx validation set (mean RMSE 1.54) but the learned representations do not transfer to Poseidon spectra. The evidence points to three overlapping failure modes:

1. **Spectral domain shift**: TauREx and Poseidon generate spectra with systematically different noise profiles, opacity models, or spectral shapes. The model latched onto generator-specific artifacts rather than physical atmospheric features.

2. **Variance collapse on OOD inputs**: When confronted with unfamiliar spectral patterns, the model's quantum circuit + classical head defaults to outputting near-constant values, collapsing to a biased attractor rather than the true mean.

3. **CO2-specific catastrophic failure**: The CO2 head has collapsed to a single-point attractor near -10.7 (3.6 units below the training mean), suggesting a failure mode where one target's gradient signal dominated during training, pushing the shared representation away from the CO2-relevant features.

---

## Recommendations

1. **Add Poseidon data to training** or use domain adaptation techniques (e.g., gradient reversal for generator-invariant features)
2. **Evaluate on a held-out TauREx test set** to confirm the model works within-distribution before claiming cross-gen capability
3. **Monitor per-target R-squared and correlation during training**, not just RMSE, to catch collapse early
4. **Add a prediction variance penalty** to prevent collapse to near-constant outputs
5. **Investigate CO2 head specifically** -- the 3.6 unit bias suggests a bug or pathological gradient flow for that target

---

## Summary Table

| Check | Result | Severity |
|-------|--------|----------|
| Predictions in valid physical range | PASS | LOW |
| RMSE matches reported values | PASS | LOW |
| Training convergence | PASS (on TauREx) | -- |
| Model beats mean baseline | **FAIL (all 5 targets)** | CRITICAL |
| Meaningful true-pred correlation | **FAIL (all |r| < 0.05)** | CRITICAL |
| Prediction variance preserved | **FAIL (13-41% of true)** | CRITICAL |
| No systematic bias | **FAIL (CO2: -3.6 units)** | CRITICAL |
| Cross-generator generalization | **FAIL (2.2x RMSE inflation)** | CRITICAL |
| Correct cross-target independence | **FAIL (spurious r=0.12-0.55)** | HIGH |
| No catastrophic outliers | **FAIL (errors up to 9.5 log10)** | HIGH |

---

## Verdict: FAIL

The model's test predictions are scientifically meaningless. On Poseidon test data, predictions are uncorrelated with ground truth (all Pearson r < 0.05), worse than a constant-mean baseline (all R^2 negative), and exhibit severe variance collapse and systematic bias. The model has learned TauREx-specific patterns that do not generalize across spectral generators. No predictions from this model should be used for scientific inference or downstream biosignature classification.
