# Audit Report 10: Overfitting and Generalization

## Summary

The model exhibits **severe overfitting manifesting as a catastrophic generalization gap** between the inner-validation set (same generator, TauREx) and the test set (different generator, Poseidon). Mean RMSE jumps from 1.54 dex on validation to 3.46 dex on test -- a 2.24x degradation that renders predictions scientifically useless (errors span 3+ orders of magnitude in mixing ratio). The train-vs-val loss curves themselves look healthy, indicating the primary problem is not classical overfitting to the training set but rather **distribution shift between generators** compounded by inadequate regularization and the complete absence of data augmentation. CO2 is the worst-affected target with a test RMSE of 4.73 dex, meaning predictions are off by nearly 5 orders of magnitude on average.

## Methodology

Every line of `crossgen_hybrid_training.py` (987 lines) was read. The training history (30 epochs), metrics summary, test predictions (685 rows), config, and run summary were analyzed. Loss curves and test scatter plots were visually inspected. Parameter counts were computed from architecture definitions.

---

## Findings

### Finding 1: Catastrophic val-to-test generalization gap (distribution shift)
- **Severity**: CRITICAL
- **Files**:
  - `/Users/michalszczesny/projects/hack4sages/codex_model/outputs/model_crossgen_rebinned/inner_val_metrics.json`
  - `/Users/michalszczesny/projects/hack4sages/codex_model/outputs/model_crossgen_rebinned/test_metrics.json`
- **Description**: The inner-validation RMSE (mean=1.545) and test RMSE (mean=3.463) differ by a factor of 2.24x. Per-target breakdown:

| Target | Val RMSE (dex) | Test RMSE (dex) | Degradation |
|--------|---------------|-----------------|-------------|
| H2O    | 1.662         | 3.131           | 1.88x       |
| CO2    | 1.158         | 4.726           | 4.08x       |
| CO     | 2.182         | 3.283           | 1.50x       |
| CH4    | 1.219         | 3.227           | 2.65x       |
| NH3    | 1.503         | 2.948           | 1.96x       |

- **Root cause**: Train and val both come from TauREx (37,281 + 4,142 samples), while test comes from Poseidon (685 samples). The model has learned TauREx-specific spectral artifacts rather than physics-transferable features. This is a dataset design issue more than a model-level overfitting issue, but the model's lack of robustness amplifies it.
- **Impact**: The model cannot generalize across radiative transfer codes. A test RMSE of 3.46 dex means the model's average prediction is off by ~2,900x in mixing ratio -- scientifically meaningless for biosignature detection.

---

### Finding 2: CO2 predictions collapse to a narrow band on test set
- **Severity**: CRITICAL
- **File**: `/Users/michalszczesny/projects/hack4sages/codex_model/outputs/model_crossgen_rebinned/test_predictions.csv`
- **Description**: Examining test predictions for CO2, the predicted values cluster tightly around -10.5 to -11.5 regardless of the true value (which spans -2.35 to -12.0). Representative samples:

```
sample_id          true_co2     pred_co2
poseidon_000002    -4.932       -11.253
poseidon_000008    -2.759       -11.040
poseidon_000012    -2.631       -10.070
poseidon_000025    -2.353       -9.375
```

The model has effectively learned to predict the TauREx-distribution mean for CO2 and cannot recover individual values on Poseidon spectra. This is the hallmark of a model that has overfit to generator-specific correlations.
- **Impact**: CO2 test RMSE of 4.726 dex. The model's CO2 predictions carry zero useful information on out-of-distribution data.

---

### Finding 3: Train-val loss curves show no classical overfitting (misleading)
- **Severity**: MEDIUM
- **File**: `/Users/michalszczesny/projects/hack4sages/codex_model/outputs/model_crossgen_rebinned/history.csv`
- **Description**: Analyzing the 30-epoch history:

| Epoch | Train Loss | Val Loss | Gap   |
|-------|-----------|----------|-------|
| 1     | 0.8287    | 0.7139   | 0.115 |
| 10    | 0.3959    | 0.3886   | 0.007 |
| 20    | 0.3256    | 0.3225   | 0.003 |
| 30    | 0.3013    | 0.3019   | -0.001|

The train-val gap is essentially zero by epoch 20, and val loss is still decreasing at epoch 30. The learning rate never decays (scheduler_patience=5, but val loss keeps improving). No early stopping triggered (best epoch = 29 of 30). By classical overfitting metrics, this looks perfect.
- **Impact**: The healthy train-val curve creates a **false sense of security**. Because val is from the same generator as train, it cannot detect the distribution shift to Poseidon. The validation set is not representative of the deployment distribution, making the early-stopping and scheduler mechanisms ineffective as generalization guards.

---

### Finding 4: Insufficient regularization for cross-generator generalization
- **Severity**: HIGH
- **File**: `/Users/michalszczesny/projects/hack4sages/codex_model/crossgen_hybrid_training.py`:106, 89-119
- **Description**: Current regularization settings:
  - `dropout = 0.05` (lines 106, 366-399, 490-498)
  - `weight_decay = 1e-4` for classical params only (line 104, 694)
  - `weight_decay = 0.0` for quantum params (line 695)
  - No batch normalization in encoders (only LayerNorm in FusionEncoder, line 412)
  - No spectral augmentation or noise injection

```python
optimizer = torch.optim.AdamW(
    [
        {"params": classical_params, "lr": config.classical_lr, "weight_decay": config.weight_decay},
        {"params": quantum_params, "lr": config.quantum_lr, "weight_decay": 0.0},
    ]
)
```

Dropout of 5% is extremely low. Weight decay of 1e-4 is minimal. The quantum parameters receive zero weight regularization. Combined, these settings allow the model to learn arbitrarily sharp decision boundaries that fit TauREx-specific features.
- **Impact**: The model has no strong inductive bias toward smooth, generator-invariant representations. Increasing dropout to 0.15-0.30 and weight decay to 1e-3 would force the model to learn more robust features, though this alone will not solve the distribution shift problem.

---

### Finding 5: Complete absence of data augmentation
- **Severity**: HIGH
- **File**: `/Users/michalszczesny/projects/hack4sages/codex_model/crossgen_hybrid_training.py` (entire file)
- **Description**: There is no data augmentation anywhere in the pipeline. The training loop at lines 739-766 fetches raw (scaled) data directly with no transforms:

```python
for batch_idx, indices in enumerate(
    batch_indices(len(data.train.aux), config.train_batch_size, config.seed, epoch, data.train.aux.device),
    start=1,
):
    aux, spectra, targets = gather_batch(data.train, indices, device)
    optimizer.zero_grad(set_to_none=True)
    pred = model(aux, spectra)
```

Missing augmentations that would improve robustness:
  - **Gaussian noise injection** on spectra (simulates different noise realizations)
  - **Wavelength jitter** (simulates different instrument calibrations)
  - **Spectral dropout** (randomly zeroing wavelength bins)
  - **Mixup** (interpolating between training samples)
  - **Feature-space dropout** on auxiliary features

- **Impact**: Without augmentation, the model sees each training spectrum exactly once per epoch with no variability. This encourages memorization of generator-specific spectral shapes rather than learning robust spectral features. Given the distribution shift to Poseidon, augmentation is arguably the single most impactful missing component.

---

### Finding 6: Model capacity is reasonable but unconstrained
- **Severity**: LOW
- **Description**: Estimated parameter count from architecture:

| Component | Parameters (approx) |
|-----------|-------------------|
| AuxEncoder | 5x64 + 64 + 64x64 + 64 + 64x32 + 32 = ~6,576 |
| SpectralEncoder | 1x32x7 + 32 + 32x64x5 + 64 + 64x64x3 + 64 + 64x32 + 32 = ~24,480 |
| FusionEncoder | 64x48 + 48 + 48x12 + 12 + 12(LN) = ~3,660 |
| QuantumBlock | 3 x 12 x 1 = 72 (trainable weights) |
| AtmosphereHead | 88x96 + 96 + 96x96 + 96 + 96x5 + 5 + 12x5 + 5 = ~18,077 |
| **Total** | **~52,865** |

With 37,281 training samples and ~53K parameters, the parameter-to-sample ratio is ~1:0.7, which means there are more parameters than effective constraints. However, this ratio is not extreme for neural networks with implicit regularization from SGD. The real issue is not raw capacity but the lack of regularization forcing those parameters toward generalizable solutions.
- **Impact**: Model capacity alone is not the problem. The architecture is appropriately sized for 37K samples, but the combination of low regularization and no augmentation allows those parameters to encode generator-specific patterns.

---

### Finding 7: Validation set masks distribution shift (structural design flaw)
- **Severity**: HIGH
- **File**: `/Users/michalszczesny/projects/hack4sages/codex_model/crossgen_hybrid_training.py`:318-320
- **Description**: The split design uses:
  - Train: 37,281 samples from TauREx
  - Val: 4,142 samples from TauREx
  - Test: 685 samples from Poseidon

```python
inner_train_indices = np.where(labels["split"].values == "train")[0]
inner_val_indices = np.where(labels["split"].values == "val")[0]
test_indices = np.where(labels["split"].values == "test")[0]
```

The val set is drawn from the same generator as train. All model selection (early stopping, LR scheduling) is based on this in-distribution validation. The model was checkpoint-selected at epoch 29 because val loss was still improving -- but this improvement was on TauREx data and had no bearing on Poseidon generalization.

Note: the config field `internal_val_fraction = 0.10` (line 95) is **unused** in the actual code. The validation split is entirely pre-assigned in the parquet file, not carved from training data. This is confirmed in audit 01, but worth re-noting: there is no held-out validation from a different distribution.
- **Impact**: Every model selection decision (which epoch to keep, when to decay LR, when to stop) is optimized for TauREx, potentially at the expense of Poseidon performance. Without a cross-generator validation signal, there is no mechanism to detect or prevent the distribution shift degradation during training.

---

### Finding 8: No training loss memorization, but val loss plateau indicates capacity saturation
- **Severity**: LOW
- **File**: `/Users/michalszczesny/projects/hack4sages/codex_model/outputs/model_crossgen_rebinned/history.csv`
- **Description**: Training loss decreases from 0.829 to 0.301 over 30 epochs, while val loss decreases from 0.714 to 0.300. The final train loss (0.301) is **not** near zero, indicating the model has not memorized the training set. This is a positive signal: the model is underfitting the in-distribution data rather than overfitting it.

The val RMSE curve shows the same pattern: steady decrease from 2.42 to 1.54, with marginal improvement in late epochs (1.57 at epoch 25 to 1.54 at epoch 29). The model is approaching its capacity limit on the TauREx distribution.
- **Impact**: The problem is not memorization. The model genuinely learns TauREx spectral patterns but those patterns do not transfer. This is more of a bias/distributional mismatch problem than a variance problem.

---

### Finding 9: Per-target overfitting analysis reveals CO2 and CO as worst generalizers
- **Severity**: MEDIUM
- **File**: `/Users/michalszczesny/projects/hack4sages/codex_model/outputs/model_crossgen_rebinned/history.csv`, `test_metrics.json`
- **Description**: Val RMSE progression by target at epoch 29 (best):

| Target | Val RMSE (ep29) | Test RMSE | Gap (dex) | Gap ratio |
|--------|----------------|-----------|-----------|-----------|
| H2O    | 1.657          | 3.131     | 1.474     | 1.89x     |
| CO2    | 1.156          | 4.726     | 3.570     | 4.09x     |
| CO     | 2.198          | 3.283     | 1.085     | 1.49x     |
| CH4    | 1.225          | 3.227     | 2.002     | 2.64x     |
| NH3    | 1.453          | 2.948     | 1.495     | 2.03x     |

CO2 has the **lowest** val RMSE (1.156) but the **highest** test RMSE (4.726), giving the worst gap ratio of 4.09x. This suggests CO2 features are the most generator-specific -- TauREx and Poseidon likely produce systematically different CO2 absorption signatures. CO has the smallest gap (1.49x) but also has the highest val RMSE (2.198), suggesting it is inherently harder but more consistently hard across generators.
- **Impact**: CO2 and CH4 need generator-invariant feature engineering or augmentation most urgently. The per-target analysis should guide where to invest regularization or domain adaptation effort.

---

### Finding 10: Test set size (685 samples) is small but sufficient for RMSE estimation
- **Severity**: LOW
- **File**: `/Users/michalszczesny/projects/hack4sages/codex_model/outputs/model_crossgen_rebinned/preflight.json`
- **Description**: With 685 Poseidon test samples, RMSE estimates have a standard error of approximately RMSE / sqrt(2 * (N-1)) ~ 3.46 / sqrt(1368) ~ 0.094. The 95% confidence interval for mean test RMSE is roughly [3.28, 3.65]. While this is adequate for detecting the catastrophic gap vs val RMSE of 1.54, the small test size means per-target estimates (especially for tails of the distribution) carry more uncertainty.
- **Impact**: The generalization failure is statistically unambiguous despite the modest test set size. However, the test set may not fully represent the diversity of Poseidon outputs.

---

## Quantitative Summary

| Metric | Value |
|--------|-------|
| Final train loss (epoch 30) | 0.3013 |
| Final val loss (epoch 30) | 0.3019 |
| Train-val gap (epoch 30) | 0.0006 (negligible) |
| Best val loss (epoch 29) | 0.2999 |
| Val mean RMSE (best model) | 1.545 dex |
| Test mean RMSE (best model) | 3.463 dex |
| **Val-to-test degradation** | **2.24x (1.92 dex increase)** |
| Worst per-target degradation | CO2: 4.09x (3.57 dex increase) |
| Best per-target degradation | CO: 1.49x (1.09 dex increase) |
| Early stopping triggered | No (ran all 30 epochs) |
| LR decay triggered | No (scheduler patience=5 never exceeded) |
| Estimated parameters | ~53K |
| Training samples | 37,281 |
| Parameter-to-sample ratio | ~1:0.7 |

## Recommendations

1. **[CRITICAL] Add cross-generator validation**: Mix a fraction of Poseidon samples into the validation set (or use a second validation metric on held-out Poseidon data). Without this, all model selection is blind to the actual deployment distribution.

2. **[CRITICAL] Implement spectral augmentation**: Add Gaussian noise injection (matched to sigma_ppm), random wavelength shifts, spectral dropout, and/or mixup to force the model to learn noise-invariant features rather than generator-specific templates.

3. **[HIGH] Increase regularization aggressively**: Raise dropout from 0.05 to 0.20-0.30. Increase weight_decay from 1e-4 to 1e-3 or 1e-2. Add weight decay to quantum parameters. Consider adding spectral smoothness penalties.

4. **[HIGH] Train on mixed-generator data**: If Poseidon training data is available, include it in the training set. Even a small fraction (5-10%) of Poseidon spectra in training could dramatically reduce the distribution shift.

5. **[MEDIUM] Add domain-invariant learning**: Consider adversarial domain adaptation (a generator-classifier head with gradient reversal) to force the encoder to learn features that are indistinguishable across TauREx and Poseidon.

6. **[MEDIUM] Log and monitor train RMSE alongside val RMSE**: The current history.csv only records train MSE loss (in standardized space) and val RMSE (in original space). Recording train RMSE in original space would enable direct comparison and early detection of within-distribution overfitting.

---

## Verdict: FAIL

The model exhibits a **catastrophic generalization failure** from validation to test. While in-distribution (TauREx) train-val behavior is healthy with no memorization, the cross-generator test RMSE of 3.46 dex (vs. val RMSE of 1.54 dex) demonstrates that the model has not learned transferable atmospheric physics. The 2.24x degradation, reaching 4.09x for CO2, means predictions on out-of-distribution spectra are scientifically meaningless. The lack of data augmentation, minimal regularization (5% dropout, 1e-4 weight decay), and same-generator validation set all contribute to this failure. The model's strong val performance is an illusion created by evaluating on in-distribution data.
