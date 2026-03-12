# Audit Report 01: Data Leakage -- Train/Val/Test Contamination

## Summary

The data pipeline is free of train/val/test contamination. All standardizers are fit exclusively on training data, splits are pre-assigned in the dataset and physically separated by radiative transfer generator, and no cross-sample information flows between splits at any stage. Two minor findings (unused config field, per-sample normalization design choice) are noted but neither constitutes data leakage.

## Methodology

Every line of `crossgen_hybrid_training.py` (987 lines) was read. The complete data flow was traced from raw file loading through normalization, splitting, training, and evaluation. Supporting files (`preflight.json`, `config.json`, `scalers.json`, `run_training.py`, `train.ipynb`) were also reviewed.

---

## Findings

### Finding 1: Split integrity -- pre-assigned column, no runtime splitting
- **Severity**: PASS (no issue)
- **File**: `/Users/michalszczesny/projects/hack4sages/exobiome_model/crossgen_hybrid_training.py`:318-320
- **Description**: Splits are determined by the `split` column in `labels.parquet`, not by any runtime random splitting. The code uses `np.where` to extract indices for each split value ("train", "val", "test"). There is no shuffling, no random sampling, and no re-splitting logic anywhere in the codebase.
- **Code snippet**:
```python
inner_train_indices = np.where(labels["split"].values == "train")[0]
inner_val_indices = np.where(labels["split"].values == "val")[0]
test_indices = np.where(labels["split"].values == "test")[0]
```
- **Impact**: None. Pre-assigned splits are the gold standard for reproducibility and prevent accidental overlap.
- **Verification**: `preflight.json` confirms the counts: train=37,281, val=4,142, test=685. Their sum (42,108) equals the total row count, confirming a complete partition with no overlap.

---

### Finding 2: Standardizer fitting scope -- train-only
- **Severity**: PASS (no issue)
- **File**: `/Users/michalszczesny/projects/hack4sages/exobiome_model/crossgen_hybrid_training.py`:325-327
- **Description**: All three standardizers (`aux_scaler`, `target_scaler`, `spectral_scaler`) are fit exclusively on training indices. The `ArrayStandardizer.fit()` and `SpectralStandardizer.fit()` methods compute `mean(axis=0)` and `std(axis=0)` from the array passed to them, with no global state or side effects.
- **Code snippet**:
```python
aux_scaler = ArrayStandardizer.fit(aux_raw[inner_train_indices])
target_scaler = ArrayStandardizer.fit(targets_raw[inner_train_indices])
spectral_scaler = SpectralStandardizer.fit(spectra_raw[inner_train_indices, 0, :])
```
- **Impact**: None. This is correct practice. Val and test data are transformed using train-derived statistics only.
- **Verification**: The `scalers.json` file records the saved scaler values. The `aux_scaler` mean for `temperature_k` is 1149.58, which matches the train-only mean shown in the notebook output. If val/test data were included, the mean would differ (test set uses a different generator with potentially different parameter distributions).

---

### Finding 3: Per-sample spectral normalization -- no cross-sample leakage
- **Severity**: LOW (design note, not leakage)
- **File**: `/Users/michalszczesny/projects/hack4sages/exobiome_model/crossgen_hybrid_training.py`:278-280
- **Description**: Before global standardization, each spectrum is divided by its own mean across the wavelength axis. This is a strictly per-sample operation (`axis=1` with `keepdims=True`). No information flows between samples. The operation is applied to the full `noisy_spectra` array before split indices are extracted, but since each row is processed independently, this is mathematically identical to applying it after splitting.
- **Code snippet**:
```python
per_sample_mean = noisy_spectra.mean(axis=1, keepdims=True)
per_sample_mean = np.where(per_sample_mean == 0, 1.0, per_sample_mean)
spectra = (noisy_spectra / per_sample_mean)[:, None, :].astype(np.float32)
```
- **Impact**: No leakage. However, this normalization removes absolute transit depth magnitude, which carries physical information about atmospheric scale height. This is a modeling design choice, not a leakage concern.
- **Recommendation**: Document the rationale for mean-normalization in any publication. The absolute transit depth encodes atmospheric extent and is physically meaningful.

---

### Finding 4: Rebinning (spectres) -- per-sample, no cross-sample interaction
- **Severity**: PASS (no issue)
- **File**: `/Users/michalszczesny/projects/hack4sages/exobiome_model/crossgen_hybrid_training.py`:56-59, 264
- **Description**: The `spectres.spectres()` function rebins spectra from 218 to 44 wavelength bins using flux-conserving resampling. It operates along the wavelength axis independently for each sample. The shared wavelength grid is a fixed physical constant (Ariel instrument grid), not a data-derived quantity.
- **Code snippet**:
```python
wavelength_um, noisy_spectra = rebin_spectra(wavelength_um_raw, noisy_spectra_raw)
```
- **Impact**: None. Rebinning uses only the input and output wavelength grids, not any cross-sample statistics.

---

### Finding 5: No data augmentation or noise injection
- **Severity**: PASS (no issue)
- **File**: `/Users/michalszczesny/projects/hack4sages/exobiome_model/crossgen_hybrid_training.py`:686-828
- **Description**: The training loop contains no data augmentation, noise injection, mixup, or any transformation that could introduce cross-split information. The only randomness during training is:
  1. Batch index permutation via `batch_indices()` (line 597-604), which shuffles only training indices using a deterministic seed.
  2. Dropout layers in the neural network (standard regularization, not data-dependent).
- **Impact**: None. The absence of augmentation eliminates an entire class of potential leakage vectors.

---

### Finding 6: Test set generator isolation (Poseidon vs TauREx)
- **Severity**: PASS (no issue)
- **File**: `/Users/michalszczesny/projects/hack4sages/exobiome_model/crossgen_hybrid_training.py`:318-320
- **Description**: The notebook output confirms the split-generator correspondence: train and val are exclusively TauREx-generated spectra; test is exclusively Poseidon-generated spectra. This is a cross-generator evaluation, which is methodologically stronger than a random split because it tests generalization to spectra from a different radiative transfer code.
- **Impact**: None. The test set is completely isolated by construction.
- **Corroboration**: The large test-vs-val RMSE gap (3.46 vs ~1.54) is consistent with a genuine domain gap between generators, which is expected and not indicative of leakage. If leakage existed, the test performance would be closer to val performance, not worse.

---

### Finding 7: Early stopping on inner_val -- standard practice, no leakage
- **Severity**: PASS (no issue)
- **File**: `/Users/michalszczesny/projects/hack4sages/exobiome_model/crossgen_hybrid_training.py`:768-769, 800-814
- **Description**: Both the learning rate scheduler and early stopping use the inner_val loss. The val set influences model selection (which epoch's weights to keep) and learning rate schedule, but the model never sees val labels during gradient computation. Val evaluation happens in `torch.inference_mode()` (line 652) with `model.eval()` (line 646).
- **Code snippet**:
```python
inner_val_metrics = evaluate_split(model, data.inner_val, data.target_scaler, config.eval_batch_size)
scheduler.step(inner_val_metrics["loss"])
```
- **Impact**: None. This is standard ML practice. The test set (Poseidon) is evaluated only once, after training completes (line 946 in `run_training_experiment()`).
- **Note**: The best epoch was 29 out of 30 max epochs. This means early stopping (patience=6) never triggered; training ran nearly to completion. The model had not begun overfitting to the val set when training ended.

---

### Finding 8: Sequential train/val split within TauREx data
- **Severity**: LOW (advisory)
- **File**: Data-level concern (not a code bug)
- **Description**: The train/val split within TauREx data is sequential by `sample_id`: train gets `tau_000001` through `tau_037281`, val gets `tau_037282` through `tau_041423`. If the TauREx generator had systematic drift over generation order, this could create a distributional difference between train and val beyond what a random split would produce.
- **Impact**: Negligible. The notebook output shows that feature ranges for train and val overlap closely (e.g., `temperature_k` ranges [500, 1800] for both). TauREx generators typically use i.i.d. parameter sampling, so sequential ordering is functionally equivalent to a random 90/10 split.
- **Recommendation**: Document the sequential split method in any publication. Optionally, verify i.i.d. properties of the TauREx generation by checking for autocorrelation in parameter sequences.

---

### Finding 9: Dead `internal_val_fraction` config parameter
- **Severity**: MEDIUM (code quality, not leakage)
- **File**: `/Users/michalszczesny/projects/hack4sages/exobiome_model/crossgen_hybrid_training.py`:95
- **Description**: The `TrainingConfig` dataclass contains `internal_val_fraction: float = 0.10` which is never referenced anywhere in the codebase. The actual split comes from the `split` column in `labels.parquet`. This field is serialized to `config.json` (confirmed: `"internal_val_fraction": 0.1`), which could mislead a reviewer into thinking a runtime 90/10 random re-split was performed.
- **Code snippet**:
```python
internal_val_fraction: float = 0.10
```
- **Impact**: No data leakage, but a reproducibility and clarity concern. A reviewer seeing this in `config.json` may question whether the splits are truly pre-assigned.
- **Recommendation**: Remove this field from `TrainingConfig`, or rename it with a comment explaining it is a legacy/unused field. At minimum, do not serialize it to `config.json`.

---

### Finding 10: `log10_sigma_ppm` derived feature computed before splitting
- **Severity**: PASS (no issue)
- **File**: `/Users/michalszczesny/projects/hack4sages/exobiome_model/crossgen_hybrid_training.py`:268
- **Description**: The `log10_sigma_ppm` auxiliary feature is computed as `log10(clip(sigma_ppm))` and injected into the labels dataframe before splits are determined. This is a strictly per-sample monotonic transformation (`log10` of each sample's own noise level), so no cross-sample information is used.
- **Code snippet**:
```python
labels["log10_sigma_ppm"] = np.log10(np.clip(sigma_ppm, 1e-10, None))
```
- **Impact**: None. Per-sample transformations applied before splitting are safe by definition.

---

### Finding 11: `SpectralStandardizer.transform` shape handling
- **Severity**: PASS (no issue)
- **File**: `/Users/michalszczesny/projects/hack4sages/exobiome_model/crossgen_hybrid_training.py`:177-185, 327
- **Description**: `SpectralStandardizer.fit()` receives a 2D array `spectra_raw[inner_train_indices, 0, :]` of shape `[N_train, 44]` and computes per-bin statistics of shape `[44]`. The `transform()` method receives 3D arrays `spectra_raw[indices]` of shape `[N, 1, 44]` and applies `self.mean[None, None, :]` which broadcasts correctly to `[1, 1, 44]`. The fit and transform are dimensionally consistent and operate only along the sample axis during fitting, then broadcast correctly during transformation.
- **Impact**: None. The implementation is correct.

---

## Summary Table

| # | Vector | Verdict | Severity |
|---|--------|---------|----------|
| 1 | Split integrity / sample overlap | CLEAN | None |
| 2 | Standardizer fitting scope | CLEAN | None |
| 3 | Per-sample spectral normalization | CLEAN | LOW (design note) |
| 4 | Rebinning cross-sample leakage | CLEAN | None |
| 5 | Data augmentation / noise injection | CLEAN (none present) | None |
| 6 | Test set generator isolation | CLEAN | None |
| 7 | Early stopping / indirect val overfitting | CLEAN | None |
| 8 | Sequential train/val split | CLEAN | LOW (advisory) |
| 9 | Dead `internal_val_fraction` config | Not leakage | MEDIUM (code quality) |
| 10 | `log10_sigma_ppm` derived at runtime | CLEAN | None |
| 11 | SpectralStandardizer shape handling | CLEAN | None |

## Verdict: PASS

The data pipeline contains no data leakage. Train, validation, and test splits are strictly separated at every stage: file loading, per-sample normalization, global standardization (fit on train only), and model evaluation. The test set uses a different spectrum generator (Poseidon) and is never accessed during training or model selection. The only actionable finding is the dead `internal_val_fraction` config parameter (MEDIUM severity for code clarity, not for scientific validity), which should be removed or documented to avoid misleading reviewers.
