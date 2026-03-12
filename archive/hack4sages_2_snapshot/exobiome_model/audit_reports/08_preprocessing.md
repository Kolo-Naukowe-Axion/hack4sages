# Audit Report 08: Data Preprocessing Pipeline

**Scope**: Full transform chain from raw HDF5 to model input tensors, including spectral rebinning, per-sample normalization, standardizers, scaler serialization, and inference reproducibility.

**Files audited**:
- `/Users/michalszczesny/projects/hack4sages/exobiome_model/crossgen_hybrid_training.py` (lines 1-362, 639-686)
- `/Users/michalszczesny/projects/hack4sages/exobiome_model/outputs/model_crossgen_rebinned/scalers.json`
- `/Users/michalszczesny/projects/hack4sages/exobiome_model/outputs/model_crossgen_rebinned/config.json`

---

## 1. Full Transform Sequence (Traced)

The exact pipeline from raw data to model input:

```
HDF5 load
  wavelength_um_raw: float64 (218,)
  noisy_spectra_raw: float32 (42108, 218)
  sigma_ppm:         float32 (42108,)
    |
    v
rebin_spectra() via SpectRes           [line 58]
  wavelength_um: float32 (44,)
  noisy_spectra: float32 (42108, 44)
    |
    v
log10_sigma_ppm = log10(clip(sigma_ppm, 1e-10, None))   [line 268]
  -> added to labels DataFrame
    |
    v
build_raw_arrays()                     [lines 273-281]
  aux_raw = labels[5 cols]             -> float32 (42108, 5)
  per_sample_mean = spectra.mean(axis=1, keepdims=True)  -> (42108, 1)
  per_sample_mean = where(==0, 1.0, per_sample_mean)
  spectra_raw = (spectra / per_sample_mean)[:, None, :]  -> float32 (42108, 1, 44)
    |
    v
Split by labels["split"]: train/val/test integer indices  [lines 318-320]
    |
    v
Fit scalers on TRAIN ONLY:            [lines 325-327]
  aux_scaler     = ArrayStandardizer.fit(aux_raw[train])           2D (N, 5)
  target_scaler  = ArrayStandardizer.fit(targets_raw[train])       2D (N, 5)
  spectral_scaler = SpectralStandardizer.fit(spectra_raw[train, 0, :])  2D (N, 44)
    |
    v
Transform all splits with same scalers [lines 329-337]
  aux:     (x - mean) / scale           -> float32
  spectra: (x - mean[None,None,:]) / scale[None,None,:]  -> float32  (3D broadcast)
  targets: (x - mean) / scale           -> float32
    |
    v
Wrap in SplitTensors (torch.Tensor)    [lines 297-310]
```

---

## 2. Findings

### FINDING 2.1: No NaN/Inf validation after SpectRes rebinning

**Severity**: MEDIUM

**File**: `/Users/michalszczesny/projects/hack4sages/exobiome_model/crossgen_hybrid_training.py`, lines 56-59

```python
def rebin_spectra(old_wavelengths: np.ndarray, spectra: np.ndarray,
                  new_wavelengths: np.ndarray = ARIEL_WAVELENGTH_GRID) -> tuple[np.ndarray, np.ndarray]:
    rebinned = spectres.spectres(new_wavelengths, old_wavelengths, spectra, verbose=False)
    return new_wavelengths.astype(np.float32), rebinned.astype(np.float32)
```

SpectRes produces NaN for output bins whose wavelength range falls outside the input grid. There is no `np.isnan` / `np.isinf` check after rebinning. If the raw 218-bin grid does not fully cover the ARIEL grid range (0.95-4.91 um), edge bins will silently become NaN, propagating through all downstream normalization and training.

The successful training run empirically indicates this did not occur with the current dataset, but the code has no guard against it for future data.

**Recommendation**: Add a post-rebin assertion:
```python
assert np.all(np.isfinite(rebinned)), "SpectRes produced non-finite values in rebinned spectra"
```

---

### FINDING 2.2: No NaN/Inf validation anywhere in the pipeline

**Severity**: MEDIUM

**File**: `/Users/michalszczesny/projects/hack4sages/exobiome_model/crossgen_hybrid_training.py`, lines 255-362

The entire `prepare_data()` function has zero NaN/Inf checks at any stage: after HDF5 load, after rebinning, after per-sample normalization, after standardization, and after tensor conversion. A single NaN in the raw data would silently propagate through the entire pipeline and corrupt training loss without any diagnostic message.

The only `np.clip` call in the pipeline is at line 268 for `sigma_ppm` (preventing `log10(0)`), which shows awareness of the issue in one specific case but not generally.

**Recommendation**: Add a validation function called at key pipeline stages:
```python
def assert_finite(arr, name):
    if not np.all(np.isfinite(arr)):
        n_bad = np.sum(~np.isfinite(arr))
        raise ValueError(f"{name} contains {n_bad} non-finite values")
```

---

### FINDING 2.3: SpectralStandardizer lacks `inverse_transform()`

**Severity**: LOW

**File**: `/Users/michalszczesny/projects/hack4sages/exobiome_model/crossgen_hybrid_training.py`, lines 171-188

`ArrayStandardizer` has both `transform()` and `inverse_transform()` (lines 161-165). `SpectralStandardizer` only has `transform()` (line 184). There is no way to invert the spectral preprocessing to recover physical transit depth values from model internals.

This is acceptable because the model predicts target gas concentrations (not spectra), so spectral inverse transform is never needed during evaluation. However, it limits interpretability analysis (e.g., gradient-based feature attribution in physical units).

```python
# ArrayStandardizer -- has inverse_transform (line 164)
def inverse_transform(self, values: np.ndarray) -> np.ndarray:
    return (values * self.scale + self.mean).astype(np.float32)

# SpectralStandardizer -- missing inverse_transform
```

**Impact**: Does not affect training or evaluation. Would block spectral interpretability workflows.

---

### FINDING 2.4: Per-sample normalization does not handle negative transit depths

**Severity**: LOW

**File**: `/Users/michalszczesny/projects/hack4sages/exobiome_model/crossgen_hybrid_training.py`, lines 278-280

```python
per_sample_mean = noisy_spectra.mean(axis=1, keepdims=True)
per_sample_mean = np.where(per_sample_mean == 0, 1.0, per_sample_mean)
spectra = (noisy_spectra / per_sample_mean)[:, None, :].astype(np.float32)
```

The code handles mean=0 but not negative means. Noisy transit depths can theoretically go negative (especially at low SNR), which would make `per_sample_mean` negative. Dividing by a negative mean flips all spectral features, creating an inverted spectrum that the SpectralStandardizer would treat as an extreme outlier.

With the current dataset (transit depths are positive physical quantities with SNR >10 based on the sigma_ppm range of ~20-100 ppm), this is extremely unlikely. But it is an unguarded edge case.

**Recommendation**: Use `np.abs(per_sample_mean)` or `np.clip(per_sample_mean, eps, None)`.

---

### FINDING 2.5: Double normalization is intentional and correct

**Severity**: None (informational)

**File**: `/Users/michalszczesny/projects/hack4sages/exobiome_model/crossgen_hybrid_training.py`, lines 278-280 and 177-185

The pipeline applies two levels of spectral normalization:
1. **Per-sample**: `spectrum / mean(spectrum)` -- removes absolute transit depth scale, centers values around 1.0
2. **Per-bin (global)**: `SpectralStandardizer` subtracts per-bin training mean and divides by per-bin training std

This is a valid two-stage approach:
- Stage 1 removes the dominant source of inter-sample variance (absolute depth, which is a function of Rp/Rs already captured by aux features). Verified: spectral scaler means are all in [0.975, 1.012], confirming values are tightly centered around 1.0 after stage 1.
- Stage 2 standardizes the remaining per-bin variance (molecular absorption signatures), giving each wavelength bin comparable scale for the neural network.

The resulting spectral scale values (0.008-0.031) confirm that after per-sample normalization, the residual variance is small and physically meaningful (absorption features).

---

### FINDING 2.6: Scaler serialization is faithful

**Severity**: None (informational)

**File**: `/Users/michalszczesny/projects/hack4sages/exobiome_model/crossgen_hybrid_training.py`, lines 932-939, and `/Users/michalszczesny/projects/hack4sages/exobiome_model/outputs/model_crossgen_rebinned/scalers.json`

The serialization chain:
1. Scalers computed in float64, cast to float32 (lines 155-159, 178-182)
2. `state_dict()` calls `.tolist()` which converts float32 to Python float (float64 precision)
3. JSON stores full float64 precision
4. Loading from JSON gives float64 values

Verified empirically: round-tripping through float64->float32->float64 produces zero error for all saved scaler values. The float32 values in the scaler are exactly representable in float64 JSON, so no precision is lost.

Dimensions confirmed:
- `aux_scaler`: mean(5), scale(5) -- matches 5 aux features
- `target_scaler`: mean(5), scale(5) -- matches 5 target columns
- `spectral_scaler`: mean(44), scale(44) -- matches 44 ARIEL bins

---

### FINDING 2.7: ArrayStandardizer handles std=0 correctly

**Severity**: None (informational)

**File**: `/Users/michalszczesny/projects/hack4sages/exobiome_model/crossgen_hybrid_training.py`, lines 154-158

```python
@classmethod
def fit(cls, values: np.ndarray) -> "ArrayStandardizer":
    values64 = values.astype(np.float64, copy=False)
    mean = values64.mean(axis=0)
    scale = values64.std(axis=0)
    scale = np.where(scale == 0.0, 1.0, scale)
    return cls(mean=mean.astype(np.float32), scale=scale.astype(np.float32))
```

Both `ArrayStandardizer` and `SpectralStandardizer` compute in float64 for numerical stability, then cast to float32 for storage. The `scale = np.where(scale == 0.0, 1.0, scale)` guard prevents division by zero. If a feature has zero variance, dividing by 1.0 means the feature just gets mean-subtracted, which is correct behavior.

Verified: none of the saved scale values are 1.0 (all are >> 0), confirming no zero-variance features were encountered.

---

### FINDING 2.8: Train/val/test receive identical transforms

**Severity**: None (informational)

**File**: `/Users/michalszczesny/projects/hack4sages/exobiome_model/crossgen_hybrid_training.py`, lines 329-337

```python
def transform(indices: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    aux_scaled = aux_scaler.transform(aux_raw[indices])
    spectra_scaled = spectral_scaler.transform(spectra_raw[indices])
    targets_scaled = target_scaler.transform(targets_raw[indices])
    return aux_scaled, spectra_scaled, targets_scaled

train_aux, train_spectra, train_targets = transform(inner_train_indices)
inner_val_aux, inner_val_spectra, inner_val_targets = transform(inner_val_indices)
test_aux, test_spectra, test_targets = transform(test_indices)
```

All three splits pass through the same `transform()` closure using the same scaler objects (fitted on train only). This is correct -- no data leakage from val/test into scaler fitting.

---

### FINDING 2.9: ARIEL wavelength grid is valid

**Severity**: None (informational)

**File**: `/Users/michalszczesny/projects/hack4sages/exobiome_model/crossgen_hybrid_training.py`, lines 43-53

The 44-element ARIEL wavelength grid was verified:
- Strictly monotonically increasing: YES
- Range: 0.95 - 4.91 um (covers Ariel Tier 1 AIRS channels)
- No duplicate values
- Minimum bin spacing: 0.040 um (at the densely sampled NIR end)
- Maximum bin spacing: 0.312 um (gap at 3.72-4.03 um, between Ariel NIR and MIR channels)

The large gap at index 39-40 (3.72 to 4.03 um) reflects a real gap in Ariel instrument coverage. SpectRes handles non-uniform grids correctly.

---

### FINDING 2.10: Dead config parameter `internal_val_fraction`

**Severity**: LOW

**File**: `/Users/michalszczesny/projects/hack4sages/exobiome_model/crossgen_hybrid_training.py`, line 95

```python
internal_val_fraction: float = 0.10
```

This config parameter is defined but never referenced in the codebase. The validation split comes entirely from the `labels["split"]` column in the parquet file (line 319). This is misleading -- a user might change this value expecting it to affect the val split size, but it would have no effect.

**Recommendation**: Remove the unused parameter or add a comment marking it as deprecated.

---

### FINDING 2.11: No inference pipeline to consume saved scalers

**Severity**: MEDIUM

**File**: `/Users/michalszczesny/projects/hack4sages/exobiome_model/outputs/model_crossgen_rebinned/scalers.json`

The scalers are correctly saved to `scalers.json`, but there is no corresponding deserialization code. Neither `ArrayStandardizer` nor `SpectralStandardizer` has a `from_dict()` or `load()` class method. There is no inference script that loads the model + scalers to make predictions on new data.

An inference user would need to manually reconstruct the scalers:
```python
import json, numpy as np
with open("scalers.json") as f:
    d = json.load(f)
aux_scaler = ArrayStandardizer(
    mean=np.array(d["aux_scaler"]["mean"], dtype=np.float32),
    scale=np.array(d["aux_scaler"]["scale"], dtype=np.float32),
)
```

This is error-prone (e.g., forgetting the per-sample normalization step, using wrong dtype, applying transforms in wrong order).

**Recommendation**: Add `from_dict()` classmethods to both standardizer classes, and create a minimal inference function that applies the full pipeline (rebin, per-sample norm, standardize, predict, inverse-transform targets).

---

### FINDING 2.12: Per-sample normalization loses absolute transit depth -- compensated by aux features

**Severity**: LOW

**File**: `/Users/michalszczesny/projects/hack4sages/exobiome_model/crossgen_hybrid_training.py`, lines 278-280

Dividing each spectrum by its mean removes the absolute transit depth scale. For a spectrum `S`, the normalized version `S/mean(S)` is identical for `S` and `k*S` for any scalar `k > 0`. This means the model cannot distinguish deep-atmosphere signals from shallow ones based on spectra alone.

This information IS available through the aux features: `planet_radius_rjup` and `star_radius_rsun` together determine the baseline transit depth (Rp/Rs)^2. The `FusionEncoder` concatenates aux and spectral features, so the model can in principle learn to combine them.

This is a deliberate and reasonable design choice, not a bug, but users should be aware that spectral predictions depend on the aux features providing accurate planetary/stellar parameters.

---

### FINDING 2.13: Rebinning applied to ALL data before splitting

**Severity**: None (informational)

**File**: `/Users/michalszczesny/projects/hack4sages/exobiome_model/crossgen_hybrid_training.py`, lines 264, 315

```python
# In load_crossgen_dataset (line 264):
wavelength_um, noisy_spectra = rebin_spectra(wavelength_um_raw, noisy_spectra_raw)

# In prepare_data (line 315):
aux_raw, spectra_raw = build_raw_arrays(labels, noisy_spectra)
```

Rebinning and per-sample normalization are applied to the entire dataset before splitting. This is NOT a data leakage issue because:
- `rebin_spectra` is a per-sample operation (each spectrum is rebinned independently using only the wavelength grid)
- Per-sample normalization uses only each sample's own mean

No cross-sample statistics are computed before the train/val/test split. Scaler fitting happens only after splitting (line 325-327).

---

## 3. Invertibility Assessment

| Component | Invertible? | Notes |
|-----------|-------------|-------|
| SpectRes rebinning | No | Lossy downsampling (218 -> 44 bins), information destroyed |
| Per-sample norm (spectrum/mean) | Yes* | *Only if you save the per-sample mean; it is not saved |
| SpectralStandardizer | Yes** | **No `inverse_transform()` method exists, but mathematically `x * scale + mean` |
| ArrayStandardizer (aux) | Yes | `inverse_transform()` implemented and used |
| ArrayStandardizer (targets) | Yes | `inverse_transform()` implemented; used in `evaluate_split()` |

The target pipeline is fully invertible (model predictions are inverse-transformed to physical log10 VMR units in `evaluate_split()`, line 664). The spectral pipeline is intentionally non-invertible (lossy compression is a feature, not a bug).

---

## 4. Summary Table

| # | Finding | Severity | Status |
|---|---------|----------|--------|
| 2.1 | No NaN/Inf check after SpectRes rebinning | MEDIUM | Open |
| 2.2 | No NaN/Inf validation anywhere in pipeline | MEDIUM | Open |
| 2.3 | SpectralStandardizer lacks inverse_transform() | LOW | Open |
| 2.4 | Negative transit depth not guarded | LOW | Open |
| 2.5 | Double normalization is correct | None | OK |
| 2.6 | Scaler serialization is faithful | None | OK |
| 2.7 | std=0 handling is correct | None | OK |
| 2.8 | Train/val/test get identical transforms | None | OK |
| 2.9 | ARIEL wavelength grid is valid | None | OK |
| 2.10 | Dead config param `internal_val_fraction` | LOW | Open |
| 2.11 | No inference deserialization for scalers | MEDIUM | Open |
| 2.12 | Per-sample norm loses absolute depth (compensated) | LOW | OK |
| 2.13 | Rebinning before split is safe | None | OK |

---

## 5. Verdict

**PASS** -- with reservations.

The preprocessing pipeline is mathematically correct and produces scientifically valid normalized inputs. Scalers are fit exclusively on training data, serialized without precision loss, and applied identically to all splits. The double normalization strategy (per-sample then per-bin) is a sound design choice for this domain.

The reservations are:
1. **Defensive programming**: Zero NaN/Inf guards mean a corrupted input file would produce silent garbage (Findings 2.1, 2.2).
2. **Inference gap**: No deserialization code or inference pipeline exists, making it error-prone for anyone trying to deploy the model on new observations (Finding 2.11).
3. **Code hygiene**: Dead parameter and missing inverse method are minor but could mislead users (Findings 2.3, 2.10).

None of the findings indicate correctness bugs in the current training pipeline with the current dataset.
