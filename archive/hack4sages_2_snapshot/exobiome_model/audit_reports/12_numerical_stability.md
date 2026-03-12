# Audit Report 12: Numerical Stability and Edge Cases

**Auditor:** Scientific Rigor Audit
**Date:** 2026-03-12
**Scope:** `crossgen_hybrid_training.py`, `outputs/model_crossgen_rebinned/scalers.json`
**Focus:** Division-by-zero, log domain errors, dtype mismatches, quantum circuit numerics, gradient stability, loss function edge cases

---

## Finding 1: Per-sample spectral mean normalization — division by zero guard is incomplete

**Severity: MEDIUM**
**File:** `crossgen_hybrid_training.py`, lines 278-280

```python
per_sample_mean = noisy_spectra.mean(axis=1, keepdims=True)
per_sample_mean = np.where(per_sample_mean == 0, 1.0, per_sample_mean)
spectra = (noisy_spectra / per_sample_mean)[:, None, :].astype(np.float32)
```

**Issue:** The guard only checks for an exactly zero mean. A near-zero mean (e.g., 1e-38) will pass the check and produce extreme normalized values that blow up downstream. Transit depth spectra are physically positive, so a true zero mean is extremely unlikely but a near-zero mean from corrupted or edge-case data is more plausible.

**Recommendation:** Replace exact equality check with a threshold:
```python
per_sample_mean = np.where(np.abs(per_sample_mean) < 1e-10, 1.0, per_sample_mean)
```

**Risk in practice:** LOW for the current dataset (transit depths are physically ~1e-4 to 1e-2), but the code lacks defensive depth.

---

## Finding 2: log10(sigma_ppm) clipping — adequate but asymmetric

**Severity: LOW**
**File:** `crossgen_hybrid_training.py`, line 268

```python
labels["log10_sigma_ppm"] = np.log10(np.clip(sigma_ppm, 1e-10, None))
```

**Issue:** The lower bound clip (1e-10) prevents log(0) and log(negative), which is correct. However, there is no upper bound clip. If sigma_ppm contains extremely large values (e.g., corrupted data with 1e30), `log10(1e30) = 30`, which after standardization could still be a large outlier. The `np.clip(..., None)` for the upper bound is intentional pass-through.

**Assessment:** Adequate for well-formed data. The 1e-10 floor is scientifically sound (noise below 1e-10 ppm is unphysical). No upper guard exists but log compression naturally limits the range.

---

## Finding 3: ArrayStandardizer / SpectralStandardizer zero-std guard

**Severity: LOW**
**File:** `crossgen_hybrid_training.py`, lines 157-158 and 180-181

```python
scale = np.where(scale == 0.0, 1.0, scale)
```

**Issue:** Like Finding 1, this uses exact equality. A scale of 1e-40 (effectively zero but not exactly) would pass the check and produce standardized values on the order of 1e40, causing NaN/Inf downstream.

**Assessment from scalers.json:** The saved scaler values show all scales are well above zero:
- Aux scales: 0.189 to 377.0
- Spectral scales: 0.0084 to 0.031
- Target scales: 2.88 to 2.90

These are all healthy ranges. No near-zero scales exist in practice.

**Recommendation:** For robustness, use a minimum threshold:
```python
scale = np.where(scale < 1e-7, 1.0, scale)
```

---

## Finding 4: float64 -> float32 precision loss in standardizer fitting

**Severity: LOW**
**File:** `crossgen_hybrid_training.py`, lines 155-159

```python
values64 = values.astype(np.float64, copy=False)
mean = values64.mean(axis=0)
scale = values64.std(axis=0)
scale = np.where(scale == 0.0, 1.0, scale)
return cls(mean=mean.astype(np.float32), scale=scale.astype(np.float32))
```

**Issue:** Mean and std are computed in float64 (good), but then truncated to float32 for storage and use. The transform at line 162 operates in float32. For the temperature feature (mean=1149.58, scale=377.0), the standardized value `(x - 1149.58) / 377.0` involves catastrophic cancellation when x is close to the mean. In float32, this yields ~6 significant digits, so the subtraction `x - 1149.58` loses roughly 3 digits of precision when `x ~ 1150`.

**Assessment:** float32 has ~7.2 decimal digits of precision. For `(1150.0 - 1149.584) / 377.0 = 0.001103`, the result retains only ~3 significant digits. This is acceptable for neural network training but is worth noting for reproducibility.

**Recommendation:** If higher precision is ever needed, perform the transform in float64 before casting back.

---

## Finding 5: SpectralStandardizer transform broadcasting shape assumption

**Severity: MEDIUM**
**File:** `crossgen_hybrid_training.py`, line 185

```python
def transform(self, spectra: np.ndarray) -> np.ndarray:
    return ((spectra - self.mean[None, None, :]) / self.scale[None, None, :]).astype(np.float32)
```

**Issue:** The transform hard-codes `[None, None, :]` broadcasting, assuming input shape is always `(N, 1, W)` (batch, channel, wavelengths). If someone calls this with a 2D array `(N, W)`, the broadcasting would silently produce incorrect results by treating the first dimension as a spurious batch-of-1 and wavelengths along the wrong axis. The `fit()` method accepts a 2D array (line 327: `spectra_raw[inner_train_indices, 0, :]`), creating an asymmetry between fit and transform interfaces.

**Assessment:** Currently the pipeline always passes 3D arrays to transform (line 331), so this is safe. But the API inconsistency (fit takes 2D, transform assumes 3D) is a latent bug.

---

## Finding 6: tanh saturation in FusionEncoder

**Severity: MEDIUM**
**File:** `crossgen_hybrid_training.py`, lines 408-417

```python
class FusionEncoder(nn.Module):
    def __init__(self, aux_dim: int, spec_dim: int, hidden_dim: int, out_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(aux_dim + spec_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(self, aux_feat: torch.Tensor, spectral_feat: torch.Tensor) -> torch.Tensor:
        fused = torch.cat([aux_feat, spectral_feat], dim=-1)
        return torch.tanh(self.net(fused)) * math.pi
```

**Issue:** The LayerNorm at line 412 outputs values roughly in [-2, 2] (unit variance, zero mean). Applying tanh to values of magnitude >2 enters the saturation regime where `dtanh/dx < 0.07`. Since tanh is applied after LayerNorm, the typical output magnitude is ~1, where tanh gradient is ~0.42. This is not catastrophic but creates a systematic gradient attenuation.

**Mitigating factor:** The LayerNorm ensures outputs stay in a reasonable range, preventing extreme saturation. The `* math.pi` scaling maps to [-pi, pi] for quantum rotation angles, which is the correct domain.

**Assessment:** The design is intentional (mapping to rotation angles). The gradient attenuation from tanh is moderate and compensated by the separate quantum learning rate. Not a bug, but a training efficiency concern.

---

## Finding 7: Quantum circuit angle numerical stability

**Severity: LOW**
**File:** `crossgen_hybrid_training.py`, lines 460-478

```python
for qubit in range(n_qubits):
    qml.RY(inputs[..., qubit], wires=qubit)
# ...
for qubit in range(n_qubits):
    qml.RY(weights[param_idx], wires=qubit)
# ...
for qubit in range(n_qubits):
    qml.RZ(weights[param_idx], wires=qubit)
# ...
for qubit in range(n_qubits):
    qml.CRX(weights[param_idx], wires=[qubit, (qubit + 1) % n_qubits])
```

**Issue:** Rotation gates (RY, RZ, CRX) take angle parameters. Due to the tanh * pi mapping from FusionEncoder, input angles are bounded to [-pi, pi]. Weight parameters are initialized at `0.5 * randn` (line 450), so they start in roughly [-1.5, 1.5] and can drift during training. All angles are periodic with period 2*pi, so extreme values wrap around without numerical issues.

For the adjoint differentiation method (line 459: `diff_method="adjoint"`), the gradient computation involves sin/cos of these angles, which are numerically stable for all finite float32 values.

**Assessment:** No numerical issues. Rotation gate angles are inherently periodic and well-conditioned.

---

## Finding 8: complex64 quantum state precision

**Severity: LOW**
**File:** `crossgen_hybrid_training.py`, lines 443-444

```python
if quantum_device_name.startswith("lightning."):
    device_kwargs["c_dtype"] = np.complex64
```

**Issue:** The quantum simulation uses complex64 (pairs of float32) rather than the default complex128. For a 12-qubit circuit, the state vector has 4096 amplitudes. With depth=2, the circuit applies ~84 gates (12 RY + 2*(12 RY + 12 CNOT + 12 RZ + 12 CRX) = 12 + 2*48 = 108 gates). Each gate multiplication introduces float32 rounding. Over 108 gates, accumulated error is bounded by roughly `108 * 2^-24 ~ 6.4e-6` per amplitude, which is acceptable for ML purposes.

**Assessment:** Acceptable. The reduced precision is a deliberate speed/memory tradeoff. For 12 qubits this is well within safe bounds. Would become concerning at >20 qubits or >10 depth.

---

## Finding 9: MSELoss with no reduction edge cases

**Severity: LOW**
**File:** `crossgen_hybrid_training.py`, lines 647, 690

```python
loss_fn = nn.MSELoss()
```

**Issue:** `nn.MSELoss()` uses default `reduction='mean'`. This is numerically stable for all finite inputs. If predictions become NaN (from upstream instability), the loss will also be NaN, which will propagate through backward() and corrupt all parameters.

There is no NaN/Inf check on model outputs or loss values. The training loop at line 752:
```python
batch_losses.append(float(loss.item()))
```
will silently store NaN values without raising an error.

**Recommendation:** Add a NaN guard after loss computation:
```python
if not torch.isfinite(loss):
    raise RuntimeError(f"Non-finite loss detected: {loss.item()}")
```

---

## Finding 10: Gradient clipping — separate norms for classical and quantum

**Severity: LOW**
**File:** `crossgen_hybrid_training.py`, lines 749-750

```python
torch.nn.utils.clip_grad_norm_(classical_params, config.gradient_clip_norm)  # 5.0
torch.nn.utils.clip_grad_norm_(quantum_params, 1.0)
```

**Assessment:** This is well-designed. Quantum parameters use a tighter clip (1.0) since quantum gradients are bounded by the Pauli expectation value range [-1, 1] and should not have large norms. Classical parameters use 5.0 which is standard. Both calls return the total norm before clipping, which is discarded (could be logged for diagnostics).

---

## Finding 11: AMP (mixed precision) interaction with quantum block

**Severity: MEDIUM**
**File:** `crossgen_hybrid_training.py`, lines 534-548

```python
def forward(self, aux: torch.Tensor, spectra: torch.Tensor) -> torch.Tensor:
    autocast_enabled = self.classical_device.type == "cuda" and self.amp_dtype is not None
    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=self.amp_dtype)
        if autocast_enabled
        else nullcontext()
    )
    with autocast_ctx:
        aux_feat = self.aux_encoder(aux)
        spectral_feat = self.spectral_encoder(spectra)
        latent = self.fusion_encoder(aux_feat, spectral_feat)

    latent = latent.float()
    quantum_feat = self.quantum_block(latent)
    head_in = torch.cat([quantum_feat, latent, aux_feat.float(), spectral_feat.float()], dim=-1)
    return self.head(head_in, quantum_feat)
```

**Issue:** The classical encoders run under AMP (float16/bfloat16 on CUDA), then `latent.float()` at line 545 casts back to float32 before the quantum block. This is correct. However, the `head` forward pass at line 548 runs outside autocast context in float32. This means the head never benefits from AMP acceleration.

More critically: if `aux_feat` or `spectral_feat` are in float16 under autocast, the `.float()` calls at line 547 correctly upcast them. But if autocast is disabled (CPU path), `.float()` is a no-op on already-float32 tensors. This is safe.

**Assessment:** The dtype handling between AMP and quantum is correct. No precision loss occurs at the boundary.

---

## Finding 12: HDF5 dtype handling

**Severity: LOW**
**File:** `crossgen_hybrid_training.py`, lines 259-262

```python
with h5py.File(data_root / "spectra.h5", "r") as handle:
    wavelength_um_raw = np.asarray(handle["wavelength_um"][:], dtype=np.float64)
    noisy_spectra_raw = np.asarray(handle["transit_depth_noisy"][:], dtype=np.float32)
    sigma_ppm = np.asarray(handle["sigma_ppm"][:], dtype=np.float32)
```

**Issue:** Explicit dtype casting is applied, which is correct. `np.asarray(..., dtype=...)` will convert from whatever the HDF5 dataset stores. If the HDF5 stores float64 spectra, truncation to float32 loses precision but this is intentional (spectra have noise at the ~1% level, so float32's 7 digits is more than sufficient).

Wavelengths are read as float64, which is correct since they serve as the rebinning grid where higher precision matters.

**Assessment:** Correct and well-handled.

---

## Finding 13: AdaptiveAvgPool1d with variable-length inputs

**Severity: LOW**
**File:** `crossgen_hybrid_training.py`, line 392

```python
nn.AdaptiveAvgPool1d(1),
```

**Issue:** `AdaptiveAvgPool1d(1)` reduces the temporal dimension to size 1 regardless of input length. This works correctly for any positive input length. For the current architecture, the input to this layer is determined by the Conv1d chain:
- Input: (B, 1, 44) [44 wavelength bins]
- After Conv1d(1, 32, k=7, p=3): (B, 32, 44)
- After Conv1d(32, 64, k=5, s=2, p=2): (B, 64, 22)
- After Conv1d(64, 64, k=3, p=1): (B, 64, 22)
- After AdaptiveAvgPool1d(1): (B, 64, 1)

The input length is always 44 (fixed by the Ariel wavelength grid), so there is no variable-length concern.

**Assessment:** No issue. Input length is fixed by the rebinning grid.

---

## Finding 14: Integer overflow in index calculations

**Severity: LOW**
**File:** `crossgen_hybrid_training.py`, lines 597-604

```python
def batch_indices(length: int, batch_size: int, seed: int, epoch: int, device: torch.device) -> Iterable[torch.Tensor]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + epoch)
    permutation = torch.randperm(length, generator=generator)
```

**Issue:** `seed + epoch` could theoretically overflow, but Python integers are arbitrary precision and `torch.Generator.manual_seed()` accepts any 64-bit integer. With seed=42 and max_epochs=30, the value is 72 — no concern.

`torch.randperm(length)` requires `length` to fit in a 64-bit integer. With 42,108 samples, this is safe. The indices tensor uses int64 by default.

**Assessment:** No overflow risk.

---

## Finding 15: Quantum gradient stability with adjoint differentiation

**Severity: LOW**
**File:** `crossgen_hybrid_training.py`, line 459

```python
@qml.qnode(device, interface="torch", diff_method="adjoint")
```

**Issue:** The adjoint differentiation method computes exact analytical gradients by replaying the circuit backwards. This is numerically stable for the circuit depth and qubit count used (12 qubits, depth 2). The gradients of Pauli-Z expectation values with respect to rotation angles are bounded by 1.0 in absolute value (parameter-shift rule equivalence), which prevents gradient explosion from the quantum side.

The main stability concern with adjoint is memory: it stores the full state vector during forward and replays during backward. For 12 qubits in complex64, this is 4096 * 8 bytes = 32 KB per sample — negligible.

**Assessment:** Numerically stable for this circuit configuration.

---

## Finding 16: No GradScaler for AMP training

**Severity: MEDIUM**
**File:** `crossgen_hybrid_training.py`, lines 686-828 (train_model function)

**Issue:** When `use_amp=True` and running on CUDA with float16, PyTorch best practices require using `torch.cuda.amp.GradScaler` to prevent gradient underflow in float16. The training loop does not use GradScaler:

```python
loss.backward()  # line 748 — no scaler.scale(loss).backward()
optimizer.step()  # line 751 — no scaler.step(optimizer)
```

If `amp_dtype` is `torch.bfloat16` (which is the case when bf16 is supported, line 244), GradScaler is not needed since bfloat16 has the same exponent range as float32. If it falls back to `torch.float16` (line 245), gradients can underflow to zero.

**Mitigating factor:** The head runs in float32 outside autocast, and the loss is computed in float32. Gradients flowing back through the head into the autocast region are in float32 and are only cast to float16 for intermediate activations. In practice, this is often sufficient, but it deviates from the recommended PyTorch AMP pattern.

**Recommendation:** Add GradScaler when amp_dtype is float16:
```python
scaler = torch.cuda.amp.GradScaler(enabled=(amp_dtype == torch.float16))
```

---

## Finding 17: Spectres rebinning NaN propagation

**Severity: MEDIUM**
**File:** `crossgen_hybrid_training.py`, lines 56-59

```python
def rebin_spectra(old_wavelengths: np.ndarray, spectra: np.ndarray,
                  new_wavelengths: np.ndarray = ARIEL_WAVELENGTH_GRID) -> tuple[np.ndarray, np.ndarray]:
    rebinned = spectres.spectres(new_wavelengths, old_wavelengths, spectra, verbose=False)
    return new_wavelengths.astype(np.float32), rebinned.astype(np.float32)
```

**Issue:** The `spectres` library produces NaN for output wavelength bins that fall outside the range of the input wavelength grid. The Ariel grid spans 0.95-4.91 um. If the input grid doesn't fully cover this range, edge bins will be NaN. There is no NaN check after rebinning.

NaN values would propagate silently through the entire pipeline: normalization, standardization, model forward pass, and loss computation.

**Recommendation:** Add a NaN check:
```python
rebinned = spectres.spectres(new_wavelengths, old_wavelengths, spectra, verbose=False)
if np.any(np.isnan(rebinned)):
    n_nan = np.isnan(rebinned).sum()
    raise ValueError(f"Rebinning produced {n_nan} NaN values. Check wavelength coverage.")
```

---

## Finding 18: No NaN/Inf validation on loaded data

**Severity: MEDIUM**
**File:** `crossgen_hybrid_training.py`, lines 256-270

**Issue:** The `load_crossgen_dataset` function reads from HDF5 and parquet files without validating that the loaded arrays contain finite values. Corrupted files, incomplete writes, or upstream pipeline errors could inject NaN/Inf values that silently propagate through the entire training pipeline.

**Recommendation:** Add validation after loading:
```python
assert np.all(np.isfinite(noisy_spectra_raw)), "Non-finite values in spectra"
assert np.all(np.isfinite(sigma_ppm)), "Non-finite values in sigma_ppm"
for col in SAFE_AUX_FEATURE_COLS + TARGET_COLS:
    if col in labels.columns:
        assert labels[col].notna().all(), f"NaN values in {col}"
```

---

## Finding 19: Evaluation metric RMSE with sqrt of near-zero values

**Severity: LOW**
**File:** `crossgen_hybrid_training.py`, line 666

```python
rmse_orig = np.sqrt(np.mean((pred_orig - true_orig) ** 2, axis=0))
```

**Issue:** If predictions exactly match targets (e.g., in overfitting), the squared difference is 0 and `sqrt(0) = 0`, which is fine. The mean is always non-negative since it's a mean of squared values. No numerical issue here.

**Assessment:** Safe.

---

## Finding 20: Residual connection in AtmosphereHead — dimension mismatch risk

**Severity: LOW**
**File:** `crossgen_hybrid_training.py`, lines 489-503

```python
class AtmosphereHead(nn.Module):
    def __init__(self, in_dim: int, latent_dim: int, hidden_dim: int, n_targets: int, dropout: float) -> None:
        # ...
        self.residual = nn.Linear(latent_dim, n_targets)

    def forward(self, head_in: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
        return self.mlp(head_in) + self.residual(latent)
```

**Issue:** The residual skip connection adds `mlp(head_in)` (shape: [B, 5]) and `residual(latent)` (shape: [B, 5]). Both produce the same output shape by construction (n_targets=5). No dimension mismatch is possible given the current wiring.

**Assessment:** Safe by construction.

---

## Summary Table

| # | Finding | Severity | Exploitable in Practice? |
|---|---------|----------|------------------------|
| 1 | Per-sample mean normalization: exact zero check only | MEDIUM | No (transit depths are positive) |
| 2 | log10(sigma_ppm) clip: no upper bound | LOW | No (log compresses range) |
| 3 | Standardizer zero-std: exact equality check | LOW | No (verified from scalers.json) |
| 4 | float64->float32 precision loss in scaler | LOW | Marginal effect on temperature feature |
| 5 | SpectralStandardizer fit/transform shape asymmetry | MEDIUM | No (pipeline always passes 3D) |
| 6 | tanh saturation in FusionEncoder | MEDIUM | Moderate gradient attenuation |
| 7 | Quantum rotation angle numerics | LOW | No (periodic, inherently stable) |
| 8 | complex64 quantum state precision | LOW | Acceptable for 12 qubits |
| 9 | No NaN guard on loss | LOW | Only triggers if upstream fails |
| 10 | Gradient clipping well-designed | LOW | No issue |
| 11 | AMP/quantum dtype boundary | MEDIUM | Correctly handled |
| 12 | HDF5 dtype casting | LOW | Correct |
| 13 | AdaptiveAvgPool1d fixed input | LOW | No issue |
| 14 | Integer overflow in index math | LOW | No risk |
| 15 | Adjoint differentiation stability | LOW | Stable for this circuit |
| 16 | Missing GradScaler for float16 AMP | MEDIUM | Only if bf16 unsupported |
| 17 | Spectres rebinning NaN propagation | MEDIUM | If wavelength grids don't overlap |
| 18 | No data validation after loading | MEDIUM | If source data corrupted |
| 19 | RMSE sqrt edge case | LOW | Safe |
| 20 | Residual connection dimensions | LOW | Safe by construction |

---

## Critical Findings Count

- **CRITICAL:** 0
- **HIGH:** 0
- **MEDIUM:** 6 (Findings 1, 5, 6, 16, 17, 18)
- **LOW:** 14

---

## Verdict: **PASS**

**Rationale:** No critical or high-severity numerical stability issues were found. The codebase demonstrates good practices in several areas: explicit dtype casting from HDF5, float64 computation in scaler fitting, explicit `.float()` casting before the quantum block, separate gradient clip norms for classical and quantum parameters, and zero-std guards in standardizers.

The six MEDIUM findings are all latent risks that do not manifest with the current dataset and configuration:
1. The exact-zero checks (Findings 1, 3) are adequate because transit depths and feature variances are physically constrained to be well above zero.
2. The tanh saturation (Finding 6) is an intentional design choice for mapping to rotation angles, mitigated by LayerNorm.
3. The missing GradScaler (Finding 16) only matters for float16 AMP, and the code preferentially uses bfloat16.
4. The NaN propagation risks (Findings 17, 18) are contingent on corrupted input data.

For a hackathon codebase, the numerical hygiene is above average. For production deployment, the MEDIUM findings should be addressed with defensive checks (NaN validation, threshold-based zero guards, GradScaler).
