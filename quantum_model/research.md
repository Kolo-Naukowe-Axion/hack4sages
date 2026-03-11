# Hybrid Quantum Model — Research & Diagnosis

## Problem
Model shows very high loss during training and does not learn. Dataset: `baseline/crossgenn/` (42,108 samples, 218 wavelength bins, 5 VMR targets).

**Note**: This is our own VQC-based hybrid architecture, not a reproduction of Vetrano et al. 2025 (QELM). Differences from the paper are intentional — we compare approaches.

---

## Root Causes (ranked by severity)

### 1. CRITICAL — Spectral standardization destroys molecular signal

**Lines**: `SpectralStandardizer` (155-171), `build_raw_arrays` (276)

The `SpectralStandardizer` computes per-wavelength-bin z-scores across the dataset. The dominant variance in transit depth comes from planet/star size ratios (`(Rp/Rs)^2`), not from molecular absorption features.

- Raw spectra vs targets: **max correlation = 0.016** (essentially zero)
- Per-sample normalized spectra vs targets: **max correlation = 0.602**

The molecular absorption signal is ~1-3% of the transit depth. After per-bin standardization, it's drowned out by the baseline variation. The CNN baseline trained on the same data has the same problem — it also predicts the dataset mean for every sample.

**Fix**: Divide each spectrum by its own mean before global standardization in `build_raw_arrays()`.

### 2. HIGH — Missing skip connections + QuantumProjection (diagram vs code mismatch)

**Lines**: `HybridAtmosphereModel.forward` (576-590), `build_model` (604-605)

`model.png` shows AuxEncoder and SpectralEncoder outputs feeding directly into PredictionHead. The code only passes `[quantum_feat, latent]` to the head — the 32+32=64 dim encoder features are discarded after fusion. All information must squeeze through a 12-dim FusionEncoder bottleneck with `tanh*π` saturation.

The diagram also shows a "QuantumProjection" block between FusionBlock and VQC that doesn't exist in code — the FusionEncoder outputs 12-dim directly into QuantumBlock.

Head `in_dim` is set to `qnn_qubits * 2 = 24`. With skip connections it should be `12 + 12 + 32 + 32 = 88`.

**Note**: The diagram may be outdated (it already shows "7AtmosphereTargets" vs 5 in code). Need to decide whether to implement the diagram's architecture or treat the current code as the intended design. Either way, the 12-dim bottleneck without skip connections severely limits capacity.

**Fix**: Add skip connections (concatenate `aux_feat` and `spectral_feat` into head input, update `head.in_dim`). Optionally add a QuantumProjection layer if a different embedding dim is desired for the quantum input.

### 3. MEDIUM — Missing GradScaler with AMP (hardware-dependent)

**Lines**: config (101), forward (577-583), training loop (779-785)

`use_amp=True` (default) enables autocast in the forward pass, but the training loop never uses `torch.cuda.amp.GradScaler`. On GPUs without bfloat16 (float16 fallback), gradients can underflow to zero. However, RTX 4090 supports bfloat16 via `resolve_amp_dtype()` (lines 226-231), and bfloat16 has the same exponent range as float32 — no GradScaler needed. **This is only a real issue on older GPUs using float16.**

**Fix**: Add GradScaler for float16 path, or just disable AMP on non-bfloat16 hardware.

### 4. HIGH — Quantum weight initialization too small

**Line**: 493

`0.01 * torch.randn(...)` — all rotation angles ~0.01 radians. RY(0.01) ≈ identity. The circuit does almost nothing at startup, gradients w.r.t. weights are proportional to sin(θ) ≈ 0.01 — vanishingly small. The quantum block starts as dead weight and has no gradient signal to escape.

**Fix**: Initialize with `0.5 * torch.randn(...)` or `Uniform(0, 2π)`.

### 5. HIGH — FusionEncoder `tanh*π` saturates and kills gradients

**Line**: 460

`torch.tanh(self.net(fused)) * math.pi` — tanh saturates for inputs with |x| > 2. No LayerNorm before tanh. Early in training with random weights, linear outputs frequently overshoot, pushing tanh into saturation where gradient ≈ 0. This blocks gradient flow from quantum block back through classical encoders.

**Fix**: Add LayerNorm before tanh, or replace with softer bounding (e.g., `π * sigmoid(x) * 2 - π`).

### 6. HIGH — Residual bypass makes quantum block easy to ignore

**Lines**: `AtmosphereHead` (532-546), forward (590)

`AtmosphereHead` has a `nn.Linear(latent, targets)` residual that provides a direct classical path from FusionEncoder output to predictions, bypassing the quantum block. The optimizer can find this shortcut early, making quantum gradients negligible.

This isn't necessarily wrong architecturally — residual connections help training. But combined with the near-zero quantum init (#4), the quantum block never gets a chance to contribute.

**Fix**: Either route the residual through `quantum_feat` instead of `latent`, or temporarily remove it during initial experiments to force the quantum path to learn.

### 7. MEDIUM — LR scheduler too aggressive

**Lines**: 84-85, 742-747

`scheduler_patience=2` halves LR after just 2 non-improving epochs. By early stopping (patience=6), the quantum LR can drop from 6e-4 → 7.5e-5 through 3 reductions. Quantum learning may be killed before it starts.

**Fix**: Increase scheduler patience to 4-5.

### 8. MEDIUM — Gradient clipping tight for classical params

**Lines**: 783-784, config (89)

`clip_norm=1.0` applied separately to classical and quantum parameter groups. The classical set has many more parameters, so its L2 gradient norm is naturally larger and clips frequently.

**Fix**: Increase to 5.0 for classical, keep 1.0 for quantum.

### 9. MEDIUM — Batch size large for quantum path

**Line**: config (80)

`train_batch_size=1024` with adjoint differentiation means sequential circuit evaluations per sample (no native batching in adjoint mode). This is both slow (2048 circuit runs per step) and averages out weak quantum gradient signals.

**Fix**: Reduce to 128-256 for stronger per-step gradient signal and faster iteration.

### 10. LOW — Notebook import path mismatch

**File**: `model_quant_sketch.ipynb`, config-code cell

The notebook imports `from models.crossgen_hybrid_training` but the directory is `quantum_model/`, not `models/`. The fallback check tests `PROJECT_ROOT.name == "models"` which doesn't match.

**Fix**: Fix the fallback check to also handle `"quantum_model"`, or change import to `from crossgen_hybrid_training import ...`.

### 11. LOW — Diagram shows 7 targets, code has 5

`model.png` labels output as "7AtmosphereTargets" but `TARGET_COLS` has 5. Diagram is outdated.

---

## Dataset Facts

| Property | Value |
|---|---|
| Total samples | 42,108 |
| Wavelength bins | 218 (0.60–5.22 μm, R~100) |
| Splits | tau/train=37,281, tau/val=4,142, poseidon/test=685 |
| Aux features | 5: planet_radius, log_g, temperature, star_radius, log10_sigma_ppm |
| Targets | 5: log10_vmr_{h2o, co2, co, ch4, nh3} |
| Target range | [-12.0, -2.0] (bimodal: present vs absent boundary at ~-8) |
| Target cross-correlation | < 0.01 (nearly independent) |

## Quantum Circuit Facts

| Property | Value |
|---|---|
| Qubits | 12 |
| Depth | 2 (→ 1 variational block) |
| Trainable params | 36 (12 RY + 12 RZ + 12 CRX) |
| Ansatz | RY encoding → [RY, CNOT ring, RZ, CRX ring] × 1 → PauliZ measurements |
| Backend | lightning.qubit (CPU) or lightning.gpu (CUDA) |
| Diff method | adjoint (sequential per sample) |

---

## Recommended Fix Priority

1. **Per-sample spectral normalization** — without this, the model cannot see molecular features at all
2. **Quantum weight init** — one-line fix, high impact on quantum gradient signal
3. **Add skip connections** — implement architecture from model.png (or decide it's outdated and keep bottleneck but widen it)
4. **Soften FusionEncoder activation** — add LayerNorm or use softer bounding
5. **Adjust residual connection** — route through quantum_feat or remove temporarily
6. **Tune hyperparameters** — scheduler patience↑, gradient clip↑ for classical, batch size↓
7. **Fix AMP / GradScaler** — only needed on non-bfloat16 GPUs
