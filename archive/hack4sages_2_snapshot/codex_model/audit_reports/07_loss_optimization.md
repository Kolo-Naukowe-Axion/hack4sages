# Audit Report 07: Loss Function and Optimization

**Auditor**: Claude Opus 4.6 (automated scientific rigor audit)
**Date**: 2026-03-12
**Scope**: Loss function choice, optimizer configuration, learning rate scheduling, gradient clipping, early stopping, NaN guards, and training stability for the hybrid quantum-classical biosignature detection model.

**Files reviewed**:
- `/Users/michalszczesny/projects/hack4sages/codex_model/crossgen_hybrid_training.py` (987 lines, every line read)
- `/Users/michalszczesny/projects/hack4sages/codex_model/outputs/model_crossgen_rebinned/config.json`
- `/Users/michalszczesny/projects/hack4sages/codex_model/outputs/model_crossgen_rebinned/history.csv` (30 epochs)
- `/Users/michalszczesny/projects/hack4sages/codex_model/outputs/model_crossgen_rebinned/run_summary.json`
- `/Users/michalszczesny/projects/hack4sages/codex_model/outputs/model_crossgen_rebinned/scalers.json`
- `/Users/michalszczesny/projects/hack4sages/codex_model/outputs/model_crossgen_rebinned/inner_val_metrics.json`
- `/Users/michalszczesny/projects/hack4sages/codex_model/outputs/model_crossgen_rebinned/test_metrics.json`
- `/Users/michalszczesny/projects/hack4sages/codex_model/outputs/model_crossgen_rebinned/preflight.json`

---

## Executive Summary

The loss function and optimization pipeline is fundamentally sound but has three issues worth addressing. MSE on standardized targets is a defensible choice given the nearly identical target distributions. The dual-optimizer setup with separate learning rates for classical and quantum parameters follows standard practice. However, the learning rate scheduler never fired during the entire 30-epoch run (a wasted mechanism), the checkpoint resume logic omits optimizer and scheduler state (corrupting warm-restart semantics), and there are no NaN/Inf guards anywhere in the training loop. The training run itself was numerically stable, but the code is not robust to future instabilities.

---

## Finding 1: MSE Loss on Standardized log10 VMR Targets

### Severity: LOW

**What**: The loss function is `nn.MSELoss()` applied to standardized (z-scored) log10 VMR targets (`crossgen_hybrid_training.py`, line 690):
```python
loss_fn = nn.MSELoss()
```

Targets are standardized at lines 326 and 332:
```python
target_scaler = ArrayStandardizer.fit(targets_raw[inner_train_indices])
...
targets_scaled = target_scaler.transform(targets_raw[indices])
```

**Analysis**: MSE is a reasonable choice here for several specific reasons:

1. **Targets are already log-transformed**. The raw targets are `log10_vmr_*`, so a multiplicative error in VMR translates to an additive error in the target space. MSE on log-space is equivalent to penalizing geometric mean ratio errors in linear VMR, which is physically appropriate for volume mixing ratios spanning many orders of magnitude.

2. **Target distributions are nearly identical**. The target scaler reveals:
   - Means: [-6.981, -6.980, -6.989, -7.009, -7.013] (all within 0.03 of each other)
   - Stds: [2.890, 2.881, 2.887, 2.891, 2.902] (ratio max/min = 1.007)

   Because the standard deviations differ by less than 1%, the standardized MSE weights all 5 targets approximately equally. No target dominates the loss.

3. **Potential concern with outliers**: MSE is sensitive to outlier predictions. Given the uniform distribution of log10 VMR values (typical for synthetic training sets), this is not a major issue, but Huber loss would provide robustness if real-world data with extreme concentrations were introduced.

**Verdict**: Acceptable. MSE on standardized log10 VMR is a well-motivated baseline. The near-identical target scales mean equal weighting is automatic without needing per-target weights.

---

## Finding 2: Loss Computed on Standardized Scale — Evaluation RMSE on Original Scale

### Severity: LOW

**What**: Training loss is MSE on standardized targets (line 747), while the reported RMSE metrics are inverse-transformed to original log10 VMR units (lines 664-666):
```python
pred_orig = target_scaler.inverse_transform(pred_scaled)
true_orig = target_scaler.inverse_transform(true_scaled)
rmse_orig = np.sqrt(np.mean((pred_orig - true_orig) ** 2, axis=0))
```

**Analysis**: This is correct practice. Training on standardized scale ensures stable gradients and balanced target weighting. Reporting on original scale (log10 VMR) provides physically interpretable metrics. The two are connected by a simple linear transformation: `RMSE_orig = RMSE_standardized * scale`.

One subtle note: the validation loss used for early stopping and scheduling (`inner_val_metrics["loss"]`, lines 769 and 800) is on the **standardized** scale, while the RMSE reported to the user is on the **original** scale. These track the same underlying error (since the scaler is linear), so there is no inconsistency.

**Verdict**: Clean. No issue found.

---

## Finding 3: ReduceLROnPlateau Scheduler Never Fired

### Severity: MEDIUM

**What**: The scheduler is configured at lines 698-703:
```python
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=config.scheduler_factor,   # 0.5
    patience=config.scheduler_patience, # 5
)
```

**Evidence from history.csv**: Both learning rates remain constant for all 30 epochs:
- Classical LR: 0.002 (unchanged)
- Quantum LR: 0.0006 (unchanged)

The longest streak without val loss improvement was 2 consecutive epochs (epochs 23-24), far below the patience threshold of 5. The scheduler never had reason to reduce the learning rate.

**Impact**: This means the learning rate schedule was effectively a constant. The scheduler configuration is dead weight — it neither helped nor hurt. However, the `scheduler_patience=5` combined with `early_stop_patience=6` creates a narrow 1-epoch window where the LR could be reduced before early stopping kills training. In the actual run, early stopping also never fired (the model improved steadily), so both mechanisms were inert.

**Recommendation**: Either (a) lower `scheduler_patience` to 2-3 to give the scheduler a chance to fire before early stopping, or (b) use `CosineAnnealingLR` which guarantees LR reduction over the training horizon. A warmup phase could also help — the first-epoch train loss (0.829) is much higher than subsequent epochs, suggesting the initial learning rate may be slightly aggressive.

---

## Finding 4: Dual Learning Rate Setup (Classical vs Quantum)

### Severity: LOW

**What**: Two parameter groups with different learning rates (`crossgen_hybrid_training.py`, lines 692-697):
```python
optimizer = torch.optim.AdamW(
    [
        {"params": classical_params, "lr": config.classical_lr, "weight_decay": config.weight_decay},
        {"params": quantum_params,  "lr": config.quantum_lr,  "weight_decay": 0.0},
    ]
)
```
- Classical LR: 2e-3
- Quantum LR: 6e-4 (3.3x smaller)

**Analysis**: This is well-justified and follows standard practice in hybrid quantum-classical optimization:

1. **Quantum parameters have a fundamentally different loss landscape**. PennyLane circuit parameters are rotation angles; their gradients have different scale characteristics (bounded between -1 and 1 for PauliZ expectation values) compared to classical neural network gradients.

2. **The 3.3x ratio is reasonable**. Literature values range from 2x to 10x lower quantum LR. Vetrano et al. 2025 (the QELM atmospheric retrieval paper this work builds on) uses a similar ratio.

3. **Zero weight decay on quantum params is correct** (line 695). Quantum parameters are rotation angles, not traditional weights. L2 regularization would bias angles toward 0, which has no physical or mathematical significance in the quantum circuit context.

**Verdict**: Correct design choice. Well-aligned with quantum ML best practices.

---

## Finding 5: Gradient Clipping — Asymmetric Values

### Severity: LOW

**What**: Two different gradient clip norms are applied (`crossgen_hybrid_training.py`, lines 749-750):
```python
torch.nn.utils.clip_grad_norm_(classical_params, config.gradient_clip_norm)  # 5.0
torch.nn.utils.clip_grad_norm_(quantum_params, 1.0)  # kept tight for quantum stability
```

**Analysis**: The asymmetric clipping is appropriate:

1. **Classical clip at 5.0**: Standard for deep learning. The classical encoder has many parameters (AuxEncoder + SpectralEncoder + FusionEncoder + AtmosphereHead), so per-parameter gradient norms aggregate to larger total norms. A clip of 5.0 is conservative enough to prevent explosion while allowing reasonable gradient flow.

2. **Quantum clip at 1.0**: The quantum block has only 36 parameters (3 layers x 12 qubits x 1 block). Quantum gradients via the adjoint method can exhibit spikes (barren plateaus phenomenon, shot noise on hardware). A tight clip of 1.0 prevents any single unstable quantum gradient update from destabilizing the model.

3. **Note**: The quantum clip norm of 1.0 is hardcoded (not configurable via `TrainingConfig`), while the classical clip is configurable via `gradient_clip_norm`. This is a minor inconsistency but acceptable given the stability rationale.

**Verdict**: Sound engineering choice. The hardcoded quantum clip value is a minor style issue only.

---

## Finding 6: No NaN/Inf Guards in Training Loop

### Severity: MEDIUM

**What**: The training loop (lines 739-766) contains no checks for NaN or Inf values in the loss, gradients, or model outputs. The code goes directly from `loss.backward()` to `optimizer.step()` without validation:

```python
pred = model(aux, spectra)
loss = loss_fn(pred, targets)
loss.backward()
torch.nn.utils.clip_grad_norm_(classical_params, config.gradient_clip_norm)
torch.nn.utils.clip_grad_norm_(quantum_params, 1.0)
optimizer.step()
batch_losses.append(float(loss.item()))
```

**Risk**: If the quantum circuit produces NaN (possible with certain PennyLane backends during differentiation, or with extreme input angles), or if a classical encoder layer explodes, the NaN will silently propagate through the optimizer and corrupt all model weights. The training would continue logging NaN losses without halting or reverting.

**Evidence from this run**: No NaN/Inf occurred (verified from `history.csv` — all 30 epochs have finite losses). However, the absence of problems in one run does not guarantee future stability, especially when:
- Moving to actual quantum hardware (IQM Spark) where shot noise is real
- Changing hyperparameters (larger LR, deeper circuits)
- Using AMP (float16), which has a much narrower dynamic range

**Recommended guard** (after line 747):
```python
if not torch.isfinite(loss):
    print(f"WARNING: Non-finite loss {loss.item()} at batch {batch_idx}, skipping update")
    optimizer.zero_grad(set_to_none=True)
    continue
```

---

## Finding 7: Checkpoint Resume Does Not Restore Optimizer/Scheduler State

### Severity: HIGH

**What**: The resume logic (lines 713-720) only restores model weights and metadata, not the optimizer or scheduler state:

```python
if resume_from is not None and resume_from.exists():
    ckpt = torch.load(resume_from, weights_only=False, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    start_epoch = ckpt["epoch"]
    best_val_loss = ckpt["val_loss"]
    best_epoch = ckpt["epoch"]
    best_state = ckpt["model_state_dict"]
```

**What is missing**: `optimizer.load_state_dict()` and `scheduler.load_state_dict()` are never called.

**Impact**: When resuming training:
1. **AdamW momentum buffers (m, v) are reset to zero**. AdamW relies on running first and second moment estimates to compute adaptive learning rates per parameter. Resetting these means the optimizer loses all accumulated gradient history, causing the first several epochs after resume to behave like a fresh start with potentially large, noisy updates.

2. **The scheduler loses its internal counter**. ReduceLROnPlateau tracks how many epochs the metric has not improved. After resume, this resets to 0, so the scheduler will wait the full patience window again before reducing LR, even if it was about to fire before the interruption.

3. **The checkpoint itself does not save optimizer/scheduler state** (line 808):
   ```python
   torch.save({"epoch": best_epoch, "val_loss": best_val_loss, "model_state_dict": best_state}, ckpt_path)
   ```

**Note**: In the actual completed run, resume was not used (`resume=False` is the default in `run_training_experiment`). This finding is about code correctness for future use.

**Recommendation**: Save and restore full training state:
```python
torch.save({
    "epoch": best_epoch,
    "val_loss": best_val_loss,
    "model_state_dict": best_state,
    "optimizer_state_dict": optimizer.state_dict(),
    "scheduler_state_dict": scheduler.state_dict(),
}, ckpt_path)
```

---

## Finding 8: Early Stopping Logic

### Severity: LOW

**What**: Early stopping is based on raw val loss comparison with no threshold (lines 800-814):
```python
if inner_val_metrics["loss"] < best_val_loss:
    best_val_loss = float(inner_val_metrics["loss"])
    ...
    patience_left = config.early_stop_patience
else:
    patience_left -= 1
    if patience_left <= 0:
        print(f"Early stopping at epoch {epoch + 1}.", flush=True)
        break
```

**Analysis**:

1. **No minimum delta threshold**: Any improvement, even 1e-7, resets patience. In the current run, epoch 29 improved over epoch 27 by only 0.001 (0.2999 vs 0.3010). This is a genuine improvement, but in noisier settings (hardware quantum circuits, smaller datasets), floating-point-level improvements could keep training alive indefinitely.

2. **Patience of 6 with max_epochs of 30**: The early stopping never fired. The model improved at epoch 29 (of 30), so training ran to completion. This suggests either (a) the model could benefit from more epochs, or (b) the patience is appropriately generous. Given the val loss was still slowly decreasing at epoch 30 (0.2999 at epoch 29 vs 0.3019 at epoch 30), the model likely has not fully converged.

3. **Early stopping monitors standardized val loss, not RMSE**: This is fine because they are monotonically related (same linear transform for all samples).

**Interaction with scheduler**: With `scheduler_patience=5` and `early_stop_patience=6`, the scheduler can fire at most once before early stopping could trigger. This is a tight window. In practice, neither mechanism fired.

**Verdict**: Functional but conservative. Consider adding `min_delta=1e-4` to avoid resetting patience on noise-level improvements.

---

## Finding 9: Training Stability — History Analysis

### Severity: LOW (informational)

**What**: Comprehensive analysis of the 30-epoch training history.

**Loss trajectory**:
- Train loss: 0.829 (epoch 1) -> 0.301 (epoch 30), monotonically decreasing with one micro-increase at epoch 29 (+6.6e-5)
- Val loss: 0.714 (epoch 1) -> 0.300 (epoch 29, best), with 4 minor increases
- Largest val spike: +0.0068 at epoch 24 (2.2% relative increase, immediately recovered)
- No NaN, no Inf, no catastrophic spikes

**Train-Val gap**:
- Val loss < train loss for 27 of 30 epochs. This is expected because: (a) dropout (0.05) is active during training but disabled during eval, (b) train loss is averaged across batches during an epoch while val loss uses the converged end-of-epoch model, and (c) LayerNorm in FusionEncoder uses different statistics in train vs eval mode.
- The gap narrows from -0.115 (epoch 1) to approximately 0 by epoch 30, indicating the model is approaching but not past the overfitting point.

**Per-target convergence** (inner-val RMSE, best epoch):
| Target | Best RMSE | Best Epoch | Rate of Improvement |
|--------|-----------|------------|-------------------|
| H2O | 1.655 | 27 | Steady |
| CO2 | 1.150 | 27 | Steady |
| CO | 2.176 | 26 | Slow convergence |
| CH4 | 1.219 | 30 | Still improving |
| NH3 | 1.453 | 29 | Still improving |

CO has the worst RMSE (2.18 log10 orders of magnitude), suggesting this molecule's spectral features are hardest to disentangle in the 0.95-4.91 um wavelength range, consistent with CO having relatively weak absorption features compared to the other species.

**Verdict**: Training was numerically stable. No instabilities detected. The model was still improving at epoch 30, suggesting max_epochs could be increased.

---

## Finding 10: Val-Test Generalization Gap

### Severity: HIGH

**What**: The test RMSE is dramatically worse than validation RMSE:

| Split | RMSE Mean | H2O | CO2 | CO | CH4 | NH3 |
|-------|-----------|-----|-----|----|-----|-----|
| Inner val | 1.545 | 1.662 | 1.158 | 2.182 | 1.219 | 1.503 |
| Test | 3.463 | 3.131 | 4.726 | 3.283 | 3.227 | 2.948 |
| Ratio | 2.24x | 1.88x | 4.08x | 1.50x | 2.65x | 1.96x |

**Analysis**: This is not directly a loss function issue, but the optimization setup may be contributing:

1. **Train and val are both TauREx-generated**, while test is Poseidon-generated. The optimizer is fitting to TauREx-specific patterns.

2. **CO2 degrades 4.08x** — by far the worst. This suggests the TauREx and Poseidon simulators disagree most on CO2 spectral features, and the model has overfit to TauREx's CO2 implementation.

3. **The loss function (MSE) does not incorporate any domain-adaptation term**. For cross-generator generalization, adversarial domain adaptation or a secondary loss term penalizing distribution shift could help.

**Note**: This finding is primarily a data/generalization issue rather than a loss function bug, but the optimization is directly responsible for the degree of TauREx overfitting. The lack of any regularization beyond mild weight decay (1e-4) and minimal dropout (0.05) means the model has no incentive to learn generator-invariant features.

---

## Finding 11: AMP Interaction with Loss Computation

### Severity: LOW

**What**: Automatic mixed precision is enabled (`use_amp=True` in config), but the model's forward pass manually manages precision (lines 534-545):
```python
autocast_ctx = (
    torch.autocast(device_type="cuda", dtype=self.amp_dtype)
    if autocast_enabled
    else nullcontext()
)
with autocast_ctx:
    aux_feat = self.aux_encoder(aux)
    spectral_feat = self.spectral_encoder(spectra)
    latent = self.fusion_encoder(aux_feat, spectral_feat)

latent = latent.float()  # explicit float32 cast before quantum
```

**Analysis**: The AMP scope is correctly limited to the classical encoders. The quantum block receives float32 input (line 545), and `loss_fn(pred, targets)` operates on float32 outputs. This means the loss computation is always in full precision, which is correct.

However, the training loop does NOT use `torch.amp.GradScaler`. When AMP is active on CUDA, float16 gradients can underflow. A GradScaler is the standard solution. In the actual run, `lightning.qubit` was used (CPU), so AMP was disabled and this did not matter. But the code structure implies CUDA+AMP support that is incomplete.

**Verdict**: Not a problem for the current CPU-based run. Would need GradScaler for correct CUDA AMP usage.

---

## Summary Table

| # | Finding | Severity | Status |
|---|---------|----------|--------|
| 1 | MSE on standardized log10 VMR | LOW | Appropriate choice |
| 2 | Loss scale vs reporting scale | LOW | Correctly implemented |
| 3 | ReduceLROnPlateau never fired | MEDIUM | Wasted mechanism, patience too high |
| 4 | Dual learning rate (classical/quantum) | LOW | Well-justified |
| 5 | Asymmetric gradient clipping | LOW | Appropriate |
| 6 | No NaN/Inf guards | MEDIUM | Risk for future runs |
| 7 | Checkpoint resume missing optimizer/scheduler state | HIGH | Broken warm-restart semantics |
| 8 | Early stopping logic | LOW | Functional, no min_delta |
| 9 | Training stability | LOW | Stable, still converging at epoch 30 |
| 10 | Val-test generalization gap (2.24x) | HIGH | Domain shift not addressed by loss |
| 11 | AMP without GradScaler | LOW | Not triggered in current run |

---

## Verdict: CONDITIONAL PASS

The loss function and optimization pipeline produced a numerically stable, converging training run. MSE on standardized log10 VMR is appropriate, the dual optimizer is correctly configured, and gradient clipping values are defensible. However, two HIGH-severity findings must be addressed:

1. **Checkpoint resume is broken** (Finding 7) — optimizer and scheduler state are not saved or restored. Any future training resumption will produce incorrect optimization behavior.

2. **The 2.24x val-test RMSE degradation** (Finding 10) — while not a bug in the loss function, the optimization has no mechanism to prevent overfitting to the TauREx generator. This is the most impactful issue for scientific credibility.

The MEDIUM-severity findings (scheduler never firing, no NaN guards) are engineering debt rather than correctness issues, but should be addressed before production use or real hardware deployment.
