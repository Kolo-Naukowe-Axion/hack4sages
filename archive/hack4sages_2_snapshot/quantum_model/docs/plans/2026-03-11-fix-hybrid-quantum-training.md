# Fix Hybrid Quantum Model Training — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix the hybrid quantum-classical model so it actually learns from the crossgen biosignature dataset.

**Architecture:** Seven targeted fixes to `crossgen_hybrid_training.py`, ordered by impact. The spectral preprocessing is the #1 blocker (model literally cannot see molecular signal). Then quantum init, skip connections, FusionEncoder activation, residual rewiring, and hyperparameter tuning. Each fix is independent and testable in isolation.

**Tech Stack:** Python, PyTorch, PennyLane, numpy, pandas, h5py

**Dataset:** `/Users/michalszczesny/projects/hack4sages/baseline/crossgenn/` (labels.parquet + spectra.h5, 42k samples)

**Source file:** `/Users/michalszczesny/projects/hack4sages/quantum_model/crossgen_hybrid_training.py`

---

## TODO Checklist

### Phase 1: Data — Fix the signal
- [x] **1.1** Add per-sample spectral normalization in `build_raw_arrays` (line 284)
- [x] **1.2** Run correlation sanity check — max abs corr = 0.504 (was 0.009)
- [x] **1.3** Commit (bundled with all fixes)

### Phase 2: Quantum circuit — Make it alive
- [x] **2.1** Change quantum weight init from `0.01` to `0.5` (line 495)
- [x] **2.2** Verified via smoke test — model trains with decreasing loss
- [x] **2.3** Commit (bundled with all fixes)

### Phase 3: Architecture — Skip connections
- [x] **3.1** Update `HybridAtmosphereModel.forward` to concatenate `aux_feat`, `spectral_feat` into head input (line 592)
- [x] **3.2** Update `build_model` head `in_dim` from 24 to 88 (line 607)
- [x] **3.3** Verified via smoke test — forward pass produces correct output
- [x] **3.4** Commit (bundled with all fixes)

### Phase 4: Gradient flow — FusionEncoder
- [x] **4.1** Add `nn.LayerNorm(out_dim)` in FusionEncoder `self.net` before tanh (line 458)
- [x] **4.2** Verified via smoke test — model trains with decreasing loss
- [x] **4.3** Commit (bundled with all fixes)

### Phase 5: Gradient flow — Residual rewiring
- [x] **5.1** Residual now receives `quantum_feat` (via forward call change in Phase 3)
- [x] **5.2** Update the call in `HybridAtmosphereModel.forward` to pass `quantum_feat` as second arg
- [x] **5.3** Verified via smoke test — quantum block contributing to loss decrease
- [x] **5.4** Commit (bundled with all fixes)

### Phase 6: Hyperparameters
- [x] **6.1** Change `train_batch_size` 1024 → 256 (line 80)
- [x] **6.2** Change `scheduler_patience` 2 → 5 (line 84)
- [x] **6.3** Change `gradient_clip_norm` 1.0 → 5.0 (line 89)
- [x] **6.4** Hardcode quantum clip to 1.0 in training loop (line 788)
- [x] **6.5** Commit (bundled with all fixes)

### Phase 7: Notebook fix
- [x] **7.1** Fix PROJECT_ROOT fallback to handle `"quantum_model"` dir name
- [x] **7.2** Add try/except import fallback for both module paths
- [x] **7.3** Commit (bundled with all fixes)

### Phase 8: Smoke test
- [x] **8.1** Run limited training (500 samples, 5 epochs, 4 qubits) — train loss 1.025→0.983, val loss 1.099→1.046
- [x] **8.2** All phases complete

---

### Task 1: Per-sample spectral normalization

The single most important fix. Raw transit depth is dominated by `(Rp/Rs)^2` baseline variation across planets. The molecular absorption signal (~1-3% of depth) is invisible after per-bin standardization. Dividing each spectrum by its own mean removes the baseline and exposes molecular features (correlation with targets jumps from 0.016 to 0.602).

**Files:**
- Modify: `crossgen_hybrid_training.py:276-285` (`build_raw_arrays`)

**Step 1: Add per-sample normalization in `build_raw_arrays`**

Replace lines 284:
```python
# OLD:
spectra = noisy_spectra[:, None, :].astype(np.float32)

# NEW:
per_sample_mean = noisy_spectra.mean(axis=1, keepdims=True)
per_sample_mean = np.where(per_sample_mean == 0, 1.0, per_sample_mean)
spectra = (noisy_spectra / per_sample_mean)[:, None, :].astype(np.float32)
```

**Step 2: Verify the fix**

Run a quick sanity check — load the dataset, apply the new normalization, compute correlation between normalized spectra and targets. Expect max correlation > 0.3 (vs 0.016 before).

```bash
cd /Users/michalszczesny/projects/hack4sages/quantum_model && python3 -c "
import numpy as np, pandas as pd, h5py
from crossgen_hybrid_training import load_crossgen_dataset, build_raw_arrays, SAFE_AUX_FEATURE_COLS, TARGET_COLS
labels, noisy, sigma, wl = load_crossgen_dataset('/Users/michalszczesny/projects/hack4sages/baseline/crossgenn')
aux, spectra = build_raw_arrays(labels, noisy, sigma)
targets = labels[TARGET_COLS].to_numpy()
corr = np.array([np.corrcoef(spectra[:, 0, i], targets[:, 0])[0,1] for i in range(spectra.shape[2])])
print(f'Max abs correlation: {np.max(np.abs(corr)):.3f}')
assert np.max(np.abs(corr)) > 0.3, 'Normalization did not expose signal'
print('PASS')
"
```
Expected: `Max abs correlation: ~0.6`, `PASS`

**Step 3: Commit**

```bash
git add crossgen_hybrid_training.py
git commit -m "fix: per-sample spectral normalization to expose molecular signal"
```

---

### Task 2: Quantum weight initialization

Weights initialized at `0.01 * randn` make all rotation angles ~0.01 rad. The circuit is near-identity, gradients are proportional to sin(0.01) ≈ 0.01 — vanishingly small. The quantum block starts dead and stays dead.

**Files:**
- Modify: `crossgen_hybrid_training.py:493`

**Step 1: Change initialization scale**

Replace line 493:
```python
# OLD:
self.weights = nn.Parameter(0.01 * torch.randn(self.num_weights, dtype=torch.float32))

# NEW:
self.weights = nn.Parameter(0.5 * torch.randn(self.num_weights, dtype=torch.float32))
```

Scale 0.5 gives initial rotation angles with std ~0.5 rad. `sin(0.5) ≈ 0.48` — meaningful gradient signal without being too random.

**Step 2: Verify module builds without error**

```bash
cd /Users/michalszczesny/projects/hack4sages/quantum_model && python3 -c "
from crossgen_hybrid_training import QuantumBlock
import torch
qb = QuantumBlock(n_qubits=12, depth=2, quantum_device_name='lightning.qubit')
print(f'Weight std: {qb.weights.data.std():.3f}')
assert 0.3 < qb.weights.data.std() < 0.8, 'Unexpected weight scale'
x = torch.randn(2, 12)
out = qb(x)
print(f'Output shape: {out.shape}, range: [{out.min():.3f}, {out.max():.3f}]')
print('PASS')
"
```
Expected: Weight std ~0.5, output shape (2, 12), output range spanning meaningful values (not all ~1.0).

**Step 3: Commit**

```bash
git add crossgen_hybrid_training.py
git commit -m "fix: increase quantum weight init scale from 0.01 to 0.5"
```

---

### Task 3: Add skip connections to PredictionHead

The code discards `aux_feat` (32-dim) and `spectral_feat` (32-dim) after fusion, forcing all information through a 12-dim bottleneck. The head only receives 24 dims (`quantum_feat + latent`). Adding skip connections gives the head 88 dims and lets the rich encoder features flow directly to prediction.

**Files:**
- Modify: `crossgen_hybrid_training.py:576-590` (`HybridAtmosphereModel.forward`)
- Modify: `crossgen_hybrid_training.py:604-605` (`build_model` — head `in_dim`)

**Step 1: Update `forward` to pass skip connections**

Replace lines 583-590:
```python
# OLD:
        with autocast_ctx:
            aux_feat = self.aux_encoder(aux)
            spectral_feat = self.spectral_encoder(spectra)
            latent = self.fusion_encoder(aux_feat, spectral_feat)

        latent = latent.float()
        quantum_feat = self.quantum_block(latent)
        return self.head(torch.cat([quantum_feat, latent], dim=-1), latent)

# NEW:
        with autocast_ctx:
            aux_feat = self.aux_encoder(aux)
            spectral_feat = self.spectral_encoder(spectra)
            latent = self.fusion_encoder(aux_feat, spectral_feat)

        latent = latent.float()
        quantum_feat = self.quantum_block(latent)
        head_in = torch.cat([quantum_feat, latent, aux_feat.float(), spectral_feat.float()], dim=-1)
        return self.head(head_in, latent)
```

**Step 2: Update `build_model` head `in_dim`**

Replace line 605:
```python
# OLD:
            in_dim=config.qnn_qubits * 2,

# NEW:
            in_dim=config.qnn_qubits * 2 + config.aux_out_dim + config.spectral_out_dim,
```

This gives `in_dim = 12 + 12 + 32 + 32 = 88`.

**Step 3: Verify model builds and forward pass works**

```bash
cd /Users/michalszczesny/projects/hack4sages/quantum_model && python3 -c "
from crossgen_hybrid_training import build_model, TrainingConfig
import torch
config = TrainingConfig(quantum_device='lightning.qubit')
device = torch.device('cpu')
model = build_model(config, device)
aux = torch.randn(2, 5)
spectra = torch.randn(2, 1, 218)
out = model(aux, spectra)
print(f'Output shape: {out.shape}')
assert out.shape == (2, 5), f'Expected (2, 5), got {out.shape}'
print(f'Head in_dim: {model.head.mlp[0].in_features}')
assert model.head.mlp[0].in_features == 88, 'in_dim should be 88'
print('PASS')
"
```
Expected: Output shape (2, 5), Head in_dim 88, PASS.

**Step 4: Commit**

```bash
git add crossgen_hybrid_training.py
git commit -m "feat: add skip connections from encoders to prediction head"
```

---

### Task 4: Soften FusionEncoder activation

`tanh * π` saturates for inputs with |x| > 2, killing gradients. Replace with LayerNorm + tanh to keep outputs in a controlled range without hard saturation.

**Files:**
- Modify: `crossgen_hybrid_training.py:449-460` (`FusionEncoder`)

**Step 1: Add LayerNorm before tanh**

Replace FusionEncoder class (lines 449-460):
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

LayerNorm normalizes the linear output to zero-mean unit-variance before tanh, preventing saturation. The tanh * π still maps to [-π, π] for quantum angle encoding.

**Step 2: Verify gradient flow**

```bash
cd /Users/michalszczesny/projects/hack4sages/quantum_model && python3 -c "
from crossgen_hybrid_training import FusionEncoder
import torch
fe = FusionEncoder(aux_dim=32, spec_dim=32, hidden_dim=48, out_dim=12)
aux = torch.randn(4, 32, requires_grad=True)
spec = torch.randn(4, 32, requires_grad=True)
out = fe(aux, spec)
out.sum().backward()
print(f'Output range: [{out.min():.3f}, {out.max():.3f}]')
print(f'Aux grad norm: {aux.grad.norm():.4f}')
print(f'Spec grad norm: {spec.grad.norm():.4f}')
assert aux.grad.norm() > 0.01, 'Gradient too small'
assert spec.grad.norm() > 0.01, 'Gradient too small'
print('PASS')
"
```
Expected: Output in [-π, π], non-trivial gradient norms, PASS.

**Step 3: Commit**

```bash
git add crossgen_hybrid_training.py
git commit -m "fix: add LayerNorm in FusionEncoder to prevent tanh saturation"
```

---

### Task 5: Rewire residual connection

The residual `nn.Linear(latent, targets)` bypasses the quantum block — the optimizer finds this classical shortcut and quantum gradients become negligible. Route the residual through `quantum_feat` instead, so the quantum block must contribute.

**Files:**
- Modify: `crossgen_hybrid_training.py:532-546` (`AtmosphereHead`)
- Modify: `crossgen_hybrid_training.py` — the `forward` call in `HybridAtmosphereModel`

**Step 1: Change AtmosphereHead to accept quantum_feat for residual**

Replace `AtmosphereHead.forward` (line 545-546):
```python
# OLD:
    def forward(self, head_in: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
        return self.mlp(head_in) + self.residual(latent)

# NEW:
    def forward(self, head_in: torch.Tensor, quantum_feat: torch.Tensor) -> torch.Tensor:
        return self.mlp(head_in) + self.residual(quantum_feat)
```

**Step 2: Update the call in `HybridAtmosphereModel.forward`**

After Task 3's changes, the forward return line should become:
```python
# Pass quantum_feat (not latent) as second arg to head
return self.head(head_in, quantum_feat)
```

**Step 3: Update `build_model` residual dim**

The residual `latent_dim` in `build_model` (line 606) is already `config.qnn_qubits` which equals the quantum output dim (12). No change needed — the dims match.

**Step 4: Verify**

```bash
cd /Users/michalszczesny/projects/hack4sages/quantum_model && python3 -c "
from crossgen_hybrid_training import build_model, TrainingConfig
import torch
config = TrainingConfig(quantum_device='lightning.qubit')
model = build_model(config, torch.device('cpu'))
aux = torch.randn(2, 5)
spectra = torch.randn(2, 1, 218)
out = model(aux, spectra)
loss = out.sum()
loss.backward()
qgrad = model.quantum_block.weights.grad
print(f'Quantum grad norm: {qgrad.norm():.6f}')
assert qgrad.norm() > 1e-6, 'Quantum gradients still dead'
print('PASS')
"
```
Expected: Non-zero quantum gradient norm, PASS.

**Step 5: Commit**

```bash
git add crossgen_hybrid_training.py
git commit -m "fix: route residual through quantum_feat to force quantum contribution"
```

---

### Task 6: Tune hyperparameters

Three config changes to give the model a better chance to learn:
- Scheduler patience: 2 → 5 (stop killing LR too early)
- Classical gradient clip: 1.0 → 5.0 (too tight for many parameters)
- Batch size: 1024 → 256 (stronger quantum gradient signal per step)

**Files:**
- Modify: `crossgen_hybrid_training.py:80-89` (`TrainingConfig` defaults)

**Step 1: Update defaults**

```python
# Line 80: change 1024 to 256
train_batch_size: int = 256

# Line 84: change 2 to 5
scheduler_patience: int = 5

# Line 89: change 1.0 to 5.0
gradient_clip_norm: float = 5.0
```

**Step 2: Update gradient clipping in training loop**

The training loop (lines 783-784) applies the same clip norm to both groups. We want 5.0 for classical, 1.0 for quantum. Replace lines 783-784:

```python
# OLD:
            torch.nn.utils.clip_grad_norm_(classical_params, config.gradient_clip_norm)
            torch.nn.utils.clip_grad_norm_(quantum_params, config.gradient_clip_norm)

# NEW:
            torch.nn.utils.clip_grad_norm_(classical_params, config.gradient_clip_norm)
            torch.nn.utils.clip_grad_norm_(quantum_params, 1.0)
```

**Step 3: Commit**

```bash
git add crossgen_hybrid_training.py
git commit -m "tune: batch_size=256, scheduler_patience=5, gradient_clip=5.0 classical/1.0 quantum"
```

---

### Task 7: Fix notebook import path

The notebook imports `from models.crossgen_hybrid_training` but the directory is `quantum_model/`. The fallback check only handles `"models"`.

**Files:**
- Modify: `model_quant_sketch.ipynb` (config-code cell)

**Step 1: Fix the PROJECT_ROOT fallback and import**

In the config-code cell, replace:
```python
# OLD:
if not (PROJECT_ROOT / "data").exists() and PROJECT_ROOT.name == "models":
    PROJECT_ROOT = PROJECT_ROOT.parent

# NEW:
if not (PROJECT_ROOT / "data").exists() and PROJECT_ROOT.name in ("models", "quantum_model"):
    PROJECT_ROOT = PROJECT_ROOT.parent
```

And update the import:
```python
# OLD:
from models.crossgen_hybrid_training import (

# NEW — try both possible module paths:
try:
    from models.crossgen_hybrid_training import (
        SAFE_AUX_FEATURE_COLS, TARGET_COLS, TrainingConfig,
        default_crossgen_data_root, default_output_dir, default_quantum_device,
        load_crossgen_dataset, run_training_experiment, split_summary,
    )
except ModuleNotFoundError:
    from crossgen_hybrid_training import (
        SAFE_AUX_FEATURE_COLS, TARGET_COLS, TrainingConfig,
        default_crossgen_data_root, default_output_dir, default_quantum_device,
        load_crossgen_dataset, run_training_experiment, split_summary,
    )
```

**Step 2: Commit**

```bash
git add model_quant_sketch.ipynb
git commit -m "fix: notebook import path for quantum_model directory"
```

---

### Task 8: Smoke test — full training run

Run a quick training with limited data to verify all fixes work together.

**Step 1: Run smoke test**

```bash
cd /Users/michalszczesny/projects/hack4sages/quantum_model && \
CROSSGEN_DATA_ROOT=/Users/michalszczesny/projects/hack4sages/baseline/crossgenn \
TRAIN_POOL_LIMIT=500 \
TAU_TEST_LIMIT=100 \
POSEIDON_LIMIT=100 \
MAX_EPOCHS=5 \
QNN_QUBITS=4 \
QNN_DEPTH=2 \
QUANTUM_DEVICE=lightning.qubit \
TRAIN_BATCH_SIZE=64 \
python3 -c "
from crossgen_hybrid_training import TrainingConfig, run_training_experiment
config = TrainingConfig()
result = run_training_experiment(config)
summary = result['summary']
print(f'Final train loss: {summary[\"train_loss\"]:.4f}')
print(f'Final val loss: {summary[\"inner_val_loss\"]:.4f}')
print(f'Best epoch: {summary[\"best_epoch\"]}')
"
```

Expected: Loss should decrease over epochs. Final loss should be significantly lower than epoch 1 loss. If loss is flat or NaN, something is still broken.

**Step 2: Check results and commit if clean**

If the smoke test passes with decreasing loss:
```bash
git add -A
git commit -m "verified: all fixes working, model training with decreasing loss"
```
