# Prepare codex_model for Training — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make codex_model understandable and trainable end-to-end on the crossgen_biosignatures dataset.

**Architecture:** Keep the 984-line monolith as-is. Fix one path function, add documentation, improve the notebook, add a CLI entrypoint, and smoke-test the pipeline.

**Tech Stack:** Python 3, PyTorch, PennyLane, h5py, pandas, spectres, matplotlib. Venv at `../quantum_model/.venv`.

---

### Task 1: Create venv symlink

The working venv lives at `../quantum_model/.venv`. Symlink it so all commands work from `codex_model/`.

**Files:**
- Create: `codex_model/.venv` (symlink)

**Step 1: Create the symlink**

```bash
cd /Users/michalszczesny/projects/hack4sages/codex_model
ln -s ../quantum_model/.venv .venv
```

**Step 2: Verify imports work**

Run: `.venv/bin/python -c "from crossgen_hybrid_training import TrainingConfig; print('OK')"`
Expected: `OK`

**Step 3: Add .venv to .gitignore**

`.venv/` is already in `.gitignore` — verify this.

---

### Task 2: Fix default_data_root()

The function at line 71-75 doesn't know about our `data/` symlink.

**Files:**
- Modify: `crossgen_hybrid_training.py:71-75`

**Step 1: Edit the function**

Replace lines 71-75:

```python
def default_data_root(project_root: Path) -> Path:
    candidate = project_root / "crossgen_biosignatures_20260311"
    if candidate.exists():
        return candidate
    return project_root.parent / "quantum_model_crossgen" / "crossgen_biosignatures_20260311"
```

With:

```python
def default_data_root(project_root: Path) -> Path:
    candidate = project_root / "data"
    if candidate.exists():
        return candidate
    candidate = project_root / "crossgen_biosignatures_20260311"
    if candidate.exists():
        return candidate
    return project_root.parent / "quantum_model_crossgen" / "crossgen_biosignatures_20260311"
```

**Step 2: Verify resolution**

Run: `.venv/bin/python -c "from crossgen_hybrid_training import TrainingConfig; c = TrainingConfig(); print(c.resolved_data_root())"`
Expected: path ending in `crossgen_biosignatures_20260311` (resolved through symlink)

---

### Task 3: Write CLAUDE.md

**Files:**
- Create: `CLAUDE.md`

**Step 1: Write the file**

```markdown
# codex_model

Hybrid quantum-classical neural network for biosignature detection in exoplanet transmission spectra. Predicts log10 VMR of 5 atmospheric gases (H2O, CO2, CO, CH4, NH3) using a QELM approach.

## Quick Start

```bash
# Activate venv
source .venv/bin/activate

# CLI training
python run_training.py

# Or use the notebook
jupyter notebook train.ipynb
```

## crossgen_hybrid_training.py — Section Map

| Lines | What |
|-------|------|
| 1-54 | Imports, constants (SAFE_AUX_FEATURE_COLS, TARGET_COLS, ARIEL_WAVELENGTH_GRID) |
| 56-59 | `rebin_spectra()` — 218→44 bin rebinning via spectres |
| 62-80 | Path resolution helpers (`resolve_project_root`, `default_data_root`, `default_output_dir`) |
| 86-142 | `TrainingConfig` dataclass — all hyperparameters |
| 145-206 | Standardizers (`ArrayStandardizer`, `SpectralStandardizer`) + data containers (`SplitTensors`, `PreparedData`) |
| 209-227 | Runtime setup (`set_runtime_seed`, `configure_runtime`) |
| 229-250 | Device resolution helpers |
| 253-359 | **Data pipeline** — `load_crossgen_dataset()`, `build_raw_arrays()`, `prepare_data()` |
| 362-414 | **Classical encoders** — `AuxEncoder`, `SpectralEncoder`, `FusionEncoder` |
| 417-483 | **Quantum block** — PennyLane circuit (12 qubits, RY/CNOT/RZ/CRX) |
| 486-568 | **Model assembly** — `AtmosphereHead`, `HybridAtmosphereModel`, `build_model()` |
| 571-634 | Device/batch utilities |
| 637-681 | `evaluate_split()` — validation/test evaluation |
| 683-825 | `train_model()` — full training loop with early stopping |
| 828-984 | `run_training_experiment()` — end-to-end orchestrator + artifact saving |

## Dataset

Only two files matter from `data/`:
- `labels.parquet` — 42,108 rows: sample_id, split, generator, 5 aux features, 5 targets
- `spectra.h5` — wavelength_um(218), transit_depth_noisy(42108,218), sigma_ppm(42108)

Splits: train=37,281 (TauREx) | val=4,142 (TauREx) | test=685 (Poseidon)

## Key Hyperparameters

| Param | Default | What to tune |
|-------|---------|-------------|
| qnn_qubits | 12 | Quantum circuit width (must match fusion output) |
| qnn_depth | 2 | Circuit depth (must be even) |
| classical_lr | 2e-3 | Classical encoder learning rate |
| quantum_lr | 6e-4 | Quantum parameter learning rate |
| train_batch_size | 256 | Larger = faster but less gradient noise |
| max_epochs | 30 | With early stopping (patience=6) |
```

---

### Task 4: Improve train.ipynb

**Files:**
- Modify: `train.ipynb`

**Step 1: Rewrite notebook cells**

Cell 0 (markdown): Keep existing header.

Cell 1 (code) — Imports + config:
```python
from crossgen_hybrid_training import (
    TrainingConfig,
    run_training_experiment,
    prepare_data,
    load_crossgen_dataset,
    TARGET_COLS,
    SAFE_AUX_FEATURE_COLS,
)

config = TrainingConfig(data_root="data")
print(f"Data root:  {config.resolved_data_root()}")
print(f"Output dir: {config.resolved_output_dir()}")
print(f"Quantum:    {config.qnn_qubits} qubits, depth {config.qnn_depth}")
print(f"Device:     {config.quantum_device}")
```

Cell 2 (code) — Data sanity checks:
```python
labels, noisy_spectra, sigma_ppm, wavelength_um = load_crossgen_dataset(config.resolved_data_root())

print(f"Labels shape:    {labels.shape}")
print(f"Spectra shape:   {noisy_spectra.shape}")
print(f"Wavelength bins: {len(wavelength_um)} ({wavelength_um[0]:.2f} - {wavelength_um[-1]:.2f} um)")
print(f"\nSplit counts:")
print(labels["split"].value_counts().to_string())
print(f"\nTarget ranges (log10 VMR):")
for col in TARGET_COLS:
    print(f"  {col}: [{labels[col].min():.2f}, {labels[col].max():.2f}], mean={labels[col].mean():.2f}")
print(f"\nAux feature ranges:")
for col in SAFE_AUX_FEATURE_COLS:
    print(f"  {col}: [{labels[col].min():.4f}, {labels[col].max():.4f}]")
```

Cell 3 (code) — Run training:
```python
result = run_training_experiment(config)
```

Cell 4 (code) — Results:
```python
print(f"\nBest epoch:     {result['summary']['best_epoch']}")
print(f"Best val loss:  {result['summary']['best_inner_val_loss']:.5f}")
print(f"Test RMSE mean: {result['summary']['test_rmse_mean']:.4f}")
print(f"\nPer-target test RMSE:")
print(result['test_metrics_frame'].to_string(index=False))
```

Cell 5 (code) — Scatter plots:
```python
import matplotlib.pyplot as plt
import numpy as np

test_preds = result["test_metrics_frame"]
fig, axes = plt.subplots(1, 5, figsize=(20, 4))
pred_csv = __import__("pandas").read_csv(result["artifacts"]["test_predictions_csv"])

for i, col in enumerate(TARGET_COLS):
    ax = axes[i]
    true_vals = pred_csv[f"true_{col}"]
    pred_vals = pred_csv[f"pred_{col}"]
    ax.scatter(true_vals, pred_vals, alpha=0.3, s=8)
    lims = [min(true_vals.min(), pred_vals.min()), max(true_vals.max(), pred_vals.max())]
    ax.plot(lims, lims, "r--", linewidth=1)
    ax.set_xlabel("True")
    ax.set_ylabel("Predicted")
    ax.set_title(col.replace("log10_vmr_", "").upper())
    ax.set_aspect("equal")

plt.suptitle("Test Set: Predicted vs True (log10 VMR)")
plt.tight_layout()
plt.savefig("outputs/model_crossgen_rebinned/test_scatter.png", dpi=150)
plt.show()
```

---

### Task 5: Create run_training.py

**Files:**
- Create: `run_training.py`

**Step 1: Write the entrypoint**

```python
"""CLI entrypoint for training the hybrid quantum-classical model."""

from crossgen_hybrid_training import TrainingConfig, run_training_experiment

if __name__ == "__main__":
    config = TrainingConfig(data_root="data")
    result = run_training_experiment(config)
    print(f"\nDone. Test RMSE mean: {result['summary']['test_rmse_mean']:.4f}")
    print(f"Artifacts saved to: {result['summary']['output_dir']}")
```

**Step 2: Verify it parses**

Run: `.venv/bin/python -c "import run_training; print('OK')"` — should not crash (won't run training because of `__name__` guard).

---

### Task 6: Smoke test the pipeline

Verify the full pipeline works up to the first forward pass without running actual training.

**Files:** None (read-only test)

**Step 1: Test data loading + model build**

Run from `codex_model/`:

```bash
.venv/bin/python -c "
from crossgen_hybrid_training import TrainingConfig, prepare_data, build_model, resolve_training_device
config = TrainingConfig(data_root='data', train_pool_limit=100, test_limit=50)
data = prepare_data(config)
print(f'Train: {data.train.aux.shape}, Spectra: {data.train.spectra.shape}, Targets: {data.train.targets.shape}')
print(f'Val:   {data.inner_val.aux.shape}')
print(f'Test:  {data.test.aux.shape}')
device = resolve_training_device(config)
model = build_model(config, device)
pred = model(data.train.aux[:4], data.train.spectra[:4])
print(f'Forward pass output shape: {pred.shape}')
print('Smoke test PASSED')
"
```

Expected output:
```
Rebinned spectra: 218 bins -> 44 bins (0.95 - 4.91 um)
Train: torch.Size([100, 5]), ...
...
Forward pass output shape: torch.Size([4, 5])
Smoke test PASSED
```

If this passes, the model is ready to train.

---

### Task 7: Update .gitignore

**Files:**
- Modify: `.gitignore`

**Step 1: Ensure completeness**

Verify `.gitignore` contains:
```
data/
outputs/
__pycache__/
*.pyc
.DS_Store
.venv/
*.pt
docs/plans/
```

Note: `docs/plans/` added since plans are local working docs, not part of the model.

---
