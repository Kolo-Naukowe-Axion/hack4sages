# Quantum Advantage Testing — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Determine whether the 8-qubit quantum layer in `ariel_quantum_regression` provides a real, measurable advantage over purely classical alternatives on the Ariel ADC 2023 dataset.

**Architecture:** One self-contained notebook (`quantum_advantage_test.ipynb`) runs all tests sequentially: load model and data once, then execute each test as a clearly separated section with its own metrics, tables, and plots. The notebook evaluates the trained checkpoint (`trained_weights/best_model.pt`) on the Ariel ADC dataset splits (validation + holdout), comparing quantum-enabled vs quantum-disabled/ablated conditions. Final section aggregates all evidence into a verdict.

**Tech Stack:** Python 3.11+, PyTorch, PennyLane (lightning.qubit), scikit-learn, h5py, pandas, numpy, matplotlib, scipy

---

## Context for the Implementer

### The model

`HybridArielRegressor` predicts 5 gas abundances from exoplanet spectra. It has a classical backbone (SpectralEncoder + AuxEncoder + FusionEncoder + ClassicalHead) and a quantum branch (QuantumProjector → 8-qubit circuit → QuantumHead). The final output is:

```python
output = classical_pred + quantum_scale * tanh(quantum_gate) * quantum_correction
```

The quantum branch is additive — setting `enable_quantum=False` gives pure classical output from the same weights. The `quantum_gate` (5 learnable scalars) controls per-target contribution.

### The checkpoint

`trained_weights/best_model.pt` contains:
- `model_state_dict` — all weights
- `config` — training config dict (qnn_qubits=8, qnn_depth=2, classical_only=False)
- `feature_cols`, `target_cols` — column names
- `best_epoch` (6), `best_val_rmse` (0.2908)

Trained with two-stage strategy: classical pre-training, then quantum fine-tuning with backbone frozen for 6 epochs. Best checkpoint at epoch 6, quantum_scale=0.5.

### The dataset

Ariel ADC 2023 at `../../datasets/ariel-ml-dataset/` (relative to `quantum_advantage/`):
- `TrainingData/AuxillaryTable.csv` — 8 auxiliary features per planet
- `TrainingData/Ground Truth Package/FM_Parameter_Table.csv` — 5 target columns
- `TrainingData/SpectralData.hdf5` — 52-bin spectra (4 channels per planet)
- `TestData/` — same structure, no ground truth

The model's `prepare_data()` handles loading, splitting (80/10/10 stratified, seed=42), scaling. **Use seed=42 for all data prep to match the training splits.**

### The scalers

`trained_weights/scalers.json` contains fitted `aux_scaler`, `target_scaler`, `spectral_scaler` from training. For inference-only tests (Tests 1-4), load these directly instead of re-fitting.

### How to load the model for inference

**Import paths:** The module lives at `quantum_advantage/ariel_quantum_regression/`. From notebooks in `quantum_advantage/tests/`, add the parent directory to `sys.path` and import from `ariel_quantum_regression.*` (NOT `models.ariel_quantum_regression.*`).

```python
import sys, json, torch
import numpy as np
from pathlib import Path

# Add quantum_advantage/ to path so we can import ariel_quantum_regression
sys.path.insert(0, str(Path.cwd().parent))

from ariel_quantum_regression.model import ModelConfig, build_model
from ariel_quantum_regression.dataset import (
    ArrayStandardizer, SpectralStandardizer, PreparedData,
    prepare_data
)
from ariel_quantum_regression.constants import TARGET_COLUMNS

CHECKPOINT_PATH = Path("../trained_weights/best_model.pt")
DATA_ROOT = Path("../../../datasets/ariel-ml-dataset")

# Build model
ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
cfg = ckpt["config"]
model_cfg = ModelConfig(
    spectral_input_channels=4,
    dropout=cfg["dropout"],
    qnn_qubits=cfg["qnn_qubits"],
    qnn_depth=cfg["qnn_depth"],
    qnn_init_scale=cfg["qnn_init_scale"],
    quantum_device="lightning.qubit",  # CPU inference
    classical_only=False,
)
device = torch.device("cpu")
model = build_model(model_cfg, device)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

# Prepare data (same splits as training)
data = prepare_data(
    data_root=DATA_ROOT,
    output_dir=Path("./outputs"),
    dataset_format="adc",
    seed=42,
)

# CRITICAL: Verify splits match the training splits exactly
assert data.train.rows == 33138, f"Train rows mismatch: {data.train.rows} != 33138"
assert data.val.rows == 4142, f"Val rows mismatch: {data.val.rows} != 4142"
assert data.holdout.rows == 4143, f"Holdout rows mismatch: {data.holdout.rows} != 4143"
```

### How to evaluate

```python
from ariel_quantum_regression.training import evaluate_labeled_split
import torch.nn as nn

loss_fn = nn.MSELoss()

# Quantum ON
metrics_q = evaluate_labeled_split(
    model, data.val, data.target_scaler, batch_size=128, loss_fn=loss_fn,
    enable_quantum=True, quantum_scale=0.5,  # match checkpoint's scale
)

# Quantum OFF
metrics_c = evaluate_labeled_split(
    model, data.val, data.target_scaler, batch_size=128, loss_fn=loss_fn,
    enable_quantum=False,
)
```

Both return dicts with: `rmse_mean`, `mae_mean`, `rmse_orig` (per-target array), `mae_orig`, `pred_orig`, `true_orig`.

---

## File Structure

```
quantum_advantage/
├── ariel_quantum_regression/    # model code (already here)
├── trained_weights/             # checkpoint (already here)
├── quantum_advantage_test.ipynb # THE notebook — all tests in one place
├── docs/plans/
│   └── 2026-03-12-quantum-advantage-testing.md  (this file)
└── research.md
```

---

## Task 0: Setup — first cells of the notebook

**File:** `quantum_advantage_test.ipynb` — Section 0

**What:** The opening cells of the notebook that load everything once and define reusable helpers:

1. **Imports & path setup cell** — add `quantum_advantage/` to `sys.path`, import from `ariel_quantum_regression.*` (NOT `models.ariel_quantum_regression.*`)
2. **Load checkpoint cell** — load `best_model.pt`, build `HybridArielRegressor` with `quantum_device="lightning.qubit"`, load state dict
3. **Load data cell** — call `prepare_data(dataset_format="adc", seed=42)`, verify splits: `train=33138, val=4142, holdout=4143`
4. **Helper functions cell** — define:
   - `evaluate(model, split, target_scaler, enable_quantum, quantum_scale)` → metrics dict
   - `compare_table(results_dict)` → prints pandas DataFrame comparison
   - `plot_per_target_comparison(results_dict)` → grouped bar chart
   - `TARGET_NAMES = ["H₂O", "CO₂", "CO", "CH₄", "NH₃"]`
   - `QUANTUM_SCALE_AT_BEST = 0.5`

**Important detail:** The saved checkpoint was trained with `quantum_scale=0.5` at best epoch. When testing quantum ON, use `quantum_scale=0.5` (the value the model was optimized at), NOT 1.0. Also test at 1.0 for comparison.

All subsequent sections reuse `model`, `data`, and the helper functions — no redundant loading.

---

## Task 1: Quantum ON vs OFF (same checkpoint)

**Notebook section:** `## Test 1: Quantum ON vs OFF`

**Purpose:** The most direct test. Load the trained model, evaluate on validation and holdout with `enable_quantum=True` vs `enable_quantum=False`. The forward pass already supports this — no weight modification needed.

**What to measure:**
- mRMSE (mean across 5 targets) for both conditions
- Per-target RMSE (H2O, CO2, CO, CH4, NH3) for both conditions
- Delta = quantum_on - quantum_off (negative = quantum helps)
- Test at quantum_scale=0.5 (checkpoint optimum) AND quantum_scale=1.0 (full quantum)
- Quantum scale sweep: evaluate at scale 0.0, 0.1, 0.2, ..., 1.5, 2.0 to find the optimal scale and show the contribution curve

**Expected structure:**
1. Load model and data using helpers
2. Evaluate validation set: quantum ON (scale=0.5), quantum ON (scale=1.0), quantum OFF
3. Evaluate holdout set: same three conditions
4. Print comparison table
5. Bar chart: per-target RMSE comparison (3 bars per target)
6. Scatter plot: predicted vs true for best and worst targets, quantum ON vs OFF
7. Line plot: mRMSE vs quantum_scale (0.0 to 2.0) — shows whether quantum helps, hurts, or is flat

The scale sweep takes ~2 minutes extra (20 forward passes on validation) and produces the single most compelling visualization for the presentation.

**Key question answered:** Does the quantum branch improve predictions on data the model was trained on?

---

## Task 2: Gate Inspection

**Notebook section:** `## Test 2: Gate Inspection`

**Purpose:** The `quantum_gate` parameter (5 scalars) controls how much quantum correction each target receives. If `tanh(gate) ≈ 0`, the model learned to ignore quantum for that target.

**What to measure:**
- Raw gate values (5 floats from `model.quantum_gate.data`)
- `tanh(gate)` values — the effective gating per target
- Effective contribution = `quantum_scale * tanh(gate)` at scale=0.5

**Expected structure:**
1. Load checkpoint, extract `quantum_gate` parameter
2. Table: target name | raw gate | tanh(gate) | effective at scale=0.5
3. Bar chart of tanh(gate) values per target
4. Interpretation: which targets does the model trust quantum for?

**Key question answered:** Did the model learn to use or suppress the quantum branch?

---

## Task 3: Quantum Output Analysis

**Notebook section:** `## Test 3: Quantum Output Analysis`

**Purpose:** Examine what the quantum circuit actually outputs. If the 8 PauliZ expectations are near-constant across samples, the circuit isn't doing sample-dependent computation.

**What to measure:**
- Run all validation samples through the model, hook into `QuantumBlock` output
- Statistics per qubit: mean, std, min, max across samples
- Correlation matrix: qubit outputs vs each other
- Correlation: qubit outputs vs true targets (do quantum features predict targets?)
- PCA of the 8-dim quantum output: how many effective dimensions?
- Histogram: distribution of each qubit's output across samples

**Expected structure:**
1. Load model and data
2. Register forward hook on `model.quantum_block` to capture outputs
3. Run validation data through model in eval mode
4. Collect all quantum outputs (N_val × 8 matrix)
5. Statistics table
6. Correlation heatmap (8 qubits × 5 targets)
7. PCA explained variance plot
8. Histograms of qubit outputs (2×4 grid)

**Also capture:** The `quantum_angles` (output of QuantumProjector) — are the input angles to the circuit diverse or collapsed?

**Also compute: Residual correlation analysis.** Take the classical-only prediction error (from `enable_quantum=False`) and check if the quantum correction vector correlates with that residual. If `quantum_correction ∝ (true - classical_pred)`, the quantum branch learned to fix the classical model's mistakes — strong evidence of useful quantum contribution. If uncorrelated, the quantum correction is essentially noise that got lucky during training.

**Key question answered:** Is the quantum circuit computing something meaningful and sample-dependent?

---

## Task 4: Noise Replacement Ablation

**Notebook section:** `## Test 4: Noise Replacement Ablation`

**Purpose:** Replace the quantum circuit output with random noise of the same distribution. If performance is similar, the circuit's specific computation doesn't matter — the QuantumHead MLP can extract value from random features + classical context.

**What to do:**
1. Load the trained model
2. Monkey-patch `model.quantum_block.forward` to return:
   - **Random uniform [-1, 1]** (same range as PauliZ expectations)
   - **Random with same mean/std** as the real quantum outputs (from Test 3)
   - **Zeros** (constant, no quantum info)
3. Evaluate on validation and holdout sets under each condition
4. Compare mRMSE: real quantum vs random vs zeros vs quantum OFF

**Expected structure:**
1. Load model and data
2. First, run real quantum to get baseline metrics and quantum output stats
3. Define replacement forward functions
4. For each replacement: patch, evaluate, record metrics
5. Comparison table: condition | val_mRMSE | holdout_mRMSE
6. Bar chart comparison

**Key question answered:** Does the quantum circuit provide signal beyond what random features would give the QuantumHead?

---

## Task 5: Classical MLP Replacement ⚠️ REQUIRES GPU / SLOW ON CPU

**Notebook section:** `## Test 5: Classical MLP Replacement` (GPU-only, skip if no GPU)

**Purpose:** Replace the quantum circuit (24 params) with a classical MLP of similar capacity. Train this replacement from the same init checkpoint with the same config. If it matches or beats the quantum model, the quantum circuit adds no advantage.

**What to do:**
1. Define `ClassicalReplacementBlock(nn.Module)`: a small MLP that takes 8 inputs (from the projector) and outputs 8 values in [-1, 1] (tanh activation). Match parameter count: ~24 params → Linear(8, 3) + tanh (24+3=27 params, close enough).
2. Build the full model with `classical_only=False` but swap `QuantumBlock` for `ClassicalReplacementBlock`
3. Initialize from the same stage-1 classical checkpoint (`init_checkpoint_path` from config)
4. Train with the same config (quantum_warmup=0, ramp=12, backbone_freeze=6, max_epochs=30)
5. Compare best validation mRMSE and holdout mRMSE against the quantum model

**Important:** This test requires retraining (~12 min per epoch × 8+ epochs on CPU, or faster on GPU). If no GPU available, reduce `train_limit` to e.g. 5000 for a faster comparison.

**Alternative fast version:** Instead of retraining, just:
1. Take the trained quantum model
2. Replace QuantumBlock with ClassicalReplacementBlock (random init)
3. Freeze everything except the replacement block + QuantumHead + quantum_gate
4. Fine-tune for a few epochs on the same data
5. Compare

**Key question answered:** Can a classical function of equal parameter count replace the quantum circuit?

---

## Task 6: Classical-Only Retrain ⚠️ REQUIRES GPU / SLOW ON CPU

**Notebook section:** `## Test 6: Classical-Only Retrain` (GPU-only, skip if no GPU)

**Purpose:** Train the model from scratch with `classical_only=True`. This removes the entire quantum branch (Projector + Circuit + QuantumHead + gate ≈ 69k params). The model uses only ClassicalHead(head_context). Compare against the hybrid model.

**What to do:**
1. Use `run_ariel_quantum_regression.py` with `--classical-only` flag
2. Same dataset, same seed, same hyperparams (but classical_lr applies to all params)
3. Train for 30 epochs with early stopping
4. Compare: classical-only holdout mRMSE vs hybrid holdout mRMSE (0.299)

**Important consideration:** The classical-only model has fewer parameters (~118k vs ~187k). To make a fair comparison:
- Also train a "fat classical" variant where ClassicalHead is enlarged to ~187k total (e.g., hidden_dim=384 instead of 192)
- This controls for the parameter count difference

**Execution:** Run from the `quantum_advantage/` directory (or adapt `--data-root` path):
```bash
cd quantum_advantage
python -m ariel_quantum_regression.run_ariel_quantum_regression \
    --project-root .. \
    --data-root ../datasets/ariel-ml-dataset \
    --output-dir tests/outputs/classical_only \
    --classical-only \
    --seed 42 \
    --max-epochs 30
```

Note: The `run_ariel_quantum_regression.py` internally imports from `models.ariel_quantum_regression` (hardcoded). You may need to adjust `sys.path` or run from the project root with `python -m models.ariel_quantum_regression.run_ariel_quantum_regression` if the repo has a `models/` directory. Alternatively, just use `TrainingConfig` directly in the notebook.

**Key question answered:** Does a model without any quantum components perform worse, equal, or better?

---

## Task 7: Statistical Significance

**Notebook section:** `## Test 7: Statistical Significance`

**Purpose:** For any mRMSE differences found in Tests 1-6, determine if they are statistically significant.

**What to do:**
1. Load predictions from quantum ON and quantum OFF (from Test 1)
2. Per-sample squared error: `(pred - true)^2` for each condition
3. **Paired t-test**: for each target, test if mean squared error differs between conditions
4. **Bootstrap confidence interval**: resample holdout predictions 10,000 times, compute mRMSE each time, get 95% CI for the difference
5. **Effect size**: Cohen's d for the mRMSE difference
6. **Per-target analysis**: some targets might benefit from quantum while others don't

**Expected structure:**
1. Load predictions from Test 1 (or recompute)
2. Compute per-sample errors for both conditions
3. Paired t-test table: target | t-stat | p-value | significant?
4. Bootstrap CI plot: distribution of mRMSE difference with 95% CI band
5. Effect size table
6. Conclusion: is the difference real or noise?

**Key question answered:** Is the measured quantum advantage (if any) statistically robust?

---

## Task 8: Summary Notebook

**Notebook section:** `## Verdict: Does Quantum Advantage Exist?`

**Purpose:** Collect all results into a single view with a clear verdict.

**Structure:**
1. Executive summary: one paragraph answer to "does quantum advantage exist?"
2. Results table: all tests, key metric, verdict per test
3. Combined visualization: multi-panel figure showing the key comparisons
4. Interpretation for the hackathon presentation
5. Recommended talking points (what to claim, what to be honest about)

| Test | What it shows | Verdict criteria |
|---|---|---|
| 1. ON/OFF | Direct quantum contribution | mRMSE_on < mRMSE_off |
| 2. Gate | Model's learned quantum trust | tanh(gate) >> 0 |
| 3. Outputs | Circuit computes something | high variance, target correlation |
| 4. Noise | Circuit > random | mRMSE_real < mRMSE_random |
| 5. MLP swap | Circuit > classical equivalent | mRMSE_quantum < mRMSE_mlp |
| 6. Retrain | Quantum branch helps overall | mRMSE_hybrid < mRMSE_classical |
| 7. Significance | Results are real | p < 0.05, CI excludes 0 |

---

## Notebook Execution

**Single notebook, run top to bottom.** Model and data are loaded once in Section 0. Each test section produces its own metrics and plots. Results accumulate into a `results` dict that feeds the final verdict section.

**Notebook sections in order:**
1. Section 0: Setup (imports, model, data) — ~3 min (data loading)
2. Test 1: ON/OFF + scale sweep — ~3 min
3. Test 2: Gate inspection — instant
4. Test 3: Quantum output analysis + residual correlation — ~2 min
5. Test 4: Noise replacement — ~3 min
6. Test 5: MLP replacement — ⚠️ skip if no GPU
7. Test 6: Classical retrain — ⚠️ skip if no GPU
8. Test 7: Statistical significance — ~1 min
9. Verdict — instant (aggregation + final plots)

**Fast path (CPU, ~15 min):** Sections 0→1→2→3→4→7→Verdict. Skip Tests 5 and 6.

**Full path (GPU):** All sections including Tests 5 and 6.

---

## Environment Setup

```bash
cd quantum_advantage
python -m venv .venv
source .venv/bin/activate
pip install torch pennylane pennylane-lightning numpy pandas scikit-learn h5py matplotlib scipy jupyter
```

Or if using existing environment, just ensure `pennylane` and `torch` are installed.

---

## What Each Outcome Means for the Hackathon

**If quantum helps (mRMSE improves, statistically significant):**
- Claim: "Hybrid quantum-classical model outperforms classical-only by X%"
- Show: per-target breakdown, which molecules benefit most
- Honest framing: advantage is on classically simulated circuit, but demonstrates the architecture works

**If quantum is neutral (no significant difference):**
- Claim: "Quantum layer integrates seamlessly without degrading performance"
- Pivot: "Architecture is quantum-ready — designed for real hardware deployment on Odra 5 / VTT Q50"
- Show: the architecture and quantum circuit design as the contribution

**If quantum hurts (mRMSE worse with quantum):**
- Don't hide it — show you tested rigorously
- Frame: "We identified that the classical backbone captures the signal; quantum layer needs deeper circuits / more qubits"
- Show: the testing methodology as scientific rigor
