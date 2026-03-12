# Quantum Advantage Research — ariel_quantum_regression

## Model Overview

**Task**: Regression of 5 atmospheric gas abundances (log10 VMR of H2O, CO2, CO, CH4, NH3) from exoplanet transmission spectra (Ariel ADC 2023 format, 52 wavelength bins).

**Architecture**: `HybridArielRegressor` — classical backbone with an additive quantum residual correction.

**Trained checkpoint**: `ariel_quantum_best_v4_epoch6` — 8 qubits, depth 2, best at epoch 6, holdout mRMSE = 0.299.

---

## Architecture Breakdown

### Classical backbone (always active)

| Component | Structure | Output dim |
|---|---|---|
| SpectralEncoder | Conv1d stem (4→32, k=5) → 3 residual blocks (32→32→64→96) with GroupNorm + GELU, attention+mean pooling, linear projection | 96 |
| AuxEncoder | MLP: 8→32→32 (8 auxiliary planet/star features, log10-transformed) | 32 |
| FusionEncoder | MLP: 128→128 (concat of spectral + aux) | 128 |
| ClassicalHead | MLP: 256→192→5 (input = fused 128 + spectral 96 + aux 32 = 256) | 5 targets |

### Quantum branch (additive correction)

| Component | Structure | Output dim |
|---|---|---|
| QuantumProjector | MLP: 128→128→8 + LayerNorm + tanh * π → rotation angles | 8 |
| QuantumBlock | PennyLane circuit: 8 qubits, 1 variational block (depth=2 means depth//2=1 block) | 8 (PauliZ expectations) |
| QuantumHead | MLP: 264→192→5 (input = head_context 256 + quantum_features 8 = 264) | 5 targets |
| quantum_gate | Learnable 5-dim parameter, applied as `tanh(gate)` per target | scalar gate per target |

### Forward pass equation

```
output = classical_pred + quantum_scale * tanh(quantum_gate) * quantum_head(concat(head_context, quantum_features))
```

Where `quantum_features = QuantumBlock(QuantumProjector(fused_features))`.

---

## Quantum Circuit Analysis

### Circuit structure (1 variational block)

```
For each qubit q in [0..7]:
    RY(input_angle[q])          # data encoding

For each qubit q in [0..7]:
    RY(weight[q])               # variational rotation
For each qubit q in [0..7]:
    CNOT(q, (q+1) % 8)         # ring entanglement
For each qubit q in [0..7]:
    RZ(weight[8+q])             # variational rotation
For each qubit q in [0..7]:
    CRX(weight[16+q], q, (q+1) % 8)  # parameterized ring entanglement

Measure: PauliZ expectation on all 8 qubits → 8 float outputs in [-1, 1]
```

### Key numbers

- **Qubits**: 8
- **Trainable circuit parameters**: 24 (3 rotations × 8 qubits × 1 block)
- **Hilbert space dimension**: 2^8 = 256
- **Entanglement topology**: ring (nearest-neighbor)
- **Classical simulation cost**: trivial — PennyLane `lightning.qubit` is a statevector simulator running on CPU

### Expressivity concerns

The circuit has limited entanglement depth (one layer of CNOT + one layer of CRX, both ring topology). This is well within classical simulability — no exponential advantage is possible with this structure and qubit count. The circuit is equivalent to a specific family of 256-dim unitary transformations parameterized by 24 values, which is a tiny fraction of all possible 8-qubit unitaries.

---

## Training Strategy for the Saved Checkpoint

The checkpoint was trained in two stages:

### Stage 1: Classical pre-training (separate run)
- Classical backbone trained alone (init_checkpoint from `ariel_quantum_two_stage_v2`)
- This produces a strong classical baseline before quantum fine-tuning

### Stage 2: Quantum fine-tuning (the saved checkpoint)
- `quantum_warmup_epochs = 0` — quantum active from epoch 1
- `quantum_ramp_epochs = 12` — quantum_scale linearly ramps from 1/12 ≈ 0.083 to 1.0
- `quantum_backbone_freeze_epochs = 6` — backbone frozen for first 6 epochs, only quantum components train
- `classical_lr = 5e-5`, `quantum_lr = 2e-4` — quantum gets 4× higher learning rate
- Best epoch = **6** (last epoch with backbone frozen), quantum_scale = **0.5** at that point
- Training stopped at epoch 8 (early stopping, patience 8 but performance degraded)

### Training history

| Epoch | quantum_scale | backbone_frozen | val_rmse_mean | delta from epoch 1 |
|---|---|---|---|---|
| 1 | 0.083 | yes | 0.2933 | — |
| 2 | 0.167 | yes | 0.2924 | -0.0009 |
| 3 | 0.250 | yes | 0.2919 | -0.0015 |
| 4 | 0.333 | yes | 0.2916 | -0.0018 |
| 5 | 0.417 | yes | 0.2913 | -0.0020 |
| 6 | 0.500 | yes | **0.2908** | **-0.0025** |
| 7 | 0.583 | no | 0.3287 | +0.0354 |
| 8 | 0.667 | no | 0.3218 | +0.0285 |

**Observation**: The total quantum-attributed improvement during frozen-backbone training is **0.0025 mRMSE** (from 0.2933 to 0.2908). This is very small. When the backbone unfreezes at epoch 7, performance crashes — the quantum gradient signal disrupts the pre-trained classical weights.

---

## Critical Issues for Quantum Advantage Claim

### 1. The model is classically simulated
The entire training and inference uses PennyLane's `lightning.qubit` (or `lightning.gpu`) — both are classical statevector simulators. No quantum hardware is involved. The circuit with 8 qubits is trivially simulable. There is no computational quantum advantage here.

### 2. Additive residual design limits quantum contribution
The quantum branch can only add a correction to the classical prediction: `classical_pred + scale * gate * quantum_correction`. If `gate ≈ 0`, the model is effectively classical. The architecture makes it easy for the model to "turn off" the quantum branch entirely.

### 3. Massive information bottleneck
128-dim fused features → projected to 8 rotation angles → 8 PauliZ expectations → concatenated with 256 classical features (8 out of 264 = 3% of QuantumHead input). The quantum information is drowned in classical context.

### 4. The 0.0025 mRMSE improvement is not validated
- We don't know the val_rmse of the stage 1 classical-only model (the init checkpoint) — it may already be ~0.293
- The improvement could come entirely from the QuantumProjector + QuantumHead MLPs (classical parameters), not from the quantum circuit
- No ablation was done: removing the circuit and replacing with random or zero features would test this
- No statistical significance test (the val set has 4142 samples — standard error of mRMSE is non-trivial)

### 5. quantum_gate values unknown but architecturally suspicious
The `quantum_gate` is initialized at zeros (tanh(0) = 0, no quantum contribution). During training, if the gate values remain small, the model learned to mostly ignore the quantum branch. We need to inspect the saved checkpoint to check this.

### 6. QuantumHead has its own classical capacity
The QuantumHead is a separate MLP (264→192→5) that sees both classical context AND quantum features. Even if the quantum circuit outputs noise, the QuantumHead could learn useful patterns from the 256 classical features it receives alongside the 8 quantum outputs. Any improvement could be from this extra classical capacity, not from quantum features.

---

## What We Need to Test

### A. Ablation: quantum off vs quantum on (same checkpoint)
Load the saved checkpoint and evaluate with `enable_quantum=False` vs `enable_quantum=True`. The forward pass already supports this flag. This tells us how much the trained quantum branch contributes at inference time.

### B. Inspect quantum_gate values
Load `best_model.pt` and read the `quantum_gate` parameter. If `tanh(gate) ≈ 0` for any target, the quantum branch is effectively disabled for that target.

### C. Inspect QuantumBlock output distribution
Run the model on validation data and record the 8-dim quantum_features output. Check: are they near-constant? Do they correlate with targets? What's their variance across samples?

### D. Ablation: replace quantum circuit with noise
Replace `QuantumBlock.forward()` with random uniform [-1, 1] of the same shape. If performance is similar, the circuit adds nothing beyond what the QuantumHead MLP can extract from classical context.

### E. Ablation: replace quantum circuit with classical MLP
Replace the entire quantum branch (Projector + Circuit + QuantumHead) with a classical MLP of equivalent parameter count (~24 params in circuit + projector + quantum_head params). Train from scratch with same config. If classical MLP matches or beats the quantum model, there's no quantum advantage.

### F. Classical-only retraining baseline
Train the same model with `--classical-only` flag on the same data/splits/seed. Compare holdout mRMSE directly. The `classical_only=True` mode skips the entire quantum branch and uses only `ClassicalHead(head_context)`.

### G. Statistical significance
For any mRMSE difference found, compute confidence intervals (bootstrap or paired t-test across samples) to determine if the difference is statistically significant.

---

## Parameter Count Comparison

| Component | Parameters (approx) |
|---|---|
| SpectralEncoder | ~50k |
| AuxEncoder | ~1.3k |
| FusionEncoder | ~17k |
| ClassicalHead (256→192→5) | ~50k |
| **Classical total** | **~118k** |
| QuantumProjector (128→128→8) | ~17k |
| QuantumBlock (circuit weights) | **24** |
| QuantumHead (264→192→5) | ~52k |
| quantum_gate | 5 |
| **Quantum branch total** | **~69k** |
| **Full model total** | **~187k** |

The quantum circuit itself contributes only **24 trainable parameters** out of ~187k total (~0.01%). The quantum branch adds ~69k parameters, but 99.96% of those are classical (projector + quantum_head + gate).

---

## Summary

The model has a well-engineered classical backbone (SpectralEncoder + AuxEncoder + FusionEncoder + ClassicalHead) that does the heavy lifting. The quantum branch is an additive residual correction with a tiny circuit (8 qubits, 24 params) surrounded by much larger classical adapters. The training history shows only 0.0025 mRMSE improvement attributable to the quantum fine-tuning stage, and this hasn't been validated against proper ablations. The circuit is classically simulated and architecturally bottlenecked. The key question is whether the 8-qubit circuit provides any signal that couldn't be replicated by a simple classical function of the same dimensionality.
