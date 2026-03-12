# Audit 09 -- Model Architecture Sanity

**Scope**: Hybrid quantum-classical neural network (`HybridAtmosphereModel`) for predicting log10 VMR of 5 atmospheric gases from rebinned exoplanet transmission spectra.

**Source file**: `/Users/michalszczesny/projects/hack4sages/codex_model/crossgen_hybrid_training.py`
**Config**: `/Users/michalszczesny/projects/hack4sages/codex_model/outputs/model_crossgen_rebinned/config.json`
**Run summary**: `/Users/michalszczesny/projects/hack4sages/codex_model/outputs/model_crossgen_rebinned/run_summary.json`

---

## 1. Tensor Shape Trace (end-to-end)

### Input shapes
- `aux`: `(B, 5)` -- 5 auxiliary features
- `spectra`: `(B, 1, 44)` -- 1-channel, 44 wavelength bins (rebinned from 218)

### AuxEncoder (lines 365-379)
```
Linear(5, 64)  -> GELU -> Dropout(0.05)
Linear(64, 64) -> GELU
Linear(64, 32) -> GELU
```
Output: `(B, 32)` -- correct.

### SpectralEncoder (lines 382-402)
```
Conv1d(1, 32, k=7, p=3)   -> GELU        # (B, 32, 44)
Conv1d(32, 64, k=5, s=2, p=2) -> GELU     # (B, 64, 22)
Conv1d(64, 64, k=3, p=1)  -> GELU         # (B, 64, 22)
AdaptiveAvgPool1d(1)                       # (B, 64, 1)
Flatten                                    # (B, 64)
Linear(64, 32) -> GELU -> Dropout(0.05)
```
Output: `(B, 32)` -- correct.

**Conv1d shape verification**:
- Layer 1: input `(B, 1, 44)`, `k=7, p=3` -> output length = `(44 + 2*3 - 7)/1 + 1 = 44`. Output: `(B, 32, 44)`. Correct.
- Layer 2: input `(B, 32, 44)`, `k=5, s=2, p=2` -> output length = `floor((44 + 2*2 - 5)/2) + 1 = floor(43/2) + 1 = 22`. Output: `(B, 64, 22)`. Correct.
- Layer 3: input `(B, 64, 22)`, `k=3, p=1` -> output length = `(22 + 2*1 - 3)/1 + 1 = 22`. Output: `(B, 64, 22)`. Correct.
- AdaptiveAvgPool1d(1): `(B, 64, 22)` -> `(B, 64, 1)`. Correct.
- Flatten: `(B, 64, 1)` -> `(B, 64)`. Correct.
- Linear(64, 32): `(B, 64)` -> `(B, 32)`. Correct.

### FusionEncoder (lines 405-417)
```
cat([aux_feat(B,32), spectral_feat(B,32)], dim=-1)  -> (B, 64)
Linear(64, 48) -> GELU
Linear(48, 12) -> LayerNorm(12)
tanh(...) * pi
```
Output: `(B, 12)` -- matches `qnn_qubits=12`. Correct.

### QuantumBlock (lines 430-486)
Input: `(B, 12)` -- 12 angles for 12 qubits via RY encoding.
Circuit: 1 block (depth=2, num_blocks=1): RY + CNOT ring + RZ + CRX ring.
Number of trainable weights: `3 * 12 * 1 = 36`.
Output: 12 Pauli-Z expectations -> `(B, 12)`. Correct.

### AtmosphereHead (lines 489-503)
```python
head_in = cat([quantum_feat(B,12), latent(B,12), aux_feat(B,32), spectral_feat(B,32)], dim=-1)
# head_in shape: (B, 12+12+32+32) = (B, 88)
```
Config at line 563: `in_dim = qnn_qubits*2 + aux_out_dim + spectral_out_dim = 12*2 + 32 + 32 = 88`. Correct.

```
MLP:
  Linear(88, 96) -> GELU -> Dropout(0.05)
  Linear(96, 96) -> GELU
  Linear(96, 5)
Residual: Linear(12, 5)   # latent_dim = qnn_qubits = 12

Output = MLP(head_in) + Residual(quantum_feat)
```
Output: `(B, 5)` -- 5 targets. Correct.

---

## 2. Findings

### FINDING 2.1 -- Skip Connection from Quantum Output May Dominate or Be Bypassed

**Severity**: MEDIUM

**Location**: lines 489-503

```python
class AtmosphereHead(nn.Module):
    def __init__(self, in_dim, latent_dim, hidden_dim, n_targets, dropout):
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),  # (88 -> 96)
            ...
            nn.Linear(hidden_dim, n_targets),  # (96 -> 5)
        )
        self.residual = nn.Linear(latent_dim, n_targets)  # (12 -> 5)

    def forward(self, head_in, latent):
        return self.mlp(head_in) + self.residual(latent)
```

The `self.residual` skip connection takes `quantum_feat` (12-dim PauliZ expectations in [-1, 1]) and linearly maps it to 5 outputs. Meanwhile, `quantum_feat` is *also* included in `head_in` that feeds the MLP. This means:
1. The quantum output has two independent gradient pathways to the loss. The skip provides a direct linear path.
2. During early training, before the MLP converges, the skip connection can dominate predictions. This is a standard residual design choice and is not necessarily harmful.
3. However, the skip also means the classical MLP path (which sees all features) can learn to ignore the quantum block entirely, reducing the quantum contribution to a simple linear projection.

**Risk**: If the MLP learns to compensate for the quantum output being noisy/uninformative, the quantum block's gradient signal weakens. The final model may effectively be a classical model with 36 quantum parameters that contribute little. This is not a correctness bug but a potential scientific validity concern for claiming quantum advantage.

**Recommendation**: After training, inspect `self.residual.weight` magnitudes vs. the MLP final layer weights. If the residual pathway dominates, the quantum block is doing most of the work through a simple linear transform (no non-linearity), which undermines the motivation for a deep MLP head. If it is near-zero, the quantum block contributes nothing beyond noise.

---

### FINDING 2.2 -- LayerNorm Followed by tanh Compression in FusionEncoder

**Severity**: LOW

**Location**: lines 405-417

```python
class FusionEncoder(nn.Module):
    def __init__(self, aux_dim, spec_dim, hidden_dim, out_dim):
        self.net = nn.Sequential(
            nn.Linear(aux_dim + spec_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(self, aux_feat, spectral_feat):
        fused = torch.cat([aux_feat, spectral_feat], dim=-1)
        return torch.tanh(self.net(fused)) * math.pi
```

LayerNorm normalizes activations to approximately zero mean and unit variance. The subsequent `tanh * pi` then maps this range. Since LayerNorm outputs are already concentrated near zero with unit variance, `tanh` on those values will operate mostly in its linear regime (tanh(x) ~ x for |x| < 1), meaning the output will cluster in approximately [-pi, +pi] but effectively scale as a near-linear transform of the LayerNorm output.

**Assessment**: This is intentional -- the goal is to produce rotation angles for RY gates in the quantum circuit, bounded in [-pi, pi]. LayerNorm prevents dead saturation of tanh. The design is sound for this purpose. The only minor concern is that angles will rarely approach +/-pi (since LayerNorm keeps values near unit variance), meaning the full Bloch sphere rotation range is underutilized. This is a minor expressivity limitation, not a bug.

---

### FINDING 2.3 -- Dropout Placement is Sparse but Acceptable

**Severity**: LOW

**Location**: lines 365-402, 489-498

Dropout (p=0.05) is applied in:
1. **AuxEncoder**: After the first `Linear(5,64) -> GELU`, i.e., only after the first hidden layer (line 371).
2. **SpectralEncoder**: After the projection `Linear(64,32) -> GELU`, at the very end (line 398).
3. **AtmosphereHead**: After the first `Linear(88,96) -> GELU` (line 495).
4. **FusionEncoder**: No dropout at all.
5. **QuantumBlock**: No dropout (not applicable -- quantum noise acts as implicit regularization).

With p=0.05, the regularization effect is minimal. For 37k training samples and ~30k parameters (see Finding 2.5), dropout at this level is more of a token presence than a meaningful regularizer. The `weight_decay=1e-4` is doing more regularization work.

**Assessment**: Acceptable. The low dropout rate combined with early stopping (patience=6) and weight decay provides sufficient regularization for this dataset size. No layers are missing dropout in a way that creates a correctness issue. `model.eval()` correctly disables dropout at inference time via standard PyTorch semantics (confirmed at line 646 via `model.eval()` in `evaluate_split`).

---

### FINDING 2.4 -- Fusion by Simple Concatenation is Adequate

**Severity**: LOW

**Location**: lines 405-417

The FusionEncoder concatenates `aux_feat (B,32)` and `spectral_feat (B,32)` into `(B,64)`, then passes through a 2-layer MLP (64->48->12). This is a simple concatenation-based fusion.

**Assessment**: For a model of this size with these input modalities, concatenation fusion is standard and appropriate. More sophisticated fusion mechanisms (cross-attention, bilinear pooling, gating) would add parameters and complexity without clear benefit given:
- The two modalities are low-dimensional (32 each).
- The fusion MLP has a hidden layer that can learn interactions.
- The dataset is only 37k samples.

No issue here.

---

### FINDING 2.5 -- Parameter Count Analysis

**Severity**: LOW (informational)

**Estimated parameter count**:

| Component | Calculation | Params |
|-----------|------------|--------|
| AuxEncoder | (5*64+64) + (64*64+64) + (64*32+32) = 384 + 4160 + 4160 + 64 + 2080 = 6,912 | 6,912 |
| SpectralEncoder (conv) | (1*32*7+32) + (32*64*5+64) + (64*64*3+64) = 256 + 10304 + 12352 = 22,912 | 22,912 |
| SpectralEncoder (proj) | (64*32+32) = 2,080 | 2,080 |
| FusionEncoder | (64*48+48) + (48*12+12) + LayerNorm(12*2) = 3,120 + 588 + 24 = 3,732 | 3,732 |
| QuantumBlock | 36 trainable weights | 36 |
| AtmosphereHead (MLP) | (88*96+96) + (96*96+96) + (96*5+5) = 8,544 + 9,312 + 485 = 18,341 | 18,341 |
| AtmosphereHead (residual) | (12*5+5) = 65 | 65 |
| **Total** | | **~54,078** |

**Detailed breakdown**:
- Classical parameters: ~54,042
- Quantum parameters: 36
- Quantum fraction: 0.07%

**Ratio of parameters to training samples**: 54,078 / 37,281 ~ 1.45 parameters per sample.

**Assessment**: The model is appropriately sized. A ratio of ~1.5 parameters per training sample is moderate for a regression task with regularization (weight decay, early stopping, dropout). There is no risk of massive overfitting. The quantum block contributes a vanishingly small fraction of the total parameters, which raises questions about the quantum block's capacity to contribute meaningfully (see also Finding 2.1 and 2.7).

---

### FINDING 2.6 -- Potential for Vanishing Gradients Through Quantum Block

**Severity**: MEDIUM

**Location**: lines 430-486, especially line 462

```python
for qubit in range(n_qubits):
    qml.RY(inputs[..., qubit], wires=qubit)
```

The input angles come from `FusionEncoder` output: `tanh(LayerNorm(x)) * pi`. As noted in Finding 2.2, these values concentrate around [-pi, pi] with most values near zero. The RY gate produces `|psi> = cos(theta/2)|0> + sin(theta/2)|1>`, so PauliZ expectation = `cos(theta)`. The gradient of `cos(theta)` w.r.t. theta is `-sin(theta)`, which vanishes at theta=0 and theta=pi. Since the LayerNorm + tanh compression keeps many angles near zero, some qubits may experience weak gradients early in training.

**Mitigating factors**:
- The circuit has trainable weights (RY, RZ, CRX rotations) that are initialized with `0.5 * randn` (line 450), breaking symmetry.
- CNOT and CRX entangling gates create correlations that distribute gradient signal.
- The skip connection (Finding 2.1) provides an alternative gradient path.
- Gradient clipping at 1.0 for quantum params (line 750) prevents instability but may also slow learning.

**Assessment**: This is a known issue in variational quantum circuits. The combination of LayerNorm + tanh angle encoding and parameter-shift rule gradients makes it possible but not certain that quantum gradients are weak. The training curve (history.csv) shows monotonic improvement across 30 epochs with no plateau, suggesting gradients are flowing, though we cannot distinguish classical-only from quantum contributions from the loss curve alone.

---

### FINDING 2.7 -- Massive Val-to-Test Performance Gap Suggests Generalization Failure

**Severity**: CRITICAL

**Location**: Run outputs

| Metric | Inner-Val (TauREx) | Test (Poseidon) |
|--------|-------------------|-----------------|
| RMSE mean | 1.54 | 3.46 |
| H2O | 1.66 | 3.13 |
| CO2 | 1.16 | 4.73 |
| CO | 2.18 | 3.28 |
| CH4 | 1.22 | 3.23 |
| NH3 | 1.50 | 2.95 |

The test RMSE is 2.25x the validation RMSE. For log10 VMR predictions, an RMSE of 3.46 means the model's predictions are off by approximately 3.5 orders of magnitude on average -- the model is essentially non-predictive on the Poseidon test set. CO2 is the worst at 4.73 dex RMSE.

**Root cause analysis**: This is a distribution shift problem, not an architecture bug per se. Train/val are TauREx-generated spectra; test is Poseidon-generated. The model memorizes TauREx-specific artifacts rather than learning generalizable atmospheric physics. However, the architecture could be contributing:
1. The SpectralEncoder's 3-layer ConvNet with relatively large receptive field (kernel sizes 7, 5, 3) could overfit to TauREx-specific spectral patterns.
2. The low dropout (0.05) and no data augmentation leave no defense against domain-specific memorization.
3. The model trained for all 30 epochs with the best checkpoint at epoch 29 and no clear convergence (last val loss at epoch 30: 0.3019 vs best at epoch 29: 0.2999), suggesting it may still be slowly overfitting.

**Recommendation**: This is the most pressing issue. Architectural mitigations include stronger dropout, spectral augmentation (noise injection, random wavelength masking), or domain-adversarial training. However, the core problem is likely dataset-level and beyond the architecture audit scope. See also the fact that early stopping patience=6 never triggered -- the model kept improving on val for all 30 epochs.

---

### FINDING 2.8 -- No Batch Normalization in Convolutional Layers

**Severity**: LOW

**Location**: lines 385-393

```python
self.conv = nn.Sequential(
    nn.Conv1d(in_channels, 32, kernel_size=7, padding=3),
    nn.GELU(),
    nn.Conv1d(32, hidden_dim, kernel_size=5, stride=2, padding=2),
    nn.GELU(),
    nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
    nn.GELU(),
    nn.AdaptiveAvgPool1d(1),
)
```

The SpectralEncoder uses 3 Conv1d layers with no normalization between them. BatchNorm1d or LayerNorm after each convolution is standard practice for stabilizing training and improving generalization.

**Assessment**: With only 44-length input and AdaptiveAvgPool at the end, the network is shallow enough that this omission is unlikely to cause training instability. The spectral standardizer applied to input data (per-channel zero-mean, unit-variance) helps. However, adding BatchNorm1d after each Conv1d could improve generalization, particularly for the domain shift problem noted in Finding 2.7.

---

### FINDING 2.9 -- AMP Autocast Scope Does Not Cover the Prediction Head

**Severity**: LOW

**Location**: lines 533-548

```python
def forward(self, aux, spectra):
    autocast_ctx = ...
    with autocast_ctx:
        aux_feat = self.aux_encoder(aux)
        spectral_feat = self.spectral_encoder(spectra)
        latent = self.fusion_encoder(aux_feat, spectral_feat)

    latent = latent.float()
    quantum_feat = self.quantum_block(latent)
    head_in = torch.cat([quantum_feat, latent, aux_feat.float(), spectral_feat.float()], dim=-1)
    return self.head(head_in, quantum_feat)
```

The AMP autocast covers encoders but not the prediction head. The head runs in float32. This is intentional -- the quantum block requires float32, and maintaining float32 for the head avoids precision issues in the final regression output. The explicit `.float()` casts on line 545 and 547 ensure type consistency.

**Assessment**: Correct design. The quantum circuit (PennyLane) requires float32 inputs. Running the head in float32 is appropriate for regression output precision. No issue.

---

### FINDING 2.10 -- Quantum Block is Functionally a Fixed-Width Bottleneck

**Severity**: MEDIUM

**Location**: lines 430-486

The quantum circuit takes 12-dim input, processes through 12 qubits with 36 trainable parameters, and outputs 12 PauliZ expectations. The information bottleneck is the 12-dim representation. The circuit architecture (1 block of RY + CNOT ring + RZ + CRX ring) provides limited entanglement depth.

With `qnn_depth=2` -> `num_blocks=1`, there is exactly one round of entanglement. The CRX ring and CNOT ring each create nearest-neighbor entanglement in a ring topology, which limits the circuit's ability to create long-range correlations.

**Assessment**: For 12 qubits, a single entanglement block is shallow. The circuit's expressibility is limited -- it can represent a subset of 12-qubit unitaries, but far from all. This is a deliberate design trade-off (deeper circuits = slower simulation and worse trainability due to barren plateaus). However, combined with the skip connection (Finding 2.1), there is a real risk that the model's performance comes almost entirely from the classical components, with the quantum block contributing marginally through its linear residual pathway.

---

### FINDING 2.11 -- Dead Neuron Risk Assessment

**Severity**: LOW

**Location**: All encoder modules

All activation functions use GELU (lines 370, 372, 374, 387, 389, 391, 396, 410, 493, 497). GELU has a non-zero gradient for all inputs (unlike ReLU which has zero gradient for negative inputs), making dead neuron issues extremely unlikely. The only "hard" nonlinearity is `tanh` in the FusionEncoder (line 417), which saturates at +/-1 but is protected by LayerNorm as discussed in Finding 2.2.

**Assessment**: No dead neuron risk. GELU is an appropriate choice throughout.

---

### FINDING 2.12 -- eval() vs train() Mode Handling

**Severity**: LOW (no issue found)

**Location**: lines 646, 732

- `model.eval()` is called at line 646 in `evaluate_split()` before inference. This disables dropout.
- `model.train()` is called at line 732 at the start of each training epoch. This re-enables dropout.
- `torch.inference_mode()` is used at line 652 for evaluation, which is more efficient than `torch.no_grad()`.

The only modules affected by train/eval mode are the Dropout layers (3 total). LayerNorm in FusionEncoder does not use running statistics (unlike BatchNorm), so it behaves identically in both modes.

**Assessment**: Correct. No issue.

---

### FINDING 2.13 -- Scheduler Never Reduced Learning Rate

**Severity**: LOW (informational)

**Location**: history.csv

Across all 30 epochs, `classical_lr` remained at `0.002` and `quantum_lr` remained at `0.0006`. The `ReduceLROnPlateau` scheduler with `patience=5` never triggered because the validation loss improved almost every epoch (the longest non-improving streak was 2 epochs: 19->20 and 23->24).

**Assessment**: This means the learning rate schedule was irrelevant for this run. The model may benefit from a longer training run with eventual LR reduction, or from cosine annealing which would reduce LR regardless of val loss trajectory. Not an architecture issue but worth noting for training optimization.

---

## 3. Summary Table

| # | Finding | Severity | Category |
|---|---------|----------|----------|
| 2.1 | Skip connection may bypass or subsume quantum block | MEDIUM | Design risk |
| 2.2 | LayerNorm + tanh compresses angle range | LOW | Design choice |
| 2.3 | Dropout sparse but acceptable at p=0.05 | LOW | Regularization |
| 2.4 | Concatenation fusion is adequate | LOW | Design choice |
| 2.5 | ~54k params for 37k samples (ratio 1.45) | LOW | Capacity |
| 2.6 | Vanishing gradients possible in quantum block | MEDIUM | Trainability |
| 2.7 | Val RMSE 1.54 vs Test RMSE 3.46 (2.25x gap) | CRITICAL | Generalization |
| 2.8 | No BatchNorm in convolutional layers | LOW | Regularization |
| 2.9 | AMP scope excludes head (intentional) | LOW | Correctness |
| 2.10 | Shallow quantum circuit (1 entanglement block) | MEDIUM | Expressibility |
| 2.11 | No dead neuron risk (GELU throughout) | LOW | Correctness |
| 2.12 | eval/train mode handled correctly | LOW | Correctness |
| 2.13 | LR scheduler never triggered | LOW | Training |

---

## 4. Verdict

**CONDITIONAL PASS**

The architecture is internally consistent: all tensor shapes match, Conv1d dimensions are correct, the AdaptiveAvgPool1d produces the expected output, dropout and eval/train mode are handled properly, and the model is appropriately sized for the dataset. There are no correctness bugs in the architecture.

However, one CRITICAL finding (2.7 -- the 2.25x val/test RMSE gap with effectively non-predictive test performance at 3.46 dex RMSE) and three MEDIUM findings (quantum skip connection risk, vanishing quantum gradients, shallow circuit expressibility) collectively raise serious concerns about whether the quantum component provides meaningful contribution and whether the model generalizes beyond its training distribution.

The architecture is mechanically sound but scientifically unvalidated in its current form. The test RMSE of 3.46 log10 VMR means predictions are wrong by ~3 orders of magnitude on out-of-distribution data, which is unacceptable for any claim of biosignature detection capability on real or cross-generator data.
