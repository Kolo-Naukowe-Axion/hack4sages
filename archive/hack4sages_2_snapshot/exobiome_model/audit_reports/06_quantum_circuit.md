# Audit Report 06: Quantum Circuit Correctness

**Auditor:** Scientific Rigor Auditor
**Date:** 2026-03-12
**Scope:** QuantumBlock class, PennyLane circuit construction, quantum-classical integration
**Source file:** `/Users/michalszczesny/projects/hack4sages/exobiome_model/crossgen_hybrid_training.py`
**Config:** `/Users/michalszczesny/projects/hack4sages/exobiome_model/outputs/model_crossgen_rebinned/config.json`

---

## 1. Circuit Architecture Summary

The `QuantumBlock` (lines 430-486) implements a 12-qubit variational quantum circuit with the following structure:

1. **Input encoding layer:** 12 RY gates, one per qubit, driven by the classical fusion encoder output
2. **Variational blocks** (repeated `depth/2 = 1` time with default config):
   - RY rotation layer (12 trainable params)
   - CNOT ring entanglement (circular ladder: 0->1, 1->2, ..., 11->0)
   - RZ rotation layer (12 trainable params)
   - CRX ring entanglement (circular ladder: same topology)
3. **Measurement:** Pauli-Z expectation value on each of the 12 qubits

**Total trainable quantum parameters:** `3 * 12 * (2/2) = 36`

---

## 2. Findings

### FINDING 2.1 -- Trainable Parameter Count

**Severity:** LOW
**Status:** CORRECT

The code at line 449:
```python
self.num_weights = 3 * n_qubits * self.num_blocks
```

With `n_qubits=12` and `num_blocks = depth//2 = 1`, this yields `3 * 12 * 1 = 36` trainable parameters.

The circuit loop (lines 465-476) consumes parameters as:
- `param_idx` advances by `n_qubits` for RY (12 params)
- `param_idx` advances by `n_qubits` for RZ (12 params)
- `param_idx` advances by `n_qubits` for CRX (12 params)
- Total per block: 36. With 1 block: 36 total.

This matches the `nn.Parameter` allocation at line 450:
```python
self.weights = nn.Parameter(0.5 * torch.randn(self.num_weights, dtype=torch.float32))
```

**Verification:** Parameter count is consistent and correct.

---

### FINDING 2.2 -- Input Encoding: tanh(x) * pi Range

**Severity:** MEDIUM
**Location:** `FusionEncoder.forward()` (line 417), `QuantumBlock.forward()` (lines 461-462)

The FusionEncoder output is:
```python
return torch.tanh(self.net(fused)) * math.pi   # line 417
```

This produces values in `(-pi, +pi)`. These values are fed directly as RY rotation angles:
```python
qml.RY(inputs[..., qubit], wires=qubit)   # line 462
```

**Analysis:**
- RY(theta) has a period of `4*pi`. The encoding maps to `(-pi, pi)`, which covers half the rotation range.
- The `tanh` function saturates at +/-1, so extreme fusion values are compressed nonlinearly. This creates a soft ceiling on the encoding rotation.
- Using `tanh * pi` is a **reasonable and common** choice in variational quantum ML. It prevents unbounded rotations that would wrap around and create discontinuities in the gradient landscape.
- However, `tanh` squashes the tails: a fusion output of 3.0 maps to `0.995*pi` while 5.0 maps to `0.9999*pi` -- virtually identical angles. If the fusion encoder produces large-magnitude outputs, information is lost.

**Recommendation:** This is acceptable but could benefit from a learnable scaling factor `alpha * tanh(x) * pi` to allow the model to control encoding range. Not a correctness bug, but a mild expressivity constraint.

---

### FINDING 2.3 -- Entanglement Pattern: Custom Ansatz, Not Standard

**Severity:** LOW
**Location:** Lines 469-476

The circuit uses a custom ansatz with two distinct entanglement sub-layers per block:

```python
# Sub-layer 1: CNOT ring
for qubit in range(n_qubits):
    qml.CNOT(wires=[qubit, (qubit + 1) % n_qubits])   # lines 469-470

# Sub-layer 2: CRX ring (parameterized)
for qubit in range(n_qubits):
    qml.CRX(weights[param_idx], wires=[qubit, (qubit + 1) % n_qubits])   # lines 474-475
```

**Analysis:**
- This is a **custom ansatz**, not a standard named one (not Hardware-Efficient, not Strongly Entangling Layers, not QAOA-style).
- The CNOT ring provides fixed entanglement; the CRX ring adds parameterized entanglement on the same topology.
- Having both fixed and parameterized entanglement on the same ring topology is **redundant in topology** but not in function -- the CRX adds a tunable rotation conditioned on the control qubit, which is strictly more expressive than CNOT alone.
- The circular ring topology (qubit n-1 -> qubit 0) ensures all qubits are connected, which is good for a 12-qubit circuit. However, the graph diameter is 6, meaning information from qubit 0 needs 6 layers to reach qubit 6 directly. With only `num_blocks=1`, long-range correlations are limited.

**Recommendation:** Consider adding an additional entanglement pattern (e.g., all-to-all or alternating odd-even pairs) for better expressivity, or increasing depth to 4. Not a correctness issue.

---

### FINDING 2.4 -- PennyLane QNode Configuration

**Severity:** LOW
**Status:** CORRECT
**Location:** Line 459

```python
@qml.qnode(device, interface="torch", diff_method="adjoint")
```

**Analysis:**
- `interface="torch"` is correct for PyTorch integration, enabling autograd-compatible quantum gradients.
- `diff_method="adjoint"` is the optimal choice for `lightning.qubit` (the device used in the actual run per config.json line 19). Adjoint differentiation is O(n) in memory and fast for statevector simulators.
- The `lightning.qubit` device (line 443-444) is correctly configured with `c_dtype=np.complex64` for memory efficiency.

**Potential issue:** If someone changes `quantum_device` to `default.qubit` or a hardware device, `diff_method="adjoint"` may not be supported. This is a minor robustness concern, not a bug in current usage.

---

### FINDING 2.5 -- Quantum Advantage Question

**Severity:** HIGH
**Category:** Scientific Rigor

The quantum block has **36 trainable parameters** and produces a 12-dimensional output. A classical equivalent would be:

- A single `nn.Linear(12, 12)` layer: `12*12 + 12 = 156` parameters (already much more)
- A constrained linear layer with 36 parameters could trivially match the parameter count

**Critical analysis:**

1. **Parameter efficiency:** The quantum circuit uses 36 parameters to map 12 inputs to 12 outputs. A classical MLP layer with similar capacity would need fewer parameters for the same transformation (a 12->12 linear map with no bias = 144 params, but with structured matrices like low-rank or diagonal, as few as 12-36 params).

2. **The circuit depth is too shallow for quantum advantage.** With `num_blocks=1`, the circuit has exactly one round of entanglement. The resulting quantum state has limited entanglement entropy. Theoretical results (e.g., Abbas et al. 2021, "The power of quantum neural networks") suggest that quantum advantage in expressibility requires sufficient circuit depth relative to qubit count.

3. **The residual bypass architecture further weakens the quantum contribution.** Looking at `HybridAtmosphereModel.forward()` (lines 533-548):
```python
quantum_feat = self.quantum_block(latent)
head_in = torch.cat([quantum_feat, latent, aux_feat.float(), spectral_feat.float()], dim=-1)
return self.head(head_in, quantum_feat)
```
The head receives `quantum_feat` (12 dims), `latent` (12 dims), `aux_feat` (32 dims), and `spectral_feat` (32 dims) -- total 88 dimensions. The quantum contribution is only 12/88 = **13.6%** of the head input. Furthermore, the `AtmosphereHead` has a residual connection `self.residual(latent)` at line 503 where `latent` is `quantum_feat`:
```python
return self.mlp(head_in) + self.residual(latent)   # latent = quantum_feat
```
This residual path allows the head to learn a direct linear projection from quantum outputs, which could either amplify or bypass them.

4. **Training evidence suggests marginal quantum contribution.** The learning rate for quantum parameters never decayed from its initial `6e-4` across all 30 epochs (history.csv shows `quantum_lr=0.0006` constant), while the scheduler is configured to reduce on plateau with patience=5. This means the quantum loss landscape contributed to consistent improvement -- the scheduler did not trigger. However, the classical LR also remained constant at `0.002`, which means both were contributing.

5. **No ablation study was performed.** Without a classical-only baseline (replacing QuantumBlock with an equivalent classical layer), the quantum advantage claim is **unsubstantiated**.

**Recommendation:** This is the most significant scientific finding. For a paper or competition submission, an ablation study comparing the hybrid model against a purely classical model (replacing QuantumBlock with `nn.Linear(12, 12)` + `nn.Tanh()`) is essential. Without it, the quantum component cannot be demonstrated to provide any advantage. The current architecture makes it easy for the classical head to learn around the quantum block entirely.

---

### FINDING 2.6 -- Barren Plateau Risk Assessment

**Severity:** MEDIUM
**Category:** Trainability

**Analysis of barren plateau risk for 12 qubits, depth 1 block:**

1. **Qubit count = 12:** The variance of cost function gradients scales as `O(1/2^n)` for random quantum circuits (McClean et al. 2018). With n=12, the gradient variance suppression factor is `~1/4096`. This is at the boundary of trainability.

2. **Mitigating factors:**
   - The depth is very shallow (1 block), which significantly reduces barren plateau severity. Deep circuits are the primary cause.
   - The input encoding via RY is data-dependent (not random), which breaks the 2-design symmetry that causes barren plateaus.
   - The parameter initialization at line 450 uses `0.5 * torch.randn(...)`, giving initial angles near zero. This is a good practice -- small initial rotations keep the circuit close to the identity, avoiding the random circuit regime.
   - The gradient clipping at line 750 is set to 1.0 for quantum parameters (tighter than the 5.0 for classical), which suggests awareness of small quantum gradients.

3. **Evidence from training:** The training loss decreased steadily from 0.829 (epoch 1) to 0.301 (epoch 30), and the inner_val_loss decreased from 0.714 to 0.300. This monotonic decrease suggests the quantum parameters are not stuck in a barren plateau. However, we cannot distinguish whether the improvement is driven by classical or quantum parameter updates without the ablation.

4. **Concerning sign:** The quantum learning rate is 3.3x lower than the classical LR (`6e-4` vs `2e-3`). If this was set empirically because higher quantum LR caused instability, it could indicate partial barren plateau effects where gradients are small and noisy.

**Verdict:** Barren plateaus are a moderate concern but the shallow depth and good initialization likely mitigate the worst effects. The 12-qubit circuit is at the edge of the regime where barren plateaus become problematic.

---

### FINDING 2.7 -- Expectation Value Range and Downstream Usage

**Severity:** LOW
**Status:** CORRECT
**Location:** Lines 478, 482-486, 547-548

Pauli-Z expectation values (line 478):
```python
return [qml.expval(qml.PauliZ(qubit)) for qubit in range(n_qubits)]
```

These are guaranteed to be in `[-1, +1]`.

In `QuantumBlock.forward()` (lines 482-486):
```python
outputs = self.qnode(x.to(dtype=torch.float32), self.weights)
if isinstance(outputs, (list, tuple)):
    outputs = torch.stack(tuple(outputs), dim=-1)
return outputs.to(dtype=torch.float32)
```

The outputs are stacked and returned as float32. They enter the head at line 547:
```python
head_in = torch.cat([quantum_feat, latent, aux_feat.float(), spectral_feat.float()], dim=-1)
```

**Analysis:** The [-1, 1] range of quantum outputs is well-behaved for concatenation with other features. The `latent` (fusion encoder output) is in `(-pi, pi)` due to `tanh * pi`, and the aux/spectral features are unbounded GELU outputs. The scale mismatch between quantum features (max range 2) and other features is handled implicitly by the head MLP, but explicit normalization of the concatenated vector could improve training dynamics.

**Status:** Correct but suboptimal. The head MLP can learn to rescale, so this is not a bug.

---

### FINDING 2.8 -- Residual Connection Architecture

**Severity:** MEDIUM
**Category:** Design Concern
**Location:** Lines 489-503, 547-548

The `AtmosphereHead` receives two inputs:
```python
# Line 547-548 in HybridAtmosphereModel.forward()
head_in = torch.cat([quantum_feat, latent, aux_feat.float(), spectral_feat.float()], dim=-1)
return self.head(head_in, quantum_feat)
```

Inside `AtmosphereHead.forward()` (line 502-503):
```python
def forward(self, head_in: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
    return self.mlp(head_in) + self.residual(latent)
```

Here, `latent` is `quantum_feat` (the Pauli-Z expectation values). So the residual path is:
```
self.residual = nn.Linear(12, 5)   # maps quantum_feat directly to targets
```

**Analysis:**
- The residual connection creates a **linear shortcut** from quantum outputs to predictions. This means the model can learn `predictions = Linear(quantum_feat) + MLP(all_features)`.
- This is architecturally sound for gradient flow -- it ensures quantum gradients always have a direct path to the loss.
- However, it also means the quantum block's contribution can be trivially small if the MLP dominates. The residual becomes a bias-like correction term.
- The naming is misleading: the parameter is called `latent` in the head but receives `quantum_feat` from the model. In `build_model()` (line 563), `latent_dim=config.qnn_qubits` confirms this is the quantum output dimension.

**Status:** Not incorrect, but the architecture allows the model to effectively bypass quantum computation. This compounds the concern in Finding 2.5.

---

### FINDING 2.9 -- Quantum Parameter Gradient Flow

**Severity:** LOW
**Status:** CORRECT
**Location:** Lines 459, 545-548, 748-750

Gradient flow path:
1. Loss is computed on `head` output (line 747)
2. `loss.backward()` propagates through `AtmosphereHead` -> `quantum_feat` -> `QuantumBlock.forward()` -> `self.qnode()`
3. PennyLane's adjoint differentiation (line 459) computes analytical gradients of expectation values w.r.t. quantum parameters
4. These gradients flow into `self.weights` (nn.Parameter), which is in the `quantum_params` optimizer group

**Verification points:**
- `self.weights` is an `nn.Parameter` (line 450) -- will be tracked by autograd.
- `diff_method="adjoint"` provides exact gradients, not finite-difference approximations.
- The quantum parameters are in a separate optimizer group (line 695) with their own learning rate and zero weight decay -- this is correct practice.
- Gradient clipping is applied separately (line 750): `torch.nn.utils.clip_grad_norm_(quantum_params, 1.0)`.

**Concern:** The `latent = latent.float()` cast at line 545 could potentially break the gradient tape if the autocast context produced a different dtype. However, `.float()` preserves gradient tracking in PyTorch, so this is safe. On CPU (where this model was run), autocast is disabled anyway.

**Status:** Gradient flow is correct. The adjoint method provides exact gradients, and the parameter groups are properly configured.

---

### FINDING 2.10 -- AMP and Quantum Interaction

**Severity:** LOW
**Location:** Lines 534-546

```python
with autocast_ctx:
    aux_feat = self.aux_encoder(aux)
    spectral_feat = self.spectral_encoder(spectra)
    latent = self.fusion_encoder(aux_feat, spectral_feat)

latent = latent.float()
quantum_feat = self.quantum_block(latent)
```

**Analysis:** The quantum block is intentionally placed **outside** the autocast context. The explicit `latent.float()` cast ensures the quantum circuit always receives float32 inputs, which is necessary because PennyLane's adjoint method requires float32/float64 precision. This is correct.

In the actual run, the device was CPU with `lightning.qubit`, and `use_amp=true` was set in config but `resolve_amp_dtype()` returns `None` for CPU (line 242), so autocast was effectively disabled. This is correct behavior.

---

### FINDING 2.11 -- Batch Handling in Quantum Circuit

**Severity:** MEDIUM
**Category:** Performance / Correctness
**Location:** Lines 460-462, 482-486

The circuit signature is:
```python
def circuit(inputs: torch.Tensor, weights: torch.Tensor) -> list[torch.Tensor]:
    for qubit in range(n_qubits):
        qml.RY(inputs[..., qubit], wires=qubit)
```

The `inputs[..., qubit]` notation with the Ellipsis suggests this should handle batched inputs. PennyLane supports parameter broadcasting, where a batch of inputs is processed in a single call.

**Analysis:**
- PennyLane's parameter broadcasting (since v0.32) allows batched execution where `inputs` has shape `(batch_size, n_qubits)`.
- The `...` indexing correctly handles both single-sample `(n_qubits,)` and batched `(batch_size, n_qubits)` inputs.
- However, whether this actually results in efficient batched execution depends on the device. `lightning.qubit` supports batched execution via parameter broadcasting, so this is efficient.
- The `torch.stack(tuple(outputs), dim=-1)` at line 485 correctly handles the batched output from PennyLane, producing shape `(batch_size, n_qubits)`.

**Status:** Correct implementation of batched quantum execution.

---

### FINDING 2.12 -- Weight Initialization

**Severity:** LOW
**Location:** Line 450

```python
self.weights = nn.Parameter(0.5 * torch.randn(self.num_weights, dtype=torch.float32))
```

The scale factor `0.5` means initial weights are drawn from `N(0, 0.25)`. For rotation gates (RY, RZ, CRX), this means initial rotations are small, keeping the circuit close to the identity transformation at initialization.

**Analysis:** This is a good practice. Research on quantum circuit initialization (Grant et al. 2019, "An initialization strategy for addressing barren plateaus") recommends initializing near the identity to avoid barren plateaus. The `0.5` scale is a reasonable choice -- small enough to avoid random circuit behavior, large enough to break symmetry.

**Status:** Correct and well-motivated.

---

### FINDING 2.13 -- Training Evidence Analysis

**Severity:** HIGH (for scientific claims)
**Category:** Empirical Validation

From `run_summary.json`:
- Best epoch: 29 (out of 30 max)
- Best inner_val_loss: 0.2999
- Test RMSE mean: **3.4631**

From `history.csv` analysis:
- Training ran for all 30 epochs without early stopping (patience=6 never triggered because the model kept improving, best at epoch 29).
- The model was still improving at epoch 30 (train_loss=0.301, val_loss=0.302 vs best val_loss=0.300 at epoch 29).
- Neither the classical LR nor the quantum LR was reduced by the scheduler.

**Concerns:**

1. **The model did not converge.** Training for all 30 epochs with continuous improvement and no LR decay suggests the model was under-trained. The `max_epochs` should be increased or the learning rates should be higher.

2. **Test RMSE of 3.46 (in log10 VMR units) is very poor.** This means on average the model's predictions are off by a factor of ~10^3.46 ~ 2884x in volume mixing ratio. For context, atmospheric VMRs for biosignature gases range from ~10^-10 to 10^-2, so an error of 3.46 dex spans nearly the entire dynamic range. This is essentially uninformative.

3. **Val-test gap is enormous.** The inner_val RMSE mean at best epoch is 1.538, while the test RMSE is 3.463. This 2x gap suggests severe overfitting or a train/test distribution shift (the test set uses Poseidon-generated spectra while training uses TauREx). The quantum block is unlikely to help with cross-generator generalization.

**Verdict:** The model's predictive performance is too poor to draw any conclusions about quantum advantage. The primary issue is likely the train/test domain shift, not the quantum circuit.

---

## 3. Summary Table

| # | Finding | Severity | Status |
|---|---------|----------|--------|
| 2.1 | Parameter count (36) matches formula | LOW | CORRECT |
| 2.2 | tanh * pi encoding range | MEDIUM | ACCEPTABLE, mild expressivity limit |
| 2.3 | Custom CNOT+CRX ring ansatz | LOW | VALID, limited depth |
| 2.4 | PennyLane QNode configuration | LOW | CORRECT |
| 2.5 | No demonstrated quantum advantage | HIGH | UNSUBSTANTIATED |
| 2.6 | Barren plateau risk (12 qubits) | MEDIUM | MITIGATED by shallow depth |
| 2.7 | Expectation value range [-1,1] | LOW | CORRECT |
| 2.8 | Residual bypass enables quantum skip | MEDIUM | DESIGN CONCERN |
| 2.9 | Quantum gradient flow | LOW | CORRECT |
| 2.10 | AMP/quantum interaction | LOW | CORRECT |
| 2.11 | Batch handling in circuit | MEDIUM | CORRECT |
| 2.12 | Weight initialization near identity | LOW | CORRECT, well-motivated |
| 2.13 | Training evidence / model performance | HIGH | UNDER-TRAINED, poor test RMSE |

---

## 4. Critical Issues Requiring Action

### ACTION 1: Ablation Study Required (Finding 2.5)
Run a classical-only baseline replacing `QuantumBlock` with `nn.Sequential(nn.Linear(12, 12), nn.Tanh())` (same parameter count ~156, or constrained to 36). Compare test RMSE. Without this, claiming quantum benefit is scientifically unsound.

### ACTION 2: Increase Training Duration (Finding 2.13)
The model was still improving at epoch 30. Increase `max_epochs` to at least 100 and consider increasing learning rates. The current results represent an under-trained model.

### ACTION 3: Address Train-Test Domain Shift (Finding 2.13)
Test RMSE (3.46) is 2.25x worse than val RMSE (1.54). The Poseidon test set appears fundamentally different from the TauREx training set. This cross-generator gap must be addressed before quantum circuit design matters.

---

## 5. Verdict

**CONDITIONAL PASS**

The quantum circuit implementation is **technically correct**: parameter counts are consistent, PennyLane integration is properly configured, gradient flow is verified, batch handling works, and the adjoint differentiation method is appropriate. The code is well-structured and free of bugs.

However, the audit identifies two HIGH-severity scientific concerns that prevent an unconditional PASS:

1. **No ablation evidence** that the quantum block provides any advantage over an equivalent classical layer. The architecture's residual connections and concatenation of classical features make it easy for the model to learn around the quantum block entirely.

2. **The model has not converged**, and the test RMSE of 3.46 dex is too poor to draw meaningful conclusions about any architectural component, quantum or otherwise.

The circuit itself is sound. The scientific claims enabled by this circuit are not yet substantiated.
