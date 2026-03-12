# Audit Report 13: Systematic Code Bug Hunt

**Scope**: `crossgen_hybrid_training.py` (987 lines), `run_training.py` (9 lines)
**Date**: 2026-03-12
**Auditor**: Claude Opus 4.6

---

## Findings

### Finding 1: Checkpoint resume crashes with `KeyError` after a completed run

**Severity**: CRITICAL
**File**: `crossgen_hybrid_training.py`
**Lines**: 713-719 (resume read), 808 (mid-training save), 911-921 (post-training overwrite)

During training, `best_model.pt` is saved with keys `{"epoch", "val_loss", "model_state_dict"}` (line 808). After training completes, `run_training_experiment` overwrites the same file with different keys: `{"config", "feature_cols", "target_cols", "best_epoch", "best_val_loss", "model_state_dict"}` (lines 911-921).

The resume logic (lines 714-719) reads `ckpt["epoch"]` and `ckpt["val_loss"]`, which do not exist in the post-training checkpoint (which has `best_epoch` and `best_val_loss`).

```python
# Resume reads these keys (line 716-717):
start_epoch = ckpt["epoch"]
best_val_loss = ckpt["val_loss"]

# But the final save (line 911-921) stores:
"best_epoch": training["best_epoch"],   # not "epoch"
"best_val_loss": training["best_val_loss"],  # not "val_loss"
```

**Impact**: Any attempt to resume training after a completed run raises `KeyError: 'epoch'`. The resume path is completely broken for the common case where a user re-runs with `resume=True` against a previously finished experiment.

**Fix**: Either (a) make the final save use the same keys as the mid-training save, or (b) make the resume logic handle both key schemas, or (c) save to a different filename for the final artifact.

---

### Finding 2: Optimizer and scheduler state not saved/restored on resume

**Severity**: HIGH
**File**: `crossgen_hybrid_training.py`
**Lines**: 692-703 (optimizer/scheduler creation), 713-720 (resume logic), 808 (checkpoint save)

When resuming, only `model_state_dict` is loaded. The optimizer state (momentum buffers, adaptive learning rates from AdamW) and the scheduler state (internal step counters, best value, number of bad epochs) are discarded and re-initialized from scratch.

```python
# Checkpoint save (line 808) -- no optimizer or scheduler state:
torch.save({"epoch": best_epoch, "val_loss": best_val_loss,
            "model_state_dict": best_state}, ckpt_path)

# Resume load (lines 714-719) -- no optimizer or scheduler restore:
ckpt = torch.load(resume_from, ...)
model.load_state_dict(ckpt["model_state_dict"])
start_epoch = ckpt["epoch"]
```

**Impact**: Resumed training restarts with fresh AdamW momentum and fresh scheduler counters. The learning rate jumps back to the initial value, and the optimizer loses all accumulated gradient statistics. This effectively corrupts the training trajectory and can cause a loss spike or divergence after resume.

**Fix**: Save `optimizer.state_dict()` and `scheduler.state_dict()` in the checkpoint, and restore them on resume.

---

### Finding 3: Validation loss computed as unweighted mean of per-batch losses (last-batch bias)

**Severity**: MEDIUM
**File**: `crossgen_hybrid_training.py`
**Lines**: 647-669

`evaluate_split` computes `nn.MSELoss()` (which is `reduction='mean'`) per batch, then takes `np.mean(losses)` across batches. If the last batch is smaller than the rest, it contributes equally to the final average despite representing fewer samples.

```python
losses.append(float(loss_fn(pred, targets).item()))  # line 660
# ...
"loss": float(np.mean(losses)),  # line 669 -- unweighted mean
```

With `eval_batch_size=8192` and `inner_val_rows=4142`, there is only one batch so this bug is dormant for inner_val. For `test_rows=685`, also one batch. However, the code is fragile: if the dataset grows or `eval_batch_size` decreases, the loss will be biased toward the last (smaller) batch.

**Impact**: Currently dormant due to dataset/batch size ratio, but will silently produce biased validation loss if data sizes change. This could affect early stopping and model selection.

**Fix**: Use sample-weighted average: accumulate `loss * batch_size` and divide by total samples, or use `reduction='sum'` and divide by total count at the end.

---

### Finding 4: `best_state` loaded onto wrong device after training

**Severity**: MEDIUM
**File**: `crossgen_hybrid_training.py`
**Lines**: 803, 820

The best model state is saved as CPU tensors (line 803), then loaded back into the model at end of training (line 820). `model.load_state_dict()` with CPU tensors into a CUDA model works (PyTorch copies to the correct device), so this is not a crash. However, the returned `model` and `training["best_state"]` have an inconsistency: `model` is on CUDA with the best weights, but `training["best_state"]` is a dict of CPU tensors.

```python
best_state = {name: value.detach().cpu() for name, value in model.state_dict().items()}  # line 803
# ...
model.load_state_dict(best_state)  # line 820 -- loads CPU tensors into CUDA model
```

When `run_training_experiment` then calls `evaluate_split(model, ...)` (lines 945-946), the model is on CUDA (because `load_state_dict` does device-aware loading for parameters already on CUDA), so evaluation works correctly.

**Impact**: No runtime crash. The inconsistency between model device and state dict device is confusing but functionally harmless. PyTorch's `load_state_dict` handles the CPU-to-CUDA mapping transparently for `nn.Module`.

---

### Finding 5: `quantum_device` default evaluated at import time, not instantiation time

**Severity**: LOW
**File**: `crossgen_hybrid_training.py`
**Line**: 115

```python
quantum_device: str = default_quantum_device()  # evaluated once at class definition
```

`default_quantum_device()` calls `torch.cuda.is_available()`. In a dataclass, field defaults are evaluated when the class body is executed (import time), not when instances are created. If CUDA availability changes between import and instantiation (unlikely but possible in environments with lazy CUDA initialization, dynamic GPU allocation, or late driver loading), the default would be stale.

**Impact**: Negligible in practice. The module is typically imported and used immediately. If needed, the user can always pass `quantum_device` explicitly.

**Fix**: Use `dataclasses.field(default_factory=default_quantum_device)`.

---

### Finding 6: `configure_runtime` may crash if interop threads already set

**Severity**: LOW
**File**: `crossgen_hybrid_training.py`
**Lines**: 220-223

```python
torch.set_num_threads(cpu_threads)
torch.set_num_interop_threads(max(1, min(cpu_threads // 2, 8)))
```

`torch.set_num_interop_threads` must be called before any inter-op parallel work has started. If this function is called after PyTorch has already launched inter-op threads (e.g., in a Jupyter notebook that has already done some PyTorch work), it raises `RuntimeError: Cannot set number of interop threads after parallel work has started`.

**Impact**: The `run_training.py` CLI script calls this early enough. However, `train.ipynb` users who have already run PyTorch operations in prior cells will crash when calling `configure_runtime()`.

**Fix**: Wrap `set_num_interop_threads` in a try/except to degrade gracefully.

---

### Finding 7: Dead config parameters `internal_val_fraction` and `test_limit`

**Severity**: LOW
**File**: `crossgen_hybrid_training.py`
**Lines**: 95, 119

```python
internal_val_fraction: float = 0.10   # line 95 -- never read
test_limit: Optional[int] = None       # line 119 -- never read
```

`internal_val_fraction` is defined but never used anywhere in the code. The train/val split is entirely determined by the `split` column in `labels.parquet`, so this parameter is vestigial.

`test_limit` is defined but never consumed. Only `train_pool_limit` is applied in `prepare_data` (line 322). If a user sets `test_limit=100` expecting to subsample the test set, it silently does nothing.

**Impact**: Misleading API. Users may set these parameters expecting behavior that never happens.

**Fix**: Either implement the functionality or remove the parameters.

---

### Finding 8: `model.train()` not restored after `evaluate_split` calls during training

**Severity**: LOW (correctly handled)
**File**: `crossgen_hybrid_training.py`
**Lines**: 646, 732, 768

`evaluate_split` calls `model.eval()` (line 646). In the training loop, `evaluate_split` is called at the end of each epoch (line 768), and `model.train()` is called at the start of the next epoch (line 732). This is correct -- the model is set back to training mode at the start of each epoch.

**Impact**: None. This is handled correctly. Noting for completeness.

---

### Finding 9: Training loop AMP context does not wrap backward pass

**Severity**: LOW
**File**: `crossgen_hybrid_training.py`
**Lines**: 534-548 (forward), 745-751 (training loop)

The `autocast` context in `HybridAtmosphereModel.forward` only wraps the classical encoder forward passes (lines 540-543). The quantum block and head run in float32 outside autocast. This is intentional and correct for quantum circuits that need float32 precision.

However, in the training loop, AMP gradient scaling (`torch.cuda.amp.GradScaler`) is not used despite `use_amp=True` being the default. On CUDA with float16 autocast, the backward pass can produce float16 gradients that underflow to zero without a GradScaler.

```python
# Training loop (lines 745-751) -- no GradScaler:
pred = model(aux, spectra)
loss = loss_fn(pred, targets)
loss.backward()              # no scaler.scale(loss).backward()
optimizer.step()             # no scaler.step(optimizer)
```

**Impact**: On CUDA with float16 (non-bfloat16 GPUs), gradient underflow may silently degrade training quality. On bfloat16 GPUs (which the code prefers at line 243-244), this is a non-issue because bfloat16 has the same exponent range as float32. On CPU, autocast is disabled entirely.

**Fix**: Either add `GradScaler` when `amp_dtype` is `torch.float16`, or disable float16 AMP (only allow bfloat16 or None).

---

### Finding 10: `batch_indices` total-batches logging can be off by one

**Severity**: LOW
**File**: `crossgen_hybrid_training.py`
**Lines**: 710, 758-759

```python
total_batches = math.ceil(len(data.train.aux) / config.train_batch_size)  # line 710
# ...
if batch_idx % config.log_every_batches == 0 or batch_idx == total_batches:  # line 759
```

`batch_idx` starts at 1 (due to `start=1` on line 741). The actual number of batches yielded by `batch_indices` is `math.ceil(length / batch_size)`. So `batch_idx` ranges from 1 to `total_batches` inclusive. The condition `batch_idx == total_batches` correctly catches the last batch. No actual off-by-one here.

**Impact**: None. Logging is correct.

---

### Finding 11: No gradient clipping reported for quantum parameters norm

**Severity**: LOW
**File**: `crossgen_hybrid_training.py`
**Line**: 750

```python
torch.nn.utils.clip_grad_norm_(quantum_params, 1.0)  # kept tight for quantum stability
```

The quantum gradient clip norm (1.0) is hardcoded rather than being a configurable hyperparameter, unlike the classical clip norm which comes from `config.gradient_clip_norm` (5.0). This is a design choice rather than a bug, but it makes it invisible to hyperparameter tuning.

**Impact**: Minor inflexibility. Not a correctness issue.

---

### Finding 12: `save_json` does not handle numpy types

**Severity**: LOW
**File**: `crossgen_hybrid_training.py`
**Line**: 832

```python
def save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
```

If any numpy scalar (e.g., `np.float32`, `np.int64`) ends up in the payload without explicit `float()` or `int()` conversion, `json.dumps` will raise `TypeError: Object of type float32 is not JSON serializable`. The callers are generally careful about conversion (e.g., `float(metrics["rmse_mean"])`), but this is fragile.

**Impact**: Low risk of runtime crash if future code changes introduce unconverted numpy values.

**Fix**: Add a custom JSON encoder that handles numpy types, e.g., `json.dumps(payload, indent=2, sort_keys=True, default=str)` or a proper `NumpyEncoder` subclass.

---

## Issues NOT Found (Verified Correct)

1. **`model.eval()` before validation**: Correctly called at line 646 before every `evaluate_split`.
2. **`torch.no_grad()` during evaluation**: Uses `torch.inference_mode()` (line 652), which is even stricter than `no_grad()` -- correct.
3. **Last batch handling**: `batch_indices` yields all indices via slicing (line 604: `permutation[start : start + batch_size]`), which naturally handles the last smaller batch. `evaluate_split` similarly handles it via `min(start + batch_size, len(split.aux))` (line 654).
4. **Device mismatches in gather_batch**: Explicitly checks and moves indices to the correct device (lines 612-613) and moves data to the target device (lines 619-622).
5. **Tensor reshaping**: Spectra enter as `(N, 1, 44)` from `build_raw_arrays` (line 280), matching `SpectralEncoder`'s `Conv1d(in_channels=1, ...)` expectation. All concatenation dimensions in `forward` are verified correct.
6. **Copy vs reference**: `best_state` is a deep copy via `{name: value.detach().cpu() ...}` (line 803), so later model mutations do not corrupt the saved state.
7. **Non-deterministic behavior**: Seeds are set for `random`, `numpy`, `torch`, and `torch.cuda` (lines 212-217). Batch shuffling uses a seeded generator (lines 598-600). The quantum circuit is seeded through `torch.manual_seed`. The `cudnn.benchmark = True` (line 228) can introduce non-determinism on CUDA but this is a standard trade-off for speed.
8. **Mutable default arguments**: No mutable defaults (lists, dicts) in function signatures. The `ARIEL_WAVELENGTH_GRID` global is a numpy array but is only read, never mutated.
9. **Silent failures**: No bare `except:` clauses. The only try/except is `import_pennylane` (line 421-427) which re-raises with a clear message.
10. **Print/logging accuracy**: All logged metrics correspond to the correct computed values. `train_loss` is correctly the mean of batch losses for that epoch. `inner_val_loss` comes from `evaluate_split` results.

---

## Summary Table

| # | Severity | Finding |
|---|----------|---------|
| 1 | CRITICAL | Checkpoint resume `KeyError` due to key mismatch after completed run |
| 2 | HIGH | Optimizer/scheduler state not saved/restored on resume |
| 3 | MEDIUM | Validation loss unweighted across unequal batches (dormant) |
| 4 | MEDIUM | Inconsistent device for `best_state` dict vs model (harmless) |
| 5 | LOW | `quantum_device` default evaluated at import time |
| 6 | LOW | `set_num_interop_threads` can crash in notebook environments |
| 7 | LOW | Dead config parameters `internal_val_fraction` and `test_limit` |
| 8 | LOW | (False positive -- `model.train()` correctly restored each epoch) |
| 9 | LOW | Missing `GradScaler` for float16 AMP (bfloat16 path unaffected) |
| 10 | LOW | (False positive -- batch logging arithmetic is correct) |
| 11 | LOW | Quantum gradient clip norm hardcoded, not configurable |
| 12 | LOW | `save_json` lacks numpy type handling (fragile) |

---

## Verdict: FAIL

**Rationale**: Finding 1 is a CRITICAL bug -- the checkpoint resume path is guaranteed to crash with `KeyError` after any completed training run because the post-training save overwrites `best_model.pt` with incompatible key names. Finding 2 is a HIGH-severity issue where resumed training silently loses all optimizer/scheduler state, effectively corrupting the training trajectory without any error or warning. These two findings together mean the resume functionality is broken at both the crash and correctness levels.

The core training-from-scratch path is sound. The model architecture, data pipeline, evaluation logic, device handling, and seed management are all correctly implemented. The remaining LOW/MEDIUM findings are either dormant, cosmetic, or unlikely to trigger in the current setup.
