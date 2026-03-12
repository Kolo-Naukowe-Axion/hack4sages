# Audit Report 05: Reproducibility

**Target**: Hybrid quantum-classical model for exoplanet biosignature detection
**Auditor**: Scientific Rigor Audit (automated)
**Date**: 2026-03-12
**Scope**: Seed propagation, determinism guarantees, config completeness, artifact sufficiency, environment pinning

---

## Summary

The codebase demonstrates a solid baseline of reproducibility practices: a single seed parameter is propagated to Python, NumPy, and PyTorch RNGs, all hyperparameters are serialized to `config.json`, scalers are saved, and batch shuffling uses epoch-indexed deterministic generators. However, several gaps prevent guaranteed bitwise reproducibility: cuDNN benchmark mode is enabled without deterministic mode, PennyLane's internal RNG is not seeded, no library versions are recorded anywhere, and there is no `requirements.txt` or lock file. The data symlink creates an implicit environment dependency that is not captured in saved artifacts.

---

## Finding 1: cuDNN benchmark=True Without deterministic=True

**Severity**: HIGH

**File**: `/Users/michalszczesny/projects/hack4sages/exobiome_model/crossgen_hybrid_training.py`
**Lines**: 226-229

```python
def configure_runtime() -> None:
    # ...
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")
```

`cudnn.benchmark = True` causes cuDNN to auto-tune convolution algorithms per input shape. The selected algorithm can vary between runs and across hardware, producing different floating-point results. The code never sets:

- `torch.backends.cudnn.deterministic = True`
- `torch.use_deterministic_algorithms(True)`

This means that any CUDA-based training run is **not bitwise reproducible** even with the same seed. The Conv1d layers in `SpectralEncoder` (lines 385-392) are directly affected.

**Impact**: On GPU, re-running training with `seed=42` may yield different loss trajectories and final metrics. The saved `run_summary.json` results cannot be guaranteed to be reproduced.

**Recommendation**: Add `torch.backends.cudnn.deterministic = True` and `torch.backends.cudnn.benchmark = False` when reproducibility is required. Optionally guard this behind a `deterministic: bool` config flag to allow toggling for speed.

---

## Finding 2: PennyLane RNG Not Seeded

**Severity**: HIGH

**File**: `/Users/michalszczesny/projects/hack4sages/exobiome_model/crossgen_hybrid_training.py`
**Lines**: 212-217 (seed function) and 430-480 (QuantumBlock)

```python
def set_runtime_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
```

The seeding function covers `random`, `numpy`, `torch` (CPU and CUDA) but does **not** seed PennyLane. PennyLane has its own `qml.numpy.random` module and, depending on backend, may use independent RNG state for circuit simulation. The `lightning.qubit` device uses a C++ backend whose RNG behavior is not governed by `np.random.seed()`.

The quantum circuit weights are initialized via `torch.randn` (line 450), which *is* governed by `torch.manual_seed`. However, the forward pass through PennyLane's device simulation may involve internal randomness (e.g., for certain measurement types or stochastic gradient methods) that is unseeded.

**Impact**: The quantum forward/backward pass may not be fully reproducible across runs, even on CPU.

**Recommendation**: Add `pennylane.numpy.random.seed(seed)` in `set_runtime_seed()`. For lightning devices, verify whether the C++ backend respects any external seed or has its own seeding API.

---

## Finding 3: No Library Version Pinning or Recording

**Severity**: HIGH

**File**: Project root `/Users/michalszczesny/projects/hack4sages/exobiome_model/`

There is no `requirements.txt`, `pyproject.toml`, `Pipfile`, `environment.yml`, `setup.py`, `setup.cfg`, or any lock file in the project. The virtual environment is a symlink to a sibling project (`.venv -> ../quantum_model/.venv`), which itself has no version manifest.

The saved artifacts (`config.json`, `preflight.json`, `run_summary.json`, `best_model.pt`) contain **zero** information about library versions. Key libraries whose versions directly affect numerical results include:

| Library | Impact |
|---------|--------|
| PyTorch | Backward pass numerics, optimizer behavior |
| PennyLane | Quantum circuit simulation, gradient computation |
| PennyLane-Lightning | C++ backend numerics for adjoint differentiation |
| NumPy | Array operations, standardizer fitting |
| spectres | Rebinning algorithm |
| h5py / pandas | Data loading behavior |

**Impact**: A future attempt to reproduce this experiment would have no guidance on which versions were used. Even minor version bumps in PyTorch or PennyLane can change gradient computation and produce different results.

**Recommendation**:
1. Generate `pip freeze > requirements.txt` immediately and save alongside the model.
2. Record library versions in `config.json` or a separate `environment.json` artifact during training.
3. At minimum, log `torch.__version__`, `pennylane.__version__`, `numpy.__version__`, `spectres.__version__` into saved artifacts.

---

## Finding 4: PYTHONHASHSEED Not Set

**Severity**: MEDIUM

**File**: `/Users/michalszczesny/projects/hack4sages/exobiome_model/crossgen_hybrid_training.py`
**Lines**: 212-217

The `set_runtime_seed()` function does not set `os.environ["PYTHONHASHSEED"]`. Python's hash randomization (enabled by default since Python 3.3) affects the iteration order of sets and dicts. While this is unlikely to affect the core training loop (which uses ordered structures), it could affect:

- Any future code that iterates over `dict` keys in a version-sensitive way
- Potential non-determinism in library internals that rely on dict ordering

**Impact**: Low practical impact for the current codebase, but it is a standard reproducibility measure that is missing.

**Recommendation**: Add `os.environ["PYTHONHASHSEED"] = str(seed)` at the top of `set_runtime_seed()`, noting that it must be set before the Python interpreter starts to be fully effective (i.e., as an environment variable before `python` is invoked).

---

## Finding 5: Hardcoded Paths via Data Symlink

**Severity**: MEDIUM

**File**: `/Users/michalszczesny/projects/hack4sages/exobiome_model/data` (symlink)
**Config**: `config.json` line 5

The `data` directory is a symlink:
```
data -> /Users/michalszczesny/projects/hack4sages/datasets/crossgen_biosignatures_20260311
```

The saved `config.json` records:
```json
"data_root": "/Users/michalszczesny/projects/hack4sages/exobiome_model/data"
```

This is a machine-specific absolute path. The `data_root` fallback logic in `default_data_root()` (lines 71-78) also hardcodes a relative path assumption (`crossgen_biosignatures_20260311`). Anyone attempting to reproduce on a different machine would need to:

1. Recreate the symlink or directory structure
2. Override `data_root` in the config

The saved config does not record a data hash or checksum, so there is no way to verify that the same data files are being used.

**Impact**: Reproduction requires manual path setup. No cryptographic verification that the correct data is being used.

**Recommendation**:
1. Compute and save SHA-256 hashes of `labels.parquet` and `spectra.h5` in `preflight.json`.
2. Consider making `data_root` a required parameter rather than resolving via heuristics.

---

## Finding 6: `internal_val_fraction` Config Field Is Unused

**Severity**: MEDIUM

**File**: `/Users/michalszczesny/projects/hack4sages/exobiome_model/crossgen_hybrid_training.py`
**Lines**: 95, 313-361

```python
@dataclass
class TrainingConfig:
    # ...
    internal_val_fraction: float = 0.10
```

This parameter is defined in the config and serialized to `config.json` (`"internal_val_fraction": 0.1`), but it is **never referenced** in the `prepare_data()` function. The actual train/val split is determined entirely by the `"split"` column in `labels.parquet` (lines 318-320):

```python
inner_train_indices = np.where(labels["split"].values == "train")[0]
inner_val_indices = np.where(labels["split"].values == "val")[0]
test_indices = np.where(labels["split"].values == "test")[0]
```

**Impact**: A user reading `config.json` would believe `internal_val_fraction=0.10` was used to create the val split. The actual split is hardcoded in the parquet file and happens to be approximately 10% (4142/41423 = 10.0%), but this is coincidental and misleading. This is a config-code desync that undermines reproducibility documentation.

**Recommendation**: Either use this parameter in the code or remove it from `TrainingConfig`.

---

## Finding 7: Quantum Gradient Clip Norm Not in Config

**Severity**: MEDIUM

**File**: `/Users/michalszczesny/projects/hack4sages/exobiome_model/crossgen_hybrid_training.py`
**Line**: 750

```python
torch.nn.utils.clip_grad_norm_(quantum_params, 1.0)  # kept tight for quantum stability
```

The classical gradient clip norm is a configurable parameter (`gradient_clip_norm = 5.0`, line 105, used at line 749). However, the quantum gradient clip is hardcoded to `1.0` and not saved in `config.json`. This is a training hyperparameter that affects the optimization trajectory.

Similarly, the quantum weight initialization scale factor `0.5` (line 450) is hardcoded:

```python
self.weights = nn.Parameter(0.5 * torch.randn(self.num_weights, dtype=torch.float32))
```

Both values would need to be discovered by reading the source code to fully reproduce the experiment.

**Impact**: Someone reproducing from `config.json` alone would miss these two hyperparameters. If either value were changed, it would produce different results.

**Recommendation**: Add `quantum_gradient_clip_norm: float = 1.0` and `quantum_weight_init_scale: float = 0.5` to `TrainingConfig`.

---

## Finding 8: No Saved Model Verification Mechanism

**Severity**: MEDIUM

**File**: `/Users/michalszczesny/projects/hack4sages/exobiome_model/crossgen_hybrid_training.py`
**Lines**: 911-921

The saved `best_model.pt` contains the model state dict, config, and best epoch/loss. However, there is no mechanism to verify that loading and evaluating the saved model reproduces the reported test metrics.

The final model evaluation (lines 945-946) happens after loading `best_state` back into the model:
```python
inner_val_metrics = evaluate_split(model, data.inner_val, data.target_scaler, cfg.eval_batch_size)
test_metrics = evaluate_split(model, data.test, data.target_scaler, cfg.eval_batch_size)
```

But the saved `best_model.pt` checkpoint (lines 911-921) overwrites the mid-training checkpoint that was saved earlier (lines 807-808) at the same path. The final checkpoint does not include the scalers, so loading just `best_model.pt` requires also loading `scalers.json` separately.

The `run_summary.json` reports `best_inner_val_loss: 0.2999088764190674` matching epoch 29 from `history.csv` (0.2999088764190674 at epoch 29). However, the test metrics are computed at epoch 30 (the last state was epoch 30, but `best_state` from epoch 29 was loaded). The history CSV shows epoch 30 val loss = 0.30185... > 0.29991..., confirming the best_state is indeed from epoch 29. The test metrics in `test_metrics.json` correspond to the **best checkpoint** evaluation, not the final epoch, which is correct but not explicitly documented.

There is also a discrepancy: `run_summary.json` reports `best_inner_val_loss: 0.2999088764190674`, while `inner_val_metrics.json` reports `rmse_mean: 1.544762372970581`. These are different metrics (MSE loss vs. RMSE in original scale), but the naming could confuse someone trying to verify reproducibility.

**Impact**: No standalone script to load a checkpoint and verify it produces the same metrics. An evaluator must reconstruct the full pipeline.

**Recommendation**: Provide a `run_evaluation.py` script or a `verify_checkpoint()` function that loads `best_model.pt` + `scalers.json` + data and confirms metrics match.

---

## Finding 9: No Data Integrity Checksums in preflight.json

**Severity**: MEDIUM

**File**: `/Users/michalszczesny/projects/hack4sages/exobiome_model/outputs/model_crossgen_rebinned/preflight.json`

The preflight file records:
```json
{
  "row_count": 42108,
  "inner_train_rows": 37281,
  "inner_val_rows": 4142,
  "test_rows": 685,
  "wavelength_bins": 44
}
```

This provides split sizes but no checksums or hashes for the input data files. Without SHA-256 hashes of `labels.parquet` and `spectra.h5`, there is no way to verify that a future reproduction attempt uses identical data.

The split counts (37281 + 4142 + 685 = 42108) are verifiable, which is good. But the actual content of each split cannot be verified from preflight alone.

**Impact**: Data file corruption or subtle modifications (e.g., re-generation with different random seeds) would go undetected.

**Recommendation**: Add file hashes and optionally target distribution statistics (mean, std per target column per split) to `preflight.json`.

---

## Finding 10: `test_limit` Config Parameter Not Reflected in Data Pipeline

**Severity**: LOW

**File**: `/Users/michalszczesny/projects/hack4sages/exobiome_model/crossgen_hybrid_training.py`
**Lines**: 119, 313-361

```python
@dataclass
class TrainingConfig:
    # ...
    test_limit: Optional[int] = None
```

The `test_limit` parameter is defined in config and saved to `config.json`, but `prepare_data()` never applies it. Only `train_pool_limit` is used (line 322-323):

```python
if config.train_pool_limit is not None:
    inner_train_indices = maybe_limit_indices(inner_train_indices, config.train_pool_limit, config.seed)
```

There is no corresponding limit applied to `test_indices` or `inner_val_indices`. This is another config-code desync.

**Impact**: Minor. Both `train_pool_limit` and `test_limit` are `null` in the saved config, so no subsampling occurred. But the dead parameter is misleading.

**Recommendation**: Either implement `test_limit` or remove it from the config.

---

## Finding 11: AMP Behavior Is Platform-Dependent

**Severity**: LOW

**File**: `/Users/michalszczesny/projects/hack4sages/exobiome_model/crossgen_hybrid_training.py`
**Lines**: 117, 240-245, 534-539

```python
def resolve_amp_dtype(device: torch.device) -> Optional[torch.dtype]:
    if device.type != "cuda":
        return None
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16
```

The AMP dtype selection is hardware-dependent: bfloat16 on Ampere+ GPUs, float16 on older GPUs, disabled on CPU. The config saves `use_amp: true` but not the **resolved** dtype. The actual run used `lightning.qubit` (CPU), so AMP was disabled (`amp_dtype = None`). However, if someone re-runs on GPU, the AMP dtype would differ based on GPU generation, producing different numerics.

**Impact**: Cross-platform reproduction would use different precision without any record of what was used originally.

**Recommendation**: Record the resolved `amp_dtype` (or "disabled") in `config.json` or `run_summary.json`.

---

## Finding 12: Epoch-Based Batch Shuffling Is Correctly Deterministic

**Severity**: PASS (positive finding)

**File**: `/Users/michalszczesny/projects/hack4sages/exobiome_model/crossgen_hybrid_training.py`
**Lines**: 597-604

```python
def batch_indices(length: int, batch_size: int, seed: int, epoch: int, device: torch.device) -> Iterable[torch.Tensor]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + epoch)
    permutation = torch.randperm(length, generator=generator)
```

The batch shuffling uses an isolated `torch.Generator` with a deterministic seed derived from `config.seed + epoch`. This is well-designed:
- It does not rely on global RNG state
- It produces the same permutation for the same (seed, epoch) pair
- The CPU generator ensures consistency regardless of training device

**Impact**: This is a strong positive for reproducibility.

---

## Finding 13: Data Loading Order Is Deterministic

**Severity**: PASS (positive finding)

**File**: `/Users/michalszczesny/projects/hack4sages/exobiome_model/crossgen_hybrid_training.py`
**Lines**: 256-270, 313-361

Data loading reads the full `labels.parquet` and `spectra.h5` into memory at once. Splits are determined by `np.where(labels["split"].values == "...")`, which returns sorted indices. The `maybe_limit_indices` function also sorts its output (line 250). HDF5 reads are sequential full-array loads, not chunked iteration. This ensures deterministic data ordering.

**Impact**: Positive. No data-loading non-determinism.

---

## Finding 14: Config Completeness Assessment

**Severity**: Informational

Comparing `TrainingConfig` fields against `config.json`:

| Config Field | In config.json | Matches Code Default | Notes |
|---|---|---|---|
| seed | 42 | Yes | |
| internal_val_fraction | 0.1 | Yes | **Unused in code** (Finding 6) |
| train_batch_size | 256 | Yes | |
| eval_batch_size | 8192 | Yes | |
| max_epochs | 30 | Yes | |
| early_stop_patience | 6 | Yes | |
| scheduler_patience | 5 | Yes | |
| scheduler_factor | 0.5 | Yes | |
| classical_lr | 0.002 | Yes | |
| quantum_lr | 0.0006 | Yes | |
| weight_decay | 0.0001 | Yes | |
| gradient_clip_norm | 5.0 | Yes | |
| dropout | 0.05 | Yes | |
| aux_hidden_dim | 64 | Yes | |
| aux_out_dim | 32 | Yes | |
| spectral_hidden_dim | 64 | Yes | |
| spectral_out_dim | 32 | Yes | |
| fusion_hidden_dim | 48 | Yes | |
| head_hidden_dim | 96 | Yes | |
| qnn_qubits | 12 | Yes | |
| qnn_depth | 2 | Yes | |
| quantum_device | lightning.qubit | Yes | |
| log_every_batches | 1 | Yes | |
| use_amp | true | Yes | |
| train_pool_limit | null | Yes | |
| test_limit | null | Yes | **Unused in code** (Finding 10) |

**Missing from config.json**:
- Quantum gradient clip norm: `1.0` (hardcoded line 750)
- Quantum weight init scale: `0.5` (hardcoded line 450)
- SpectralEncoder conv kernel sizes: `7, 5, 3` (hardcoded lines 386-390)
- SpectralEncoder first conv channels: `32` (hardcoded line 386)
- SpectralEncoder conv stride: `2` (hardcoded line 388)
- Resolved AMP dtype: not recorded
- Library versions: not recorded
- Data file checksums: not recorded

---

## Findings Summary

| # | Finding | Severity |
|---|---------|----------|
| 1 | cuDNN benchmark=True without deterministic=True | HIGH |
| 2 | PennyLane RNG not seeded | HIGH |
| 3 | No library version pinning or recording | HIGH |
| 4 | PYTHONHASHSEED not set | MEDIUM |
| 5 | Hardcoded paths via data symlink, no data checksums | MEDIUM |
| 6 | `internal_val_fraction` config field unused (config-code desync) | MEDIUM |
| 7 | Quantum gradient clip norm and weight init scale not in config | MEDIUM |
| 8 | No saved model verification mechanism | MEDIUM |
| 9 | No data integrity checksums in preflight.json | MEDIUM |
| 10 | `test_limit` config parameter unused (config-code desync) | LOW |
| 11 | AMP behavior is platform-dependent and not recorded | LOW |
| 12 | Batch shuffling is correctly deterministic | PASS |
| 13 | Data loading order is deterministic | PASS |

---

## Verdict: FAIL

The codebase has a reasonable reproducibility foundation (deterministic batch shuffling, config serialization, scaler saving, split-based data organization). However, three HIGH-severity issues prevent a passing grade:

1. **No deterministic CUDA mode**: GPU training cannot be reproduced bitwise.
2. **PennyLane RNG unseeded**: The quantum component is not fully reproducibility-controlled.
3. **No version pinning**: There is zero record of which library versions produced the saved results. No `requirements.txt`, no lock file, no version logging in any artifact.

Additionally, four MEDIUM-severity config-code desyncs and missing checksums erode confidence that the experiment can be faithfully reconstructed from saved artifacts alone.

**Can training be reproduced from saved config alone?** No. The config is missing hardcoded hyperparameters (quantum grad clip, weight init scale, conv architecture details), library versions, and data checksums. A researcher would need both the source code at the exact commit and the same library environment.

**Does setting seed=42 guarantee identical results?** On CPU with `lightning.qubit`: likely yes for the classical portion, uncertain for PennyLane internals. On GPU: no, due to cuDNN non-determinism and TF32 variability.
