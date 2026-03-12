# Crossgen Dataset Training — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Adapt the fixed hybrid quantum model to train on the crossgen dataset (TauREx train/val, POSEIDON test)

**Architecture:** Same model as ADC2023 (all 8 fixes intact). Only the data loading and feature columns change. The model architecture, training loop, hyperparameters, and evaluation pipeline stay identical.

**Tech Stack:** Same — torch, pennylane, pandas, h5py, numpy, scikit-learn

---

## Key Differences: ADC2023 vs Crossgen

| | ADC2023 | Crossgen |
|---|---|---|
| Data format | 3 CSVs + per-planet HDF5 groups | 1 parquet + flat HDF5 |
| Spectra shape | (N, 52) via per-planet groups | (42108, 218) flat array |
| Noise | per-bin array (N, 52) | scalar `sigma_ppm` per sample |
| Aux features | 8 (star_mass_kg, star_radius_m, ...) | 4 (planet_radius_rjup, log_g_cgs, temperature_k, star_radius_rsun) + derived log10_sigma_ppm |
| Targets | log_H2O etc (from FM_Parameter_Table) | log10_vmr_h2o etc (in labels.parquet) |
| Splits | random 90/10/10 | pre-defined: tau/train, tau/val, poseidon/test |
| ID column | planet_ID | sample_id |
| Wavelength bins | 52 | 218 |

## Source Code

The fixed model lives on branch `feat/quantum-model` at `quantum_model/crossgen_hybrid_training.py`. Copy it into the working directory first.

---

### Task 1: Copy fixed model code into working directory

**Files:**
- Create: `quantum_model/crossgen_hybrid_training.py` (copy from feat/quantum-model branch)

**Step 1:** Extract the fixed .py from the branch

```bash
cd /Users/michalszczesny/projects/hack4sages
git show feat/quantum-model:quantum_model/crossgen_hybrid_training.py > quantum_model/crossgen_hybrid_training.py
```

**Step 2:** Verify the file has the fixes (skip connections, LayerNorm, 0.5*randn init, etc.)

```bash
grep -n "aux_feat.float(), spectral_feat.float()" quantum_model/crossgen_hybrid_training.py
grep -n "LayerNorm" quantum_model/crossgen_hybrid_training.py
grep -n "0.5 \* torch.randn" quantum_model/crossgen_hybrid_training.py
```

Expected: all three match (skip connections at ~546, LayerNorm at ~411, init at ~449).

---

### Task 2: Update constants for crossgen dataset

**Files:**
- Modify: `quantum_model/crossgen_hybrid_training.py` — lines 1, 27-44

**Step 1:** Change docstring

```python
# Line 1: change from
"""Hybrid quantum training helpers for the ADC2023 (Ariel Data Challenge) dataset."""
# to
"""Hybrid quantum training helpers for the crossgen biosignatures dataset."""
```

**Step 2:** Update SAFE_AUX_FEATURE_COLS (lines 27-36)

```python
SAFE_AUX_FEATURE_COLS = [
    "planet_radius_rjup",
    "log_g_cgs",
    "temperature_k",
    "star_radius_rsun",
    "log10_sigma_ppm",
]
```

5 features (4 from parquet + 1 derived from sigma_ppm).

**Step 3:** Update TARGET_COLS (lines 38-44)

```python
TARGET_COLS = [
    "log10_vmr_h2o",
    "log10_vmr_co2",
    "log10_vmr_co",
    "log10_vmr_ch4",
    "log10_vmr_nh3",
]
```

---

### Task 3: Replace data loading function

**Files:**
- Modify: `quantum_model/crossgen_hybrid_training.py` — replace `load_adc_dataset` (~lines 233-264)

**Step 1:** Replace `load_adc_dataset` with `load_crossgen_dataset`

```python
def load_crossgen_dataset(data_root: Path) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    labels = pd.read_parquet(data_root / "labels.parquet")

    with h5py.File(data_root / "spectra.h5", "r") as handle:
        wavelength_um = np.asarray(handle["wavelength_um"][:], dtype=np.float32)
        noisy_spectra = np.asarray(handle["transit_depth_noisy"][:], dtype=np.float32)
        sigma_ppm = np.asarray(handle["sigma_ppm"][:], dtype=np.float32)

    labels["log10_sigma_ppm"] = np.log10(np.clip(sigma_ppm, 1e-10, None))

    return labels, noisy_spectra, sigma_ppm, wavelength_um
```

**Step 2:** Update `default_data_root` (~line 56) to point to crossgen

```python
def default_data_root(project_root: Path) -> Path:
    return project_root / "quantum_model_crossgen" / "crossgen_biosignatures_20260311"
```

---

### Task 4: Replace `prepare_data` split logic

**Files:**
- Modify: `quantum_model/crossgen_hybrid_training.py` — `prepare_data` function (~lines 305-360)

The crossgen dataset has pre-defined splits in the `split` column (train/val/test) and `generator` column (tau/poseidon). Use these instead of random `train_test_split`.

**Step 1:** Replace prepare_data

```python
def prepare_data(config: TrainingConfig) -> PreparedData:
    labels, noisy_spectra, sigma_ppm, wavelength_um = load_crossgen_dataset(config.resolved_data_root())
    aux_raw, spectra_raw = build_raw_arrays(labels, noisy_spectra)
    targets_raw = labels[TARGET_COLS].to_numpy(dtype=np.float32, copy=True)

    # Use pre-defined splits: tau/train, tau/val, poseidon/test
    inner_train_indices = np.where(labels["split"] == "train")[0]
    inner_val_indices = np.where(labels["split"] == "val")[0]
    test_indices = np.where(labels["split"] == "test")[0]

    if config.train_pool_limit:
        inner_train_indices = inner_train_indices[: config.train_pool_limit]
    if config.test_limit:
        test_indices = test_indices[: config.test_limit]

    preflight = split_summary(labels, wavelength_um)
    preflight["inner_train_rows"] = int(len(inner_train_indices))
    preflight["inner_val_rows"] = int(len(inner_val_indices))
    preflight["test_rows"] = int(len(test_indices))

    # Fit scalers on TRAIN only
    aux_scaler = ArrayStandardizer.fit(aux_raw[inner_train_indices])
    target_scaler = ArrayStandardizer.fit(targets_raw[inner_train_indices])
    spectral_scaler = SpectralStandardizer.fit(spectra_raw[inner_train_indices, 0, :])

    def transform(indices):
        a = aux_scaler.transform(aux_raw[indices])
        s = spectral_scaler.transform(spectra_raw[indices])
        t = target_scaler.transform(targets_raw[indices])
        return a, s, t

    train_aux, train_spectra, train_targets = transform(inner_train_indices)
    inner_val_aux, inner_val_spectra, inner_val_targets = transform(inner_val_indices)
    test_aux, test_spectra, test_targets = transform(test_indices)

    return PreparedData(
        train=make_split_tensors(
            labels.iloc[inner_train_indices],
            train_aux, train_spectra, train_targets, targets_raw[inner_train_indices],
        ),
        inner_val=make_split_tensors(
            labels.iloc[inner_val_indices],
            inner_val_aux, inner_val_spectra, inner_val_targets, targets_raw[inner_val_indices],
        ),
        test=make_split_tensors(
            labels.iloc[test_indices],
            test_aux, test_spectra, test_targets, targets_raw[test_indices],
        ),
        aux_scaler=aux_scaler,
        target_scaler=target_scaler,
        spectral_scaler=spectral_scaler,
        preflight=preflight,
    )
```

---

### Task 5: Update `make_split_tensors` ID column

**Files:**
- Modify: `quantum_model/crossgen_hybrid_training.py` — `make_split_tensors` (~line 296)

Change `planet_ID` to `sample_id`:

```python
def make_split_tensors(
    labels: pd.DataFrame,
    aux_values: np.ndarray,
    spectra_values: np.ndarray,
    targets_scaled: np.ndarray,
    raw_targets: np.ndarray,
) -> SplitTensors:
    return SplitTensors(
        sample_ids=labels["sample_id"].to_numpy(dtype="U32"),
        ...
    )
```

---

### Task 6: Update output directory default

**Files:**
- Modify: `quantum_model/crossgen_hybrid_training.py` — `TrainingConfig.resolved_output_dir` or default

Change output dir from `outputs/model_quant_sketch_adc` to `outputs/model_crossgen`.

---

### Task 7: Adjust model input dimension

**Files:**
- Modify: `quantum_model/crossgen_hybrid_training.py` — `build_model` function

The AuxEncoder input changes from 8 features to 5. This is handled automatically by `len(SAFE_AUX_FEATURE_COLS)` — no code change needed, just verify.

The SpectralEncoder input changes from 52 bins to 218 bins. Conv1d with AdaptiveAvgPool1d handles variable input length — no code change needed, just verify.

**Step 1:** Verify no hardcoded dimensions

```bash
grep -n "52\|218" quantum_model/crossgen_hybrid_training.py
```

Expected: no hardcoded spectral dimensions.

---

### Task 8: Smoke test — verify data loads correctly

**Step 1:** Run a quick sanity check

```bash
cd /Users/michalszczesny/projects/hack4sages/quantum_model
.venv/bin/python -c "
from crossgen_hybrid_training import TrainingConfig, prepare_data
data = prepare_data(TrainingConfig(qnn_qubits=12))
print(f'Train: {data.train.aux.shape}')
print(f'Val: {data.inner_val.aux.shape}')
print(f'Test: {data.test.aux.shape}')
print(f'Spectra bins: {data.train.spectra.shape[-1]}')
print(f'Aux features: {data.train.aux.shape[-1]}')
"
```

Expected output:
```
Train: torch.Size([37281, 5])
Val: torch.Size([4142, 5])
Test: torch.Size([685, 5])
Spectra bins: 218
Aux features: 5
```

---

### Task 9: Run training

**Step 1:** Launch training in terminal (user watches progress)

```bash
cd /Users/michalszczesny/projects/hack4sages/quantum_model
.venv/bin/python -u -c "
from crossgen_hybrid_training import TrainingConfig, run_training_experiment
result = run_training_experiment(TrainingConfig(qnn_qubits=12))
"
```

This is run by the user in their terminal, not in background. Do NOT run this — just print the command.
