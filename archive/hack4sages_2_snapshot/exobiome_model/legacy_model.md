# exobiome_model

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
| 62-82 | Path resolution helpers (`resolve_project_root`, `default_data_root`, `default_output_dir`) |
| 86-144 | `TrainingConfig` dataclass — all hyperparameters |
| 147-208 | Standardizers (`ArrayStandardizer`, `SpectralStandardizer`) + data containers (`SplitTensors`, `PreparedData`) |
| 211-229 | Runtime setup (`set_runtime_seed`, `configure_runtime`) |
| 231-252 | Device resolution helpers |
| 255-361 | **Data pipeline** — `load_crossgen_dataset()`, `build_raw_arrays()`, `prepare_data()` |
| 364-416 | **Classical encoders** — `AuxEncoder`, `SpectralEncoder`, `FusionEncoder` |
| 419-485 | **Quantum block** — PennyLane circuit (12 qubits, RY/CNOT/RZ/CRX) |
| 488-570 | **Model assembly** — `AtmosphereHead`, `HybridAtmosphereModel`, `build_model()` |
| 573-636 | Device/batch utilities |
| 639-683 | `evaluate_split()` — validation/test evaluation |
| 685-827 | `train_model()` — full training loop with early stopping |
| 830-986 | `run_training_experiment()` — end-to-end orchestrator + artifact saving |

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
