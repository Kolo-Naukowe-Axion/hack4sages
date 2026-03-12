# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Hybrid quantum-classical neural network for exoplanet biosignature detection. Part of the HACK-4-SAGES hackathon (ETH Zurich). Takes exoplanet transmission spectra and predicts atmospheric abundances (log VMR) for 5 molecules: H2O, CO2, CO, CH4, NH3.

## Running

**Notebook** (primary workflow):
```bash
jupyter notebook model_quant_sketch.ipynb
```

**Programmatic**:
```python
from crossgen_hybrid_training import TrainingConfig, run_training_experiment
result = run_training_experiment(TrainingConfig())
```

**GPU training** (Vast.ai):
```bash
cd /workspace && python run_training.py
```

**Environment variable overrides** (all optional):
```bash
DATA_ROOT  OUTPUT_DIR  SEED  TRAIN_BATCH_SIZE  EVAL_BATCH_SIZE
MAX_EPOCHS  QNN_QUBITS  QNN_DEPTH  QUANTUM_DEVICE  TRAIN_POOL_LIMIT
LOG_EVERY_BATCHES
```

There is no test suite, Makefile, linter config, or build system. Validation happens inside the training loop.

## Architecture

See `model.png` for the architecture diagram.

All code lives in two files:
- `crossgen_hybrid_training.py` — single-module implementation (all classes, training loop, evaluation)
- `model_quant_sketch.ipynb` — notebook that imports and drives training

**Model pipeline** (class per stage, all `nn.Module`):
```
AuxMetadata → AuxEncoder (FFN)         — planet/star properties (8 features) → 32-dim
SpectralInput → SpectralEncoder (Conv1d) — transmission spectrum (52 bins) → 32-dim
                    ↓ both feed into ↓
                FusionBlock (FFN)        — concatenated → LayerNorm → tanh*π scaled → 16-dim
                    ↓
                QuantumProjection → VQC  — variational circuit (16 qubits, RY/CNOT/RZ/CRX)
                    ↓
                PredictionHead (FFN)     — receives: quantum output + skip from both encoders
                    ↓
                5 AtmosphereTargets      — VMR predictions
```

Skip connections feed AuxEncoder and SpectralEncoder outputs directly into PredictionHead alongside the quantum path. Wrapped by `HybridAtmosphereModel`.

**Quantum specifics**: PennyLane with `lightning.qubit` (CPU) or `lightning.gpu` (CUDA). Adjoint differentiation. Quantum path forced to float32 even under AMP. Separate optimizer param groups with different LRs for classical (2e-3) vs quantum (6e-4).

## Data

ADC2023 (Ariel Data Challenge) dataset at `ariel-ml-dataset/`:
- `TrainingData/AuxillaryTable.csv` — 41,423 samples, 8 stellar/planetary features + planet_ID
- `TrainingData/Ground Truth Package/FM_Parameter_Table.csv` — targets: log_H2O, log_CO2, log_CO, log_CH4, log_NH3
- `TrainingData/SpectralData.hdf5` — per-planet groups with instrument_spectrum, instrument_noise, instrument_wlgrid (52 bins)
- `TestData/` — 685 test samples (same format, no ground truth)

**Splits**: 90/10/10 train/val/test from training data (simple random split).

**Scalers**: `ArrayStandardizer` for aux features and targets, `SpectralStandardizer` for per-wavelength-bin spectrum normalization. Fit on training set only.

## Output Artifacts

Saved to `outputs/model_quant_sketch_adc/`:
- `best_model.pt`, `last_model.pt` — state dicts
- `scalers.json` — fitted scaler parameters
- `history.csv` — per-epoch losses and metrics
- `test_predictions.csv`
- `metrics_summary.csv`, `loss_curve.png`, `rmse_curve.png`

## Key Constants

- `SAFE_AUX_FEATURE_COLS`: 8 features (star_mass_kg, star_radius_m, star_temperature, planet_mass_kg, planet_orbital_period, planet_distance, planet_surface_gravity, log10_noise_mean)
- `TARGET_COLS`: 5 log VMR targets (log_H2O, log_CO2, log_CO, log_CH4, log_NH3)

## Dependencies

torch, pennylane, numpy, pandas, h5py, scikit-learn, matplotlib

## Conventions

- `TrainingConfig` dataclass holds all hyperparameters; env vars override defaults
- Path resolution uses `resolve_project_root()` with fallback logic for different working directories
- AMP (mixed precision) auto-enabled on CUDA, disabled on CPU
