# Design: Prepare exobiome_model for Training

**Date**: 2026-03-11
**Goal**: Clean up exobiome_model so it's understandable and trainable on the crossgen_biosignatures dataset.

## Approach: Document-in-place

Keep the 984-line monolith (`crossgen_hybrid_training.py`) as-is. Make minimal code changes. Focus on documentation and making it runnable.

## Steps

### 1. Fix path resolution
Change `default_data_root()` (line 71-75) to check for `data/` symlink first, then fall back to original paths. This is the only code change to the model file.

### 2. Add legacy_model.md
Project-level docs with:
- One-liner project description
- Section map of crossgen_hybrid_training.py (line ranges to purpose)
- How to run training (CLI + notebook)
- Dataset: only labels.parquet + spectra.h5 matter
- Key tunable hyperparameters

### 3. Improve train.ipynb
Expand from 4 cells to include:
- Config with explicit data_root="data"
- Data loading sanity checks (shapes, split counts, sample ranges)
- Training execution
- Results display
- Predicted vs true scatter plot

### 4. Add run_training.py
~10-line CLI entrypoint that imports from the monolith and runs with data_root="data".

### 5. Verify it runs
Smoke test: load data, check shapes, verify imports. No full training run (hours on CPU), just confirm the pipeline works up to first forward pass.

## Final Structure
```
exobiome_model/
├── .legacy_model/
│   └── research.md
├── .gitignore
├── legacy_model.md
├── crossgen_hybrid_training.py    # one path fix
├── run_training.py                # new CLI entrypoint
├── train.ipynb                    # improved notebook
├── docs/plans/
│   └── 2026-03-11-prepare-training-design.md
└── data -> symlink to dataset
```

## What we're NOT doing
- No file splitting/refactoring of the monolith
- No feature engineering additions
- No model architecture changes
- No use of latents.parquet or shards/ (only labels.parquet + spectra.h5)

## Hardware
- Local CPU (lightning.qubit) for now
- Code already auto-detects CUDA; vast.ai GPU possible later
