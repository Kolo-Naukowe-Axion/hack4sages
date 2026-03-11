# ADC2023 FMPE Folder

This folder is the self-contained non-quantum training path for ADC2023 five-gas retrieval.

Contents:
- `prepare_dataset.py`: reads raw ADC2023 CSV/HDF5 files and writes prepared arrays
- `dataset.py`: loads prepared train, validation, holdout, and test splits
- `raw_dataset.py`: raw CSV/HDF5 loading plus stratification helpers
- `dingo_compat.py`: local Dingo/FMPE model builder and checkpoint helpers
- `runtime.py`: local PyTorch training utilities
- `training.py`, `train.py`, `evaluate.py`: train, resume, and evaluate the FMPE model
- `settings/adc2023_rtx4090.yaml`: default 4090-oriented training config
- `run_train_ubuntu4090.sh`: end-to-end launcher for Linux/Vast-style machines
- `adc2023_fmpe_runbook.ipynb`: one notebook with setup, prepare, train, monitor, and evaluate steps

The only thing expected outside this folder is the raw dataset itself, typically at `data/full-ariel`.
