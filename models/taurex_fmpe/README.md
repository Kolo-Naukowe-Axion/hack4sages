# TauREx FMPE

This folder is the self-contained non-quantum FMPE / flow-matching training path for the repo's TauREx bundle.

What it does:
- trains the Dingo-based `sbi-ariel` style FMPE posterior model on the TauREx rows in `data/TauREx set`
- ignores POSEIDON entirely
- uses the embedded TauREx split membership already saved in `labels.parquet`
- mirrors `tau/val` into `holdout` and `testdata` so the FMPE training/evaluation stack keeps the same contract as the ADC2023 path

Contents:
- `prepare_dataset.py`: reads `labels.parquet` + `spectra.h5` and writes prepared arrays
- `dataset.py`: loads prepared train, validation, holdout, and test splits
- `raw_dataset.py`: TauREx parquet/HDF5 loading and auxiliary-feature synthesis
- `dingo_compat.py`: Dingo/FMPE model builder and checkpoint helpers
- `runtime.py`: local PyTorch training utilities
- `training.py`, `train.py`, `evaluate.py`: train, resume, and evaluate the FMPE model
- `settings/taurex_rtx4090.yaml`: default 4090-oriented training config
- `run_train_ubuntu4090.sh`: end-to-end launcher for Linux/Vast-style machines

Input contract:
- `data/TauREx set/labels.parquet`
- `data/TauREx set/spectra.h5`

Prepared context layout:
- `218` per-spectrum mean-normalized transit-depth bins
- `218` log10 white-noise bins expanded from `sigma_ppm`
- `8` synthesized ADC-style auxiliary features derived from TauREx labels

Default target set:
- `log10_vmr_h2o`
- `log10_vmr_co2`
- `log10_vmr_co`
- `log10_vmr_ch4`
- `log10_vmr_nh3`
