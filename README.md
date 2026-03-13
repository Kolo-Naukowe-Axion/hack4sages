# ExoBiome

ExoBiome is Axion's Hack4SAGES 2026 project on biosignature retrieval from exoplanet transmission spectra. The core idea is a hybrid quantum/classical regression stack for estimating five atmospheric gases (`H2O`, `CO2`, `CO`, `CH4`, `NH3`) from Ariel-style spectra, with classical baselines and winner-model ports alongside it.

The hackathon presentation is here: [ExoBiome - Hybrid Quantum ML for Biosignature Identification in Transmission Spectroscopy](https://youtu.be/3fCZmm0QsE4?si=l1vFH72Xhi8hS9G2). Authors: Iwo Wojtakajtis, Iwo Smura, Maria Platek, and Michal Szczesny.

## Current repo shape

- [models](/Users/iwosmura/projects/hack4sages/models): active training and evaluation packages.
- [data](/Users/iwosmura/projects/hack4sages/data): dataset generators, validators, and benchmark preparation code.
- [reports](/Users/iwosmura/projects/hack4sages/reports): generated evaluation summaries and audit notes.
- [archive](/Users/iwosmura/projects/hack4sages/archive): preserved exploratory notebooks and the deduped pre-cleanup snapshot imported from `/Users/iwosmura/Downloads/hack4sages`.

## Model map

- [models/ariel_exobiome](/Users/iwosmura/projects/hack4sages/models/ariel_exobiome): hybrid quantum regressor for the ADC2023 Ariel dataset.
- [models/taurex_exobiome](/Users/iwosmura/projects/hack4sages/models/taurex_exobiome): TauREx/POSEIDON cross-generator version of the hybrid quantum model.
- [models/taurex_exobiome_without_quant](/Users/iwosmura/projects/hack4sages/models/taurex_exobiome_without_quant): classical control / ablation of the TauREx model.
- [models/five_qubit_exobiome](/Users/iwosmura/projects/hack4sages/models/five_qubit_exobiome): five-qubit variant of the same family.
- [models/ariel_winner_nf](/Users/iwosmura/projects/hack4sages/models/ariel_winner_nf), [models/ariel_winner_on_taurex](/Users/iwosmura/projects/hack4sages/models/ariel_winner_on_taurex), and [models/ariel_winner_trace_nf](/Users/iwosmura/projects/hack4sages/models/ariel_winner_trace_nf): winner-style normalizing-flow baselines and reruns.
- [models/sbi_ariel_adc2023](/Users/iwosmura/projects/hack4sages/models/sbi_ariel_adc2023) and [models/taurex_fmpe](/Users/iwosmura/projects/hack4sages/models/taurex_fmpe): FMPE / flow-matching baselines.
- [models/garnet_ariel_quantum_regression](/Users/iwosmura/projects/hack4sages/models/garnet_ariel_quantum_regression): IQM Garnet evaluation path for the quantum branch.

## What was cleaned up

- The nonsense package name `models/ariel_winner_rerun_model_please` was renamed to [models/ariel_winner_trace_nf](/Users/iwosmura/projects/hack4sages/models/ariel_winner_trace_nf).
- Stray root-level experiment files were moved into [archive/model_sketch_experiment](/Users/iwosmura/projects/hack4sages/archive/model_sketch_experiment) and [archive/ariel_runtime_data_scaled.ipynb](/Users/iwosmura/projects/hack4sages/archive/ariel_runtime_data_scaled.ipynb).
- The CPU sweep bundle `fair_small_experiment_cpu` was moved under [outputs/fair_small_experiment_cpu](/Users/iwosmura/projects/hack4sages/outputs/fair_small_experiment_cpu), because it is a generated benchmark run, not source code.
- A deduped authored-file snapshot from `/Users/iwosmura/Downloads/hack4sages` now lives in [archive/early_prototype_snapshot](/Users/iwosmura/projects/hack4sages/archive/early_prototype_snapshot).
- A second curated snapshot from `/Users/iwosmura/Downloads/hack4sages 2` now lives in [archive/hack4sages_2_snapshot](/Users/iwosmura/projects/hack4sages/archive/hack4sages_2_snapshot). It preserves the useful parts of that dump: baseline code, audit reports, alternate website prototypes, video source, research docs, and quantum-advantage notebooks.
- Heavy generated outputs, raw dataset blobs, virtual environments, and obvious duplicate files were intentionally not copied from `Downloads/`.

## Where to start

- Read [archive/early_prototype_snapshot/README.md](/Users/iwosmura/projects/hack4sages/archive/early_prototype_snapshot/README.md) if you want the notebook-era prototype history.
- Read [archive/hack4sages_2_snapshot/README.md](/Users/iwosmura/projects/hack4sages/archive/hack4sages_2_snapshot/README.md) if you want the later mixed snapshot with web prototypes, audit docs, and presentation assets.
- Read the package README nearest the model family you care about in [models](/Users/iwosmura/projects/hack4sages/models).
- Open [data/README.md](/Users/iwosmura/projects/hack4sages/data/README.md) if you want the dataset-generation pipeline.

## Verified result snapshot

The current reports and saved artifacts point to the same central claim: the best verified hybrid checkpoint is the epoch-6 model with validation `mRMSE` about `0.2936` and holdout `mRMSE` about `0.2994`. Cross-model comparison tables elsewhere in the repo still mix verified numbers with placeholder presentation values, so treat those sections accordingly.
