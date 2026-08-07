# ExoBiome

ExoBiome is KN Axion's Hack4SAGES 2026 project on atmospheric-gas retrieval from exoplanet transmission spectra. It evaluates hybrid quantum-classical regression alongside classical and normalizing-flow baselines, including an execution path for the 20-qubit IQM Garnet processor.

Target gases: `H2O`, `CO2`, `CO`, `CH4`, and `NH3`.

Authors: Iwo Wojtakajtis, Iwo Smura, Maria Płatek, and Michał Szczęsny. [Watch the project presentation](https://youtu.be/3fCZmm0QsE4).

## Repository

- [`models/`](models) — training and evaluation packages.
- [`data/`](data) — dataset generation, validation, and benchmark preparation.
- [`reports/`](reports) — evaluation summaries and audit notes.
- [`models/garnet_ariel_quantum_regression/`](models/garnet_ariel_quantum_regression) — IQM Garnet evaluation path.
- [`archive/`](archive) — preserved exploratory notebooks and earlier prototypes.

## Evaluation

The project studies small-data performance, transfer across independently generated atmosphere datasets, and execution of the quantum branch on physical hardware.

The best currently verified saved hybrid checkpoint reports validation `mRMSE ≈ 0.2936` and holdout `mRMSE ≈ 0.2994`. Exploratory and presentation artifacts are kept separate from verified result snapshots.

Dataset preparation is documented in [`data/README.md`](data/README.md). Model-specific setup lives beside each model family.
