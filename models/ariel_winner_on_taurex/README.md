# Ariel Winner NSF On TauREx

This package ports the strongest externally-supported non-quantum ADC2023 architecture I found onto the repo's TauREx/POSEIDON cross-generator bundle:

- source family: the AstroAI winning ADC2023 normalizing-flow solution
- implemented variant: the winner paper and poster's noised-spectrum, input-parameter path
- model form: independent conditional neural spline flows built with Zuko
- training domain: TauREx rows from `data/TauREx set`
- validation domain: TauREx `val`
- holdout domain: POSEIDON `test`

Primary sources:

- official leaderboard: <https://www.ariel-datachallenge.space/ML/leaderboard_final/>
- winner paper: <https://arxiv.org/abs/2309.09337>
- winner code: <https://github.com/AstroAI-CfA/Ariel_Data_Challenge_2023_solution>
- third-place reference: <https://arxiv.org/abs/2406.10771>

What is implemented here:

- five-gas targets only: `log10_vmr_h2o`, `log10_vmr_co2`, `log10_vmr_co`, `log10_vmr_ch4`, `log10_vmr_nh3`
- winner-style preprocessing:
  - noisy transit-depth spectra from `spectra.h5`
  - per-spectrum mean/std normalization with mean/std retained as features
  - synthesized ADC-style auxiliary inputs derived from the TauREx labels and fixed dataset-generation constants
  - engineered radius features from spectrum and gravity
  - per-bin white-noise vector expanded from the dataset's scalar `sigma_ppm`
- one independent conditional NSF per gas target

Entry points:

- prepare data:
  - `python -m models.ariel_winner_on_taurex.prepare_dataset --data-root "data/TauREx set" --output data/generated-data/ariel_winner_on_taurex_prepared --overwrite`
- train:
  - `python -m models.ariel_winner_on_taurex.train --settings models/ariel_winner_on_taurex/settings/winner_noised_independent_nsf.yaml --prepared-data data/generated-data/ariel_winner_on_taurex_prepared --run-dir local_runs/ariel_winner_on_taurex_test`
- launcher:
  - `bash models/ariel_winner_on_taurex/run_train_ubuntu4090.sh`

Dataset split behavior:

- uses the saved split membership already embedded in `labels.parquet`
- `tau/train` becomes the training split (`37,281` rows in the current bundle)
- `tau/val` becomes the validation split (`4,142` rows)
- `poseidon/test` becomes both:
  - labeled `holdout` for final metrics
  - unlabeled `testdata` mirror for compatibility with the existing training stack

Default context dimension is `450`:

- `12` engineered auxiliary features
- `2` spectral summary features
- `218` normalized spectral bins
- `218` normalized noise bins
