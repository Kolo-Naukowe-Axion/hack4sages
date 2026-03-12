# Ariel Winner NSF

This package implements the strongest externally-supported non-quantum ADC2023 architecture I found for atmospheric retrieval:

- source family: the AstroAI winning ADC2023 normalizing-flow solution
- implemented variant: the winner paper and poster's noised-spectrum, input-parameter path
- model form: independent conditional neural spline flows built with Zuko

Primary sources:

- official leaderboard: <https://www.ariel-datachallenge.space/ML/leaderboard_final/>
- winner paper: <https://arxiv.org/abs/2309.09337>
- winner code: <https://github.com/AstroAI-CfA/Ariel_Data_Challenge_2023_solution>
- third-place reference: <https://arxiv.org/abs/2406.10771>

What is implemented here:

- five-gas targets only: `log_H2O`, `log_CO2`, `log_CO`, `log_CH4`, `log_NH3`
- winner-style preprocessing:
  - noised spectra
  - per-spectrum mean/std normalization with mean/std retained as features
  - normalized auxiliary variables
  - engineered radius features from spectrum and gravity
  - normalized uncertainty vector as part of the context
- one independent conditional NSF per gas target

Entry points:

- prepare data:
  - `python -m models.ariel_winner_nf.prepare_dataset --data-root data/full-ariel --split-source data/val_dataset --output data/generated-data/ariel_winner_nf_prepared --overwrite`
- train:
  - `python -m models.ariel_winner_nf.train --settings models/ariel_winner_nf/settings/winner_noised_independent_nsf.yaml --prepared-data data/generated-data/ariel_winner_nf_prepared --run-dir local_runs/ariel_winner_nf_test`
- launcher:
  - `bash models/ariel_winner_nf/run_train_ubuntu4090.sh`

Saved split behavior:

- if `data/val_dataset` exists, preparation uses its exact `train/validation/holdout` `planet_ID` membership
- the current saved split is `33,138 / 4,142 / 4,143` rows, for `41,423` labeled planets total
- the launcher fails fast if those split files are missing, rather than silently reshuffling rows

Default context dimension is `118`:

- `12` engineered auxiliary features
- `2` spectral summary features
- `52` normalized spectral bins
- `52` normalized noise bins
