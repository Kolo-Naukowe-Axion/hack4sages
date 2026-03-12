# Ariel Winner Trace NF

This package used to live under the placeholder name `ariel_winner_rerun_model_please`. It was renamed because this folder is specifically the trace-data normalizing-flow rerun of the public AstroAI winner family.

This package is a clean rerun path for the actual AstroAI winning model family on the local Ariel dataset.

What this package is:

- winner-family architecture: independent conditional neural spline flows, one flow per atmospheric parameter
- winner-family supervision: full per-planet `Tracedata.hdf5` samples with their weights
- winner-family targets: all `7` posterior parameters
  - `planet_radius`
  - `planet_temp`
  - `log_H2O`
  - `log_CO2`
  - `log_CO`
  - `log_CH4`
  - `log_NH3`
- deterministic local split: intersection of the repo's saved `data/val_dataset` split with the `6,766` planets that actually have valid tracedata/quartiles

What this package is not:

- it is not the archived `.55 mRMSE` five-gas surrogate in `models/adc_winner_on_ariel`
- it is not a claim of exact leaderboard reproduction
  - the public AstroAI repo does not ship the original hyperopt logs or the saved ensemble members that defined the final winning submission
  - this rerun therefore uses the real winner training target and architecture family, but an explicit local `ensemble_size = 1`

Primary references used while building this package:

- official winner repo: <https://github.com/AstroAI-CfA/Ariel_Data_Challenge_2023_solution>
- winner paper: <https://arxiv.org/abs/2309.09337>

The public repo's own comparison script labels `Ensembles__independent_flows.pth` as the winning model, and treats the complete-flow variants as alternative models. This package follows that independent-flow path.

Prepared split sizes in this repo:

- train tracedata rows: `5,440`
- validation tracedata rows: `663`
- holdout tracedata rows: `663`
- unlabeled testdata rows: `685`

Entry points:

- prepare data:
  - `python -m models.ariel_winner_trace_nf.prepare_dataset --data-root data/full-ariel --split-source data/val_dataset --output data/generated-data/ariel_winner_trace_nf_prepared --overwrite`
- train:
  - `python -m models.ariel_winner_trace_nf.train --settings models/ariel_winner_trace_nf/settings/winner_trace_independent_nsf.yaml --prepared-data data/generated-data/ariel_winner_trace_nf_prepared --run-dir "local_runs/ariel_winner_trace_nf"`
- evaluate a finished run:
  - `python -m models.ariel_winner_trace_nf.evaluate --settings <run-dir>/settings_resolved.yaml --prepared-data data/generated-data/ariel_winner_trace_nf_prepared --run-dir <run-dir>`

Saved outputs from training include:

- per-target checkpoints in `targets/<target_name>/`
- bundled checkpoint in `best_independent_bundle.pt`
- validation and holdout metrics with:
  - weighted-trace K-S statistics
  - posterior-median RMSE against FM point targets
  - posterior-median RMSE against quartile medians
- validation, holdout, and testdata posterior-median CSVs
