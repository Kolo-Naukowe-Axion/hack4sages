# Ariel Quantum Best Checkpoint

This directory contains the best confirmed hybrid checkpoint from the `2026-03-11` Ubuntu run.

## Summary

- Dataset rows:
  - train `33138`
  - validation `4142`
  - holdout `4143`
  - test `685`
- Best checkpoint:
  - run `ariel_quantum_stage2_restart_v4_20260311_185231`
  - epoch `6`
- Metrics:
  - training-phase best validation mRMSE `0.29081112146377563`
  - re-evaluated validation mRMSE `0.29361358284950256`
  - re-evaluated holdout mRMSE `0.2993761897087097`

## Files

- `best_model.pt`: full PyTorch checkpoint with model weights and checkpoint metadata
- `config.json`: training configuration used for the saved run
- `scalers.json`: auxiliary, target, and spectral standardizers
- `prepared_manifest.json`: cached-preparation metadata
- `split_manifest.json`: dataset split metadata
- `history.csv`: per-epoch training history
- `training_state.json`: final recorded trainer state before the run was stopped
- `validation_metrics.json`: best-checkpoint validation metrics re-evaluated after stopping
- `holdout_metrics.json`: best-checkpoint holdout metrics re-evaluated after stopping
- `validation_predictions.csv`: validation predictions from `best_model.pt`
- `holdout_predictions.csv`: holdout predictions from `best_model.pt`
- `testdata_predictions.csv`: test-set predictions from `best_model.pt`
- `run_summary.json`: compact summary of the best checkpoint

## Architecture

The architecture for this checkpoint is documented in:

- `/Users/iwosmura/projects/hack4sages/docs/ariel_quantum_architecture.md`

High-level form:

1. Per-sample-normalized spectra into a residual 1D encoder with mean and attention pooling.
2. Auxiliary features into a small MLP.
3. Fused latent plus direct encoder skip connections into a classical regression head.
4. A quantum residual branch with an `8`-qubit variational block and per-target quantum gating.
5. Final prediction as classical output plus gated quantum correction.
