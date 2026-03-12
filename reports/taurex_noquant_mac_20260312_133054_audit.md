# TauREx Noquant Mac Retrain Audit

Run directory: `/Users/iwosmura/projects/hack4sages/outputs/taurex_noquant_mac_20260312_133054/train`

## Cold-look findings

- The previous noquant checkpoint was effectively a mean predictor: validation mRMSE was almost identical to the train-mean baseline, and prediction standard deviations were near zero.
- The H200 run used a very large batch size (`8192`) for a `37281`-row TauREx train split, which reduced each epoch to only a handful of optimizer steps and made the early plateau misleadingly fast.
- The noquant rewrite had regressed away from the proven classical TauREx backbone in this repo.
- Final evaluation also had a bug when warmup/ramp was enabled: the best checkpoint could be selected at one refinement scale and then re-evaluated at `1.0`, which distorted the reported summary.

## Fixes applied

- Replaced the transformer-heavy noquant architecture with the simpler conv-attention backbone that already demonstrated strong TauREx learning in the repo.
- Replaced the quantum block with a classical residual MLP refinement path.
- Restored stable training defaults:
  - `batch_size=256`
  - `eval_batch_size=512`
  - `loss_name=mse`
  - `dropout=0.05`
  - `max_epochs=60`
  - `early_stop_patience=12`
  - `scheduler_patience=4`
  - `quantum_warmup_epochs=3`
  - `quantum_ramp_epochs=6`
- Fixed checkpoint reporting so final validation uses the refinement scale associated with the best epoch.
- Trained on TauREx only with POSEIDON excluded.

## Result

- Previous noquant best validation mRMSE: `2.8852927684783936`
- New noquant best validation mRMSE: `1.4230316877365112`
- Absolute improvement: `1.4622610807418823`
- Relative improvement: `50.68%`
- Best epoch: `59`

## Interpretation

The current noquant model now behaves like a real learner again and lands in the same general performance band as the stronger earlier TauREx runs in this repository, instead of collapsing to the train-set mean.
