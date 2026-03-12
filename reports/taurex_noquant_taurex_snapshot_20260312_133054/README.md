# TauREx Noquant TauREx Snapshot

This directory captures the best nonquant checkpoint from the local Mac TauREx-only training run on March 12, 2026, plus a local evaluation on the Poseidon split from `data/TauREx set`.

Snapshot summary:

- Checkpoint source: local `train/best_model.pt`
- Best epoch at snapshot time: `59`
- Checkpoint refinement active: `true`
- Checkpoint refinement scale: `1.0`
- TauREx validation mRMSE: `1.4230316877365112`
- TauREx validation MAE: `0.9856079816818237`
- Poseidon holdout mRMSE: `3.2795588970184326`
- Poseidon holdout MAE: `2.7480854988098145`
- Poseidon vs TauREx mRMSE gap: `1.8565272092819214`
- Poseidon vs TauREx mRMSE ratio: `2.3046281578135006`
- Poseidon vs TauREx relative increase: `130.46281578135006%`

Notes:

- This snapshot comes from the repaired nonquant architecture and training loop after removing the earlier mean-predictor collapse.
- This checkpoint was trained on TauREx only; POSEIDON was excluded from training and used here only for out-of-generator evaluation.
- `best_model_epoch059.pt` is the copied best checkpoint for this snapshot.
