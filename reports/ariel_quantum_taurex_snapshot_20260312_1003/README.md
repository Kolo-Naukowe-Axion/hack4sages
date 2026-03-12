# Ariel Quantum TauREx Snapshot

This directory captures the best stage-2 checkpoint downloaded from the live TauREx training run on March 12, 2026, plus a local evaluation on the Poseidon split from `data/TauREx set`.

Snapshot summary:

- Checkpoint source: live `stage2_hybrid/best_model.pt`
- Best epoch at snapshot time: `5`
- Checkpoint quantum active: `true`
- Checkpoint quantum scale: `0.625`
- TauREx validation mRMSE: `1.4490022659301758`
- TauREx validation MAE: `1.0452070236206057`
- Poseidon holdout mRMSE: `3.2156150341033936`
- Poseidon holdout MAE: `2.701436996459961`
- Poseidon vs TauREx mRMSE gap: `1.7666127681732178`
- Poseidon vs TauREx mRMSE ratio: `2.21919255042652`

Notes:

- The remote TauREx training continued after this snapshot was taken.
- This checkpoint is a point-in-time artifact, not necessarily the final best model from the full run.
- `stage2_best_model_epoch005.pt` is the downloaded best-so-far checkpoint at the time of capture.
