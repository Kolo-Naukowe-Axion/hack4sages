This folder stores the finished TauREx run artifacts from `ariel_winner_on_taurex_20260312_112940`.

The raw checkpoint files are saved locally but are not committed directly because Git LFS was unavailable and GitHub rejects files over 100 MB.

Committed checkpoint bundles:
- `best_model_by_mrmse.pt.part-00`
- `best_model_by_mrmse.pt.part-01`
- `best_model_by_nll.pt.part-00`
- `best_model_by_nll.pt.part-01`

To restore the original checkpoints:

```bash
cat best_model_by_mrmse.pt.part-00 best_model_by_mrmse.pt.part-01 > best_model_by_mrmse.pt
cat best_model_by_nll.pt.part-00 best_model_by_nll.pt.part-01 > best_model_by_nll.pt
shasum -a 256 best_model_by_mrmse.pt best_model_by_nll.pt
```

Expected SHA-256 hashes are listed in `checkpoint_sha256.txt`.

Evaluation artifacts:
- `holdout_metrics.json`: remote final Poseidon holdout evaluation produced by the training run after early stopping
- `best_model_by_mrmse_poseidon_eval.json`: local rerun on the Poseidon holdout using the epoch 30 mRMSE-best checkpoint
- `best_model_by_nll_poseidon_eval.json`: local rerun on the Poseidon holdout using the epoch 49 NLL-best checkpoint
- `checkpoint_summary.json`: checkpoint epochs plus summarized Poseidon results
