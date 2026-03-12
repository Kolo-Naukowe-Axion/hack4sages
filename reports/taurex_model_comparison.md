# TauREx Model Comparison

This comparison is based only on the TauREx-related report artifacts currently in `reports/`.

For this document:

- `TauREx mRMSE` means the reported TauREx validation `mRMSE`
- `Poseidon mRMSE` means the reported Poseidon holdout `mRMSE`
- `Gap` means `Poseidon mRMSE - TauREx mRMSE`
- Lower is better for all three columns, but a small gap alone is not enough if the TauREx score is weak overall

## Main comparison: quantum, noquant, and Ariel winner

| Model | Source report | TauREx mRMSE | Poseidon mRMSE | Gap |
| --- | --- | ---: | ---: | ---: |
| TauREx noquant snapshot | [`taurex_noquant_taurex_snapshot_20260312_133054/README.md`](./taurex_noquant_taurex_snapshot_20260312_133054/README.md) | 1.423032 | 3.279559 | 1.856527 |
| Ariel quantum snapshot | [`ariel_quantum_taurex_snapshot_20260312_1003/README.md`](./ariel_quantum_taurex_snapshot_20260312_1003/README.md) | 1.449002 | 3.215615 | 1.766613 |
| Ariel winner on TauREx | [`ariel_winner_on_taurex_20260312_112940_results_summary.md`](./ariel_winner_on_taurex_20260312_112940_results_summary.md) | 1.482191 | 3.453121 | 1.970929 |

## Ranking summary

### By TauREx validation mRMSE

1. TauREx noquant snapshot: `1.423032`
2. Ariel quantum snapshot: `1.449002`
3. Ariel winner on TauREx: `1.482191`

### By Poseidon holdout mRMSE

1. Ariel quantum snapshot: `3.215615`
2. TauREx noquant snapshot: `3.279559`
3. Ariel winner on TauREx: `3.453121`

### By Poseidon-vs-TauREx gap

1. Ariel quantum snapshot: `1.766613`
2. TauREx noquant snapshot: `1.856527`
3. Ariel winner on TauREx: `1.970929`

## Direct takeaways

- The repaired **noquant** model has the best pure TauREx validation `mRMSE`, beating the quantum snapshot by `0.025971` and the Ariel winner by `0.059160`.
- The **quantum** snapshot has the best Poseidon transfer and the smallest cross-generator gap among the three main contenders. Compared with the repaired noquant snapshot, it improves Poseidon `mRMSE` by `0.063944` and shrinks the gap by `0.089914`.
- The **Ariel winner** trails both the repaired noquant and the quantum snapshot on TauREx `mRMSE`, Poseidon `mRMSE`, and the Poseidon-vs-TauREx gap.

## Reference baseline from the reports folder

There is one additional TauREx noquant report worth keeping in view:

| Model | Source report | TauREx mRMSE | Poseidon mRMSE | Gap |
| --- | --- | ---: | ---: | ---: |
| TauREx noquant H200 baseline | [`taurex_noquant_h200_20260312_poseidon_eval_summary.md`](./taurex_noquant_h200_20260312_poseidon_eval_summary.md) | 2.885281 | 2.894607 | 0.009326 |

That H200 baseline has by far the smallest gap, but its own report explicitly warns that this likely reflects a **high-bias / underfit checkpoint**, not better TauREx-specialized learning. So it should be treated as a reference point, not as the best overall TauREx model.

## Bottom line

- Best TauREx fit: **TauREx noquant snapshot**
- Best Poseidon transfer: **Ariel quantum snapshot**
- Best gap among the three competitive runs: **Ariel quantum snapshot**
- Weakest of the three competitive runs: **Ariel winner on TauREx**
- Smallest overall gap in the folder: **TauREx noquant H200 baseline**, but the report itself says that result is likely underfitting-driven
