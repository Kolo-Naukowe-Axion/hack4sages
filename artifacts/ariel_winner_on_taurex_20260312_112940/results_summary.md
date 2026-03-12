TauREx winner-on-TauREx run summary for `ariel_winner_on_taurex_20260312_112940`.

Final run outcome:
- Early stopped at epoch `79`
- Best validation mRMSE checkpoint: epoch `30`
- Best validation NLL checkpoint: epoch `49`

Main metrics:
- TauREx validation mRMSE: `1.4821914434432983`
- Poseidon holdout mRMSE: `3.453120708465576`
- Absolute mRMSE gap: `1.9709292650222778`
- Poseidon / TauREx ratio: `2.329740010132285`
- Relative increase: `132.9740010132285%`

Checkpoint comparison on Poseidon:
- Epoch `30` mRMSE-best checkpoint: `3.4730594158172607`
- Epoch `49` NLL-best checkpoint: `3.565753221511841`

Interpretation:
- The saved mRMSE-best checkpoint generalized better to Poseidon than the NLL-best checkpoint.
- The official run holdout score (`3.453120708465576`) is slightly better than the local rerun of the epoch `30` checkpoint because evaluation samples observation noise stochastically.
