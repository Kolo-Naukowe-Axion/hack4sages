# TauREx No-Quant H200 Poseidon Evaluation

Checkpoint evaluated:
- `/Users/iwosmura/projects/hack4sages/outputs/taurex_noquant_h200_20260312_130354/train/best_model.pt`

Dataset comparison:
- TauREx validation rows: `4142`
- Poseidon holdout rows: `685`
- TauREx validation mRMSE: `2.8852810859680176`
- Poseidon holdout mRMSE: `2.894606828689575`
- Absolute mRMSE gap: `0.009325742721557617`
- Poseidon / TauREx ratio: `1.0032321782327938`
- Relative increase: `0.32321782327937587%`

Per-target RMSE:
- TauREx validation:
  - `log_H2O`: `2.8646974563598633`
  - `log_CO2`: `2.8946447372436523`
  - `log_CO`: `2.914268970489502`
  - `log_CH4`: `2.8537917137145996`
  - `log_NH3`: `2.8990020751953125`
- Poseidon holdout:
  - `log_H2O`: `2.8722946643829346`
  - `log_CO2`: `2.967393636703491`
  - `log_CO`: `2.8497121334075928`
  - `log_CH4`: `2.941803455352783`
  - `log_NH3`: `2.841829299926758`

Interpretation:
- The Poseidon gap is far smaller than the cross-generator gaps in the existing repo reports.
- That is more consistent with a high-bias / underfit checkpoint than with a strongly specialized TauREx model.
