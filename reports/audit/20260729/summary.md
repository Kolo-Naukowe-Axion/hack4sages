# ExoBiome audit — inventory summary

- generated: `2026-07-29T12:51:48.669695+00:00`
- revision: `e8a426173b850db945ad2c6dfb05627caef8c074-dirty`
- counts: **FAIL** 17, **WARN** 1, **INFO** 1, **PASS** 2
- uruchomiono 21 z 21 checkow (pelna suita)

| status | check | finding | pass criterion | record |
|---|---|---|---|---|
| FAIL | `a01_spectral_variation` | K1 — all 685 POSEIDON spectra are wavelength-constant; the cross-generator axis has no signal | zero bit-constant rows AND median feature/sigma > 1 AND median std_bins/|mean_bins| > 1e-4 | `reports/audit/20260729/a01_spectral_variation.json` |
| FAIL | `a02_trivial_baseline` | K2 — every cross-generator model is worse than a constant predictor; no table reports a baseline | skill = 1 - reported/(train-mean baseline) > 0 for every number presented as a model result | `reports/audit/20260729/a02_trivial_baseline.json` |
| FAIL | `a03_input_convention` | K3 — the ExoBiome-vs-SOTA comparison mixes two input conventions; the ranking reverses when they are equalised | sign(ExoBiome - baseline) is invariant to the input convention | `reports/audit/20260729/a03_input_convention.json` |
| FAIL | `a04_quantum_scale_provenance` | K4 — reported metrics use quantum_scale=1.0 (never validated); selection ran at 0.5 | some swept scale reproduces a published number AND it equals the selection-time scale from history.csv | `reports/audit/20260729/a04_quantum_scale_provenance.json` |
| FAIL | `a05_training_completeness` | K5 / K3 — the flagship checkpoint comes from an 8/30-epoch aborted run; the baseline stopped at 79/300 on a different metric | terminated by its own rule AND reported metric plateaued AND selection metric == reported metric AND full-split selection | `reports/audit/20260729/a05_training_completeness.json` |
| FAIL | `a06_param_accounting` | K6 — 24 of 69,434 'quantum branch' parameters are quantum; the branch is a second classical head | quantum branch adds <1% extra classical parameters over the classical-only model | `reports/audit/20260729/a06_param_accounting.json` |
| FAIL | `a07_gate_dynamics` | K7 — 'the model learned to silence the quantum branch' is unsupported: the gate is zero-initialised and was still growing | gate is converged (run terminated by its own rule) AND per-gas gates agree in sign | `reports/audit/20260729/a07_gate_dynamics.json` |
| FAIL | `a08_reference_posterior` | K8 — the reported 0.30 dex beats the reference nested-sampling retrieval ~3.8x; the metric measures simulator inversion, not retrieval | model mRMSE >= reference-run mRMSE, both on the same 663 planets that carry a reference retrieval | `reports/audit/20260729/a08_reference_posterior.json` |
| FAIL | `a09_noise_realization` | W14 / K3 — the stored ADC spectra carry high-frequency scatter well below the quoted sigma | R statistic distribution centred at >= 1 (consistent with noise present at full sigma) | `reports/audit/20260729/a09_noise_realization.json` |
| FAIL | `a10_split_integrity` | P1 / P2 / U4 — taurex_fmpe holdout == validation; noquant copies val->holdout; testdata == holdout; no in-domain TauREx test split | validation and holdout selectors differ AND row overlap is zero in every package | `reports/audit/20260729/a10_split_integrity.json` |
| FAIL | `a11_pairing_audit` | P3 — the quantum-vs-noquant comparison is unmatched on ~10 axes, including two different losses under one name | differences confined to the declared factor under study | `reports/audit/20260729/a11_pairing_audit.json` |
| FAIL | `a12_significance_power` | P7 / P4 — all 'significance' is row bootstrap on one checkpoint; no seed variance, no multiplicity control | effect claims backed by >=5 seeds (derived from repo files) and larger than the across-seed spread | `reports/audit/20260729/a12_significance_power.json` |
| FAIL | `a13_provenance_index` | P6 / U8 — several headline numbers have no backing artifact; the verification doc and plan-of-record are not in git | every result number is recomputable from committed predictions, and its document is tracked | `reports/audit/20260729/a13_provenance_index.json` |
| FAIL | `a14_importability` | U6 — flagship CLI entrypoints and the Garnet port import a non-existent module; the evidence script loads a missing checkpoint | all referenced modules resolve and all hard-coded artifact paths exist | `reports/audit/20260729/a14_importability.json` |
| FAIL | `a15_target_completeness` | K9 — the model predicts 5 of its benchmark's 7 parameters; T is leaked by aux on ADC and wholly absent on cross-generator | model predicts the full benchmark parameter vector, or omitted params are both weakly coupled and input-determined | `reports/audit/20260729/a15_target_completeness.json` |
| FAIL | `a21_dead_features` | U2 / U3 / K9(d) — 5 of 8 crossgen aux features are constants; temperature_k varies 500-1800 K and is read by nobody | no zero-variance input dims, no scalar-broadcast channel, no unread varying label column | `reports/audit/20260729/a21_dead_features.json` |
| FAIL | `a24_official_metrics` | K10 — scored on the metrics the challenge actually used, a point estimate is at or near the worst attainable value | point model has positive KS skill against the PRIOR arm (beats a model that ignores the spectrum) | `reports/audit/20260729/a24_official_metrics.json` |
| WARN | `a26_baseline_ladder` | K2 (extension) — how much of the reported skill survives when the spectrum is removed | upper end of the CI on the aux-only skill share < 20 % of the aux+spectrum skill, on every dataset, and every dataset has a positive aux+spectrum skill to divide | `reports/audit/20260729/a26_baseline_ladder.json` |
| INFO | `d01_poseidon_diagnosis` | K1 — root-cause diagnosis for the wavelength-constant POSEIDON spectra | the repo's exact call reproduces a spectrum with the same relative variation as tau (~1e-2) | `reports/audit/20260729/d01_poseidon_diagnosis.json` |
| PASS | `a27_pipeline_fidelity` | A0.2 — row-by-row fidelity of the reconstructed ExoBiome input pipeline (underpins K3, K4, K9) | max |diff| per row per gas < 1e-4 vs mac_holdout_predictions.csv at quantum_scale=1.0 | `reports/audit/20260729/a27_pipeline_fidelity.json` |
| PASS | `a29_smoke_baseline_recovery` | K1(b) — the team's own ridge smoke baseline ran, shipped in the bundle, and records no skill on POSEIDON | the finding is REFUTED (PASS) if an artefact is missing, or the CSV does not reproduce the JSON within 1e-9, or the two skills share a sign, or any file outside the bundle / audit/ / docs/ reads the artefacts by path; it STANDS (FAIL) otherwise | `reports/audit/20260729/a29_smoke_baseline_recovery.json` |

Re-run any single check with:

```bash
./.venv-qml/bin/python audit/<check>.py
```

