# ExoBiome audit harness

Every claim in `docs/METHODOLOGICAL_AUDIT.md` is backed by one runnable check here. A check loads
only committed artifacts, never modifies model or training code, and emits a self-describing JSON
record (git revision, env versions, input sha256, verdict) so a reviewer can re-derive it with one
command.

## Running

Run from the **main checkout** — the one that has `data/`, `artifacts/` and `.venv-qml`:

```bash
cd /Users/mariaplatek/projects/AXION/hack4sages
./.venv-qml/bin/python .claude/worktrees/<worktree>/audit/run_all.py
```

```bash
./.venv-qml/bin/python .claude/worktrees/<worktree>/audit/run_all.py --fast
```

```bash
./.venv-qml/bin/python .claude/worktrees/<worktree>/audit/a03_input_convention.py --split holdout --seeds 42,1,2
```

Outputs land in `reports/audit/<UTC-date>/`: one JSON per check plus `summary.json` / `summary.md`.
Those records are the **only** files the harness creates — redirect them with `--out` or
`EXOBIOME_AUDIT_OUT` if you want them outside the checkout; point at a different checkout with
`EXOBIOME_REPO=/path/to/checkout`.

**The harness never commits and never modifies anything that was already in the repo.** git is only ever read (`rev-parse`,
`status --porcelain`, `ls-files --error-unmatch`); there is no `add`, `commit`, `checkout`, `reset`
or `push` anywhere in the package. Verify with:

```bash
grep -rnE '"(add|commit|push|checkout|reset|rm|mv|stash)"' audit/*.py
```

## Status vocabulary

| status | meaning |
|---|---|
| `PASS` | the repo satisfies the check's criterion |
| `FAIL` | the defect is confirmed by data, and the payload contains the evidence |
| `WARN` | the criterion cannot be satisfied as posed, but the defect is not proven from artifacts alone |
| `INFO` | the check produced measurements but has no pass/fail semantics in this configuration |

## Implemented checks

| check | finding | question it answers | PASS criterion | cost |
|---|---|---|---|---|
| `a01_spectral_variation` | **K1** | Does each stored spectrum vary across wavelength bins, and above the noise it was generated for? | zero bit-constant rows ∧ median feature/σ > 1 ∧ median `std_bins/|mean_bins| > 1e-4`; checks **every** row | ~20 s |
| `a02_trivial_baseline` | **K2** | For each published mRMSE, what is the constant-predictor mRMSE on the same rows? | skill `= 1 − reported/baseline > 0` for every number presented as a result | seconds |
| `a03_input_convention` | **K3** | What is each model's mRMSE with and without eval-time `N(0,σ)` injection? | `sign(ExoBiome − baseline)` invariant to the input convention | ~10 min CPU |
| `a04_quantum_scale_provenance` | **K4** | Which `quantum_scale` reproduces the published numbers, and what is gate-off? | reproducing scale == selection-time scale from `history.csv` | ~4 min CPU |
| `a05_training_completeness` | **K5**, **K3** | Did each reported run converge, on the metric it is compared on? | terminated by its own rule ∧ metric plateaued ∧ selection metric == reported metric ∧ full-split selection | seconds |
| `a06_param_accounting` | **K6** | What fraction of the "quantum branch" is quantum? Is on/off parameter-matched? | branch adds < 1 % extra classical parameters | seconds |
| `a07_gate_dynamics` | **K7** | Is `gate≈0.046` a converged equilibrium or zero-init plus a truncated schedule? | gate converged ∧ per-gas gates agree in sign | seconds |
| `a08_reference_posterior` | **K8** | How does the model compare with the reference nested-sampling retrieval? | model mRMSE ≥ reference-posterior mRMSE | ~1 min |
| `a09_noise_realization` | **W14**, **K3** | Is the bin-to-bin scatter of `instrument_spectrum` consistent with a full-σ realization? | R-statistic distribution centred ≥ 1 | ~1 min |
| `a10_split_integrity` | **P1**, **P2**, **U4** | Does any package report as "holdout" rows used for selection? | validation ≠ holdout selectors ∧ zero row overlap | seconds |
| `a11_pairing_audit` | **P3** | Which config fields differ between two runs presented as an ablation pair? | differences confined to the declared factor | seconds |
| `a12_significance_power` | **P7**, **P4** | Is the resampling unit right for the claim? Can the set sizes resolve the effects? | ≥5 seeds ∧ effect > seed spread ∧ effect > numeric noise floor | ~30 s |
| `a13_provenance_index` | **P6**, **U8** | Can each published number be re-derived from a committed artifact? | every result `backed_full` ∧ its document tracked in git | seconds |
| `a14_importability` | **U6** | Do documented entrypoints resolve their imports and file paths? | all modules resolve, all hard-coded artifact paths exist | seconds |
| `a21_dead_features` | **U2**, **U3**, **K9(d)** | Which model inputs carry no information, and which informative label columns are never read? | no zero-variance input dims, no scalar-broadcast channel, no unread varying label column | seconds |
| `a26_baseline_ladder` | **K2** (rozszerzenie), task A0.4 | Constant vs aux-only vs spectrum-only vs both — which input carries the reported skill? | aux-only skill < 20 % of the aux+spectrum skill | ~8 min |
| `a24_official_metrics` | **K10** | What do the models score on the ADC2022/ADC2023 distributional metrics? | point model within 2x of the finite-sample floor | ~5 min |
| `a15_target_completeness` | **K9** | What does omitting temperature and planet radius from the target vector cost, per dataset? | model predicts the full benchmark parameter vector, or omitted params are both weakly coupled and input-determined | ~6 min |
| `a29_smoke_baseline_recovery` | **K1(b)** | Do the team's own two smoke artefacts agree per gas, what skill is behind them, and how wide are the predictions? | the artefacts agree per gas ∧ the smoke baseline has positive skill on both generators | seconds |

## Diagnostics (root-cause, not verdict)

`a*` checks establish *that* something is wrong. `d*` scripts find *why*.

| script | for | stages |
|---|---|---|
| `d01_poseidon_diagnosis` | **K1**, **K1(c)** | 0: environment probe (needs nothing) · 1: atmosphere geometry + `saturation_test`, which refuted the pressure-ordering and `R_p_ref`/`P_ref` hypotheses and narrowed the cause to saturated extinction — **needs POSEIDON but NOT the 72 GB opacity data** · 2: full spectra, needs the opacity database |

`d01` is **not** in `run_all.SUITE`, on purpose: POSEIDON is installed in neither `.venv-qml` nor
`.venv-cnn`, so the suite would record it as an error on every run. It needs its own interpreter, and
`-u`, because a POSEIDON segfault otherwise swallows the whole stdout:

```
python -m venv /tmp/poseidon-venv
/tmp/poseidon-venv/bin/pip install 'git+https://github.com/MartianColonist/POSEIDON.git' h5py pandas pyarrow
MPLCONFIGDIR=/tmp/mpl /tmp/poseidon-venv/bin/python -u audit/d01_poseidon_diagnosis.py --stage 1
```

A venv under `/tmp` does not survive a reboot, so the K1(c) record is reproducible only as long as
nobody clears it. Move it out of `/tmp` before treating that record as durable.

## Still to write (specs, so they are mechanical)

These prove findings that are currently established by code reading only. Each is ≤ 100 lines.

| check | finding | what it must do | PASS criterion |
|---|---|---|---|
| `a16_numeric_noise_floor.py` | **P8** | Evaluate one checkpoint under every available (device, dtype, `quantum_scale`) combination — cpu/cuda × fp32/bf16 × scales — and tabulate the spread of the *same* quantity. Compare that spread against every effect size the reports treat as a finding. | max spread of one quantity < 0.2 × the smallest claimed effect |
| `a17_runtime_fairness.py` | **P5** | Parse `fair_small_experiment_cpu/**/run_summary.json` + `settings.yaml` + `batch_metrics.jsonl`: actual wall clock, optimizer steps, patience, per-budget hyperparameter counts per arm; detect budget cells present on disk but absent from `experiment_results.csv`. | wall clock within ±10 % between arms ∧ both arms can early-stop ∧ every on-disk cell appears in the CSV |
| `a18_generator_config_diff.py` | **P9** | Extract, side by side, the forward-model configuration of each generator: opacity source and version, native resolution, reference pressure, CIA/Rayleigh contributions, stellar model, He convention, dependency pins. | only the RT implementation differs |
| `a19_pairedness.py` | **P10**, **P11** | Check whether the two generators were rendered from the **same** latent rows (exact tuple match) and report the power the test-set size affords. | ≥95 % of test latents shared between generators ∧ MDE < the smallest claimed gap |
| `a20_loss_metric_alignment.py` | **U1** | From `scalers.json`, compute the implicit per-target weight `1/σ_g²` induced by MSE on z-scored targets, normalise, and compare with the equal weights of the reported metric; do the same for every package in a comparison. | max/min implicit weight ratio < 1.25 ∧ identical objective across compared packages |
| ~~`a21_dead_features.py`~~ (zrobione) | **U2**, **U3** | After the package's own standardisation, count input dimensions with zero variance, and channels that are a single scalar broadcast; list label columns present in the data but never read. | zero dead dimensions ∧ no informative column discarded by one arm only |
| `a25_physical_amplitude.py` | **K11** | Compare each generator's measured feature amplitude against the physical expectation `2·R_p·H/R_star²` with `H = kT/(μg)` from the labels. This is the check that would have caught the missing CIA term, which `a01` passes by design. | measured amplitude within a stated factor of the physical expectation |
| `a22_qat_suite_replication.py` | **P4** | Re-run the qat notebooks' arithmetic from their committed outputs and assert each verdict sentence against the numbers: symmetric-budget check for Test 05, variance match for Test 04's surrogates, estimand match for Test 07, presence of tests 05/06 in the Test 08 tally. | every verdict sentence entailed by the numbers it cites |
| `a23_holdout_reuse_counter.py` | **P4** | Count distinct evaluations of each test split across notebooks/scripts/reports, and flag any hyperparameter whose value was chosen by an argmin on a test split. | test split evaluated once per reported number ∧ zero test-set-selected hyperparameters |

## Repair-verification scripts (separate from the audit)

The audit proves the defects; these prove the fixes. Suggested names, in dependency order.

| script | fixes | must produce |
|---|---|---|
| `fix01_regenerate_crossgen.py` | K1 | a POSEIDON shard whose spectra pass `a01`; plus a `validate_dataset.py` patch adding the spectral-variation assertion and a mandatory ridge smoke baseline, and a content hash in the prepared-cache manifest key |
| `fix02_paired_generators.py` | P10 | both generators rendered from the **same** latent rows, so `a18` passes |
| `fix03_noise_matched_training.py` | K3 | ExoBiome retrained with the same noise augmentation as the flow, and both evaluated in both conventions — the 2×2 table of `a03` filled from *matched* training, not just matched evaluation |
| `fix04_report_with_baseline.py` | K2 | a table generator that refuses to emit a row without a trivial baseline and a skill score |
| `fix05_scale_provenance_patch.py` | K4 | patch threading `best_quantum_scale` into the final eval in all four packages, plus a re-report of every quantum number |
| `fix06_param_matched_quantum.py` | K6, P3, P4 | one comparison where the residual head is identical and only the 8-dim slot changes (circuit ↔ classical map ↔ row-permuted circuit output), symmetric budget, ≥5 seeds, single pre-declared endpoint |
| `fix07_seed_campaign.py` | P7 | ≥5 seeds per arm, per-seed deltas, CI over seeds, pre-declared multiplicity correction |
| `fix08_posterior_metric.py` | K8 | evaluation against `Tracedata.hdf5`: coverage/SBC/TARP versus the reference posterior, plus per-gas information ceilings, plus the multimodality split that `a08` shows is already obtainable |
| `fix10_multitask_targets.py` | K9 | ExoBiome z 7 wyjściami (5 gazów + T + R_p) vs 5 wyjściami, wspólny trunk, ten sam budżet, >=5 seedów: czy nadzór wielozadaniowy poprawia gazy? plus raportowanie T obok abundancji |
| `fix09_freeze_and_track.py` | U8, P6 | plan-of-record, `VERIFICATION.md` and `reports/audit/` committed; every figure regenerated from `backed_full` numbers only; unbacked numbers deleted rather than annotated |
