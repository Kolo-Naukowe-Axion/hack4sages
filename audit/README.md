# ExoBiome audit harness — executable part of the research protocol

This package is the **runnable half** of `docs/PROTOKOL_BADAWCZY.md`. Claims in `docs/METHODOLOGICAL_AUDIT.md` are backed by checks here.

A check loads only committed artifacts, never modifies model or training code, and emits a
self-describing JSON record (git revision, env versions, input sha256, pass criterion, verdict) so that
a reviewer can re-derive any number with one command.

## Running

Run from the checkout that contains `data/`, `artifacts/` and `.venv-qml`:

```bash
./.venv-qml/bin/python audit/run_all.py
```

```bash
./.venv-qml/bin/python audit/run_all.py --fast
```

A single check, with its own flags:

```bash
./.venv-qml/bin/python audit/a03_input_convention.py --split holdout --seeds 42,1,2
```

Outputs land in `reports/audit/<UTC-date>/`: one JSON per check plus `summary.json` / `summary.md`.
Those records are the **only** files the harness creates. Redirect them with `--out` or
`EXOBIOME_AUDIT_OUT`; point the harness at a different checkout with `EXOBIOME_REPO=/path/to/checkout`
(that variable is the single place a path is resolved — `audit_lib.py:33`).

**The harness never commits and never modifies anything that was already in the repo.** git is only ever
read ; there is no `add`, `commit`,
`checkout`, `reset` or `push` anywhere in the package. Verify with:

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

Two checks use an **inverted** criterion, where `PASS` means *the finding was refuted*. Both say so in
their own `pass_criterion` field, and the report says so at the point of citation. However, it is worth
knowing before reading `summary.md`: `a29_smoke_baseline_recovery` and (for the consumption clause)
its `verdict.refuting_conditions`.

## Checks

`finding` refers to sections of `docs/METHODOLOGICAL_AUDIT.md`. Where a check's numbers are cited
without a dedicated finding section, the citation site is named instead.

| check | supports | question it answers | PASS criterion | cost |
|---|---|---|---|---|
| `a01_spectral_variation` | **K1** | Does each stored spectrum vary across wavelength bins, and above the noise it was generated for? | zero bit-constant rows ∧ median feature/σ > 1 ∧ median `std_bins/\|mean_bins\| > 1e-4`; checks **every** row | ~20 s |
| `a02_trivial_baseline` | constant-predictor floor (§0a) | For each published mRMSE, what is the constant-predictor mRMSE on the same rows? | skill `= 1 − reported/baseline > 0` for every number presented as a result | seconds |
| `a03_input_convention` | **K3** | What is each model's mRMSE with and without eval-time `N(0,σ)` injection? | `sign(ExoBiome − baseline)` invariant to the input convention | ~10 min CPU |
| `a04_quantum_scale_provenance` | **K4** | Which `quantum_scale` reproduces the published numbers, and what is gate-off? | reproducing scale == selection-time scale from `history.csv` | ~4 min CPU |
| `a05_training_completeness` | **K5**, **K3** | Did each reported run converge, on the metric it is compared on? | terminated by its own rule ∧ metric plateaued ∧ selection metric == reported metric ∧ full-split selection | seconds |
| `a06_param_accounting` | **K6** | What fraction of the "quantum branch" is quantum? Is on/off parameter-matched? | branch adds < 1 % extra classical parameters | seconds |
| `a07_gate_dynamics` | **K7** | Is `gate≈0.046` a converged equilibrium or zero-init plus a truncated schedule? | gate converged ∧ per-gas gates agree in sign | seconds |
| `a08_reference_posterior` | reference posterior | How does the model compare with the reference nested-sampling retrieval? | model mRMSE ≥ reference-posterior mRMSE | ~1 min |
| `a09_noise_realization` | **K3** (noise convention) | Is the bin-to-bin scatter of `instrument_spectrum` consistent with a full-σ realization? | R-statistic distribution centred ≥ 1 | ~1 min |
| `a10_split_integrity` | split integrity (§2, additional problems) | Does any package report as "holdout" rows used for selection? | validation ≠ holdout selectors ∧ zero row overlap | seconds |
| `a12_significance_power` | MDE and seed count (§7.3) | Is the resampling unit right for the claim? Can the set sizes resolve the effects? | ≥5 seeds ∧ effect > seed spread ∧ effect > numeric noise floor | ~30 s |
| `a13_provenance_index` | provenance index (§0a, **K10(b)**) | Can each published number be re-derived from a committed artifact? | every result `backed_full` ∧ its document tracked in git | seconds |
| `a14_importability` | dead imports | Do documented entrypoints resolve their imports and file paths? | all modules resolve, all hard-coded artifact paths exist | seconds |
| `a15_target_completeness` | **K9** | What does omitting temperature and planet radius from the target vector cost, per dataset? | model predicts the full benchmark parameter vector, or omitted params are both weakly coupled and input-determined | ~6 min |
| `a21_dead_features` | **K11** (H2/He background), dead inputs | Which model inputs carry no information, and which informative label columns are never read? | no zero-variance input dims, no scalar-broadcast channel, no unread varying label column | seconds |
| `a24_official_metrics` | **K10** | What do the models score on the ADC2022/ADC2023 distributional metrics? | point model within 2× of the finite-sample floor | ~5 min |
| `a26_baseline_ladder` | baseline ladder (§0a) | Constant vs aux-only vs spectrum-only vs both — which input carries the reported skill? | aux-only skill < 20 % of the aux+spectrum skill | ~8 min |
| `a27_pipeline_fidelity` | pipeline reconstruction (§1.1) | Does the audit's own reconstruction of the ExoBiome pipeline match the committed predictions? | max \|diff\| per row per gas < 1e-4 vs `mac_holdout_predictions.csv` | ~2 min |
| `a29_smoke_baseline_recovery` | **K1(b)** | Do the team's own two smoke artefacts agree per gas, what skill is behind them, and how wide are the predictions? | **inverted** — `PASS` = finding refuted; see the check's own `pass_criterion` | seconds |
| `d01_poseidon_diagnosis` | **K1**, **K1(c)** | Why are the POSEIDON spectra flat — zero opacity or saturated opacity? | stage-dependent; see below | stage 0: seconds |


## Diagnostics: `d01` and its three stages

| stage | what it does | needs |
|---|---|---|
| 0 | environment probe: is POSEIDON importable, is an opacity root configured | nothing — this is the stage `run_all.SUITE` invokes |
| 1 | atmosphere geometry + `saturation_test`, which refuted the pressure-ordering and `R_p_ref`/`P_ref` hypotheses and narrowed the cause to saturated extinction | POSEIDON, but **not** the 72 GB opacity database |
| 2 | full spectra | POSEIDON **and** the opacity database (zenodo 16107813, 72.1 GB) |

The suite calls `d01 --stage 0`, which is why it never errors there. Stage 1 — the stage K1(c) rests on
— needs POSEIDON, which is installed in neither `.venv-qml` nor `.venv-cnn`, so it runs from its own
interpreter. Pinned to the **generating** version (1.3.2), with the two system dependencies POSEIDON
needs but does not declare:

```bash
uv venv --python 3.11.9 /tmp/poseidon-venv
uv pip install --python /tmp/poseidon-venv/bin/python \
  'POSEIDON @ git+https://github.com/MartianColonist/POSEIDON.git@v1.3.2' \
  h5py pandas pyarrow open-mpi 'setuptools<81'
MPLCONFIGDIR=/tmp/mpl /tmp/poseidon-venv/bin/python -u audit/d01_poseidon_diagnosis.py --stage 1
```

`open-mpi` is required by `mpi4py`, which `POSEIDON.core` imports; `setuptools<81` because newer
versions dropped `pkg_resources`, on which `pysynphot` (imported by `POSEIDON.stellar`) depends. `-u` is
required because a POSEIDON segfault otherwise swallows the whole stdout.

A venv under `/tmp` does not survive a reboot, so **move it out of `/tmp`.

## Known limitations of this harness

- **Input hashes cover the first 64 MiB** of each file (`audit_lib.py:59-69`), flagged in the hash key
  as `…-first67108864B`. For `data/TauREx set/spectra.h5` (68.7 MB) the fingerprint does not identify
  the whole file, although the checks read every row.
- **`a27` covers one path**: the clean ExoBiome path on `holdout`. The noised path behind K3's
  `degradation_factor`, the crossgen path behind K9, and `a26`'s own feature code are not covered by it.
- **`a04`'s `--tol` (2e-5) is adopted, not derived.** The tolerance-independent formulation of its
  finding is in the report: no single scale is nearest to the committed number on both splits.
- **`a09` transfers a threshold calibrated on 218 bins with a scalar σ to data with 52 bins and a
  per-bin σ.** The record notes this via `adc_sigma_is_per_bin`; the report states it as a limitation.

## Second reproduction paths (independent of `audit_lib`)

Three findings can be re-derived from the raw artefacts alone — no `import audit_lib`, so these
verify the FINDING. The expected values live in each finding’s verification table in
`docs/METHODOLOGICAL_AUDIT.md`.

### K5 — training completeness (`history.csv`, `config.json`, `training_state.json`)

```bash
cd ../hack4sages && ./.venv-qml/bin/python -c "
import json, pandas as pd
A='artifacts/ariel_quantum_best_v4_epoch6/'
h=pd.read_csv(A+'history.csv'); c=json.load(open(A+'config.json')); s=json.load(open(A+'training_state.json'))
print(h[['epoch','val_rmse_mean','quantum_scale','backbone_frozen']].to_string(index=False))
last=int(h.epoch.max()); best=int(s['best_epoch']); v=h.val_rmse_mean.values
print(f\"T1 stop reczny? ostatnia={last}<max={c['max_epochs']}; od najlepszej={last-best}<patience={c['early_stop_patience']}\")
print(f\"T2 best_epoch({best}) == freeze_epochs({c['quantum_backbone_freeze_epochs']})\")
print(f'T3 po odmrozeniu: {v[best-1]:.5f} -> {v[best]:.5f} ({100*(v[best]/v[best-1]-1):+.1f}%)')
print(f'T4 max quantum_scale = {h.quantum_scale.max():.4f}')"
```

### K6 — parameter accounting (`best_model.pt`, `config.json`)

```bash
cd ../hack4sages && ./.venv-qml/bin/python -c "
import torch, json
A='artifacts/ariel_quantum_best_v4_epoch6/'
sd=torch.load(A+'best_model.pt',map_location='cpu',weights_only=False)['model_state_dict']
c=json.load(open(A+'config.json')); g={}
for k,v in sd.items(): g[k.split('.')[0]]=g.get(k.split('.')[0],0)+v.numel()
tot=sum(g.values()); q=sum(g.get(k,0) for k in ('projector','quantum_block','quantum_head','quantum_gate'))
circ=sd['quantum_block.weights'].numel(); W=sd['quantum_head.net.0.weight']
[print(f'   {k:18}{v:>8}') for k,v in sorted(g.items(),key=lambda x:-x[1])]
print(f\"T1 wzor 3*{c['qnn_qubits']}*{c['qnn_depth']//2}={3*c['qnn_qubits']*(c['qnn_depth']//2)} vs ksztalt {tuple(sd['quantum_block.weights'].shape)}\")
print(f'T2 udzial obwodu {circ}/{tot} = {circ/tot:.6%}')
print(f'T3 sciezka kwantowa {q}, kwantowe {circ}, KLASYCZNE {q-circ}')
print(f'T4 classical-only {tot-q}, branch doklada +{(q-circ)/(tot-q):.2%}')
print(f\"T5 wejscie quantum_head {W.shape[1]} wym., z obwodu {c['qnn_qubits']} = {c['qnn_qubits']/W.shape[1]:.2%}\")"
```

### K7 — gate dynamics (`best_model.pt`, `config.json`, `model.py` source)

```bash
cd ../hack4sages && grep -n "quantum_gate = nn.Parameter" -A 2 models/ariel_exobiome/model.py && ./.venv-qml/bin/python -c "
import torch, numpy as np, json
A='artifacts/ariel_quantum_best_v4_epoch6/'
sd=torch.load(A+'best_model.pt',map_location='cpu',weights_only=False)['model_state_dict']
raw=sd['quantum_gate'].numpy(); g=np.tanh(raw); c=json.load(open(A+'config.json'))
for t,r,v in zip(['H2O','CO2','CO','CH4','NH3'],raw,g): print(f'{t:5}{r:+11.6f}{v:+11.6f}')
print(f'T1 mean|tanh| = {np.abs(g).mean():.6f}')
print(f'T2 zgodne znaki: {len(set(np.sign(g)))==1}')
print(f'T3 mnoznik |tanh|*0.5 w [{np.abs(g).min()*0.5:.4f}, {np.abs(g).max()*0.5:.4f}]')
print(f\"T4 ~{6*33138//c['batch_size']} krokow przy lr={c['quantum_lr']}\")"
```
