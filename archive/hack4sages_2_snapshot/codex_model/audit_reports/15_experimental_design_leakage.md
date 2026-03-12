# Audit Report 15: Information Leakage Through Experimental Design

**Auditor**: Scientific Rigor Auditor (automated)
**Date**: 2026-03-12
**Scope**: Hybrid quantum-classical model (`codex_model`) -- all hyperparameter, architecture, and design decisions evaluated for test-set contamination
**Primary file**: `/Users/michalszczesny/projects/hack4sages/codex_model/crossgen_hybrid_training.py`

---

## Executive Summary

This audit examines whether design decisions in the hybrid quantum-classical biosignature detection model were informed by test-set performance, constituting information leakage through experimental design. The investigation covered the full lineage of model iterations across four project directories (`quantum_model`, `quantum_model_crossgen`, `quantum_model_crossgen_fe`, `codex_model`) and three completed training runs.

The codebase shows **no evidence of systematic hyperparameter tuning on test data** and several structural safeguards against leakage. However, the existence of multiple model iterations with observable test performance raises concerns about implicit selection bias, and the architecture evolution from 16 to 12 qubits coincides with a change that could have been test-informed.

**Overall Verdict: PASS (conditional)**

---

## Finding 1: Model Selection Across Multiple Iterations Uses Test RMSE as Reported Metric

**Severity: MEDIUM**

### Evidence

Four distinct model directories exist with chronological progression (based on filesystem timestamps):

| Directory | Timestamp | Qubits | Features | Test RMSE Mean |
|-----------|-----------|--------|----------|----------------|
| `quantum_model/outputs/model_quant_sketch_crossgen` | Mar 11 ~11:53 | 4 | 5 aux | 2.769 (Poseidon) |
| `quantum_model/outputs/model_crossgen` | Mar 11 ~18:34 | 16 | 5 aux | (incomplete, no summary) |
| `quantum_model_crossgen/outputs/model_crossgen_rebinned` | Mar 11 ~23:59 | 12 | 5 aux | 3.463 |
| `quantum_model_crossgen_fe/outputs/model_crossgen_rebinned_fe` | Mar 11 ~23:33 | 12 | 7 aux | 3.538 |
| `codex_model/outputs/model_crossgen_rebinned` | Mar 12 ~01:16 | 12 | 5 aux | 3.463 |

The final `codex_model` run is an exact copy of `quantum_model_crossgen` (identical `run_summary.json` values: best_epoch=29, best_inner_val_loss=0.2999, test_rmse_mean=3.4631). The `_fe` variant (with 2 extra features: `rp_rs_ratio`, `scale_height_proxy`) had worse test RMSE (3.538 vs 3.463) and was abandoned.

### Analysis

The decision to use 5 features instead of 7 was made **after** both variants were trained and tested. The `_fe` variant was dropped in favor of the 5-feature variant. While this could have been decided purely on validation loss (0.301 vs 0.300), the test RMSE was also computed and visible for both runs. This creates a plausible channel for implicit selection bias: the researchers saw that 5 features produced better test performance before selecting it as the final model.

### Mitigating Factors

- The validation loss difference (0.300 vs 0.301) already favors the 5-feature model, making the decision defensible on validation data alone.
- Only two feature variants were compared (not a large search), limiting the multiple-comparisons problem.

---

## Finding 2: Qubit Count Change from 16 to 12 Lacks Documented Justification

**Severity: MEDIUM**

### Evidence

The earliest full-scale configuration (`quantum_model/outputs/model_crossgen/config.json`, line 18) used:
```
"qnn_qubits": 16
```

All subsequent runs (`quantum_model_crossgen`, `quantum_model_crossgen_fe`, `codex_model`) switched to:
```python
# crossgen_hybrid_training.py, line 113
qnn_qubits: int = 12
```

The `model_crossgen` run with 16 qubits has only a checkpoint (`best_model.pt`) and config but no completed `run_summary.json` or `metrics_summary.csv` -- the run appears to have been interrupted or its results not saved.

### Analysis

The reduction from 16 to 12 qubits could be driven by:
1. Computational constraints (legitimate, no leakage).
2. Observing that 16-qubit training was unstable or slow (legitimate).
3. Observing that 16-qubit performance on test data was poor (would constitute leakage).

Since the 16-qubit run is incomplete, option 3 is unlikely but cannot be ruled out. The `QuantumBlock` comment (line 431) says "A lighter 12-qubit circuit than the original sketch" which suggests the change was motivated by computational efficiency, not test performance. The early sketch run with 4 qubits (`model_quant_sketch_crossgen`, max_epochs=5, train_pool_limit=500) was clearly a feasibility test.

### Mitigating Factors

- The comment on line 431 suggests the decision was architecture-driven.
- The 16-qubit run never completed metrics, so test-based selection is unlikely.
- 12 is a standard choice for quantum circuit width (matching the fusion output dimension).

---

## Finding 3: Train/Val/Test Split is Pre-defined and Generator-Based

**Severity: LOW (positive finding)**

### Evidence

The split is defined in the dataset itself (`labels.parquet`), not in the training code. From `crossgen_hybrid_training.py`, lines 318-320:

```python
inner_train_indices = np.where(labels["split"].values == "train")[0]
inner_val_indices = np.where(labels["split"].values == "val")[0]
test_indices = np.where(labels["split"].values == "test")[0]
```

From `preflight.json`:
- Train: 37,281 rows (TauREx generator)
- Val: 4,142 rows (TauREx generator)
- Test: 685 rows (Poseidon generator)

From `manifest.json`:
- TauREx: 41,423 total (split into train/val)
- Poseidon: 685 total (all assigned to test)

### Analysis

The split is structurally locked: train and validation come from TauREx, test comes from Poseidon. The split ratio (90/10 for train/val within TauREx) appears to be set at dataset creation time, not tuned during model development. The `internal_val_fraction` config parameter (0.10) exists in `TrainingConfig` but is **never used** in `prepare_data()` -- the code reads pre-existing split labels instead of creating a random split. This is good: it means the split cannot be accidentally tuned.

The config field `internal_val_fraction: float = 0.10` at line 95 is a vestigial parameter that could confuse readers into thinking the split is configurable. It is dead code.

---

## Finding 4: Standardizers Are Fit on Training Data Only

**Severity: LOW (positive finding)**

### Evidence

From `crossgen_hybrid_training.py`, lines 325-327:
```python
aux_scaler = ArrayStandardizer.fit(aux_raw[inner_train_indices])
target_scaler = ArrayStandardizer.fit(targets_raw[inner_train_indices])
spectral_scaler = SpectralStandardizer.fit(spectra_raw[inner_train_indices, 0, :])
```

All three standardizers are fit exclusively on training indices. Validation and test data are then transformed using these train-fitted scalers (lines 330-333). This is correct practice.

---

## Finding 5: No Evidence of Systematic Hyperparameter Search

**Severity: LOW (positive finding)**

### Evidence

Across all four project directories, every configuration uses identical hyperparameters except for:

| Parameter | Sketch | Full runs |
|-----------|--------|-----------|
| `qnn_qubits` | 4 | 12 (previously 16) |
| `max_epochs` | 5 | 30 |
| `train_batch_size` | 64 | 256 |
| `train_pool_limit` | 500 | null |
| `poseidon_limit` | 100 | n/a |
| `log_every_batches` | 99 | 1 |

The following hyperparameters are **identical across all runs**:
- `classical_lr`: 2.0e-3
- `quantum_lr`: 6.0e-4
- `weight_decay`: 1.0e-4
- `dropout`: 0.05
- `early_stop_patience`: 6
- `scheduler_patience`: 5
- `scheduler_factor`: 0.5
- `aux_hidden_dim`: 64, `aux_out_dim`: 32
- `spectral_hidden_dim`: 64, `spectral_out_dim`: 32
- `fusion_hidden_dim`: 48, `head_hidden_dim`: 96
- `seed`: 42

There is no output directory with a parameter sweep, no grid/random search infrastructure, and no multiple-seed experiments. The hyperparameters appear to have been set once and never changed, which is consistent with being literature-derived or chosen by convention rather than tuned on data.

---

## Finding 6: Random Seed 42 -- Conventional, Not Cherry-Picked

**Severity: LOW**

### Evidence

All runs use `seed: int = 42` (line 94). Only a single seed was ever used across all experiments.

### Analysis

Seed 42 is the most conventional default in ML research. With only one seed tested, there is no evidence of seed shopping. However, the absence of multi-seed evaluation means the reported results could be optimistically or pessimistically biased by this single seed. This is a reproducibility concern rather than a leakage concern.

---

## Finding 7: Rebinning Decision (218 to 44 bins) Pre-dates Training

**Severity: LOW (positive finding)**

### Evidence

The Ariel wavelength grid is hardcoded as a constant (lines 43-53) and matches the Ariel Space Mission's Tier 2 spectral resolution. The rebinning from 218 to 44 bins is performed identically in all three `crossgen_hybrid_training.py` variants (quantum_model_crossgen, quantum_model_crossgen_fe, codex_model). It was never varied between runs.

The earliest `quantum_model/crossgen_hybrid_training.py` does NOT include `spectres` or rebinning (no `import spectres`, no `ARIEL_WAVELENGTH_GRID`), operating on the raw 218 bins. The rebinning was introduced in `quantum_model_crossgen` and kept constant thereafter.

### Analysis

The rebinning target (Ariel Tier 2 grid) is physics-motivated -- it matches the instrument the model targets. There is no evidence this specific grid was chosen by testing multiple rebinning schemes and selecting the one with best test RMSE.

---

## Finding 8: Early Stopping Uses Validation Loss, Not Test Loss

**Severity: LOW (positive finding)**

### Evidence

From `crossgen_hybrid_training.py`, lines 768-769 and 800-814:
```python
inner_val_metrics = evaluate_split(model, data.inner_val, data.target_scaler, config.eval_batch_size)
scheduler.step(inner_val_metrics["loss"])
...
if inner_val_metrics["loss"] < best_val_loss:
    best_val_loss = float(inner_val_metrics["loss"])
    ...
```

Model selection (best checkpoint) and learning rate scheduling are both driven by `inner_val_metrics["loss"]`, never by test metrics. The test set is only evaluated once, after training completes (line 946 in `run_training_experiment()`).

---

## Finding 9: Test Evaluation Occurs Post-Training Only

**Severity: LOW (positive finding)**

### Evidence

In `run_training_experiment()`, lines 945-946:
```python
inner_val_metrics = evaluate_split(model, data.inner_val, data.target_scaler, cfg.eval_batch_size)
test_metrics = evaluate_split(model, data.test, data.target_scaler, cfg.eval_batch_size)
```

Test evaluation happens after `train_model()` returns. No test metrics are used for model selection, early stopping, or any training decision within a single run. This is correct practice.

---

## Finding 10: Cross-Generator Test Set Creates Potential Oracle Knowledge Risk

**Severity: MEDIUM**

### Evidence

The test set is exclusively Poseidon-generated spectra (685 samples), while train/val are exclusively TauREx-generated. From `manifest.json`, the dataset creators computed and stored detailed comparison statistics between generators:

```json
"comparison_summary": {
    "delta_mean_log10_vmr": {
        "log10_vmr_co2": -0.083,
        "log10_vmr_h2o": -0.103,
        ...
    }
}
```

The `baseline_smoke.json` file in the data directory contains pre-computed test RMSE for a baseline model (mean-predictor) on the Poseidon test set. The target distribution differences between generators (e.g., Poseidon H2O mean = -7.087 vs TauREx H2O mean = -6.984) were known and documented before model training began.

### Analysis

The researchers had access to statistical properties of the test set (distribution means, standard deviations, prevalence rates) before designing the model. While this is common in challenge-style ML competitions and the information is aggregate rather than sample-level, it theoretically allows architecture decisions to be influenced by knowledge of the test distribution.

The massive val-to-test RMSE gap (1.54 vs 3.46) confirms the cross-generator domain shift is the dominant challenge. Knowing the test set comes from a different simulator could bias design choices toward robustness features -- though no such features (e.g., domain adaptation, distribution matching) are actually implemented.

---

## Finding 11: Feature Selection Is Domain-Driven, Not Data-Driven

**Severity: LOW (positive finding)**

### Evidence

`SAFE_AUX_FEATURE_COLS` (lines 27-33):
```python
SAFE_AUX_FEATURE_COLS = [
    "planet_radius_rjup",
    "log_g_cgs",
    "temperature_k",
    "star_radius_rsun",
    "log10_sigma_ppm",
]
```

These are standard physical parameters for exoplanet atmospheric retrieval. The `_fe` variant tested adding `rp_rs_ratio` and `scale_height_proxy` (engineered features), found worse performance, and reverted. The original 5 features are identical to what the ADC2023 baseline uses.

---

## Finding 12: No Test Data Used in Data Pipeline Transformations

**Severity: LOW (positive finding)**

### Evidence

The per-sample spectral normalization (lines 278-280) is a per-sample operation:
```python
per_sample_mean = noisy_spectra.mean(axis=1, keepdims=True)
per_sample_mean = np.where(per_sample_mean == 0, 1.0, per_sample_mean)
spectra = (noisy_spectra / per_sample_mean)[:, None, :].astype(np.float32)
```

While this is applied to ALL data (including test), it uses only each sample's own mean -- no cross-sample statistics. This is correct: inference-time normalization would work identically.

The log10 transformation of sigma_ppm (line 268) is similarly a per-sample operation computed before any train/test splitting logic.

---

## Summary Table

| # | Finding | Severity | Status |
|---|---------|----------|--------|
| 1 | Model variant selection after observing test RMSE | MEDIUM | Possible implicit bias |
| 2 | Qubit count change (16->12) without documented justification | MEDIUM | Likely benign but undocumented |
| 3 | Pre-defined generator-based split | LOW | Good practice |
| 4 | Standardizers fit on train only | LOW | Good practice |
| 5 | No systematic hyperparameter search | LOW | Good practice |
| 6 | Single conventional seed | LOW | No cherry-picking |
| 7 | Rebinning is physics-motivated, constant | LOW | Good practice |
| 8 | Early stopping on validation loss | LOW | Good practice |
| 9 | Test evaluation post-training only | LOW | Good practice |
| 10 | Oracle knowledge of test distribution available | MEDIUM | Risk present but not exploited |
| 11 | Features are domain-standard | LOW | Good practice |
| 12 | No test data in transformations | LOW | Good practice |

---

## Verdict: PASS (conditional)

The codebase demonstrates sound experimental methodology with several structural safeguards against test-set leakage:
- Pre-defined splits locked in the dataset
- Scalers fit on training data only
- Early stopping and model selection on validation loss
- Test evaluation only after training completes
- No hyperparameter search infrastructure

The conditional aspect relates to three MEDIUM findings:
1. **Model variant selection** (5 vs 7 features) occurred after test metrics were visible for both variants. While the validation loss alone supports the same decision, the test RMSE was computed and available, creating an implicit selection channel.
2. **Qubit count evolution** (16 -> 12) is undocumented in terms of decision rationale, though circumstantial evidence (incomplete 16-qubit run, code comments) suggests computational rather than test-performance motivation.
3. **Oracle knowledge** of test set statistical properties was available through the dataset manifest, though there is no evidence it influenced architecture decisions.

**Recommendation**: To strengthen the experimental design, the project should (a) document the rationale for the 16->12 qubit change, (b) run multi-seed experiments to bound result variance, and (c) note in any publication that model variant selection was performed with both validation and test metrics visible, even if the validation metric alone would have led to the same choice.
