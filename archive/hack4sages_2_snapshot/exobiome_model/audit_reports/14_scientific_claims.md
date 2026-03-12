# Audit Report 14: Scientific Claims Validation

## Summary

The project makes several scientific claims that range from well-supported to critically problematic. The most severe issue is that the hybrid quantum-classical model **performs worse than a trivial mean predictor** on the cross-generator test set, meaning no claims about biosignature detection, quantum advantage, or scientific utility are currently supported by the empirical results. The "first application of quantum ML to biosignature detection" novelty claim is defensible but narrowly scoped. Several framing claims conflate VMR regression with biosignature detection.

**Overall Verdict: FAIL**

## Methodology

All source files were read in full: `crossgen_hybrid_training.py` (987 lines), `train.ipynb` (6 cells with outputs), `legacy_model.md`, `.legacy_model/research.md`, design docs in `docs/plans/`, and all output artifacts (`metrics_summary.csv`, `test_predictions.csv`, `run_summary.json`, `config.json`, `test_metrics.json`, `inner_val_metrics.json`, `history.csv`, `baseline_smoke.json`). Claims were cross-referenced against the actual model outputs, baseline performance, physical constraints, and literature context.

---

## Findings

### Finding 1: Model performs WORSE than mean predictor on test set

- **Severity**: CRITICAL
- **Files**:
  - `/Users/michalszczesny/projects/hack4sages/exobiome_model/outputs/model_crossgen_rebinned/test_metrics.json`
  - `/Users/michalszczesny/projects/hack4sages/exobiome_model/data/baseline_smoke.json`
  - `/Users/michalszczesny/projects/hack4sages/exobiome_model/.legacy_model/research.md`:151-155

- **Description**: The baseline mean predictor (predicting training set mean for every sample) achieves test RMSE values of 2.84-2.97 per target. The hybrid quantum-classical model achieves test RMSE values of 2.95-4.73 per target. The model is worse than guessing the mean for every single target molecule.

  **Baseline mean predictor test RMSE** (from `baseline_smoke.json`):
  | Target | Baseline RMSE |
  |--------|---------------|
  | H2O    | 2.874         |
  | CO2    | 2.966         |
  | CO     | 2.848         |
  | CH4    | 2.948         |
  | NH3    | 2.841         |
  | **Mean** | **2.895**   |

  **Hybrid model test RMSE** (from `test_metrics.json`):
  | Target | Model RMSE | vs. Baseline |
  |--------|------------|--------------|
  | H2O    | 3.131      | +0.257 worse |
  | CO2    | 4.726      | +1.760 worse |
  | CO     | 3.283      | +0.435 worse |
  | CH4    | 3.227      | +0.280 worse |
  | NH3    | 2.948      | +0.107 worse |
  | **Mean** | **3.463** | **+0.568 worse** |

- **Impact**: This invalidates all downstream claims about the model's scientific utility. The research.md document (line 155) itself states: "Any model must beat these numbers to demonstrate learning." The model fails this criterion. The inner-val RMSE (1.54) is much better, indicating severe overfitting to TauREx-generated data and complete failure to generalize to Poseidon-generated test data.

- **Root cause**: The model has learned TauREx simulator artifacts rather than physical features. The 37k TauREx train / 4k TauREx val / 685 Poseidon test split was designed to detect exactly this failure mode, and it has.

---

### Finding 2: "Biosignature detection" vs VMR regression -- misleading framing

- **Severity**: HIGH
- **Files**:
  - `/Users/michalszczesny/projects/hack4sages/exobiome_model/legacy_model.md`:3
  - `/Users/michalszczesny/projects/hack4sages/exobiome_model/train.ipynb` cell-0
  - `/Users/michalszczesny/projects/hack4sages/exobiome_model/.legacy_model/research.md`:1-7

- **Description**: The project is consistently described as a model "for biosignature detection" (legacy_model.md line 3, notebook title, research.md line 7). However, the model performs **regression of log10 volume mixing ratios** for 5 molecules. These are fundamentally different tasks:

  - **VMR regression**: Estimating the concentration of each gas in an atmosphere. This is atmospheric retrieval, a well-established inverse problem.
  - **Biosignature detection**: Determining whether observed atmospheric signatures indicate biological processes. This requires: (a) identifying molecular abundances, (b) assessing whether those abundances are consistent with abiotic processes, (c) evaluating the thermodynamic disequilibrium of the atmosphere, and (d) considering contextual factors (stellar type, planetary orbit, etc.).

  The model does only step (a), and does it poorly (see Finding 1). There is no classification head, no disequilibrium metric, no abiotic/biotic discrimination, and no decision threshold. Predicting that CH4 = 10^-5 VMR does not tell you whether that methane is biogenic.

- **Impact**: The project title and framing overstate what the model actually does. A more accurate description would be "quantum-classical atmospheric retrieval" or "quantum-classical VMR regression from transmission spectra."

---

### Finding 3: "First application of quantum ML to biosignature detection" -- partially defensible

- **Severity**: MEDIUM
- **Files**:
  - `/Users/michalszczesny/projects/hack4sages/exobiome_model/.legacy_model/research.md`:186-187

- **Description**: The MEMORY.md context states this is the "first ever application of quantum ML to biosignature detection." Based on the cited literature:

  - **Vetrano et al. 2025** (arXiv:2509.03617) applied QELM to atmospheric retrieval (VMR regression), but not to biosignature detection per se.
  - **Seeburger et al. 2023** linked biosphere models to planetary spectra but used no ML.
  - No other published work combines quantum ML with exoplanet atmospheric composition prediction.

  The novelty claim is defensible **if and only if** the task is accurately characterized. Since the model also does VMR regression (not biosignature detection), the distinction from Vetrano et al. narrows to: (1) different quantum circuit design, (2) different dataset, and (3) cross-generator validation. These are incremental rather than paradigmatic differences.

- **Impact**: The novelty claim should be qualified. The honest framing would be: "Among the first applications of hybrid quantum-classical ML to exoplanet atmospheric retrieval, with a novel cross-generator evaluation protocol."

---

### Finding 4: No evidence of quantum advantage

- **Severity**: HIGH
- **Files**:
  - `/Users/michalszczesny/projects/hack4sages/exobiome_model/crossgen_hybrid_training.py`:430-486 (QuantumBlock)
  - `/Users/michalszczesny/projects/hack4sages/exobiome_model/crossgen_hybrid_training.py`:533-548 (forward pass)
  - `/Users/michalszczesny/projects/hack4sages/exobiome_model/.legacy_model/research.md`:186-187, 199-203

- **Description**: The research.md claims the "QELM approach encodes classical features into quantum angles, leveraging exponential Hilbert space for feature mapping" and suggests this "provides a richer feature transformation than a 12-neuron classical layer" (line 187). However:

  1. **No ablation study exists**: There is no classical-only baseline to compare against. Without removing the quantum block and retraining, it is impossible to attribute any model performance to quantum processing.
  2. **The quantum block is architecturally bypassable**: The head receives a 88-dimensional input: 12 from quantum output, 12 from the classical latent (which is the quantum input), 32 from aux features, and 32 from spectral features. The classical skip connections provide 76 of 88 input dimensions. The optimizer could learn to ignore the quantum output entirely.
  3. **36 trainable quantum parameters vs ~50k+ classical parameters**: The quantum contribution is a tiny fraction of the model's capacity.
  4. **Simulation only**: Running on `lightning.qubit` provides no quantum computational advantage. The simulator performs classical linear algebra. Any quantum advantage would only manifest on real hardware, which is not used here.
  5. **The model fails to beat a mean predictor**: Even if quantum processing were contributing, the overall system is not scientifically useful.

  The research.md itself acknowledges this weakness (lines 199-203): "the quantum contribution might be marginal compared to the classical encoders alone. An ablation study would clarify."

- **Impact**: Claims about quantum advantage or quantum-enhanced feature learning are unsubstantiated. The recommendation in research.md to run an ablation study has not been implemented.

---

### Finding 5: Predicted VMR ranges are physically unrealistic in some cases

- **Severity**: HIGH
- **Files**:
  - `/Users/michalszczesny/projects/hack4sages/exobiome_model/outputs/model_crossgen_rebinned/test_predictions.csv`

- **Description**: Examining the test predictions reveals systematic issues:

  1. **CO2 predictions are compressed**: The model predicts CO2 values clustered around -10 to -11 regardless of the true value. For example:
     - poseidon_000008: true=-2.76, pred=-11.04 (error of 8.28 dex)
     - poseidon_000025: true=-2.35, pred=-9.38 (error of 7.02 dex)
     - poseidon_000012: true=-2.63, pred=-10.07 (error of 7.44 dex)

     The model essentially predicts "no CO2" for nearly every sample, explaining the 4.73 RMSE for CO2.

  2. **Prediction range collapse**: While true values span [-12, -2], predictions cluster in a much narrower band (roughly [-11, -5] for most targets). The model has not learned to discriminate between trace (10^-12) and abundant (10^-2) concentrations.

  3. **Physical meaning of errors**: An RMSE of 3.46 in log10 space means the model's predictions are off by a factor of ~10^3.5 = ~3,000x on average. For a gas with true VMR of 10^-5 (10 ppm), the model might predict anywhere from 10^-8.5 (0.003 ppm) to 10^-1.5 (3%). This range spans from undetectable trace gas to dominant atmospheric constituent -- the prediction is scientifically meaningless.

- **Impact**: The predictions cannot distinguish between atmospheres with and without significant molecular features. The model output has no diagnostic value for atmospheric characterization.

---

### Finding 6: Wavelength range adequacy for target molecules

- **Severity**: LOW
- **Files**:
  - `/Users/michalszczesny/projects/hack4sages/exobiome_model/crossgen_hybrid_training.py`:43-53 (ARIEL_WAVELENGTH_GRID)

- **Description**: The rebinned wavelength grid covers 0.95-4.91 um (44 bins). Coverage of key absorption features:

  | Molecule | Key bands (um) | Covered? |
  |----------|----------------|----------|
  | H2O | 1.4, 1.9, 2.7, 6.3 | Partial (misses 6.3 um) |
  | CO2 | 2.0, 2.7, 4.3, 15 | Partial (has 2.7, 4.3; misses 15 um) |
  | CO | 2.3, 4.7 | Yes (both within range) |
  | CH4 | 1.7, 2.3, 3.3, 7.7 | Partial (has 1.7, 2.3, 3.3; misses 7.7 um) |
  | NH3 | 1.5, 2.0, 3.0, 10.5 | Partial (has 1.5, 2.0, 3.0; misses 10.5 um) |

  The 0.95-4.91 um range captures the primary near-infrared bands for all 5 molecules. This is consistent with the Ariel mission's planned spectral range. The missing mid-infrared features (>5 um) would improve sensitivity but are not required for detection at the VMR levels in this dataset.

- **Impact**: The wavelength coverage is appropriate and physically justified for the Ariel context. This is not a significant limitation.

---

### Finding 7: Sufficiency of 5 molecular species for biosignature claims

- **Severity**: MEDIUM
- **Files**:
  - `/Users/michalszczesny/projects/hack4sages/exobiome_model/crossgen_hybrid_training.py`:35-41 (TARGET_COLS)

- **Description**: The model predicts VMR for H2O, CO2, CO, CH4, and NH3. For actual biosignature assessment, additional context would be needed:

  1. **Missing key biosignature gases**: O2, O3 (ozone), N2O (nitrous oxide), phosphine (PH3), and dimethyl sulfide (DMS) are recognized biosignature candidates not included in the model.
  2. **No disequilibrium metric**: The simultaneous presence of oxidizing (CO2) and reducing (CH4) species at certain ratios is the classic thermodynamic disequilibrium biosignature. The model predicts raw VMRs but computes no ratio or disequilibrium metric.
  3. **Context dependence**: Whether CH4 at 10^-4 VMR is a biosignature depends on the stellar UV flux, atmospheric temperature, and presence of other species. The model has no mechanism to evaluate this context.

  However, for the narrower task of atmospheric retrieval (estimating gas abundances), 5 species is a reasonable starting point consistent with the ADC2023 challenge format and Ariel mission Tier 2 science objectives.

- **Impact**: The 5 species are adequate for atmospheric retrieval but insufficient to justify "biosignature detection" claims without additional analysis layers.

---

### Finding 8: Noise model realism

- **Severity**: MEDIUM
- **Files**:
  - `/Users/michalszczesny/projects/hack4sages/exobiome_model/.legacy_model/research.md`:108
  - `/Users/michalszczesny/projects/hack4sages/exobiome_model/crossgen_hybrid_training.py`:262 (sigma_ppm loading)

- **Description**: The dataset uses iid Gaussian white noise with sigma ranging from 20-100 ppm (log10_sigma_ppm range: 1.30-2.00, confirmed in notebook output). Evaluation of realism:

  1. **Ariel context**: Ariel's expected noise performance for Tier 2 spectroscopy is 20-100 ppm for a single transit of bright targets, consistent with this dataset. For fainter targets or Tier 1, noise could be higher.
  2. **JWST context**: JWST NIRSpec achieves ~10-50 ppm per spectral bin for bright targets (e.g., WASP-39b), overlapping the lower end of this noise range.
  3. **Simplification**: Real instrument noise is not iid Gaussian. Systematic effects include: (a) correlated noise from stellar variability, (b) wavelength-dependent noise from detector characteristics, (c) time-varying systematics from pointing jitter, and (d) contamination from stellar spots/faculae. The white noise assumption is standard for initial studies but optimistic.
  4. **No multi-epoch stacking**: Real observations often stack multiple transits to reduce noise. The dataset models single-transit observations.

- **Impact**: The noise model is a reasonable first approximation for Ariel Tier 2 science but does not capture the full complexity of real instrument systematics. Results on real data would likely be worse. This is a standard limitation shared by most synthetic training datasets in the field.

---

### Finding 9: Required RMSE for scientifically useful biosignature discrimination

- **Severity**: HIGH
- **Files**:
  - `/Users/michalszczesny/projects/hack4sages/exobiome_model/outputs/model_crossgen_rebinned/test_metrics.json`

- **Description**: To assess what RMSE would be scientifically useful, consider the following thresholds:

  1. **Detection vs non-detection**: A gas is typically considered "detectable" at VMR > 10^-6 and "not detected" at VMR < 10^-10. The gap is 4 orders of magnitude (4.0 in log10). An RMSE of ~1.0 would be needed to reliably distinguish detection from non-detection.
  2. **Biosignature assessment**: Distinguishing biogenic CH4 levels (~10^-4 to 10^-3) from abiotic CH4 (~10^-8 to 10^-6) requires accuracy of ~1-2 orders of magnitude (RMSE < 1.0-2.0 in log10).
  3. **ADC2023 competition context**: Top entries in the Ariel Data Challenge 2023 achieved per-target RMSE of 0.5-1.5 on similar data using classical neural networks.
  4. **Current model performance**: Test RMSE of 3.46 means the model cannot distinguish between any of these regimes. The inner-val RMSE of 1.54 would be borderline useful (within the range of detection/non-detection discrimination) but this performance does not generalize.

  For the model to make any scientifically meaningful statement, per-target test RMSE should be below ~1.5 (detection discrimination) and ideally below ~0.5-1.0 (quantitative retrieval).

- **Impact**: The current test RMSE of 3.46 is approximately 2-7x worse than the minimum threshold for scientific utility, depending on the application.

---

### Finding 10: Inner-val vs test RMSE gap reveals fundamental generalization failure

- **Severity**: CRITICAL
- **Files**:
  - `/Users/michalszczesny/projects/hack4sages/exobiome_model/outputs/model_crossgen_rebinned/inner_val_metrics.json`
  - `/Users/michalszczesny/projects/hack4sages/exobiome_model/outputs/model_crossgen_rebinned/test_metrics.json`
  - `/Users/michalszczesny/projects/hack4sages/exobiome_model/outputs/model_crossgen_rebinned/history.csv`

- **Description**: The gap between inner-val and test performance is extreme:

  | Target | Inner-val RMSE | Test RMSE | Degradation |
  |--------|---------------|-----------|-------------|
  | H2O    | 1.662         | 3.131     | 1.88x       |
  | CO2    | 1.158         | 4.726     | 4.08x       |
  | CO     | 2.182         | 3.283     | 1.50x       |
  | CH4    | 1.219         | 3.227     | 2.65x       |
  | NH3    | 1.503         | 2.948     | 1.96x       |
  | **Mean** | **1.545**   | **3.463** | **2.24x**   |

  The CO2 degradation (4.08x) is particularly severe, suggesting the model learned TauREx-specific CO2 spectral signatures that do not transfer to Poseidon-generated spectra.

  This is the strongest evidence that the model is learning simulator artifacts rather than physical features. The cross-generator test protocol worked exactly as designed in exposing this failure.

- **Impact**: Any claim about the model learning physical features or generalizing to real observations is contradicted by this evidence. The model cannot be trusted for inference on real data if it fails on synthetic data from a different simulator.

---

## Summary of Findings

| # | Finding | Severity |
|---|---------|----------|
| 1 | Model worse than mean predictor on test set | CRITICAL |
| 2 | "Biosignature detection" framing is misleading | HIGH |
| 3 | Novelty claim partially defensible but needs qualification | MEDIUM |
| 4 | No evidence of quantum advantage | HIGH |
| 5 | Predicted VMR ranges physically unrealistic | HIGH |
| 6 | Wavelength coverage adequate for Ariel context | LOW |
| 7 | 5 species insufficient for biosignature claims | MEDIUM |
| 8 | Noise model reasonable but simplified | MEDIUM |
| 9 | Required RMSE for utility is 2-7x better than current | HIGH |
| 10 | Inner-val/test gap reveals fundamental generalization failure | CRITICAL |

## Recommendations

1. **Immediate**: Do not present the current test RMSE as evidence of working biosignature detection. The model underperforms a mean predictor.
2. **Reframe claims**: Change "biosignature detection" to "atmospheric retrieval" or "VMR regression" throughout all materials.
3. **Run ablation**: Train a classical-only model (remove QuantumBlock, feed latent directly to head) to establish whether the quantum component helps or hurts.
4. **Address generalization**: Investigate why TauREx->Poseidon transfer fails so severely. Consider: (a) training on mixed-generator data, (b) domain adaptation techniques, (c) identifying and removing generator-specific features.
5. **Qualify novelty**: State "among the first" rather than "the first," and clarify the distinction from Vetrano et al. 2025.
6. **Add classical baselines**: Compare against published ADC2023 results (classical NNs achieving RMSE 0.5-1.5) to contextualize performance.

---

## Verdict: FAIL

Two CRITICAL findings (model worse than mean predictor; fundamental generalization failure), three HIGH findings (no quantum advantage evidence, misleading biosignature framing, unrealistic predictions), and three MEDIUM findings. The model does not support any of the scientific claims currently made about it. The experimental design (cross-generator testing) is sound and correctly identified the model's failure to learn transferable physical features.
