# ML Model Scientific Rigor Audit — 16 Parallel Sub-Agents

## Usage
Run this from the model directory you want to audit (e.g., `exobiome_model/` or `quantum_model_crossgen/`).
Paste the prompt below into legacy_model Code.

---

## PROMPT (copy everything below this line)

```
I need you to run a comprehensive scientific rigor audit of the ML model in this directory. This model is a hybrid quantum-classical neural network for exoplanet biosignature detection — it will be used in a scientific paper, so every claim must be bulletproof.

The project structure:
- Training code: `crossgen_hybrid_training.py` (main pipeline)
- Training entry: `run_training.py` or `train.ipynb`
- Dataset: `../datasets/crossgen_biosignatures_20260311/` (labels.parquet + spectra.h5)
- Baseline comparison: `../baseline/` (TF/Keras CNN)
- Outputs: `outputs/` directory with model checkpoints, metrics, predictions

Launch exactly 16 sub-agents in parallel. Each agent must:
1. Read ALL relevant source files thoroughly (not skim — read every line)
2. Write a detailed findings report as a markdown file in `audit_reports/` directory
3. Include severity ratings: CRITICAL (paper-invalidating), HIGH (must fix), MEDIUM (should fix), LOW (cosmetic)
4. Include specific file paths, line numbers, and code snippets for every finding
5. End with a PASS/FAIL verdict for their specific audit area

Create the `audit_reports/` directory first, then launch all 16 agents:

---

### Agent 1: Data Leakage — Train/Val/Test Contamination
File: `audit_reports/01_data_leakage.md`

Investigate every possible vector of data leakage:
- Read `crossgen_hybrid_training.py` line by line. Trace exactly how `labels.parquet` is loaded, how the `split` column is used, and verify train/val/test are strictly separated.
- Check if the internal validation split (`inner_val_frac`) is created BEFORE or AFTER any fitting/normalization. If standardizers are fit on data that includes val samples, that's leakage.
- Verify that `ArrayStandardizer` and `SpectralStandardizer` are fit ONLY on training data. Read the fit/transform logic carefully.
- Check if per-sample spectral normalization (dividing by mean) could leak information across splits.
- Check if any data augmentation or noise injection uses information from test/val sets.
- Look for any shuffling that might mix splits before they're separated.
- Check if `sample_id` ordering could create temporal/sequential leakage.
- Verify the test set (POSEIDON generator) is truly held out and never seen during training.
- Check if early stopping on inner_val could indirectly overfit to val distribution.
- Read the dataset files (`labels.parquet` schema, `spectra.h5` structure) to verify split integrity.

---

### Agent 2: Target Leakage — Features Containing Target Information
File: `audit_reports/02_target_leakage.md`

Investigate whether any input features contain or encode target information:
- List ALL input features used: auxiliary features (planet_radius_rjup, log_g_cgs, temperature_k, star_radius_rsun, log10_sigma_ppm) and spectral features.
- For each auxiliary feature, reason about whether it could be derived from or correlated with the targets (log10 VMR of H2O, CO2, CO, CH4, NH3).
- Check if `temperature_k` is the atmospheric temperature (which IS related to chemical abundances) vs. equilibrium temperature (which is a physical input). This matters hugely.
- Check if `planet_radius_rjup` is the fitted radius (output of retrieval, i.e., leakage) or the true/input radius.
- Verify that `log10_sigma_ppm` (noise level) doesn't encode information about atmospheric composition.
- Check if `latents.parquet` (encoder bottleneck representations) is ever used as input — if so, that's massive leakage.
- Look for any feature engineering that uses target values.
- Check if the spectral data itself has been pre-processed in a way that embeds target information.

---

### Agent 3: Cross-Generator Validity — TauREx vs POSEIDON
File: `audit_reports/03_cross_generator.md`

Evaluate the scientific validity of training on TauREx and testing on POSEIDON:
- Read the dataset labels and identify exact counts per generator per split.
- Analyze whether the train/val/test split is stratified or random. Is it purely generator-based (train=TauREx, test=POSEIDON)?
- If test is ONLY POSEIDON, assess what this means for generalization claims. Can you claim "the model works" if it's only tested on one simulator?
- Check if TauREx and POSEIDON use the same physics, opacity databases, and assumptions. Look for any documentation or comments about this.
- Analyze the cross-generator gap in results (val RMSE vs test RMSE). Is a gap of +1.92 expected or alarming?
- Check if the model is memorizing TauREx-specific artifacts rather than learning physics.
- Look for domain adaptation or any technique to handle distribution shift. Is the absence of such techniques a problem?
- Evaluate whether the wavelength grids are compatible between generators.

---

### Agent 4: Metric Validity and Statistical Rigor
File: `audit_reports/04_metrics_statistics.md`

Audit all evaluation metrics and statistical claims:
- Read how RMSE is computed. Is it on standardized or original scale? Are there any off-by-one errors?
- Check if the inverse scaling (from standardized back to original) is done correctly. Trace the math: does `y_pred * scale + mean` give correct log10 VMR values?
- Verify that per-target RMSE is computed correctly (not accidentally averaging wrong dimensions).
- Check if mean RMSE across targets is appropriate or if some targets should be weighted differently.
- Look for any cherry-picking: are "best" results reported instead of mean/median?
- Check if the reported metrics come from the best epoch (checkpoint) or last epoch. Is using best-checkpoint valid or is it a form of test-set snooping?
- Verify that no metric computation uses information from the test set (e.g., test-set statistics for normalization).
- Check sample sizes: with 685 test samples, compute confidence intervals for RMSE. Are the reported differences statistically significant?
- Look for any multiple comparison issues (testing many configurations, reporting best).

---

### Agent 5: Reproducibility Audit
File: `audit_reports/05_reproducibility.md`

Check every aspect of reproducibility:
- Is `random_seed=42` actually propagated to ALL sources of randomness? Check: Python random, numpy, PyTorch (CPU + CUDA), PennyLane.
- Does setting the seed guarantee identical results across runs? Check for non-deterministic operations (e.g., CUDA atomics, certain PyTorch ops).
- Are all hyperparameters saved in `config.json`? Compare what's in config vs what's actually used in code.
- Can the training be reproduced from the saved config alone?
- Check if the data loading order is deterministic (HDF5 access patterns, parquet reading).
- Verify that the saved model (`best_model.pt`) can reproduce the reported metrics exactly.
- Check if library versions are pinned (requirements.txt, conda env). Are there version-sensitive behaviors?
- Look for any hardcoded paths, magic numbers, or environment-dependent behavior.
- Check if `preflight.json` contains enough info to verify the data split.

---

### Agent 6: Quantum Circuit Correctness
File: `audit_reports/06_quantum_circuit.md`

Deep audit of the quantum computing component:
- Read the `QuantumBlock` class line by line. Trace the quantum circuit construction.
- Verify that the number of trainable parameters matches expectations: should be 3 * n_qubits * (depth/2). Check the actual count.
- Check if the input encoding (RY gates with fusion output) is correct. Is `tanh(x) * π` a valid encoding range?
- Verify the entanglement pattern (CNOT ladder + CRX ladder). Is this a known ansatz or custom?
- Check if PennyLane's `qml.qnode` is correctly configured (interface, diff_method). Is `best` diff method appropriate?
- Does the quantum circuit actually provide any advantage over a classical layer with the same parameter count? This is a critical scientific question.
- Check for barren plateaus: with 16 qubits and this depth, is the gradient landscape trainable?
- Verify that expectation values (Pauli-Z) are in [-1, 1] range and check how they're used downstream.
- Check if the residual connection from quantum output bypasses the quantum block effectively making it optional.
- Verify quantum parameter gradients are flowing correctly (check gradient norms in training history if available).

---

### Agent 7: Loss Function and Optimization
File: `audit_reports/07_loss_optimization.md`

Audit the training objective and optimization:
- Is MSE the right loss for log10 VMR regression? Consider: are targets normally distributed? Are there outliers?
- Check if the loss is computed on standardized or raw scale. This affects relative weighting of targets.
- Verify the dual learning rate setup: classical_lr=2e-3, quantum_lr=6e-4. Is this justified?
- Check gradient clipping values (5.0 classical, 1.0 quantum). Are these appropriate? Too aggressive clipping can prevent learning.
- Verify ReduceLROnPlateau scheduler behavior. What happens when LR is reduced — does it affect both param groups?
- Check for NaN/Inf in training — are there any guards? What happens if quantum gradients explode?
- Verify AdamW weight decay settings. Weight decay on quantum params is 0 — is this correct?
- Check early stopping logic: patience=6 on inner_val loss. Is this robust?
- Look for any training instabilities in `history.csv` (loss spikes, oscillations).
- Verify that the optimizer state is correctly handled during checkpoint loading.

---

### Agent 8: Data Preprocessing Pipeline
File: `audit_reports/08_preprocessing.md`

Thorough audit of all data transformations:
- Trace the EXACT sequence of transforms from raw HDF5 → model input.
- Check spectral rebinning (SpectRes): is the wavelength grid mapping correct? Are edge cases handled (bins at boundaries)?
- Verify per-sample normalization: `spectrum / mean(spectrum)`. Does this preserve relative feature importance? Could it introduce artifacts?
- Check `ArrayStandardizer`: verify mean/std computation is numerically stable. What if std=0 for a feature?
- Check `SpectralStandardizer`: per-bin normalization on top of per-sample normalization — is double normalization correct?
- Verify that saved scalers (`scalers.json`) exactly reproduce the training-time normalization.
- Check for any silent NaN/Inf introduction during normalization.
- Verify that train/val/test get EXACTLY the same transform (except fitting only on train).
- Check if the preprocessing is invertible — can you go from predictions back to physical units?
- Look for any off-by-one errors in wavelength bin indexing.

---

### Agent 9: Model Architecture Sanity
File: `audit_reports/09_architecture.md`

Audit the neural network architecture for correctness and appropriateness:
- Read SpectralEncoder, AuxEncoder, FusionEncoder, PredictionHead line by line.
- Check tensor shapes at every layer. Do Conv1d input/output channels match? Is padding correct?
- Verify AdaptiveAvgPool1d(1) produces the expected output shape.
- Check if dropout (0.05) is applied consistently and only where appropriate.
- Verify the fusion mechanism: is simple concatenation sufficient? Does the architecture actually use the quantum output meaningfully?
- Check the prediction head: the skip connection `Linear(qubits → 5)` from quantum output — does this create a shortcut that bypasses the main network?
- Count total parameters (classical + quantum). Is the model appropriately sized for 37k training samples?
- Check for dead neurons, vanishing gradients potential (deep concatenation with many linear layers).
- Verify that `model.eval()` vs `model.train()` correctly handles dropout.
- Check if batch normalization or layer normalization is used correctly (LayerNorm in FusionEncoder).

---

### Agent 10: Overfitting and Generalization Analysis
File: `audit_reports/10_overfitting.md`

Deep analysis of overfitting risks:
- Read `history.csv` and `metrics_summary.csv`. Plot/analyze train vs val loss curves.
- Compare validation RMSE to test RMSE. A large gap indicates potential overfitting or distribution shift.
- Check if the model capacity (parameter count) is appropriate for the dataset size (37k train).
- Analyze per-target overfitting: which targets overfit most? CO2 has the largest cross-gen gap — why?
- Check if regularization (dropout=0.05, weight_decay=1e-4) is sufficient.
- Look for memorization: does the model achieve near-zero training loss while val loss plateaus?
- Analyze the learning curves: do they suggest more data would help (high bias) or that the model is overfitting (high variance)?
- Check if data augmentation is used. For spectral data, noise augmentation could help — is it missing?
- Evaluate whether the 10% inner validation split is large enough for reliable early stopping.
- Compare against the baseline model overfitting behavior.

---

### Agent 11: Baseline Comparison Fairness
File: `audit_reports/11_baseline_comparison.md`

Audit whether the quantum model vs baseline comparison is fair:
- Read BOTH the baseline code (`../baseline/`) and quantum model code.
- Check if they use the SAME dataset, SAME splits, SAME preprocessing.
- The baseline uses 52-bin ADC2023 data while quantum uses 218-bin crossgen — this is NOT the same dataset. Is the comparison valid?
- Check if there's a `crossgen-baseline.ipynb` that runs baseline on the same crossgen data. If so, compare those results.
- Verify that the baseline has been properly tuned. An under-tuned baseline makes the quantum model look better unfairly.
- Compare parameter counts: how many parameters does the baseline have vs the quantum model?
- Check if the baseline uses MC Dropout for uncertainty while the quantum model doesn't — is this an apples-to-oranges comparison?
- Look for any claims of "quantum advantage" and verify they're supported by the evidence.
- Check if the comparison controls for training compute (epochs, time, FLOPs).

---

### Agent 12: Numerical Stability and Edge Cases
File: `audit_reports/12_numerical_stability.md`

Hunt for numerical issues:
- Check all divisions for potential division by zero (spectral normalization by mean — what if mean=0?).
- Check all log operations — are there guards against log(0) or log(negative)?
- Verify that standardization doesn't produce extreme values when std is very small.
- Check quantum circuit: are there numerical issues with very small/large angles?
- Check if tanh saturation in FusionEncoder causes vanishing gradients.
- Look for float32 vs float64 mismatches that could cause precision loss.
- Check if HDF5 reading produces the expected dtypes.
- Verify that the loss function handles edge cases (all-zero predictions, extreme targets).
- Check if AdaptiveAvgPool1d handles variable-length inputs correctly.
- Look for any potential integer overflow in index calculations.
- Check if gradient computation through the quantum circuit is numerically stable.

---

### Agent 13: Code Quality and Bug Hunt
File: `audit_reports/13_code_bugs.md`

Systematic bug hunting:
- Read `crossgen_hybrid_training.py` completely. Look for:
  - Off-by-one errors in array indexing
  - Incorrect tensor reshaping (wrong dimensions)
  - Variables used before assignment
  - Mutable default arguments
  - Silent failures (exceptions caught and ignored)
  - Copy vs reference issues (are tensors cloned when they should be?)
  - Device mismatches (CPU vs CUDA tensors mixed)
- Check if `model.eval()` is called before validation/test inference.
- Check if `torch.no_grad()` is used during evaluation (memory + correctness).
- Verify that the checkpoint saving/loading preserves all necessary state.
- Check if the batch iteration handles the last (potentially smaller) batch correctly.
- Look for any race conditions or non-deterministic behavior.
- Verify print/logging statements don't contain bugs that misreport metrics.

---

### Agent 14: Scientific Claims Validation
File: `audit_reports/14_scientific_claims.md`

Validate the scientific narrative:
- Read any notebooks, markdown files, or comments that make scientific claims.
- Check: "First application of quantum ML to biosignature detection" — verify by searching the literature context in code comments and the MEMORY.md.
- Validate the claim that QELM provides meaningful quantum advantage for this task.
- Check if the model actually detects "biosignatures" or just regresses VMR values. These are different claims.
- Verify that the target values (log10 VMR) are physically meaningful — are the predicted ranges realistic for exoplanet atmospheres?
- Check if the wavelength range (0.6-5.2 μm) covers the relevant absorption features for H2O, CO2, CO, CH4, NH3.
- Evaluate whether 5 molecular species are sufficient for biosignature detection.
- Check if the noise model in the training data is realistic for actual instruments (JWST, Ariel).
- Assess whether the model's RMSE is scientifically useful — what RMSE is needed to distinguish biosignatures?

---

### Agent 15: Information Leakage Through Experimental Design
File: `audit_reports/15_experimental_design_leakage.md`

Audit higher-level experimental design choices for subtle leakage:
- Was any hyperparameter tuning done using test set performance? Check all output directories for multiple runs.
- Is the architecture choice (12 vs 16 qubits, depth 2, etc.) justified by val performance or test performance?
- Check if the choice of which targets to predict was informed by seeing test results.
- Was the decision to rebin spectra (218→44) informed by test performance?
- Check if the train/val split ratio was tuned. Multiple different val fractions = potential leakage.
- Look for any "oracle" information: did the researchers know properties of the test set that influenced design?
- The test set is POSEIDON while train is TauREx — if the researchers knew POSEIDON's characteristics, this could bias design.
- Check if feature selection (which 5 aux features to use) was done using test data.
- Look for signs of extensive hyperparameter search that wasn't properly accounted for.
- Check if the random seed (42) was chosen because it gives good results.

---

### Agent 16: End-to-End Prediction Sanity Check
File: `audit_reports/16_prediction_sanity.md`

Verify predictions make physical sense:
- Read `test_predictions.csv`. For each target:
  - Check if predictions are in the physically valid range (log10 VMR should be roughly -12 to -1).
  - Check if predictions are constant (model collapsed to predicting the mean).
  - Check if predictions show meaningful variation that correlates with true values.
  - Compute correlation coefficients between true and predicted.
  - Look for any systematic bias (consistently over/under-predicting).
- Check for impossible predictions (e.g., log10 VMR > 0, which means >100% atmosphere).
- Verify that the per-target RMSE in `metrics_summary.csv` matches what you compute from `test_predictions.csv`.
- Check if any test samples have wildly wrong predictions — are there outliers? What do they look like?
- Cross-reference: do the worst-predicted samples have unusual input features?
- Verify the mean prediction is close to the mean true value (no systematic shift from denormalization).
- Check if the model is just learning the prior (predicting near-mean for all samples) or actually using input features.

---

After all 16 agents complete, create a final summary file `audit_reports/00_EXECUTIVE_SUMMARY.md` that:
1. Lists all CRITICAL and HIGH severity findings across all reports
2. Gives an overall PASS/FAIL verdict for paper readiness
3. Prioritizes what must be fixed before publication
4. Notes what can be acknowledged as limitations vs what invalidates results
```
