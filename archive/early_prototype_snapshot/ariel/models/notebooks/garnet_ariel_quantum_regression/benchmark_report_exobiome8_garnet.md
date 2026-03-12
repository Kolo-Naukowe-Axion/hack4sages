# ExoBiome 8-Qubit Garnet Benchmark Report

## Overview

This report summarizes the completed benchmark run from `garnet_port_tutorial.ipynb` using the `8`-qubit ExoBiome checkpoint:

- Checkpoint: `ariel/models/weights/ariel_exobiome_8.pt`
- Split: `validation`
- Evaluated rows: `2000`
- Quantum backend modes compared:
  - `classical_only`
  - `local_statevector`
  - `fake_256`
  - `real_hardware`

The checkpoint metadata embedded in `ariel_exobiome_8.pt` reports:

- `best_val_rmse = 0.29081112146377563`

Important caveat:

- This notebook benchmark used `MAX_SAMPLES = 2000`, so it evaluates a large validation subset rather than the checkpoint's full validation split.
- The benchmark therefore provides a strong experimental comparison across modes, but it is not a strict one-to-one reproduction of the checkpoint's full validation metric.

## Overall Metrics

| Mode | Shots | Rows | RMSE mean | MAE mean | Total time (s) | Sec/sample |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `classical_only` | - | 2000 | `0.284091` | `0.174520` | - | - |
| `local_statevector` | 0 | 2000 | `0.278651` | `0.173108` | `2.502571` | `0.001251` |
| `fake_256` | 256 | 2000 | `0.278666` | `0.173144` | `47.749726` | `0.023875` |
| `real_hardware` | 256 | 2000 | `0.278756` | `0.173304` | `29.981910` | `0.014991` |

## Main Findings

### 1. The hybrid quantum path improves on the classical-only baseline

Comparing the pure classical prediction head against the quantum-enabled ideal simulation:

- `classical_only` RMSE: `0.284091`
- `local_statevector` RMSE: `0.278651`
- Absolute RMSE improvement: `0.005440`
- Relative RMSE improvement: about `1.9%`

For MAE:

- `classical_only` MAE: `0.174520`
- `local_statevector` MAE: `0.173108`
- Absolute MAE improvement: `0.001412`
- Relative MAE improvement: about `0.8%`

This indicates that the learned quantum residual branch contributes measurable predictive value in this checkpoint.

### 2. Hardware closely matches ideal simulation

The most important deployment result is that the real IQM hardware output is extremely close to the ideal statevector result:

- `local_statevector` RMSE: `0.278651`
- `real_hardware` RMSE: `0.278756`
- Difference: `0.000105`

For MAE:

- `local_statevector` MAE: `0.173108`
- `real_hardware` MAE: `0.173304`
- Difference: `0.000196`

This is a very small degradation, which suggests that:

- the notebook's real hardware execution path is functioning correctly
- the learned quantum contribution is not disappearing when run on the actual backend
- the hardware implementation is a faithful approximation of the ideal simulator for this benchmark

### 3. Fake backend also reproduces the simulator result

The fake backend at `256` shots is nearly identical to the ideal statevector result:

- `fake_256` RMSE: `0.278666`
- Difference from `local_statevector`: `0.000015`

This is useful as a local dry-run validation step before using real hardware.

## Per-Target RMSE

| Target | Classical | Local statevector | Fake 256 | Real hardware |
| --- | ---: | ---: | ---: | ---: |
| `log_CH4` | `0.224302` | `0.222248` | `0.222248` | `0.222266` |
| `log_CO` | `0.203195` | `0.197955` | `0.197948` | `0.197823` |
| `log_CO2` | `0.225544` | `0.207186` | `0.207164` | `0.207104` |
| `log_H2O` | `0.381731` | `0.388421` | `0.388417` | `0.388699` |
| `log_NH3` | `0.385683` | `0.377443` | `0.377553` | `0.377888` |

## Per-Target Interpretation

Relative to the classical-only baseline, the quantum-enabled path:

- improves `log_CO2` substantially
- improves `log_NH3`
- improves `log_CO`
- improves `log_CH4` slightly
- worsens `log_H2O`

So the quantum residual branch is not uniformly better on every target. Instead, it redistributes error across the target set, with a net overall gain in average RMSE and MAE.

## Runtime Interpretation

### Relative speed

- `local_statevector` is the fastest evaluation path: `0.001251 s/sample`
- `real_hardware` is much slower than ideal simulation: `0.014991 s/sample`
- `fake_256` is the slowest in this notebook run: `0.023875 s/sample`

### Practical meaning

- The ideal simulator remains best for rapid offline experimentation.
- The real hardware path is slower, but still operationally reasonable for this benchmark size.
- The fact that hardware was faster than the local fake backend in this specific run should not be overgeneralized; it mostly reflects notebook implementation overhead and batching behavior rather than a universal hardware advantage.

## Hardware Execution Summary

The benchmark successfully executed on the real IQM backend.

- Total hardware jobs: `20`
- Circuits per job: `100`
- Covered rows: `0..2000`
- Total hardware elapsed time: `29.981910 s`
- All job chunks finished with `final_status = DONE`

This confirms that the benchmark was not only simulated but also executed successfully on real hardware at scale.

## Interpretation of Scientific Meaning

### Does this show the quantum part is relevant?

Yes, this benchmark provides credible evidence that the quantum branch is relevant in this trained model.

The strongest reasons are:

- `local_statevector` improves over `classical_only`
- `fake_256` matches `local_statevector`
- `real_hardware` also matches `local_statevector` closely

That combination supports the claim that the learned quantum residual branch contributes something real to predictive performance and that the effect survives hardware execution.

### Does this prove a strong quantum advantage?

No, not by itself.

This benchmark supports:

- relevance of the quantum branch in this checkpoint
- faithful transfer from ideal simulation to fake backend to real hardware

It does not yet establish:

- a uniquely quantum advantage over all fair classical controls
- statistical significance over multiple retrainings
- robustness across random seeds and multiple splits
- a broad advantage across all targets

The effect size is modest, so the appropriate conclusion is careful rather than overstated.

## Recommended Conclusion

The safest high-level conclusion is:

> The 8-qubit ExoBiome hybrid checkpoint achieves better validation performance than the classical-only baseline on this 2000-row benchmark subset, and the improvement is preserved with very small degradation on real IQM hardware. This supports the practical relevance of the quantum residual branch, while the modest effect size means the result should be interpreted as evidence of contribution rather than definitive proof of a strong quantum advantage.

## Suggested Next Steps

To strengthen the claim scientifically, the next most useful experiments would be:

1. Rerun the benchmark on the full validation split instead of a 2000-row subset.
2. Repeat on the holdout split.
3. Compare against a matched classical control with similar parameter count and residual structure.
4. Repeat over multiple training seeds.
5. Quantify uncertainty with confidence intervals or significance tests on the metric differences.
