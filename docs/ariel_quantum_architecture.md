# Ariel Quantum Regressor Architecture

This document describes the model that is actually training now in `/Users/iwosmura/projects/hack4sages/models/ariel_quantum_regression/`.

## End-to-end view

```mermaid
flowchart LR
    subgraph P["1. Data Preparation"]
        R["Raw per-planet inputs<br/>52 bins x 4 raw arrays<br/>+ 8 aux features"]
        P1["Keep sample-varying spectra<br/>instrument_spectrum<br/>instrument_noise"]
        P2["Per-sample normalization<br/>divide both channels by each sample's<br/>mean instrument_spectrum"]
        P3["Train-split z-score<br/>only on sample-varying channels"]
        P4["Append fixed templates<br/>normalized instrument_width<br/>normalized wavelength grid"]
        A1["Aux preprocessing<br/>log10 on scale-like columns<br/>train-split standardization"]
        R --> P1 --> P2 --> P3 --> P4
        R --> A1
    end

    subgraph M["2. Model"]
        X["Spectral tensor<br/>4 x 52"]
        S0["Stem<br/>Conv1d 4 -> 32, k=5<br/>GroupNorm + GELU"]
        S1["Residual block<br/>32 -> 32"]
        S2["Residual block<br/>32 -> 64"]
        S3["Residual block<br/>64 -> 96"]
        SP["Dual pooling<br/>global mean + attention pool"]
        SE["Spectral embedding<br/>192 -> 96"]

        AUX["Aux vector<br/>8"]
        AE["Aux encoder<br/>8 -> 32 -> 32"]

        F["Fusion encoder<br/>[spectral, aux] = 128 -> 128"]
        HC["Head context<br/>concat(fused 128,<br/>spectral 96, aux 32)<br/>= 256"]

        CH["Classical head<br/>256 -> 192 -> 5"]

        QP["Quantum projector<br/>128 -> 128 -> 8<br/>LayerNorm + tanh(pi)"]
        QB["Quantum block<br/>8 qubits, depth 2<br/>RY / CNOT / RZ / CRX<br/>returns 8 expectation values"]
        QH["Quantum correction head<br/>[head_context, qfeat] = 264<br/>264 -> 192 -> 5"]
        G["Per-target quantum gate<br/>tanh(gate[5]) * quantum_scale"]
        OUT["Final 5-gas prediction<br/>log_H2O, log_CO2, log_CO,<br/>log_CH4, log_NH3"]

        X --> S0 --> S1 --> S2 --> S3 --> SP --> SE
        AUX --> AE
        SE --> F
        AE --> F
        F --> HC
        SE --> HC
        AE --> HC
        HC --> CH --> OUT
        F --> QP --> QB --> QH
        HC --> QH
        QH --> G --> OUT
    end

    subgraph T["3. Training Schedule"]
        T1["Stage 1<br/>classical-only pretrain"]
        T2["Load best stage-1 checkpoint"]
        T3["Stage 2<br/>hybrid fine-tune"]
        T4["Warmup<br/>quantum disabled"]
        T5["Ramp<br/>quantum_scale increases gradually"]
        T6["Early stopping on<br/>validation RMSE mean"]
        T1 --> T2 --> T3 --> T4 --> T5 --> T6
    end

    P4 --> X
    A1 --> AUX
```

## Prediction equation

The current model predicts:

```text
head_context = concat(fused, spectral_feat, aux_feat)
classical_pred = ClassicalHead(head_context)
quantum_angles = Projector(fused)
quantum_features = QuantumBlock(quantum_angles)
quantum_correction = QuantumHead(concat(head_context, quantum_features))
gate = tanh(quantum_gate[5])

final_pred = classical_pred + quantum_scale * gate * quantum_correction
```

That is different from the earlier version in two important ways:

1. The prediction head now gets direct skip-connected encoder summaries instead of relying only on the fused latent.
2. The quantum branch is a gated residual correction, not the main information bottleneck.
3. The gate is now per-target, so each gas can learn a different quantum residual strength.

## Current implementation details

- Spectral input channels are:
  - `instrument_spectrum`
  - `instrument_noise`
  - `instrument_width_template`
  - `wavelength_um`
- Only the first two channels vary by sample.
- The width and wavelength channels are fixed normalized templates shared across all samples.
- Auxiliary input has `8` features. `7` scale-like columns get `log10`; `star_temperature` stays linear.
- Targets are the `5` log-abundance values:
  - `log_H2O`
  - `log_CO2`
  - `log_CO`
  - `log_CH4`
  - `log_NH3`
- Splits are deterministic `80/10/10` train/validation/holdout.
- Primary stratification uses gas-presence signatures from `log_* >= -8.0`.
- Fallback stratification uses coarse abundance bins when signatures are too sparse.
- Loss is `MSE`; model selection is by validation RMSE mean in original target units.
- RMSE is computed once per epoch, not per batch.
- Training artifacts are saved incrementally:
  - `history.csv`
  - `training_state.json`
  - `best_model.pt`
  - `last_model.pt`
  - `validation_metrics.json`
  - `holdout_metrics.json`
  - `validation_predictions.csv`
  - `holdout_predictions.csv`
  - `testdata_predictions.csv`

## What changed from the stagnating version

- Added per-sample spectral normalization before train-split standardization.
- Stopped treating `instrument_width` as a learned sample-varying channel.
- Injected fixed `instrument_width` and wavelength templates explicitly.
- Replaced plain global averaging with mean pooling plus attention pooling.
- Added direct skip connections from `spectral_feat` and `aux_feat` into both prediction heads.
- Added `LayerNorm` in the quantum projector before `tanh(pi * x)`.
- Increased quantum initialization scale from the nearly-dead tiny-init regime.
- Switched to two-stage training instead of co-training everything from scratch.
- Added per-target quantum gates instead of one shared scalar gate.
- Split optimization into backbone, quantum-adapter, and quantum-circuit parameter groups.
- Disabled async `lightning.gpu` by default to avoid host-memory blowups on the laptop GPU run.
- Added gradual ramp and temporary backbone freezing during hybrid fine-tuning.
- Aligned training loss with the metric we care about by using `MSE` and selecting by RMSE.

## Latest verified run snapshot

As of `2026-03-11`, the best confirmed hybrid checkpoint is:

- Output root:
  - `/home/iwo/hack4sages-crossgen/outputs/ariel_quantum_stage2_restart_v4_20260311_185231`
- Best epoch:
  - `6`
- Best training-phase validation RMSE mean recorded during training:
  - `0.29081112146377563`
- Post-stop validation RMSE mean from re-evaluating `best_model.pt`:
  - `0.29361358284950256`
- Holdout RMSE mean from re-evaluating `best_model.pt`:
  - `0.2993761897087097`
- Behavior after epoch 6:
  - validation worsened after the backbone unfroze, so the epoch-6 checkpoint was kept and training was stopped

This stage-2 hybrid checkpoint is the best confirmed model at the moment this file was updated.
