# Garnet Ariel Quantum Regression

This package ports the current Ariel hybrid regressor to IQM Resonance for IQM Garnet without submitting jobs by default.

Training is out of scope here. The model is assumed to be already trained, and the Garnet path is for validation-time quantum execution only.

## What it does

- Loads the current best checkpoint from `/Users/iwosmura/projects/hack4sages/artifacts/ariel_quantum_best_v4_epoch6`
- Reuses the frozen classical encoder and regression head
- Rebuilds the trained `8`-qubit quantum residual branch in Qiskit
- Transpiles circuits for IQM Garnet and supports dry validation on fake/facade backends
- Allows notebook-level `n_qubits` selection of `8` or `12`

## Current compatibility

- `8` qubits: full checkpoint-backed prediction path
- `12` qubits: scaffold, layout selection, transpilation, and mock preparation only
- Live submission: implemented behind `submit_to_iqm=False` by default

## Main entrypoints

- `load_default_checkpoint_bundle()`
- `prepare_garnet_run(...)`
- `select_garnet_layout(...)`
- `run_local_baseline(...)`
- `run_mock_evaluation(...)`
- `run_iqm_execution(...)`
- `prepare_validation_split_run(...)`

## Tutorial

Start with:

- `/Users/iwosmura/projects/hack4sages/models/garnet_ariel_quantum_regression/garnet_port_tutorial.ipynb`

That notebook is the single supported runtime entrypoint for preparing validation runs.

## Environment

Use:

- `/Users/iwosmura/projects/hack4sages/models/garnet_ariel_quantum_regression/setup_garnet_env.sh`
