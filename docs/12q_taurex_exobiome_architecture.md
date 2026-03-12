# 12q TauREx ExoBiome

This document describes the prepared 12-qubit TauREx variant in `/Users/iwosmura/projects/hack4sages/models/12q_taurex_exobiome/`.

## Summary

- It is a clean copy of `ariel_quantum_regression` with its own package directory, entrypoints, helper scripts, and output roots.
- The quantum width default is `12` qubits instead of `8`.
- The TauREx two-stage helper script targets the new package and passes `--qnn-qubits 12` during stage 2.
- External entrypoints load the package through `importlib` or direct script paths because `12q_taurex_exobiome` is not a valid dotted-import identifier in normal Python syntax.
- No training was started while preparing this variant.

## Architecture delta vs. `ariel_quantum_regression`

- Quantum projector output changes from `8` to `12`.
- Quantum block changes from `8` qubits to `12` qubits.
- Quantum correction head input width changes from `264` to `268`.
- Default output root changes to `outputs/12q_taurex_exobiome`.

## Main entrypoints

- Training CLI:
  - `/Users/iwosmura/projects/hack4sages/models/12q_taurex_exobiome/run_12q_taurex_exobiome.py`
- TauREx two-stage Ubuntu GPU helper:
  - `/Users/iwosmura/projects/hack4sages/models/12q_taurex_exobiome/run_12q_taurex_exobiome_taurex_ubuntu_gpu.sh`
- Remote sync helper:
  - `/Users/iwosmura/projects/hack4sages/models/12q_taurex_exobiome/sync_12q_taurex_exobiome_remote.sh`
- Remote launch helper:
  - `/Users/iwosmura/projects/hack4sages/models/12q_taurex_exobiome/launch_12q_taurex_exobiome_remote.sh`
