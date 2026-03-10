# ExoBiome

Quantum biosignature detection in exoplanet atmospheres. Built for HACK-4-SAGES 2026 (ETH Zurich COPL).

## What it does

ExoBiome uses quantum extreme learning machines (QELM) running on real quantum hardware to classify whether an exoplanet's transmission spectrum contains biosignature patterns. Three models are compared: two quantum (QELM Vetrano, QELM Extended) and one classical baseline (Random Forest).

## Branches

| Branch | Description |
|--------|-------------|
| `main` | Project root + EDA |
| `feat/web` | Web frontend (Next.js) — two design variants |
| `eda/ariel-challenge` | EDA for Ariel Data Challenge 2023 |

## EDA

```bash
cd eda
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter lab
# Open ariel_eda.ipynb
```

## Web App

See the [`feat/web`](../../tree/feat/web) branch for the interactive frontend with setup instructions.

## Research

- [Vetrano et al. 2025 — QELM for atmospheric retrieval](https://arxiv.org/abs/2509.03617)
- [Cardenas et al. 2025 — MultiREx dataset](https://doi.org/10.1093/mnras/stae2948)

## Team

Built by the ExoBiome team (Koło Naukowe Axion) at HACK-4-SAGES 2026.
