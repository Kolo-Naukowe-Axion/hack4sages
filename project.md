# ExoBiome - Quantum Biosignature Detection

## What is this?

ExoBiome is a web application that detects biosignatures in exoplanet atmospheres using quantum machine learning. Given a transmission spectrum of an exoplanet, it classifies whether the atmosphere contains signs of biological activity (biosignature: yes/no + type).

This is the **first ever application of quantum ML to biosignature detection**.

## The Science (short version)

When an exoplanet transits its star, starlight passes through its atmosphere. Different gases absorb different wavelengths. The resulting transmission spectrum is a fingerprint of atmospheric composition. Certain gas combinations (e.g. CH4 + O3 together) strongly suggest biological origin - these are biosignatures.

We train ML models on synthetic spectra (generated via MultiREx/TauREx - physics-based simulators using real lab-measured gas opacities) and then validate on real JWST telescope data.

## Core Approach: 3 Models Compared

We build and compare 3 models on the same dataset to provide honest benchmarking:

### Model 1: QELM (Vetrano Architecture)
- Reproduces the architecture from Vetrano et al. 2025 (arXiv:2509.03617)
- Original paper did atmospheric retrieval (regression) on hot Jupiters
- We adapt it to biosignature classification on rocky habitable planets
- Architecture: RX encoding → RY + CNOT + RY reservoir (depth 3) → Z-measurement → linear output
- 5 qubits per sub-reservoir, spectrum split into patches via PCA

### Model 2: QELM (Our Extended Topology)
- Modified quantum circuit topology (different entanglement pattern, adjusted depth)
- Explores whether circuit design impacts classification performance
- Same preprocessing pipeline, different reservoir structure
- Runs on real quantum hardware: Odra 5 (IQM Spark, 5 qubits) and VTT Q50 (53 qubits)

### Model 3: Best Classical Model
- The strongest classical baseline we can build (Random Forest, XGBoost, or CNN)
- Longest training time, most optimized hyperparameters
- Serves as the "gold standard" to compare quantum models against
- Honest comparison: even matching the classical model is a success for 5-qubit quantum

## Dataset

- **Training**: MultiREx (synthetic transmission spectra of TRAPPIST-1e-like rocky planets)
- **Gases**: CH4, CO2, H2O, O3 at varying concentrations
- **Labels**: biosignature presence/absence based on gas combination rules
- **Noise**: realistic JWST shot noise added to simulate real observations
- **Validation**: real JWST spectrum of K2-18b (controversial DMS/CH4 detection)

## Hardware

| Machine | Qubits | Location | SDK |
|---------|--------|----------|-----|
| Odra 5 (IQM Spark) | 5 | PWR Wroclaw | qiskit-on-iqm |
| VTT Q50 | 53 | Finland (remote) | qiskit-on-iqm |
| GPU cluster | - | Cloud | PyTorch/sklearn |

---

## Web Application Design

Dark space-themed aesthetic throughout. Navigation bar with 3 main pages.

---

### Page 1: Landing Page

Hero section with exoplanet transit visual. Short, accessible explanation:
- What are exoplanets
- What is a transmission spectrum
- What are biosignatures
- What this app does

Call-to-action button → navigates to Planet Explorer.

---

### Page 2: Planet Explorer + Detection

This is the core interactive page. It has a clear top-to-bottom flow:

**Section A: Interactive Timeline**

Horizontal scrollable timeline of **confirmed rocky exoplanets**, based on real NASA data, ordered by discovery date. Each planet is a clickable node showing:
- Name (e.g. TRAPPIST-1e, K2-18b, LHS 1140b, Proxima Centauri b)
- Discovery year
- Key properties: mass, radius, equilibrium temperature, orbital period
- Habitability zone status
- Whether JWST has observed it

User **clicks a planet** to select it. The selected planet highlights and its data loads below.

**Section B: Selected Planet Summary**

After clicking a planet, a panel appears/expands showing:
- Planet name + star system
- Physical parameters (mass, radius, temperature, distance)
- Spectrum preview (if available from JWST, or synthetic from MultiREx)
- Status badge: "Real JWST data available" or "Synthetic spectrum (simulated)"

*Future extension: the models will receive this planet's actual data (real JWST spectrum or generated spectrum from its known parameters via MultiREx/PSG) as input.*

**Section C: Run All Models**

A single **"Analyze Biosignatures"** submit button. When clicked, all 3 models process the planet's data simultaneously.

**Section D: Results — 3 Columns**

Results appear as **3 side-by-side columns**, one per model:

| QELM (Vetrano) | QELM (Extended) | Classical (Best) |
|---|---|---|
| Model type badge | Model type badge | Model type badge |
| Biosignature: YES/NO/UNCERTAIN | Biosignature: YES/NO/UNCERTAIN | Biosignature: YES/NO/UNCERTAIN |
| Confidence: 87% | Confidence: 91% | Confidence: 94% |
| Detected gases: CH4, O3 | Detected gases: CH4, O3, H2O | Detected gases: CH4, O3 |
| Spectrum plot with highlights | Spectrum plot with highlights | Spectrum plot with highlights |
| Processing time: 2.3s | Processing time: 1.8s | Processing time: 0.1s |

Each column is a formatted card with consistent layout for easy comparison.

**Section E: Bridge Panel → Models Page**

Below the results, a connector panel:
- "Want to understand how these models work and why they give different answers?"
- Brief one-liner per model explaining the core difference
- **"Explore Models →"** button linking to the Models page
- Visual hint showing this connects to the next page

---

### Page 3: Models

Detailed description of each model. Three expandable sections/cards:

**Model 1: QELM — Vetrano Architecture**
- What it is: Quantum Extreme Learning Machine reproducing Vetrano et al. 2025
- How it works: angle encoding → random quantum reservoir → measurement → linear output
- Architecture diagram (circuit visual)
- Training method: SVD (one-shot, no gradient descent)
- Stats (placeholder for now):
  - Training time
  - Accuracy / Precision / Recall
  - Qubits used
  - Hardware: simulator / Odra 5 / VTT Q50

**Model 2: QELM — Extended Topology**
- What it is: our modified QELM with different circuit topology
- How it differs: different entanglement pattern, adjusted depth, different reservoir structure
- Architecture diagram
- Same stats format as Model 1

**Model 3: Classical Baseline**
- What it is: best classical model (Random Forest / XGBoost / CNN)
- How it works: standard ML on same PCA-compressed features
- Training method: hyperparameter optimization, cross-validation
- Same stats format

**Comparison Table** at the bottom:

| Metric | QELM Vetrano | QELM Extended | Classical |
|--------|-------------|---------------|-----------|
| Accuracy | - | - | - |
| Precision | - | - | - |
| Recall | - | - | - |
| Training time | - | - | - |
| Inference time | - | - | - |
| Hardware | - | - | - |

*All stats are placeholder — will be filled after model training.*

**About Section** at the bottom of this page:
- Brief project motivation
- Link to Vetrano et al. 2025 paper
- Team info
- Hackathon context (HACK-4-SAGES, ETH Zurich, March 2026)
- Tech stack summary

---

## Tech Stack

**Backend**: Python (FastAPI or Flask)
**Frontend**: React / Next.js (or static HTML/JS for prototype)
**ML/Quantum**: qiskit, qiskit-on-iqm, sQUlearn, scikit-learn, multirex
**Data**: MultiREx (generated), JWST MAST Archive (real spectra)
**Visualization**: matplotlib (backend plots), D3.js or Chart.js (frontend)

## Current Status

- [x] Research complete (paper analysis, dataset selection, novelty verification)
- [x] Architecture defined (3-model comparison approach)
- [ ] Web app frontend prototype (sketch/mockup)
- [ ] MultiREx data generation pipeline
- [ ] PCA preprocessing
- [ ] QELM implementation (Vetrano reproduction)
- [ ] QELM implementation (extended topology)
- [ ] Classical baseline training
- [ ] Real JWST data validation (K2-18b)
- [ ] Model integration into web app
- [ ] Final presentation

## Hackathon

**HACK-4-SAGES** (March 9-13, 2026) organized by ETH Zurich (COPL)
Category: "Life Detection and Biosignatures"
Team: 4 people, 3 working days
Prize: Trip to ETH Zurich + presentation at Origins Federation Conference
