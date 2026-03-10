# Ariel Data Challenge - Dataset Research

Source: https://www.ariel-datachallenge.space/ML/documentation/data

## ARIEL Mission Context

ARIEL (Atmospheric Remote-sensing Infrared Exoplanet Large-survey) is an ESA medium-class mission (~500M EUR), scheduled for 2029 launch to L2. Goal: survey atmospheres of ~1000 nearby exoplanets via transit spectroscopy.

The challenge focuses on **inverse modeling** — mapping observed atmospheric signatures to planetary characteristics (water content, temperature, cloud properties, gas abundances).

### Science: Transit Spectroscopy

- Planet transits cause measurable stellar brightness dips (area ratio planet/star)
- Jupiter-sized planet orbiting sun-like star → ~1% brightness dip
- Stellar light filtering through atmosphere creates spectral fingerprints revealing molecular composition, temperature, cloud coverage
- Transit depth shifts across wavelengths based on atmospheric opacity → identifies absorbing species

### Data Generation

Generated using **Alfnoor** software combining:
- **TauREx 3** — open-source atmospheric modeling suite
- **ArielRad** — official Ariel instrument simulator

Produces large-scale atmosphere simulations reflecting realistic Ariel performance characteristics.

---

## Dataset Overview

### Our actual dataset (from ariel-datachallenge.space, 2023 edition)
- **Train**: 41,423 planets, **Test**: 685 planets
- **Quartiles + FM params**: all 41,423 train planets
- **Full posteriors (tracedata)**: only 6,766 / 41,423 (16.3%) — rest are empty/NaN
- planet_ID format: strings like "train1", "test1"
- HDF5 keys: `Planet_{planet_ID}` → `Planet_train1`
- Tracedata subkeys: lowercase `tracedata`, `weights`

---

## Data Files

### 1. AuxillaryTable.csv

Metadata for all 91,392 planetary systems. Indexed by `planet_ID`.

**9 features:**

| Feature | Description |
|---------|-------------|
| Star Distance | Distance to host star |
| Stellar Mass | Mass of host star |
| Stellar Radius | Radius of host star |
| Stellar Temperature | Temperature of host star |
| Planet Mass | Mass of planet |
| Orbital Period | Orbital period |
| Semi-Major Axis | Semi-major axis of orbit |
| Planet Radius | Radius of planet |
| Surface Gravity | Surface gravity of planet |

```python
import pandas as pd
df = pd.read_csv('AuxillaryTable.csv', index_col='planet_ID')
```

### 2. SpectralData.hdf5

Hierarchical HDF5 structure organized by planet ID. Each planet contains:

| Key | Description |
|-----|-------------|
| `instrument_wlgrid` | Wavelength grid values |
| `instrument_spectrum` | Observed spectrum (transit depth) |
| `instrument_noise` | Measurement uncertainty |
| `instrument_width` | Wavelength bin width |

**Dimensions**: (num_planets, 52, 4) — 52 spectral bins × 4 channels [wlgrid, spectrum, noise, wlwidth]

Each spectrum point: intensity measurement (transit depth), associated wavelength, wavelength bin size (spectral resolution), and measurement uncertainty.

### 3. Tracedata.hdf5 (Ground Truth — Regular Track)

Posterior probability distributions from **MultiNest Nested Sampling algorithm**.

Per planet:
- `Tracedata` array — likelihood-evaluated points (L)
- `Weights` array — importance weights (w)

**Posterior calculation:**
```
p_i = (L_i × w_i) / Σ(L_j × w_j)
```

- Variable array lengths per planet (minimum 1,500 data points)
- Available for ~24% of instances
- Nested by planet ID

### 4. QuartilesTable.csv (Ground Truth — Light Track)

Percentile summaries for ~25% of instances.

**Columns:** planet_ID, then for each parameter: `Parameter_q1`, `Parameter_q2`, `Parameter_q3`

| Percentile | Meaning |
|-----------|---------|
| q1 | 16th percentile |
| q2 | 50th percentile (median) |
| q3 | 84th percentile |

### 5. FM_Parameter_Table.csv (Supplementary)

High-resolution forward model input parameters. Available for ALL instances. **Not ground truth** — reference data only.

---

## Prediction Targets

**7 atmospheric properties** to predict:

| Parameter | Description |
|-----------|-------------|
| Planet Radius | Radius of the planet |
| T | Temperature |
| log_H2O | Log-abundance of water |
| log_CO2 | Log-abundance of carbon dioxide |
| log_CH4 | Log-abundance of methane |
| log_CO | Log-abundance of carbon monoxide |
| log_NH3 | Log-abundance of ammonia |

**Task**: Estimate the **conditional joint distribution (Bayesian posterior distribution)** of these 7 targets. Only distribution traces required, not analytical forms.

---

## Two Competition Tracks

### Regular Track
- Predict full posterior distributions matching `Tracedata.hdf5` structure
- Must provide tracedata and weights arrays per planet

### Light Track
- Predict quartiles (q1, q2, q3) for each of the 6 atmospheric parameters
- Simpler output format via `QuartilesTable.csv`

---

## Scoring System

### Final Score Formula

```
final = 0.2 × Score_Spectral + 0.8 × Score_Posterior
```

Range: 0 (minimum) to 1000 (maximum). Heavily weighted toward posterior accuracy.

### Posterior Score (80% weight)

- Two-sample **Kolmogorov-Smirnov test** (`scipy.stats.ks_2samp`)
- Tests whether predicted and ground truth samples come from same continuous distribution
- Normalized to 0-1000 scale (1000 = maximum similarity)
- Evaluates univariate distributions (diagonal of corner plot)

### Spectral Score (20% weight)

Compares:
- Median spectra (median value per wavelength bin)
- Uncertainty bounds (interquartile range per bin)

Differences quantified using **inverse Huber loss**. Combines median spectrum differences with bounds differences linearly. Max score: 1000.

The spectral score enforces physical laws by capturing covariance relationships between targets (off-diagonal elements of corner plot).

---

## Submission Requirements

- Predictions must match `Tracedata.hdf5` structure with tracedata and weights arrays per planet
- Helper function for format conversion available on GitHub repository
- Top 10 leaderboard participants run code on additional planet dataset

---

## Allowed Techniques

No algorithmic restrictions. Competitors may use:
- Data augmentation
- Pretrained models
- Domain knowledge
- Custom train/validation splits

---

## Key Resources

- Zenodo repository: Ariel Big Challenge Database
- Tutorials and Jupyter notebooks for TauREx 3 usage
- Published paper describing training dataset structure
- GitHub helper functions for submission formatting
