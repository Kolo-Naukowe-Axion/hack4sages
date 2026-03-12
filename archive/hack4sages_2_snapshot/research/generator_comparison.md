# MultiREx vs petitRADTRANS: Spectrum Generator Comparison

## TL;DR

| | MultiREx | petitRADTRANS (pRT) |
|---|---|---|
| **What it is** | High-level wrapper around TauREx 3 for massive spectrum generation | Full-featured radiative transfer + retrieval framework |
| **Primary use** | Generating large training datasets for ML | Forward modeling, retrieval, and detailed atmospheric characterization |
| **Spectroscopy modes** | Transmission only | Transmission + emission + reflection + scattering |
| **Temperature profile** | Isothermal only | Guillot, spline, gradient, isothermal, Madhusudhan-Seager |
| **Cloud models** | None (TauREx has them, but MultiREx doesn't expose them) | 6 models including physical condensates (EddySed, Mie, DHS) |
| **Chemistry** | Manual (user-specified mixing ratios) | Free, equilibrium (easyCHEM), quenching, hybrid |
| **Retrieval** | No | Yes (PyMultiNest, UltraNest) |
| **Resolution modes** | Line-by-line via TauREx | Correlated-k (R~1000) and line-by-line (R~10^6) |
| **Speed per spectrum** | Seconds (TauREx backend) | Seconds (Fortran backend) |
| **Killer feature** | `explore_multiverse()` — generate millions of spectra in parallel | Most comprehensive cloud/opacity library in the field |
| **Language** | Pure Python (TauREx backend) | Python frontend + Fortran backend |
| **Best for our project** | Training data generation | Validation, detailed modeling, if we need emission/clouds |

---

## 1. What Each Tool Actually Does

### MultiREx

MultiREx (Duque-Castano et al. 2025, MNRAS 539) is a **dataset factory**. It wraps TauREx 3 (Al-Refaie et al. 2021) and reorganizes its API to enable massive parallelized generation of synthetic transmission spectra. It was built specifically for training ML classifiers to detect biosignatures.

Core workflow:
1. Define Star (T_eff, R, M) → Planet (R, M) → Atmosphere (T, pressures, gases) → System
2. Call `system.generate_spectrum(wn_grid)` for one clean spectrum
3. Call `system.generate_observations(wn_grid, snr)` for noisy observations
4. Call `system.explore_multiverse(n_universes=1000000, n_jobs=16)` for massive ensemble generation

The "multiverse" mode is the key feature: parameters specified as `(min, max)` tuples get randomly sampled, producing diverse planetary systems automatically.

**Install:** `pip install multirex` (v0.3.2, April 2025)

### petitRADTRANS

petitRADTRANS (Molliere et al. 2019, A&A 627; updated Molliere et al. 2020) is a **full radiative transfer and retrieval framework**. It computes transmission, emission, and reflection spectra with detailed physics including multiple scattering, physical cloud models, and equilibrium chemistry. It includes a built-in Bayesian retrieval module.

Current version: 3.3.2 (pRT3 since May 2024). Uses a Fortran backend for fast computation with a Python API.

**Install:** `pip install petitRADTRANS`

---

## 2. Radiative Transfer Physics

### MultiREx (via TauREx 3)

TauREx computes 1D transmission spectra line-by-line:

- **Transit depth:** Δ_λ = [π R_p² + Σ_r (1 - exp(-τ_r^λ)) · S_r] / [π R_s²]
- **Optical depth:** τ = Σ layers [P/(k_B·T) · (Σ_molecules χ_m · σ_m,λ + Σ scattering) · Δl]
- **Atmosphere discretization:** Hypsometric equation, 100 layers typical
- **Processes:** Molecular absorption (ExoMol/ExoTransmit), Rayleigh scattering, CIA (H2-H2, H2-He)
- **Stellar models:** Phoenix spectra or blackbody

**Key limitation:** MultiREx's `Atmosphere` class only exposes **isothermal T-P profiles**, even though TauREx itself supports more complex profiles.

### petitRADTRANS

pRT has two resolution modes with distinct opacity treatments:

**Correlated-k (c-k, R~1000):**
- Uses 16-point Gaussian quadrature over the cumulative opacity distribution
- Product-of-transmissions across species (no combined k-tables needed)
- ~1-2% systematic error vs line-by-line from decorrelation between species
- Wavelength range: 0.11 - 250 μm

**Line-by-line (lbl, R up to 10^6):**
- Direct opacity summation at each wavelength
- Preserves wavelength correlations
- Range: 0.3 - 28 μm

**Emission:**
- Full radiative transfer equation integration with discrete layers
- 3-point Gaussian quadrature over zenith angles
- Optional **multiple scattering** via Feautrier method (from petitCODE heritage)

**Scattering configurations:**
- Globally-averaged (factor 1/4)
- Dayside-averaged (factor 1/2)
- Non-isotropic with specified incidence angle
- Surface scattering (Lambertian) with wavelength-dependent reflectance

Both tools use opacity data in the **80-3000 K** temperature range.

---

## 3. Atmospheric Models

### Temperature Profiles

| Model | MultiREx | pRT |
|-------|----------|-----|
| Isothermal | Yes (only option) | Yes |
| Guillot (2010) radiative-convective | No | Yes |
| Spline interpolation (configurable nodes) | No | Yes |
| Gradient (Zhang et al. 2023) | No | Yes |
| Madhusudhan & Seager (2009) piecewise | No | Yes |

**Impact:** Isothermal assumption can cause **order-of-magnitude errors** in retrieved gas abundances for planets with significant temperature gradients. For terrestrial planets with thin atmospheres this may be acceptable; for gas giants it's a serious limitation.

### Chemistry

| Feature | MultiREx | pRT |
|---------|----------|-----|
| Free chemistry (manual abundances) | Yes (log10 mixing ratios) | Yes (mass fractions) |
| Equilibrium chemistry | No | Yes (easyCHEM: T 60-4000K, P 10^-8 to 10^3 bar, C/O 0.1-1.6, [Fe/H] -1 to +3) |
| Quenching / disequilibrium | No | Yes (carbon_pressure_quench) |
| Hybrid (some free, some equilibrium) | No | Yes |
| Abundance units | Volume mixing ratios | Mass fractions (conversion utilities provided) |

### Cloud and Haze Models

| Model | MultiREx | pRT |
|-------|----------|-----|
| Gray cloud deck | No* | Yes (opaque_cloud_top_pressure) |
| Scaled Rayleigh (hazes) | No* | Yes (haze_factor) |
| Power law clouds | No* | Yes (κ = κ₀·(λ/λ₀)^γ) |
| Physical condensates (EddySed) | No* | Yes (30+ species: silicates, oxides, sulfides, metals, liquids) |
| Mie theory (spherical) | No* | Yes |
| DHS (irregular shapes) | No* | Yes (porosity P=0.25) |
| Partial cloud coverage | No* | Yes (cloud_fraction parameter) |
| Arbitrary opacity functions | No | Yes |

*TauREx has some cloud models (SimpleClouds, LeeMie, BHMie, FlatMie), but the MultiREx paper explicitly excluded clouds from their dataset, and the MultiREx API doesn't directly expose cloud configuration.

---

## 4. Performance

### Speed

Both generate individual spectra in **seconds**. The real difference is in bulk generation:

- **MultiREx:** Built for bulk. `explore_multiverse()` with `n_jobs` parameter parallelizes across cores. Generated ~10M spectra for the Duque-Castano paper. This is its entire purpose.
- **pRT:** Optimized for single-spectrum speed (Fortran backend). No built-in bulk generation mode — you'd write your own loop. Retrieval chains (~10^6 evaluations) run in days on 30 cores.

### Accuracy (cross-validated)

pRT benchmarks against petitCODE: ~1% deviation, localized CO band deviations up to 4% (~40 ppm for hot Jupiters).

TauREx 3 (MultiREx's backend) has been cross-validated against pRT, petitCODE, ATMO, and Exo-REM.

Both participate in the MALBEC intercomparison (2024) with general agreement. Discrepancies mainly from linelist choices and opacity treatment methods.

**Bottom line:** Both are accurate to ~1-2%. They agree with each other within these margins.

---

## 5. Datasets and Opacity Sources

### MultiREx

- **Opacity:** ExoMol (CO2, CH4, H2O), ExoTransmit (O3, N2)
- **Download size:** ~3 GB for gases, ~2 GB for Phoenix stellar models
- **Pre-built dataset:** ~10M spectra of TRAPPIST-1e analogs with:
  - JWST NIRSpec PRISM (R=100), 0.69-5.3 μm
  - Variable CH4, O3, H2O mixing ratios (10^-8 to 10^-1)
  - 3 temperatures (200K, 287K, 400K)
  - 10 stellar contamination scenarios (POSEIDON)
  - 5 SNR levels (0, 1, 3, 6, 10)
  - N2 fill gas, fixed CO2 at 10^-2

### petitRADTRANS

- **Opacity:** ExoMol, HITRAN, HITEMP line lists + CIA pairs
- **Download:** Automatic per-species from Keeper server (pRT3). No 12GB upfront download anymore.
- **Species count:** Dozens of molecular and atomic absorbers
- **Condensate opacities:** 30+ species with real optical constants
- **No pre-built dataset** — it's a modeling tool, not a dataset generator

---

## 6. Retrieval Capabilities

| Feature | MultiREx | pRT |
|---------|----------|-----|
| Built-in retrieval | No | Yes |
| Sampling methods | N/A | PyMultiNest (fast), UltraNest (accurate evidence) |
| Pre-built emission models | N/A | 7 models |
| Pre-built transmission models | N/A | 4 models |
| High-resolution retrieval | N/A | Yes (telluric removal, time-varying spectra) |
| Bayesian model comparison | N/A | Yes (evidence from nested sampling) |

---

## 7. Practical Considerations for Our Project

### What We Need (ExoBiome)
- Generate training data: transmission spectra with biosignature labels
- JWST NIRSpec wavelength range (~0.6-5.3 μm)
- Terrestrial planet atmospheres (N2/CO2 dominated, not H2/He)
- Biosignature molecules: CH4, O3, H2O (possibly O2)
- Feed into QELM classifier

### MultiREx Advantages for Us
1. **Purpose-built for our exact use case** — generating labeled ML training data
2. **`explore_multiverse()` out of the box** — no custom parallelization code needed
3. **Pre-existing biosignature dataset** for TRAPPIST-1e that we can use directly or extend
4. **Simple API** — Star + Planet + Atmosphere → spectra in 5 lines of code
5. **Noise model included** — SNR-based Gaussian noise addition built in
6. **Already pip-installable** and lightweight
7. **The Duque-Castano paper** is directly relevant — they did ML biosignature classification with Random Forest on MultiREx data

### MultiREx Disadvantages for Us
1. **Isothermal only** — no temperature gradients, which matters for atmospheric retrieval accuracy
2. **No clouds** — real exoplanet spectra will have clouds/hazes that affect transmission
3. **Transmission only** — can't do emission if we ever want to expand
4. **Limited molecule set** easily accessible — adding new species requires digging into TauREx
5. **Parameters assumed independent** — unrealistic for actual planetary systems
6. **Simplistic noise model** — constant Gaussian across all wavelengths (real JWST noise is wavelength-dependent)

### petitRADTRANS Advantages for Us
1. **More physically realistic** — T-P profiles, clouds, equilibrium chemistry
2. **Cloud models** — critical for realistic training data (clouds mask biosignatures)
3. **Equilibrium chemistry** — self-consistent abundance profiles instead of arbitrary mixing ratios
4. **Emission + reflection** — could model thermal emission biosignatures (e.g., vegetation red edge)
5. **Retrieval module** — if we want to do atmospheric retrieval alongside classification
6. **Huge condensate library** — 30+ cloud species with real optical constants
7. **High-resolution mode** — R up to 10^6 if needed
8. **Very actively maintained** (pRT3, large team at MPIA)

### petitRADTRANS Disadvantages for Us
1. **No bulk generation mode** — we'd have to write our own parallelization
2. **Steeper learning curve** — more parameters, more complex API
3. **Fortran compilation** — can be painful on macOS
4. **Mass fractions not mixing ratios** — less intuitive for atmospheric composition specification
5. **Designed for H2-dominated atmospheres** — equilibrium chemistry tables optimized for gas giants, not terrestrial N2/CO2 atmospheres
6. **Larger download/dependency footprint**
7. **No built-in labeling for ML** — we'd need to build the dataset pipeline from scratch

---

## 8. Recommendation

**Use MultiREx as the primary generator** for training data. Reasons:

1. It was literally built for this — generating labeled biosignature training datasets
2. The existing TRAPPIST-1e dataset is directly usable as a starting point
3. `explore_multiverse()` handles parallelization, noise, and labeling
4. The Duque-Castano paper validates this exact approach (ML on MultiREx spectra → biosignature classification)
5. For a 3-day hackathon, the simpler API means faster iteration

**Consider petitRADTRANS for:**
- Validation — generate a small set of "gold standard" spectra with realistic T-P profiles and clouds to test if our classifier generalizes
- If we find the isothermal/no-clouds limitation is killing classifier performance on real JWST data
- Future work beyond the hackathon where physical realism matters more

**Hybrid approach (if time permits):**
1. Train on MultiREx bulk data (fast, labeled, millions of spectra)
2. Validate/fine-tune on a smaller pRT-generated dataset with clouds and temperature gradients
3. Test on real JWST observations (K2-18b, WASP-39b)

---

## 9. Key References

- **MultiREx:** Duque-Castano, Zuluaga & Flor-Torres (2025), MNRAS 539, 1528-1552. [arXiv:2407.19167](https://arxiv.org/abs/2407.19167)
- **TauREx 3:** Al-Refaie et al. (2021), ApJ 917, 37. [arXiv:1912.07759](https://arxiv.org/abs/1912.07759)
- **pRT original:** Molliere et al. (2019), A&A 627, A67. [arXiv:1904.11504](https://arxiv.org/abs/1904.11504)
- **pRT scattering:** Molliere et al. (2020), A&A 640, A131. [arXiv:2006.09394](https://arxiv.org/abs/2006.09394)
- **pRT retrieval:** Nasedkin et al. (2024), JOSS 9(96), 5875
- **pRT3 SpectralModel:** Blain et al. (2024), JOSS 9(101), 7028
- **MALBEC intercomparison:** [arXiv:2402.04329](https://arxiv.org/abs/2402.04329)
