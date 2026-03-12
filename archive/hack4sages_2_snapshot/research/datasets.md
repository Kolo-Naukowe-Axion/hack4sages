# Exoplanet datasets for biosignature detection and quantum ML at HACK-4-SAGES

**The most actionable datasets for a quantum ML biosignature project are the MultiREx binary classification spectra (~10⁷ synthetic spectra with biosignature labels), the INARA dataset (3.1M rocky planet spectra on NASA Exoplanet Archive), and the ML4SCI/POSEIDON dataset (~98K spectra already tested with quantum circuits).** These three provide ready-made classification tasks perfectly suited for variational quantum classifiers or quantum kernel methods on ODRA 5's 5-qubit IQM Spark architecture. The field of quantum ML for exoplanet atmospheres is brand new — only one paper (Vetrano et al. 2025) has demonstrated quantum atmospheric retrieval on real hardware — making this a frontier opportunity for your hackathon team. Below is every relevant dataset, tool, and QML resource identified, organized by category.

---

## 1. NASA Exoplanet Archive: the foundational catalog

**Source:** Caltech/IPAC, under NASA's Exoplanet Exploration Program  
**URL:** https://exoplanetarchive.ipac.caltech.edu/  
**Current count:** **~6,128 confirmed exoplanets** (as of February 2026)  
**Access:** TAP/ADQL queries, HTTP API, web UI, Python via `astroquery.ipac.nexsci.nasa_exoplanet_archive`  
**Formats:** CSV, VOTable, JSON, FITS

### Planetary Systems table (ps) and Composite Parameters (pscomppars)

The `ps` table has one row per planet per literature reference (~29,683 rows); `pscomppars` provides one best-estimate row per planet (~6,128 rows). Both share **~546 columns** spanning:

**Planet parameters critical for habitability:**
- `pl_rade` / `pl_radj` — radius in Earth/Jupiter radii (rocky vs. gaseous classification)
- `pl_bmasse` / `pl_bmassj` — best mass estimate in Earth/Jupiter masses
- `pl_dens` — bulk density in g/cm³ (composition indicator)
- **`pl_insol`** — insolation flux in Earth units (**directly determines habitable zone membership**)
- **`pl_eqt`** — equilibrium temperature in K (**key habitability metric**)
- `pl_orbper`, `pl_orbsmax`, `pl_orbeccen` — orbital period (days), semi-major axis (AU), eccentricity

**Stellar parameters for HZ computation:**
- **`st_teff`** — effective temperature (K), determines HZ location and UV environment
- **`st_lum`** — luminosity in log(Solar), directly sets HZ boundaries
- **`st_rad`**, **`st_mass`** — radius and mass in Solar units
- **`st_met`** — metallicity [Fe/H] or [M/H], indicates system chemistry
- **`st_age`** — stellar age in Gyr (time available for life to evolve)

**Discovery and detection flags:** `discoverymethod`, `disc_year`, `disc_facility`, `tran_flag`, `rv_flag`, plus photometry across V, K, Gaia, 2MASS, WISE, TESS, and Kepler bands.

**Important note:** The archive has **no habitable zone flag column**. You must compute HZ membership yourself using `pl_insol`, `st_teff`, `st_lum`, and models like Kopparapu et al. (2013/2014).

### Atmospheric Spectroscopy table (spectra)

**TAP name:** `spectra` (replaced retired `transitspec`/`emissionspec` in 2023)  
**DOI:** 10.26133/NEA36  
**Content:** All published transmission, emission, and direct imaging spectra from peer-reviewed literature, including JWST data (K2-18 b CH₄/CO₂, TRAPPIST-1 planets, WASP-39b). Growing rapidly with ongoing JWST observations. This is the single most comprehensive repository of real observed exoplanet atmospheric spectra.

### Contributed ML-ready datasets hosted at the Archive

Two datasets from NASA's Frontier Development Lab are hosted directly:

- **INARA (Intelligent exoplaNet Atmospheric RetrievAl):** **3,112,620 synthetic rocky exoplanet spectra** generated with NASA's Planetary Spectrum Generator (PSG). Spectra have **4,379 wavelength columns** covering planets in the optimistic habitable zone around F, G, K, M dwarfs. Parameters include radius, temperature, and abundances of **12 biosignature-relevant gases (O₂, O₃, CH₄, H₂O, CO₂, and more)**. DOI: 10.26133/NEA42. Format: CSV/NPZ with web interface and bulk download.

- **PyATMOS:** **~124,314 simulated 1D Earth-like atmospheres** varying biologically mediated gas concentrations (O₂, CO₂, H₂O, CH₄, H₂, N₂). Generated using the VPL ATMOS photochemistry-climate model. Each model provides temperature, pressure, gas concentration, and gas fluxes as functions of altitude (0–80 km, 100 steps). Format: CSV + numpy (.npz), ~110 GB uncompressed.

---

## 2. Habitable zone catalogs and Earth Similarity Index datasets

### Habitable Worlds Catalog (HWC)

**Source:** Planetary Habitability Laboratory (PHL), University of Puerto Rico at Arecibo  
**URL:** https://phl.upr.edu/hwc  
**Size:** ~70 potentially habitable worlds (**29 conservative**, 0.5–1.6 R⊕; **41 optimistic**, 1.6–2.5 R⊕)  
**Key columns:** Planet name, PHL classification type (stellar type + HZ location + size class), mass (M⊕), radius (R⊕), stellar flux (Earth units), estimated surface temperature (K), orbital period, distance (ly), stellar age (Gyr), **Earth Similarity Index (ESI, 0–1)**

The downloadable **PHL Exoplanets Catalog (PHL-EC)** CSV (`phl_exoplanet_catalog.csv`) contains **60+ parameters for ALL confirmed exoplanets** including ESI, Habitable Zone Distance (HZD), and habitability classifications unavailable elsewhere. This is available at https://phl.upr.edu/projects/habitable-exoplanets-catalog/hec-data-of-potentially-habitable-worlds/phls-exoplanets-catalog. The ESI metric (geometric mean of interior and exterior similarity to Earth) provides a **ready-made continuous label for regression** or a binary threshold (ESI ≥ 0.8 = "Earth-like") for classification.

### Other HZ resources

- **Habitable Zone Gallery** (http://www.hzgallery.org/) — percentage of orbital phase spent in HZ for all known planets with orbital solutions
- **Hill et al. 2023 HZ Catalog** (AJ 165, 34; arXiv: 2210.02484) — comprehensive catalog of all planets in the HZ, available via VizieR
- **HabCat** (https://phl.upr.edu/projects/habcat) — 17,129 candidate stars suitable for harboring habitable planets
- **El-Kholy et al. 2025 Active Learning dataset** (https://github.com/rehamelkholy/ExoplanetAL) — unified HWC + NASA Archive dataset with **binary habitability labels** (potentially habitable / not habitable), designed for gradient-boosted decision trees with active learning

---

## 3. Atmospheric spectroscopy from space telescopes

### JWST data at MAST

**URL:** https://archive.stsci.edu/missions-and-data/jwst  
**Format:** FITS (multiple extensions); reduced spectra in published supplementary CSV/ASCII  
**Access:** MAST Portal, astroquery, ExoMAST interface, AWS cloud access

The **Transiting Exoplanet Community ERS Program** (Program 1366) provides the benchmark: transmission and emission spectra of **WASP-39b** across all four JWST instruments (NIRISS/SOSS 0.6–2.8 µm, NIRCam 2.4–5.0 µm, NIRSpec PRISM 0.6–5 µm, NIRSpec G395H 2.7–5.2 µm, MIRI LRS 5–12 µm). Molecules detected include **CO₂** (first unambiguous exoplanet detection), **H₂O, CO, SO₂** (first detection, photochemical), Na, K. Zero proprietary period — all data public.

Additional JWST atmospheric results include LHS 475 b (Earth-sized), K2-18 b (Hycean candidate with CH₄/CO₂), LHS 1140 b (habitable-zone super-Earth), and the **Rocky Worlds DDT Program** HLSPs at https://archive.stsci.edu/hlsp/rocky-worlds.

### HST compiled catalogs

- **Changeat et al. 2022 / Edwards et al. 2023** (https://quentchangeat.github.io/datasets/) — **70 transit + 25 eclipse HST/WFC3 spectra** uniformly reduced with Iraclis pipeline, with population-level retrievals via Alfnoor
- **David Sing's Spectral Library** (https://pages.jh.edu/dsing3/David_Sing/Spectral_Library.html) — published transmission/emission spectra and models for benchmark hot Jupiters (CSV format)

### Spitzer Heritage Archive

**URL:** https://irsa.ipac.caltech.edu/Missions/spitzer.html  
**Content:** Transit/eclipse photometry at 3.6 and 4.5 µm (overlapping CH₄ and CO₂ absorption bands); IRS spectroscopy 5.2–38 µm. ~42 million sources in SEIP catalog. Format: FITS.

---

## 4. Molecular line databases for atmospheric modeling

### HITRAN

**URL:** https://hitran.org/  
**Source:** Center for Astrophysics, Harvard & Smithsonian  
**Current edition:** HITRAN2024 (61 molecules, 156 isotopologues, **>7 million spectral lines**)

HITRAN is the foundational spectroscopic database for atmospheric retrieval codes. Each transition record contains: wavenumber (cm⁻¹), line intensity at 296 K, Einstein A-coefficient, air/self-broadened half-widths, lower-state energy, temperature dependence, pressure shift, quantum numbers. The database also provides **IR cross-sections for 644+ molecular species** and collision-induced absorption data.

Key biosignature molecules: H₂O (ID 1), CO₂ (2), **O₃** (3), **N₂O** (4), CO (5), **CH₄** (6), **O₂** (7), NH₃ (11), **PH₃** (28), SO₂ (9).

**Access:** HITRANonline web interface, **HAPI** Python API for programmatic access and spectral calculations, 160-character `.par` files. H₂, He, and CO₂ broadening parameters included for exoplanet modeling.

### HITEMP

**URL:** https://hitran.org/hitemp/  
**Content:** High-temperature extension covering **8 molecules** (H₂O, CO₂, CO, CH₄, N₂O, NO, NO₂, OH) valid to **4000+ K** with orders of magnitude more transitions than HITRAN (billions for H₂O). Essential for hot Jupiter/warm Neptune modeling. Same `.par` format.

### ExoMol

**URL:** https://www.exomol.com/  
**Source:** University College London  
**Content:** **91 molecules, 224 isotopologues, ~10¹² transitions**. Covers all biosignature molecules with line lists valid to 5000+ K. Format: `.states`/`.trans` files, pre-computed cross-sections, JSON metadata. Python tools: ExoCross, PyExoCross for cross-section calculation.

ExoMol line lists are the primary opacity source for most exoplanet retrieval codes (TauREx, petitRADTRANS, POSEIDON, CHIMERA, PICASO, NEMESIS). The 2024 release added photodissociation cross-sections and extended UV coverage.

---

## 5. TESS, Kepler, and Gaia mission data

### TESS

**URL:** https://archive.stsci.edu/missions-and-data/tess  
**Products:** Light curves (SAP_FLUX, PDCSAP_FLUX), target pixel files, full frame images  
**TOI Catalog:** **>7,700 planet candidates** at https://exofop.ipac.caltech.edu/tess/view_toi.php — columns include period, depth (ppm), planet radius (R⊕), equilibrium temperature (K), insolation flux. Format: CSV via API.  
**TIC v8.2:** **1.73 billion sources** with Teff, radius, mass, logg, metallicity, distance, magnitudes across all bands. Essential for stellar characterization of exoplanet hosts.  
**Access:** MAST Portal, Lightkurve Python package, TESScut, bulk download.

### Kepler/K2

**URL:** https://archive.stsci.edu/missions-and-data/kepler  
**KOI Catalog (DR25):** **9,564 KOIs** (4,717 candidates + 4,847 false positives) — columns include `koi_disposition` (CANDIDATE/FALSE POSITIVE/CONFIRMED), `koi_prad` (R⊕), **`koi_teq`** (equilibrium temp K), **`koi_insol`** (insolation flux), `koi_period`, `koi_depth`, stellar parameters. Format: CSV via TAP.

The **KOI disposition column provides a direct binary classification label** (planet vs. false positive) that has been widely used for ML transit classification and could be adapted for quantum classifiers.

**Kepler Stellar Properties Catalog:** 197,096 targets with Teff, logg, [Fe/H], radius, mass, density, distance.

### Gaia DR3

**URL:** https://gea.esac.esa.int/archive/  
**Content:** **1.8+ billion sources** with parallax, proper motion, G/BP/RP photometry, radial velocities (33M sources), astrophysical parameters (Teff, logg, [M/H], [α/Fe], 13 chemical abundances, radius, mass, age, luminosity) for 470M+ sources. BP/RP low-resolution spectrophotometry for ~220M sources.  
**Access:** ADQL queries via TAP, bulk ECSV/VOTable downloads.  
**Relevance:** Precise stellar characterization → accurate planetary radii → HZ boundary computation → bulk composition constraints.

### CHEOPS

**URL:** https://cheops-archive.astro.unige.ch  
**Content:** Ultra-precise transit photometry for radius measurements of known planets. Light curves in FITS format. Enables rocky vs. gaseous determination via bulk density.

---

## 6. Synthetic spectral datasets purpose-built for machine learning

This section contains the datasets most directly useful for your hackathon project.

### MultiREx biosignature classification dataset (highest relevance)

**Source:** Duque-Castaño et al. 2025, MNRAS 539(2):1528  
**URL:** GitHub (MultiREx package); arXiv: 2407.19167  
**Size:** **~10⁷ synthetic transmission spectra** of TRAPPIST-1 e-like planets  
**Content:** JWST/NIRSpec PRISM simulated spectra with stellar contamination and noise at varying SNR. Varying mixing ratios of CH₄, O₃, H₂O.  
**ML labels:** **Binary classification** ("interesting for follow-up" vs. "not interesting") AND **multilabel classification** (presence/absence of methane, ozone, water).  
**Why it matters:** This is the **only large-scale dataset with explicit binary biosignature labels on spectral data**. Classification works at SNR as low as 4. Ideal for variational quantum classifiers or quantum kernel methods.

### ML4SCI/POSEIDON quantum-ready dataset

**Source:** ML4SCI GSoC 2025 project (Sourish Phate)  
**URL:** https://github.com/ML4SCI/QMLHEP (EXXA group)  
**Size:** **~98,000 simulated transmission spectra**, 269 wavelength bins each  
**Content:** Generated with POSEIDON forward model. 11 parameters: radius, gravity, temperature, cloud opacity, log abundances of H₂O, CO₂, CO, CH₄, NH₃, cloud pressure.  
**Key advantage:** **Already tested with quantum circuits** — PCA to 8 components → quantum amplitude encoding → hybrid quantum-classical regression. R² and RMSE comparable between quantum and classical. Provides ready-made pipeline and codebase for quantum atmospheric characterization.

### INARA dataset (largest rocky planet dataset)

**Size:** 3,112,620 spectra with 4,379 wavelength columns. Rocky planets in habitable zones. 12 biosignature gases. See Section 1 for details.

### Ariel Atmospheric Big Challenge (ABC) Database

**URL:** https://zenodo.org/records/6770103  
**Size:** **105,887 synthetic Ariel Tier-2 observations** with 52 spectral bins (0.5–7.5 µm)  
**Content:** TauREx3 forward models + ArielRad noise. 26,109 complemented with full Nested Sampling retrieval posteriors. 7-dimensional target: planet radius, temperature, H₂O, CO₂, CO, CH₄, NH₃ abundances.  
**Format:** HDF5/numpy with Jupyter tutorials.

### Ariel Data Challenge competition datasets (2019–2025)

Available at https://www.ariel-datachallenge.space/ and Kaggle:
- **ADC 2024** (https://www.kaggle.com/competitions/ariel-data-challenge-2024): Extract 283-wavelength transmission spectra from simulated 2D detector images. 23,000+ submissions.
- **ADC 2025** (https://www.kaggle.com/competitions/ariel-data-challenge-2025): More realistic instrument models, wider atmospheric signatures.
- **ADC 2023:** 41,423 spectra with 6,766 Nested Sampling posteriors.

### Additional model grids and simulation tools

- **Sonora model grids** (Zenodo) — Bobcat (cloud-free), Diamondback (cloudy), Elf Owl (disequilibrium chemistry). Substellar atmosphere models, useful for brown dwarf/gas giant classification but less relevant for rocky planet biosignatures.
- **Bioverse** (https://github.com/danielapai/bioverse) — simulation framework generating synthetic exoplanet populations with **binary biosignature labels** based on hypothetical abiogenesis scenarios. Ideal for testing survey strategies.
- **LIFEsim** (https://github.com/fdannert/LIFEsim) — mid-IR nulling interferometry simulator for the LIFE mission concept. Can simulate O₃, CH₄, PH₃ detectability.
- **NASA PSG** (https://psg.gsfc.nasa.gov/) — online spectrum generator for any planetary body, supporting custom telescope configurations. Can generate training data at scale via API.
- **VPL Spectral Explorer** (https://vpl.uw.edu/models/spectral-database-tools/) — synthetic spectra of terrestrial exoplanets including Earth-through-orbit spectra, biological pigment database.

---

## 7. Quantum ML for exoplanet atmospheres: the emerging frontier

### Published QML papers directly relevant to exoplanets

**Vetrano et al. 2025 — "Exoplanetary atmospheres retrieval via a quantum extreme learning machine"**  
arXiv: 2509.03617 | The **first paper demonstrating quantum atmospheric retrieval on real quantum hardware**. Uses Quantum Extreme Learning Machines (QELMs) with TauREx-generated synthetic JWST spectra. Retrieves H₂O, CO₂, CO, CH₄ abundances with **93–100% accuracy** on clean data and **87%+** on noisy data. Demonstrated on IBM Fez quantum processor. PCA preprocessing reduces spectra to 6–7 principal components per patch — perfectly compatible with ODRA 5's 5-qubit limit. Dataset: 10,000 spectra split 80/20 train/test.

**Regadío 2024 — "Exoplanet Discovery with Variational Quantum Circuits"**  
Quantum Machine Intelligence, Springer. Uses Variational Quantum Circuits (VQCs) in Qiskit for **binary classification of Kepler transit signals**. Tests ZZFeatureMap, different ansatz structures, and training algorithms. Demonstrates VQCs are viable for astronomical classification.

**Chen et al. 2023 — "Quantum-Enhanced SVM for Large-Scale Stellar Classification"**  
arXiv: 2311.12328. QSVM for multi-class stellar spectral classification using quantum kernels. **Outperforms** KNN and Logistic Regression. Uses cuQuantum GPU acceleration. The quantum kernel + classical SVM approach transfers directly to atmospheric spectral classification.

**Deshler et al. 2026 — "Quantum Limits of Exoplanet Detection and Localization"**  
PRX Quantum (arXiv: 2403.17988). Theoretical work showing quantum processors can detect atmospheric molecules (O₂, CH₄, H₂O) at narrow spectral lines with orders-of-magnitude fewer photons via quantum signal processing.

### ML4SCI GSoC 2025 — complete QML pipeline for exoplanet spectra

The most implementation-ready resource. GSoC student Sourish Phate built a full pipeline: **PCA dimensionality reduction → classical autoencoder → quantum amplitude encoding → hybrid quantum-classical regression** for atmospheric parameter estimation from POSEIDON spectra. Code available on GitHub (https://github.com/ML4SCI/QMLHEP). The pipeline is Qiskit-based and directly transferable to ODRA 5.

### Additional QML-in-astronomy work

- **Luongo et al. 2023** (RAS Techniques and Instruments): Quantum-enhanced SVM for galaxy morphology classification, AUC = 0.946, matching classical SVM
- **Souza et al. 2025** (arXiv: 2505.15600): VQC for pulsar classification in Qiskit — provides complete VQC workflow template
- **Slabbert et al. 2025** (arXiv: 2507.07018): Quantum spectral clustering on SDSS astronomical survey data
- **Spectral Phase Encoding** (arXiv: 2602.19644, Feb 2026): DFT-based spectral preprocessing for quantum kernel SVMs — improves noise robustness, directly applicable to exoplanet spectra

---

## 8. ODRA 5 and HACK-4-SAGES hackathon context

### ODRA 5 specifications

**Location:** Wrocław Centre for Networking and Supercomputing (WCSS), Poland  
**Hardware:** IQM Spark — **5 superconducting qubits** in STAR topology, operating at 10 mK  
**Frameworks:** **Qiskit** (primary), CUDA-Q compatible  
**Remote access:** Also provides access to **20-qubit and 50+ qubit IQM machines** at IQM's Aalto (Finland) center  
**Access portal:** https://odra5.e-science.pl/

The 5-qubit local constraint is manageable: PCA reduction to 5–8 components (as demonstrated by Vetrano et al. and ML4SCI) fits well within this limit. The remote 20–50+ qubit access enables more complex circuits for the hackathon demonstration.

### HACK-4-SAGES context

**Full name:** HACK-4-SAGES: Digital Twins in Astrobiology Hackathon  
**Dates:** March 9–13, 2026 (4-day hybrid event)  
**Organizers:** ETH Zurich (lead), Warsaw University of Technology, Gdańsk University of Technology, ENHANCE network  
**Theme 2** ("Digital Twins in Life Detection and Biosignatures") is the target category. A QML biosignature classifier building a "digital twin" of an atmospheric characterization pipeline fits this theme precisely.

---

## Recommended hackathon project architecture

Given ODRA 5's constraints and the available datasets, three viable approaches emerge for combining quantum computing with biosignature detection:

- **Quantum kernel classifier on MultiREx data.** Use MultiREx's binary biosignature labels (biosignature present/absent) with PCA-reduced spectra as input to a Quantum Support Vector Machine (QSVM). The ZZFeatureMap in Qiskit maps 5 PCA components to 5 qubits, computing quantum kernel matrices for SVM classification. This requires minimal circuit depth and is robust on NISQ hardware.

- **Variational quantum classifier on POSEIDON/ML4SCI data.** Replicate and extend the ML4SCI GSoC pipeline: encode PCA-compressed spectra via angle encoding into a 5-qubit parametrized circuit, train via classical optimizer. The existing codebase provides a ready starting point. Convert the continuous atmospheric parameters to binary labels (e.g., CH₄ above/below a biogenic threshold).

- **Hybrid quantum-classical anomaly detection on ABC database.** Train a classical autoencoder on the 105,887 Ariel spectra, then use a quantum circuit to classify the latent representations as chemically "normal" vs. "anomalous" (potential biosignature candidates). This leverages quantum advantage in complex decision boundaries within compressed feature spaces.

All three approaches use **Qiskit** (ODRA 5's primary framework), require only **5 qubits** for the quantum component, and address the hackathon's "digital twin" theme by creating a quantum-enhanced computational model of the biosignature detection pipeline.

---

## Summary reference table

| Dataset | Size | Format | Labels | Biosig. relevance | Best for QML? |
|---|---|---|---|---|---|
| NASA Exoplanet Archive (pscomppars) | ~6,128 planets | CSV/TAP | None (compute HZ) | Medium | No (tabular, not spectral) |
| PHL-EC / HWC | ~70 habitable + all confirmed | CSV | ESI, HZD | High | Binary habitability classification |
| INARA | 3.1M spectra | CSV/NPZ | Continuous (12 gas abundances) | Very high | Yes — PCA + quantum regression |
| PyATMOS | 124K atmospheres | CSV/NPZ | Continuous (gas fluxes) | Very high | Possible — high dimensionality |
| MultiREx | ~10⁷ spectra | Package-generated | **Binary + multilabel** | **Highest** | **Yes — ideal for QSVM/VQC** |
| ML4SCI/POSEIDON | 98K spectra | NumPy | Continuous (11 params) | High | **Yes — already quantum-tested** |
| ABC (Ariel) | 106K spectra | HDF5 | Continuous (7 params) | High | Yes — anomaly detection |
| ADC 2024/2025 (Kaggle) | Varies | CSV/FITS | Spectra extraction | Medium | Possible preprocessing task |
| HITRAN/HITEMP/ExoMol | Millions of lines | .par/.trans | N/A (line lists) | Supporting | No (input to forward models) |
| JWST ERS (WASP-39b) | Individual spectra | FITS | Real observations | High | No (too few samples for ML) |
| Kepler KOI DR25 | 9,564 KOIs | CSV | Binary disposition | Medium | Yes — transit classification |
| Bioverse | Variable (framework) | Python-generated | Binary biosignature | Very high | Yes — custom populations |
| El-Kholy et al. 2025 | Merged catalog | CSV | Binary habitability | High | Yes — ready for classification |

The MultiREx dataset, ML4SCI/POSEIDON spectra, and INARA dataset represent the strongest starting points. MultiREx provides the explicit biosignature classification task that quantum kernels and variational classifiers are designed for. The ML4SCI codebase provides a proven Qiskit pipeline. And the Vetrano et al. QELM paper provides the published precedent demonstrating that quantum atmospheric retrieval works on real quantum hardware with as few as 5–7 PCA components — a perfect fit for ODRA 5.