# pRT ADC2023 Validation Set Method

This document describes exactly how the `petitRADTRANS`-based ADC2023 validation
set is being generated in this repository, what is treated as physical truth,
what is treated as ADC-format compatibility metadata, which official resources
are used, and which implementation issues were discovered and resolved during
the build.

It is intended to be an implementation-accurate methods record for the current
generator code in:

- `prt_adc2023_validation/`
- `scripts/build_reference_bundle.py`
- `scripts/rebin_prt_opacities.py`
- `scripts/generate_validation_set.py`
- `scripts/validate_validation_set.py`

## 1. Goal

The goal is to generate a `20,000`-sample validation set that:

- uses the `petitRADTRANS` disequilibrium emission simulator described in the
  cited papers,
- is binned into the ADC2023 Ariel challenge format,
- preserves exact generative truth for the atmosphere and noiseless spectra,
- preserves enough auxiliary and provenance information to make the validation
  set scientifically auditable.

The finished dataset is written in an ADC2023-compatible directory layout:

- `ValidationData/SpectralData.hdf5`
- `ValidationData/AuxillaryTable.csv`
- `ValidationData/Ground Truth Package/FM_Parameter_Table.csv`
- `ValidationData/Ground Truth Package/NoiselessSpectralData.hdf5`
- `ValidationData/Ground Truth Package/NativeSpectra_R400.hdf5`
- `manifest.json`

We intentionally do **not** generate:

- `QuartilesTable.csv`
- `Tracedata.hdf5`

Those are retrieval outputs, not generative truth.

## 2. Authoritative Sources

The method is based on the following official or primary sources:

1. Gebhard et al. 2025:
   `https://arxiv.org/html/2410.21477v1`
2. Vasist et al. 2023:
   `https://arxiv.org/abs/2301.06575`
3. Official ADC2023 baseline repository:
   `https://github.com/ucl-exoplanets/ADC2023-baseline`
4. `petitRADTRANS` installation/docs:
   `https://petitradtrans.readthedocs.io/en/latest/content/installation.html`
5. `sbi-ear` repository used by the paper authors:
   `https://github.com/francois-rozet/sbi-ear`
6. The `sbi-ear` pRT input-data bundle:
   `https://keeper.mpdl.mpg.de/f/78b3c66857924b5aacdd/?dl=1`

When official resources disagreed with guessed behavior, the generator was
changed to follow the official or directly verified behavior.

## 3. Local ADC Reference Data

The repository already contains a local copy of the ADC-format dataset in
`data/ariel-ml-dataset/`. We use it for two things only:

1. To extract the canonical ADC output arrays:
   - `instrument_wlgrid`
   - `instrument_width`
2. To fit an empirical conditional prior for ADC auxiliary metadata that is not
   directly produced by the pRT forward model.

We do **not** use the local ADC spectra as simulator truth. The pRT spectra are
the truth source for this validation set.

### 3.1 Canonical arrays

The local training-set HDF5 shows that every ADC sample shares the same:

- `instrument_wlgrid` (52 values)
- `instrument_width` (52 values)

Those arrays are recorded in `prt_adc2023_validation/constants.py` and are used
as the authoritative output metadata arrays for this validation set.

Important distinction:

- The **actual binning** uses the official ADC2023 baseline Ariel resolution.
- The output field `instrument_width` is copied from the canonical local ADC
  HDF5 metadata field, because that is what the challenge-format files contain.

These are not the same array.

### 3.2 Compact reference bundle

The script `scripts/build_reference_bundle.py` builds
`data/reference_data/adc2023_reference_bundle.npz` from:

- `TrainingData/AuxillaryTable.csv`
- `TrainingData/Ground Truth Package/FM_Parameter_Table.csv`
- `TrainingData/SpectralData.hdf5`

The bundle contains:

- standardized empirical feature matrix,
- empirical auxiliary values,
- canonical ADC output arrays,
- training-range bounds for the empirical query features.

## 4. Remote Environment Actually Used

The generation is running on the remote Ubuntu machine provided by the user.

The current working runtime is:

- Python `3.9.23`
- `numpy 1.26.4`
- `scipy 1.11.4`
- `pandas 2.2.3`
- `h5py 3.10.0`
- `petitRADTRANS 2.6.7`
- `taurex 3.2.4`

### 4.1 Why this exact Python stack is used

`petitRADTRANS 2.6.7` did not build cleanly on the newer Python `3.11` /
`NumPy 2.x` / `setuptools 82` stack available by default in 2026. The working
remote environment was therefore rebuilt with:

- Python `3.9`
- `numpy 1.26`
- `setuptools 59.8`
- Conda-forge `gfortran` / `g++`

This is an implementation compatibility choice only. It does not change the
scientific forward model or the dataset methodology.

## 5. pRT Input Data and Opacity Preparation

### 5.1 Input data source

The `petitRADTRANS` input data is downloaded from the same bundle referenced by
the authors’ `sbi-ear` README:

- `https://keeper.mpdl.mpg.de/f/78b3c66857924b5aacdd/?dl=1`

It is extracted on the remote machine to:

- `/home/iwo/hack4sages/input_data`

### 5.2 R=400 rebinned opacities

The script `scripts/rebin_prt_opacities.py` reproduces the authors’ `rebin.py`
pattern:

- it uses the full-resolution correlated-k species,
- calls `Radtrans.write_out_rebin(400, ...)`,
- writes the rebinned opacities into the pRT input-data tree.

Source species rebinned:

- `H2O_HITEMP`
- `CO_all_iso_HITEMP`
- `CH4`
- `NH3`
- `CO2`
- `H2S`
- `VO`
- `TiO_all_Exomol`
- `PH3`
- `Na_allard`
- `K_allard`

### 5.3 CO species naming discrepancy and resolution

The archived `sbi-ear` simulator code uses:

- `CO_all_iso_R_400`

However, the actual rebinned opacity directory produced by the current pRT input
data and `write_out_rebin()` is:

- `CO_all_iso_HITEMP_R_400`

This was resolved by direct `Radtrans` initialization tests on the remote
machine:

- `CO_all_iso_R_400` fails because the corresponding opacity directory does not
  exist,
- `CO_all_iso_HITEMP_R_400` loads successfully.

Therefore the implemented generator uses:

- `CO_all_iso_HITEMP_R_400`

This is a resolved implementation discrepancy, not a guess.

## 6. Atmospheric Forward Model

The generator uses the pRT disequilibrium emission model:

- `petitRADTRANS.retrieval.models.emission_model_diseq()`

The atmosphere is constructed with:

- line species:
  - `H2O_HITEMP_R_400`
  - `CO_all_iso_HITEMP_R_400`
  - `CH4_R_400`
  - `NH3_R_400`
  - `CO2_R_400`
  - `H2S_R_400`
  - `VO_R_400`
  - `TiO_all_Exomol_R_400`
  - `PH3_R_400`
  - `Na_allard_R_400`
  - `K_allard_R_400`
- cloud species:
  - `MgSiO3(c)_cd`
  - `Fe(c)_cd`
- Rayleigh species:
  - `H2`
  - `He`
- continuum opacities:
  - `H2-H2`
  - `H2-He`
- wavelength bounds:
  - `[0.95, 2.45] µm`
- `do_scat_emis=True`

Pressure structure:

- `pressure_scaling = 10`
- `pressure_simple = 100`
- `pressure_width = 3`

The native simulator output is a `379`-bin spectrum on the pRT native `R=400`
grid over `0.95–2.45 µm`.

### 6.1 Cloud parameter naming

Older code paths and pRT versions use slightly different parameter names for the
cloud scaling / cloud-base parameters. To avoid hard-coding the wrong pair, the
generator resolves the accepted names at worker startup by trial-calling
`emission_model_diseq()` and accepting the first valid pair among:

- `eq_scaling_Fe(c)` / `eq_scaling_MgSiO3(c)`
- `eq_scaling_Fe` / `eq_scaling_MgSiO3`
- `log_X_cb_Fe(c)` / `log_X_cb_MgSiO3(c)`

This keeps the implementation aligned with the installed `petitRADTRANS 2.6.7`
runtime instead of assuming a name from an older example.

## 7. Prior Over Atmospheric Parameters

The generator samples the 16 code-facing atmospheric parameters independently,
following the 2025 paper’s disequilibrium emission setup:

- `C/O ~ U(0.1, 1.6)`
- `[Fe/H] ~ U(-1.5, 1.5)`
- `log P_quench ~ U(-6, 3)`
- `S_eq,Fe ~ U(-2.3, 1.0)`
- `S_eq,MgSiO3 ~ U(-2.3, 1.0)`
- `f_sed ~ U(0, 10)`
- `log Kzz ~ U(5, 13)`
- `sigma_g ~ U(1.05, 3.0)`
- `log g ~ U(2.0, 5.5)` in cgs
- `R_P ~ U(0.9, 2.0)` in Jupiter radii
- `T_int ~ U(300, 2300)` K
- `T3_unit ~ U(0, 1)`
- `T2_unit ~ U(0, 1)`
- `T1_unit ~ U(0, 1)`
- `alpha ~ U(1, 2)`
- `log_delta_unit ~ U(0, 1)`

### 7.1 Internal transformed PT quantities

The truth package records the transformed quantities actually used by pRT:

- `t_connect`
- `t3`
- `t2`
- `t1`
- `delta`

These are derived from `T_int`, `T3_unit`, `T2_unit`, `T1_unit`, `alpha`, and
`log_delta_unit` using the same formulas implemented in
`prt_adc2023_validation/physics.py`.

## 8. Distance, Scaling, and Noise

### 8.1 Flux scaling distance

`star_distance` is sampled before the pRT call from the empirical auxiliary
prior and then mapped directly to pRT’s distance parameter:

- `D_pl = distance_pc_to_cm(star_distance)`

This means the stored distance is physically coupled to the generated flux.

### 8.2 Scale factor

Following the authors’ simulator convention, native pRT fluxes are multiplied by:

- `1e16`

All stored spectra and stored `instrument_noise` are in these scaled ML units.

### 8.3 Noise model

For each sample, a scalar noise level is drawn from the 2025 paper’s noise
model:

- `sigma ~ U(0.05, 0.50) × 1e-16 W m^-2 µm^-1`

Implementation details:

- `sampled_sigma_scaled` is stored in scaled units (`0.05–0.50`)
- `sampled_sigma_w_m2_um` is stored in physical flux units
- `instrument_noise` is a 52-element constant vector filled with the sampled
  scaled sigma
- `instrument_spectrum = noiseless_binned + N(0, sigma)` independently in each
  ADC bin

## 9. ADC2023 Binning

### 9.1 Authoritative binning source

The official ADC2023 baseline uses TauREx `FluxBinner` in wavenumber space.
This is the authoritative binning definition we follow.

The official baseline arrays are recorded in:

- `prt_adc2023_validation/constants.py`

These correspond to the baseline repo’s `ariel_resolution()` output.

### 9.2 Important implementation detail: native pRT grid must be converted to wavenumber

The native pRT spectrum is returned on a wavelength grid in microns.
TauREx `FluxBinner` is being used with the official ADC wavenumber grid, so the
native pRT wavelength grid must be converted before calling `bindown()`:

- `native_wavenumber = 10000 / native_wavelength_micron`

This was not left as an assumption. It was verified directly during debugging:

- passing native wavelengths to `FluxBinner.bindown()` produced all-zero
  52-bin spectra,
- passing native wavenumbers produced non-zero binned spectra,
- the corrected smoke dataset then passed validation.

### 9.3 Output ordering

The TauREx binning output is converted back to ascending wavelength order for
ADC output storage by reversing the binned flux vector after `bindown()`.

### 9.4 Output metadata arrays

For every stored ADC-format spectrum:

- `instrument_wlgrid` is written from the canonical local ADC array
- `instrument_width` is written from the canonical local ADC HDF5 array

Again, this output metadata array is not the same object as the baseline binning
width array.

## 10. Auxiliary Metadata Strategy

The pRT atmosphere is treated as the authoritative physical truth.

### 10.1 Quantities treated as direct physical truth

These are derived directly from the sampled atmosphere or used directly in the
forward model:

- `star_distance`
- `D_pl`
- `planet_surface_gravity`
- `planet_mass_kg`
- `planet_radius_r_jup`
- `planet_radius_m`
- `t_connect`, `t3`, `t2`, `t1`, `delta`
- native and binned noiseless spectra

### 10.2 Quantities treated as ADC-compatibility covariates

These ADC fields are not native pRT controls in this simulator and are therefore
sampled from an empirical conditional prior fitted on the local ADC training
set:

- `star_mass_kg`
- `star_radius_m`
- `star_temperature`
- `planet_distance`
- `planet_orbital_period`

These are documented as compatibility covariates, not native atmospheric
forward-model truth.

### 10.3 Empirical conditional prior

The empirical nearest-neighbor prior is implemented in
`prt_adc2023_validation/empirical_prior.py`.

Reference feature columns in the local ADC data:

- `planet_mass_kg`
- `planet_surface_gravity`
- `planet_radius`
- `planet_temp`

Generated query features:

- derived `planet_mass_kg`
- derived `planet_surface_gravity`
- sampled `r_p_r_jup`
- generated `t_connect`

These are log-transformed, standardized with the training-set statistics, and
query-clipped only to the empirical training range before nearest-neighbor
search. The generated truth values themselves are not clipped when written out.

For each generated sample, the truth package records:

- the selected empirical row index,
- the empirical `planet_ID`,
- the chosen neighbor rank,
- the query distance in standardized feature space.

### 10.4 Kepler consistency

`planet_orbital_period` is recomputed from:

- sampled `star_mass_kg`
- sampled `planet_distance`

using Kepler’s law, rather than copied blindly from the empirical row. This
keeps the stored orbit fields self-consistent.

## 11. Invalid-Sample Handling

Some cloud parameter combinations can trigger pRT warnings such as:

- `Cloud rescaling lead to nan opacities, skipping RT calculation!`

These cases are **not** kept in the dataset.

The generator now treats a sample as invalid and resamples it if any of the
following occur:

- the pRT call raises an exception,
- the pRT warning stream contains the cloud-rescaling nan-opacity warning,
- the native spectrum contains non-finite values,
- the binned ADC spectrum contains non-finite values,
- the binned ADC spectrum is identically zero.

For accepted samples, the truth package records:

- `generation_attempt`

This counts how many tries were needed before a valid sample was obtained.

## 12. Truth Package Contents

The truth package is intentionally richer than the public ADC-style validation
files.

### 12.1 `FM_Parameter_Table.csv`

Contains:

- the 16 sampled atmospheric parameters,
- transformed PT quantities used internally by pRT,
- sampled sigma in scaled and physical units,
- `star_distance_pc`,
- `d_pl_cm`,
- derived planet mass / radius / gravity quantities,
- generation attempt count,
- empirical-prior provenance fields.

### 12.2 `NoiselessSpectralData.hdf5`

Contains the 52-bin noiseless ADC-format spectra:

- `instrument_wlgrid`
- `instrument_width`
- `instrument_noise`
- `instrument_spectrum`

The stored `instrument_spectrum` is the noiseless binned flux.

### 12.3 `NativeSpectra_R400.hdf5`

Contains the unbinned native pRT spectra:

- `native_wlgrid`
- `native_noiseless_spectrum`

## 13. Validation and QA

The validator in `prt_adc2023_validation/validate_dataset.py` checks:

- exact ADC output schema and HDF5 key alignment,
- exact match to canonical `instrument_wlgrid`,
- exact match to canonical output `instrument_width`,
- native spectrum length `379`,
- binned spectrum length `52`,
- finite values only,
- noiseless binned spectra are not identically zero,
- `instrument_noise` is constant across bins for each sample,
- `instrument_noise` matches stored sampled sigma,
- `planet_mass_kg` matches `log g` and radius,
- `planet_surface_gravity` matches `log g`,
- `planet_orbital_period` satisfies Kepler consistency,
- `star_distance` matches `star_distance_pc`,
- `d_pl_cm` matches the stored distance,
- stored PT-transformed quantities are consistent,
- manifest file list completeness,
- manifest SHA256 checksums,
- normalized noise residual mean near zero and standard deviation near one.

## 14. Current Production Run

At the time this document is being written, the current full production run is:

- output root:
  - `/home/iwo/hack4sages-output/full_20000_20260310b`
- log file:
  - `/home/iwo/hack4sages-output/full_20000_20260310b.log`
- worker count:
  - `4`
- shard size:
  - `250`
- target sample count:
  - `20,000`

The run writes shard `.npz` files first, then assembles the final dataset, then
runs the validator automatically.

During generation, operational progress is written to:

- `work/progress.json`

This file is updated when generation starts, when resume skips an existing
shard, periodically while the current shard is being filled, after each shard is
written, when final assembly begins, when assembly completes, and if the run
fails with an exception. It is operational metadata only and does not affect the
scientific outputs.

## 15. Local Destination

Once the production run and validation complete, the finished dataset is copied
back into the local project at:

- `data/petitradtrans-adc2023-validation/`

Remote/local integrity is then checked using the generated manifest checksums.

## 16. Files in This Repo That Implement the Method

- `prt_adc2023_validation/constants.py`
- `prt_adc2023_validation/physics.py`
- `prt_adc2023_validation/reference_bundle.py`
- `prt_adc2023_validation/empirical_prior.py`
- `prt_adc2023_validation/generate_dataset.py`
- `prt_adc2023_validation/validate_dataset.py`
- `scripts/build_reference_bundle.py`
- `scripts/rebin_prt_opacities.py`
- `scripts/generate_validation_set.py`
- `scripts/validate_validation_set.py`

## 17. Summary of Resolved Implementation Issues

These are important because they change whether the dataset is scientifically
valid, even though they are implementation details rather than methodological
choices in the papers.

1. `petitRADTRANS 2.6.7` required an older Python/NumPy/setuptools build stack
   on the remote host.
2. The rebinned CO opacity name that actually loads is
   `CO_all_iso_HITEMP_R_400`, not `CO_all_iso_R_400`.
3. TauREx `FluxBinner` must receive the native pRT grid in wavenumber, not
   wavelength.
4. Invalid pRT cloud-opacity cases are rejected and resampled instead of being
   written into the validation set.

These resolutions are already reflected in the current codebase.
