# Transmission Benchmark Dataset

This benchmark generates synthetic **transmission spectroscopy** with `petitRADTRANS` and writes a canonical dataset:

- `spectra.h5`
- `labels.parquet`
- `provenance.parquet`
- `manifest.json`

## Outputs

`spectra.h5` stores:

- `sample_id`
- `split`
- `benchmark/wavelength_um`
- `benchmark/transit_depth_noisy`
- `benchmark/transit_depth_noiseless`
- `benchmark/sigma_1sigma`
- `native/wavelength_um`
- `native/transit_depth_noiseless`
- `native/transit_radius_m`

`labels.parquet` stores:

- continuous targets `log_X_*`
- detection labels `present_*`
- bulk planetary parameters
- star/system covariates
- cloud/haze parameters

`provenance.parquet` stores:

- generation attempt count
- OOD candidate flag
- sampled noise/systematic hyperparameters
- resample-reason summary JSON

## Priors

- Pressure grid: `1e-6` to `1e2` bar, `100` levels
- Reference pressure: `10 bar`
- Wavelength range: `0.5-5.0 um`
- Native pRT setup: `R~400`
- Benchmark grid: fixed `R=100`
- Temperature regimes:
  - cool: `500-900 K`
  - warm: `900-1600 K`
  - hot: `1600-2500 K`
- Chemistry:
  - broad common pRT species with regime-conditioned log-uniform mass fractions
  - reject/resample if heavy-species mass fraction exceeds `0.15`
- Clouds/hazes:
  - gray cloud deck via `Pcloud`
  - scattering haze via `kappa_zero` and `gamma_scat`

## Run Flow

1. Rebin opacities for the transmission wavelength range:

```bash
.venv-prt311/bin/python data/scripts/rebin_prt_transmission_opacities.py \
  --input-data-path /path/to/input_data
```

2. Autotune worker count on the target machine:

```bash
.venv-prt311/bin/python data/scripts/tune_transmission_workers.py \
  --input-data-path /path/to/input_data \
  --output-root data/generated-data/worker_tuning
```

3. Launch the full run:

```bash
PRT_BENCH_INPUT_DATA_PATH=/path/to/input_data \
PRT_BENCH_WORKERS=18 \
data/scripts/run_transmission_benchmark.sh
```

4. Watch progress:

```bash
.venv-prt311/bin/python data/scripts/check_generation_status.py \
  --output-root data/generated-data/transmission_benchmark_YYYYMMDD_HHMMSS --watch
```

5. Validate the assembled dataset:

```bash
.venv-prt311/bin/python data/scripts/validate_transmission_benchmark.py \
  --output-root /path/to/output_root
```

## Notes

- Generation is designed to be **CPU-bound**. Keep BLAS/OpenMP thread counts at `1` per worker.
- `pyarrow` is required for final Parquet assembly and validation.
- `--generate-only` is available for smoke runs and worker tuning.
