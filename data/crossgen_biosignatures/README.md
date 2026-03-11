# Cross-Generator Biosignatures

This package builds a brand-new transmission-spectrum dataset with:

- `TauREx` rows for `train` and `val`
- `POSEIDON` rows for `test`
- shared latent priors
- shared `0.6-5.2 um` constant-`R=100` rebinned spectra
- both continuous `log10_vmr_*` targets and binary `present_*` targets

## CLI

Generate in one environment or one backend at a time:

```bash
python data/scripts/generate_crossgen_biosignatures.py --output-root /path/to/output --mode tau
python data/scripts/generate_crossgen_biosignatures.py --output-root /path/to/output --mode poseidon
python data/scripts/generate_crossgen_biosignatures.py --output-root /path/to/output --mode assemble
```

Validate and run the baseline smoke test:

```bash
python data/scripts/validate_crossgen_biosignatures.py --output-root /path/to/output
python data/scripts/run_crossgen_baseline.py --output-root /path/to/output
```

## Remote Helpers

- `data/scripts/sync_crossgen_remote.sh`
- `data/scripts/setup_crossgen_remote_envs.sh`
- `data/scripts/bootstrap_poseidon_input_data.sh`
- `data/scripts/run_crossgen_remote_pipeline.sh`

These scripts assume the remote host already exposes the existing TauREx-capable `~/micromamba/envs/prt39` Python and create separate POSEIDON and neutral utility environments next to it.
