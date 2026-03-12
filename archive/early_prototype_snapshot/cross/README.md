## Cross-Generator Biosignature Dataset

Synced from the completed remote run on `100.103.127.124` into this local dataset directory on `2026-03-11`.

This folder contains the full output bundle for the `crossgen_biosignatures` dataset:

- `spectra.h5`: assembled feature table
- `labels.parquet`: assembled label table
- `manifest.json`: dataset summary and validation metadata
- `latents.parquet`: intermediate latent table used to generate the spectra
- `baseline_smoke.json`: baseline train/val/test smoke-test metrics
- `baseline_poseidon_predictions.csv`: baseline predictions on the POSEIDON test set
- `meta/`: per-generator generation metadata
- `shards/`: resumable generator shard files used to assemble the final dataset

Canonical split contract:

- TauREx train: `37,281`
- TauREx val: `4,142`
- POSEIDON test: `685`

Canonical generator counts:

- TauREx: `41,423`
- POSEIDON: `685`

Recommended files for model training:

- features: `spectra.h5`
- targets: `labels.parquet`

Implementation and schema source:

- `data/crossgen_biosignatures/`
