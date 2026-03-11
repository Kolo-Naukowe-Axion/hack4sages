# Data Access

Curated datasets and EDA assets are intentionally kept off `main`.

Use the dedicated data branch in a separate worktree:

```bash
git fetch origin
git worktree add ../hack4sages-data origin/iwosmu/data-artifacts
```

Published assets on `origin/iwosmu/data-artifacts`:

- `data/ariel-ml-dataset/`
- `data/petitradtrans-adc2023-validation/`
- `data/reference_data/adc2023_reference_bundle.npz`
- `data/eda/`
- `data/published/crossgen_biosignatures/20260311/`

Local scratch outputs generated from this codebase should stay under `data/generated-data/`, which is ignored on `main`.
