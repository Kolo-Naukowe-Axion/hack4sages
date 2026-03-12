# ADC Winner On Ariel

This folder is a saved local snapshot of the winner-style ADC2023 normalizing-flow model trained on the Ariel dataset with the exact split from [data/val_dataset](/Users/iwosmura/projects/hack4sages/data/val_dataset).

What is included:

- the full model package snapshot under this folder
- the exact resolved settings used for the finished run
- saved checkpoints in `trained_run/`
- training history and logs in `trained_run/`
- final validation and holdout metrics in `trained_run/`
- the saved split and prepared-data manifests in `trained_run/`

Run provenance:

- source package snapshot: `models/ariel_winner_nf`
- run name: `ariel_winner_nf_valsplit_20260312_062848`
- Vast label: `SOTA_training_valsplit`
- exact split sizes:
  - train: `33138`
  - validation: `4142`
  - holdout: `4143`
- early stopping epoch: `79`

Final metrics:

- validation `rmse_mean`: `0.5528115034103394`
- holdout `rmse_mean`: `0.5522884130477905`
- best preview subset `mRMSE` during training:
  - epoch `10`: `0.629857`
  - epoch `20`: `0.612171`
  - epoch `30`: `0.573626`
  - epoch `40`: `0.579947`
  - epoch `50`: `0.562935`
  - epoch `60`: `0.544989`
  - epoch `70`: `0.565738`

Important files:

- best checkpoint by preview `mRMSE`: [trained_run/best_model_by_mrmse.pt](/Users/iwosmura/projects/hack4sages/models/adc_winner_on_ariel/trained_run/best_model_by_mrmse.pt)
- best checkpoint by validation NLL: [trained_run/best_model_by_nll.pt](/Users/iwosmura/projects/hack4sages/models/adc_winner_on_ariel/trained_run/best_model_by_nll.pt)
- latest resumable checkpoint: [trained_run/resume_latest.pt](/Users/iwosmura/projects/hack4sages/models/adc_winner_on_ariel/trained_run/resume_latest.pt)
- resolved config: [trained_run/settings_resolved.yaml](/Users/iwosmura/projects/hack4sages/models/adc_winner_on_ariel/trained_run/settings_resolved.yaml)
- validation metrics: [trained_run/validation_metrics.json](/Users/iwosmura/projects/hack4sages/models/adc_winner_on_ariel/trained_run/validation_metrics.json)
- holdout metrics: [trained_run/holdout_metrics.json](/Users/iwosmura/projects/hack4sages/models/adc_winner_on_ariel/trained_run/holdout_metrics.json)
- exact split manifest: [trained_run/saved_split_manifest.json](/Users/iwosmura/projects/hack4sages/models/adc_winner_on_ariel/trained_run/saved_split_manifest.json)
- prepared-data manifest: [trained_run/prepared_manifest.json](/Users/iwosmura/projects/hack4sages/models/adc_winner_on_ariel/trained_run/prepared_manifest.json)

Loading the archived model:

```python
from pathlib import Path

import torch
import yaml

from models.adc_winner_on_ariel.model import IndependentNSF, ModelConfig

root = Path("models/adc_winner_on_ariel")
settings = yaml.safe_load((root / "trained_run" / "settings_resolved.yaml").read_text())
model = IndependentNSF(ModelConfig(**settings["model"]))
checkpoint = torch.load(root / "trained_run" / "best_model_by_mrmse.pt", map_location="cpu")
model.load_state_dict(checkpoint["model"])
model.eval()
```
