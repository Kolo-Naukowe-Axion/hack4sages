# ExoBiome UI (Osoba 4)

Streamlit demo for biosignature retrieval from exoplanet transmission spectra.
Pick a curated planet or upload a one-row spectrum CSV, run live inference, and
compare the predicted atmospheric composition (5 gases: H2O, CO2, CO, CH4, NH3)
against ground truth and the full quantum model.

## Run

From the repo root, with the app venv (`.venv-app`, Python 3.12):

```bash
.venv-app/bin/streamlit run app/ui/app.py
```

Then open http://localhost:8501. The dark theme is in `.streamlit/config.toml`.

## Inputs

- **Kuratowana planeta** - real ADC2023 planets (needs the data worktree, see below).
- **Przykład** - bundled example planets (`app/ui/examples/`); works with no dataset.
- **Wczytaj plik CSV** - upload a one-row CSV (8 aux columns + `flux_0..51` +
  `noise_0..51`). Use "Pobierz szablon CSV" for a valid template to edit.

## Data (optional - only for the curated dropdown)

The ADC2023 dataset lives on the data branch, checked out as a sibling worktree:

```bash
git worktree add ../hack4sages-data origin/iwosmu/data-artifacts
```

`app.py` auto-detects `../hack4sages-data` and sets `EXOBIOME_DATA`. Without the
dataset, the example and upload modes still work end to end.

## Layout (paradigm: hybrid OOP + FP)

- `app.py` - Streamlit entry: orchestration, caching (`@st.cache_resource` /
  `@st.cache_data`), error handling.
- `components.py` - pure Altair/pandas chart builders (no `st.*`, unit-tested):
  spectrum, comparison bars, comparison table.
- `inference_adapter.py` - the seam over the frozen checkpoint bridge (OOP model)
  and the functional `app.data` pipeline (FP). The only place the UI touches the
  model, so it is the single point to repoint at `app/models` / `app/inference`
  when those land.

The data flow consumes `app.data`'s immutable records + pure pipeline (FP) and
the checkpoint bridge (OOP) - the hybrid this project is about.

## Model

Live inference runs the hybrid model's **classical head** (CPU, no PennyLane).
The **full quantum** model's predictions come from the checkpoint's
`validation_predictions.csv` (real, precomputed) and appear as the second series
in the comparison view when available for the planet.

## Tests

```bash
PYTHONPATH=. .venv-app/bin/python -m unittest tests.test_o4_ui
```
