# EDA — Ariel Data Challenge 2023

Exploratory data analysis of the Ariel Space Mission ML dataset for exoplanet atmospheric spectral retrieval.

## Setup

```bash
cd eda
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
jupyter lab
# Open ariel_eda.ipynb
```

## Dataset

Download the [Ariel ML Data Challenge 2023](https://www.ariel-datachallenge.space/) dataset and place it in `ariel-ml-dataset/` (gitignored).

```
eda/
├── ariel_eda.ipynb       # Main notebook
├── ariel-ml-dataset/     # Dataset (gitignored)
├── requirements.txt      # Python dependencies
└── README.md
```
