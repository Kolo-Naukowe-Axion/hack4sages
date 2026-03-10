# ExoBiome Web App

3 pages via top navbar.

## Page 1: Landing

Hero with exoplanet transit visual. Short explanation: exoplanets, transmission spectra, biosignatures, what this app does. CTA button → Planet Explorer.

## Page 2: Planet Explorer + Detection

Top-to-bottom flow on one page:

**A. Timeline** — Horizontal scrollable timeline of confirmed rocky exoplanets (real NASA data). Each node: name, discovery year, mass, radius, temperature, habitable zone status. Click to select.

**B. Planet Summary** — Expands after selection. Shows physical params, spectrum preview, data source badge (real JWST / synthetic).

**C. Submit** — "Analyze Biosignatures" button. Sends planet data to all 3 models.

**D. Results (3 columns)** — Side-by-side cards:

| QELM Vetrano | QELM Extended | Classical |
|---|---|---|
| YES/NO/UNCERTAIN | YES/NO/UNCERTAIN | YES/NO/UNCERTAIN |
| Confidence % | Confidence % | Confidence % |
| Detected gases | Detected gases | Detected gases |
| Spectrum plot | Spectrum plot | Spectrum plot |
| Processing time | Processing time | Processing time |

**E. Bridge Panel** — "How do these models work?" + brief diff summary + link to Models page.

## Page 3: Models

3 cards with full description per model:
- **QELM Vetrano**: reproduction of paper, angle encoding → quantum reservoir → SVD
- **QELM Extended**: our modified topology, different entanglement/depth
- **Classical**: best ML baseline (RF/XGBoost/CNN)

Each card: architecture visual, training method, stats table (accuracy, precision, recall, training time, qubits, hardware). All stats placeholder until training done.

Comparison table at bottom. About/team/hackathon info at footer.
