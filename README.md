# ExoBiome

Quantum biosignature detection in exoplanet atmospheres. Built for HACK-4-SAGES 2026 (ETH Zurich COPL).

## What it does

ExoBiome uses quantum extreme learning machines (QELM) running on real quantum hardware to classify whether an exoplanet's transmission spectrum contains biosignature patterns.

## Web App

Two design variants of the interactive frontend:

- **`web-sci-journal/`** — Scientific journal aesthetic (Nature paper typography, light cream theme)
- **`web-apple-product/`** — Apple product page aesthetic (clean white, premium feel, scroll reveals)

### Tech Stack

- Next.js 16 + React 19
- Tailwind CSS v4
- Nivo (charts)
- TypeScript

### Run locally

```bash
# Scientific Journal version
cd web-sci-journal
npm install
npm run dev
# Open http://localhost:3000

# Apple Product version (in a separate terminal)
cd web-apple-product
npm install
npm run dev -- --port 3001
# Open http://localhost:3001
```

### Build for production

```bash
cd web-sci-journal  # or web-apple-product
npm run build
npm start
```

### Pages

- `/` — Landing page with project overview
- `/explorer` — Select an exoplanet, view its spectrum, run biosignature analysis
- `/models` — Compare QELM Vetrano, QELM Extended, and Classical RF architectures

## Research

- [Vetrano et al. 2025 — QELM for atmospheric retrieval](https://arxiv.org/abs/2509.03617)
- [Cardenas et al. 2025 — MultiREx dataset](https://doi.org/10.1093/mnras/stae2948)
