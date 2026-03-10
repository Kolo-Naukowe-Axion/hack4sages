# ExoBiome

Quantum biosignature detection in exoplanet atmospheres. Built for HACK-4-SAGES 2026 (ETH Zurich COPL).

## What it does

ExoBiome uses quantum extreme learning machines (QELM) running on real quantum hardware to classify whether an exoplanet's transmission spectrum contains biosignature patterns. Three models are compared side by side: two quantum approaches and one classical baseline.

## Web App

Interactive frontend for exploring exoplanets and analyzing their atmospheres.

### Tech Stack

- Next.js 16 + React 19
- Tailwind CSS v4
- Nivo (charts)
- TypeScript

### Run locally

```bash
cd web
npm install
npm run dev
```

Open http://localhost:3000

### Build for production

```bash
cd web
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

## Team

Built by the ExoBiome team at HACK-4-SAGES 2026.
