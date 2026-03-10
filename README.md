# ExoBiome — Web Frontend

Quantum biosignature detection web app. Built by **Axion** for HACK-4-SAGES 2026.

Two design variants:

- **`web/sci-journal/`** — Scientific journal aesthetic (Nature paper typography, light cream)
- **`web/apple-product/`** — Apple product page aesthetic (clean white, scroll reveals)

## Tech Stack

Next.js 16, React 19, Tailwind CSS v4, Nivo, TypeScript

## Run locally

```bash
# Scientific Journal version
cd web/sci-journal
npm install
npm run dev
# http://localhost:3000

# Apple Product version (separate terminal)
cd web/apple-product
npm install
npm run dev -- --port 3001
# http://localhost:3001
```

## Build

```bash
cd web/sci-journal  # or web/apple-product
npm run build
npm start
```

## Pages

- `/` — Landing page
- `/explorer` — Planet selection, spectrum chart, biosignature analysis
- `/models` — QELM Vetrano, QELM Extended, Classical RF comparison
