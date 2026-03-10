# ExoBiome Web App — Design Document

## Overview

3-page SPA for quantum biosignature detection. Dark "Deep Space Observatory" aesthetic — cinematic, immersive, scientific. Hackathon demo with real model integration planned later.

## Tech Stack

- **Framework**: Next.js 15 + React 19
- **Styling**: Tailwind CSS v4
- **Charts**: Recharts
- **Fonts**: Outfit (display), DM Sans (body), JetBrains Mono (data)
- **Planet data**: Hardcoded curated list (~15-20 rocky exoplanets)
- **Model results**: Realistic mock with 2-3s delay animation (swap for real API later)
- **Deployment**: Vercel

## Color System

| Token | Hex | Use |
|---|---|---|
| `--void` | `#06060c` | Page background |
| `--deep` | `#0c0c18` | Card backgrounds |
| `--surface` | `#14142a` | Elevated surfaces |
| `--border` | `#1e1e3a` | Subtle borders |
| `--muted` | `#6b6b8d` | Secondary text |
| `--text` | `#e2e2f0` | Body text |
| `--heading` | `#f0f0ff` | Headings |
| `--cyan` | `#00e5ff` | Primary accent, interactive |
| `--teal` | `#0d9488` | Secondary accent |
| `--amber` | `#f59e0b` | Uncertain verdict |
| `--green` | `#10b981` | Biosignature detected |
| `--red` | `#ef4444` | No biosignature |

## Typography

- **H1**: Outfit 700, ~72px, letter-spacing wide
- **H2**: Outfit 600, ~36px
- **H3**: Outfit 500, ~24px
- **Body**: DM Sans 400, 16px
- **Data/Mono**: JetBrains Mono 400, for percentages, times, gas formulas

## Global Elements

### Navbar
- Fixed top, 64px height
- Frosted glass: `backdrop-blur-xl`, `bg-void/80`
- Logo left: "ExoBiome" in Outfit, cyan dot replaces "o" in Bio
- Nav links right: Landing, Explorer, Models
- Active link: cyan underline with subtle glow (`box-shadow: 0 2px 8px var(--cyan)`)

### Background
- CSS radial gradient: dark navy center → void edges
- Tiny static star dots via repeating CSS background pattern (lightweight, no JS particles)

### Card Style (shared)
- `bg-deep/60`, `backdrop-blur-md`
- `border border-border`, `rounded-2xl`
- Hover: `shadow-lg shadow-cyan/5`

---

## Page 1: Landing

### Hero (100vh)
- Heading: "Detect Life Beyond Earth" — Outfit 700, ~72px, white, subtle cyan `text-shadow`
- Subheading: "Quantum-powered biosignature detection in exoplanet atmospheres" — DM Sans, muted
- Background: CSS conic-gradient simulating star transit (bright circle + dark crossing dot) or static SVG transit illustration
- CTA: "Explore Planets" pill button — cyan bg, dark text, glow shadow, hover scale-up

### Science Explainer (4-card grid)
- Card 1: "Exoplanets" — what they are
- Card 2: "Transmission Spectra" — how starlight reveals atmosphere
- Card 3: "Biosignatures" — gas combos that suggest life
- Card 4: "Our Approach" — quantum ML on spectral data
- Frosted glass cards, icon top, 2-3 sentences each
- Stagger-animate on scroll via CSS `animation-delay` + `IntersectionObserver`

### Visual Separator
- Thin horizontal line, gradient: transparent → cyan → transparent

### Stats Bar
- 3 inline stats: "3 Models Compared" | "15+ Exoplanets" | "5 Qubit Quantum Hardware"
- JetBrains Mono, cyan numbers, muted labels

---

## Page 2: Planet Explorer + Detection

### Section A: Planet Timeline
- Horizontal scrollable strip, full-width, ~200px tall
- Darker gradient background to create a "lane"
- Each planet: circular node (~60px) connected by thin horizontal line
- Node shows: planet name below, discovery year above (small mono)
- States:
  - Default: dim, `border-border`
  - Hover: cyan border glow, tooltip with key stats
  - Selected: cyan fill glow, larger, triggers panel below
- Left/right arrow buttons at edges for scroll
- ~15-20 planets ordered by discovery year:
  - TRAPPIST-1b-h, K2-18b, LHS 1140b, Proxima Cen b, TOI-700d, GJ 1214b, Kepler-442b, Kepler-186f, etc.

### Section B: Planet Summary Panel
- Slides/expands below timeline on planet selection
- Two-column layout in frosted glass card:
  - **Left**: Planet name (large), star system, data table (mass, radius, eq temp, orbital period, distance) in JetBrains Mono
  - **Right**: Recharts line chart of transmission spectrum (mock data), badge below: green "Real JWST Data" or blue "Synthetic Spectrum"

### Section C: Submit
- Centered pill button: "Analyze Biosignatures"
- Disabled (muted) until planet selected
- On click: text → "Analyzing..." + pulsing cyan dot
- 2-3s mock delay before results appear

### Section D: Results (3 columns)
- 3 equal-width cards side by side (stack on mobile)
- Each card contains:
  - **Header**: Model name + type badge ("Quantum" cyan tag / "Classical" muted tag)
  - **Verdict**: Large text with color glow:
    - "BIOSIGNATURE DETECTED" → green
    - "NO BIOSIGNATURE" → red
    - "UNCERTAIN" → amber
  - **Confidence**: Big % in JetBrains Mono + circular progress ring (`conic-gradient`)
  - **Detected Gases**: Pills/chips (CH₄, O₃, H₂O) with distinct subtle colors
  - **Spectrum Plot**: Small Recharts area chart with highlighted absorption bands (colored vertical strips)
  - **Processing Time**: Bottom, mono, e.g. "2.3s"
- Cards animate in staggered: left → center → right (slide-up + fade)

### Section E: Bridge Panel
- Full-width banner
- "How do these models work? Why do they give different answers?"
- Three inline one-liners per model
- CTA: "Explore Models →" outlined button, cyan border

---

## Page 3: Models

### 3 Model Detail Cards (stacked vertically)

Each card:
- Header bar: model name + type tag + expand/collapse
- Default: all expanded

**Card 1: QELM — Vetrano Architecture**
- "Reproduction of Vetrano et al. 2025"
- Architecture flow diagram (CSS/HTML, not image):
  `Spectrum → PCA → RX Encoding → RY+CNOT Reservoir (depth 3) → Z-Measurement → SVD → Output`
- Each step: rounded pill, connected by arrows, cyan for quantum, muted for classical
- Stats table: Accuracy, Precision, Recall, Training Time, Qubits, Hardware (all placeholder —)

**Card 2: QELM — Extended Topology**
- "Our modified circuit — different entanglement and depth"
- Same layout, different reservoir structure highlighted
- Same stats format

**Card 3: Classical Baseline**
- "Best classical ML — the gold standard benchmark"
- Flow: `Spectrum → PCA → Feature Engineering → RF/XGBoost/CNN → Output`
- Same stats format

### Comparison Table
- Full-width, frosted glass, alternating row backgrounds
- Columns: Metric | QELM Vetrano | QELM Extended | Classical
- Rows: Accuracy, Precision, Recall, F1, Training Time, Inference Time, Hardware, Qubits
- All values placeholder (—)
- Cyan left-border on quantum column headers

### Footer / About
- Horizontal divider
- Project motivation (2-3 sentences)
- Link to Vetrano et al. 2025 paper
- Team: 4 member names/roles
- Hackathon badge: "HACK-4-SAGES 2026 | ETH Zurich | Life Detection and Biosignatures"
- Tech stack icons row

---

## Responsive Strategy

- Desktop-first, breakpoints at `md` (768px) and `lg` (1024px)
- Timeline: horizontal scroll maintained on all sizes
- Result cards: 3 columns → stacked on mobile
- Model cards: full-width on all sizes
- Navbar: hamburger menu on mobile

## Animation Approach

- CSS transitions + `animation-delay` for stagger effects
- `IntersectionObserver` for scroll-triggered reveals
- No heavy animation libraries — CSS-only for performance
- Key moments: hero load, card stagger, results reveal, button state changes

## Data Architecture

- `data/planets.ts` — hardcoded planet array with all properties
- `data/mockResults.ts` — pre-defined model results per planet
- `types/` — TypeScript interfaces for Planet, ModelResult, Gas, etc.
- Components fetch from local data, no external API calls in v1
- Model API integration later: swap mock imports for `fetch()` calls to FastAPI backend
