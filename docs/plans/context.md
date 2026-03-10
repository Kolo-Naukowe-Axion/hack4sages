# ExoBiome Web App — Shared Context

> **Load this file before every task.** It contains all project specs, rules, and constraints.

## Project

3-page dark space-themed web app for quantum biosignature detection. Hackathon demo (HACK-4-SAGES 2026, ETH Zurich) with real model integration planned later.

## Tech Stack

Next.js 15 (App Router), React 19, TypeScript, Tailwind CSS v4, Recharts, lucide-react, next/font (Outfit, DM Sans, JetBrains Mono). Deploy on Vercel.

## Routes

- `/` — Landing page
- `/explorer` — Planet Explorer + Detection
- `/models` — Model details + comparison

## Skills to Use

- `frontend-design:frontend-design` — when writing component code
- Context7 — look up Tailwind CSS v4 docs before writing any styles
- Playwright — screenshot + self-review after each section

## Design Persona

You are a senior UI/UX designer who has worked at Linear, Vercel, and Stripe. You have strong opinions about whitespace, typography hierarchy, and visual polish. You refuse to ship anything that looks like a generic template.

## Color Palette (locked — do not modify)

| Token | Hex | Use |
|---|---|---|
| `void` | `#06060c` | Page background |
| `deep` | `#0c0c18` | Card backgrounds |
| `surface` | `#14142a` | Elevated surfaces |
| `border` | `#1e1e3a` | Borders |
| `muted` | `#6b6b8d` | Secondary text |
| `text` | `#e2e2f0` | Body text |
| `heading` | `#f0f0ff` | Headings |
| `cyan` | `#00e5ff` | Primary accent |
| `teal` | `#0d9488` | Secondary accent |
| `amber` | `#f59e0b` | Uncertain verdict |
| `green` | `#10b981` | Biosignature detected |
| `red` | `#ef4444` | No biosignature |

## Typography

- **Display (H1-H3)**: Outfit — `font-display`
- **Body**: DM Sans — `font-sans`
- **Data/Mono**: JetBrains Mono — `font-mono` (percentages, times, gas formulas only)

## Content Rules

- Real planet names: TRAPPIST-1e, K2-18b, Proxima Centauri b, LHS 1140b
- Realistic numbers: 94.2%, 91.7%, 96.8% — not round placeholders
- Real gas names: CH₄, O₃, H₂O, CO₂
- No lorem ipsum. No "Welcome to our platform." Every string is real copy.

## Anti-Cliché Rules

- No gradient text on everything
- No centered-everything layouts — mix full-width, split, asymmetric, overlapping
- No oversized hero with tiny content below
- No identical-size card grids — vary sizes, use asymmetric grids
- No generic "Welcome to..." or "Explore our..." copy

## UI Quality Rules

- Every section: `min-h-[60vh]` or more
- Minimum padding: cards `p-6`, sections `py-20 px-8`
- All interactive elements need `hover:` and `focus:` states
- Icons: Lucide via `lucide-react` — no emoji, no unicode symbols

## Card Style (shared)

`bg-deep/60 backdrop-blur-md border border-border rounded-2xl p-6`
Hover: `hover:border-cyan/20 hover:shadow-lg hover:shadow-cyan/5 transition-all duration-300`

## Section Review Protocol

After building each section, take a Playwright screenshot and check:
- Does it look premium (Linear/Vercel/Stripe level)?
- Enough whitespace and padding?
- Colors consistent?
- Visual hierarchy clear?
- Text readable?
- No layout breaks or overflow?
Fix issues before moving to the next task.

## File Tree (target)

```
web/src/
├── app/
│   ├── globals.css
│   ├── layout.tsx
│   ├── page.tsx              (landing)
│   ├── explorer/page.tsx
│   └── models/page.tsx
├── components/
│   ├── Navbar.tsx
│   ├── PlanetTimeline.tsx
│   ├── PlanetSummary.tsx
│   ├── AnalyzeButton.tsx
│   ├── ResultCard.tsx
│   ├── ResultsPanel.tsx
│   ├── BridgePanel.tsx
│   ├── ModelCard.tsx
│   ├── ArchitectureFlow.tsx
│   ├── ComparisonTable.tsx
│   └── Footer.tsx
├── data/
│   ├── planets.ts
│   └── mockResults.ts
└── types/index.ts
```
