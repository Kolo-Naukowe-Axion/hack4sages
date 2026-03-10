# Task 1: Project Scaffold + Theme + Data Layer

> **Pre-req:** Read `docs/plans/context.md` first.

## Files

- Create: `web/` (Next.js project)
- Create: `web/src/types/index.ts`
- Create: `web/src/data/planets.ts`
- Create: `web/src/data/mockResults.ts`
- Modify: `web/src/app/layout.tsx`
- Modify: `web/src/app/globals.css`
- Create: `.gitignore` (root)

## Steps

### 1. Create feature branch + .gitignore

```bash
cd /Users/michalszczesny/projects/hack4sages
git checkout -b feat/web
```

Create root `.gitignore`:
```
.DS_Store
node_modules/
.next/
.env*
```

### 2. Create Next.js project + install deps

```bash
npx create-next-app@latest web --typescript --tailwind --eslint --app --src-dir --turbopack --no-import-alias --yes
cd web && npm install @nivo/line @nivo/core lucide-react
```

Note: Using **Nivo** (not Recharts) — confirmed React 19 compatible. `@nivo/line` handles spectrum area/line charts.

### 3. Use Context7 to look up Tailwind v4 `@theme` syntax

Verify correct `@theme` / `@theme inline` usage for v4 before writing globals.css.

### 4. Configure fonts in `src/app/layout.tsx`

```tsx
import type { Metadata } from "next";
import { Outfit, DM_Sans, JetBrains_Mono } from "next/font/google";
import "./globals.css";

const outfit = Outfit({ variable: "--font-outfit", subsets: ["latin"], display: "swap" });
const dmSans = DM_Sans({ variable: "--font-dm-sans", subsets: ["latin"], display: "swap" });
const jetbrainsMono = JetBrains_Mono({ variable: "--font-jetbrains-mono", subsets: ["latin"], display: "swap" });

export const metadata: Metadata = {
  title: "ExoBiome — Quantum Biosignature Detection",
  description: "Detect biosignatures in exoplanet atmospheres using quantum machine learning",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className={`${outfit.variable} ${dmSans.variable} ${jetbrainsMono.variable} antialiased`}>
        {children}
      </body>
    </html>
  );
}
```

### 5. Configure theme in `src/app/globals.css`

```css
@import "tailwindcss";

@theme inline {
  --font-sans: var(--font-dm-sans);
  --font-display: var(--font-outfit);
  --font-mono: var(--font-jetbrains-mono);
}

@theme {
  --color-void: #06060c;
  --color-deep: #0c0c18;
  --color-surface: #14142a;
  --color-border: #1e1e3a;
  --color-muted: #6b6b8d;
  --color-text: #e2e2f0;
  --color-heading: #f0f0ff;
  --color-cyan: #00e5ff;
  --color-teal: #0d9488;
  --color-amber: #f59e0b;
  --color-green: #10b981;
  --color-red: #ef4444;
}

body {
  background-color: var(--color-void);
  color: var(--color-text);
  font-family: var(--font-sans);
}

body::before {
  content: "";
  position: fixed;
  inset: 0;
  z-index: -1;
  background:
    radial-gradient(ellipse at 50% 30%, #0d1117 0%, var(--color-void) 70%),
    radial-gradient(1px 1px at 10% 20%, rgba(255,255,255,0.15) 1px, transparent 0),
    radial-gradient(1px 1px at 30% 60%, rgba(255,255,255,0.1) 1px, transparent 0),
    radial-gradient(1px 1px at 50% 40%, rgba(255,255,255,0.12) 1px, transparent 0),
    radial-gradient(1px 1px at 70% 80%, rgba(255,255,255,0.08) 1px, transparent 0),
    radial-gradient(1px 1px at 90% 10%, rgba(255,255,255,0.1) 1px, transparent 0);
  background-size: 100% 100%, 200px 200px, 300px 300px, 250px 250px, 350px 350px, 150px 150px;
}
```

### 6. Create TypeScript types in `src/types/index.ts`

```ts
export interface Planet {
  id: string;
  name: string;
  starSystem: string;
  discoveryYear: number;
  massEarth: number | null;
  radiusEarth: number;
  eqTempK: number | null;
  orbitalPeriodDays: number;
  distanceLy: number;
  inHabitableZone: boolean;
  hasJWSTData: boolean;
  spectrumType: "jwst" | "synthetic";
  spectrumData: { wavelength: number; flux: number }[];
}

export type Verdict = "detected" | "none" | "uncertain";

export interface DetectedGas {
  formula: string;
  name: string;
  confidence: number;
}

export interface ModelResult {
  modelName: string;
  modelType: "quantum" | "classical";
  verdict: Verdict;
  confidence: number;
  detectedGases: DetectedGas[];
  processingTimeMs: number;
  spectrumHighlights: { start: number; end: number; gas: string }[];
}

export interface PlanetResults {
  planetId: string;
  results: [ModelResult, ModelResult, ModelResult];
}
```

### 7. Create planet data in `src/data/planets.ts`

**Keep it compact** — use a `generateSpectrum()` helper function that creates ~50 wavelength/flux points from a base curve + absorption dip positions. Do NOT hardcode 50 individual data points per planet.

```ts
// Helper: generate realistic spectrum from dip positions
function generateSpectrum(dips: { pos: number; depth: number; width: number }[]): { wavelength: number; flux: number }[] {
  // generate 50 points from 0.6 to 5.0 μm with gaussian dips at specified positions
}
```

~15-20 planets with real NASA values. Each planet entry is ~8 lines (properties) + a `generateSpectrum()` call.

Planets: TRAPPIST-1b-h, K2-18b (hasJWSTData: true), LHS 1140b, Proxima Cen b, TOI-700d, TOI-700e, Kepler-442b, Kepler-186f, GJ 1214b, LP 791-18d.

### 8. Create mock results in `src/data/mockResults.ts`

Pre-defined results per planet with realistic numbers (94.2%, 91.7%, 87.3%). Use a helper to reduce repetition. K2-18b: all detect biosignature. TRAPPIST-1e: quantum uncertain, classical none. Export `getMockResults(planetId): Promise<PlanetResults>` with 2-3s delay.

### 9. Verify + commit

```bash
npm run dev  # verify localhost:3000 works
cd .. && git add .gitignore web/ && git commit -m "scaffold: Next.js 15 with Tailwind v4, Nivo, fonts, theme, data layer"
```
