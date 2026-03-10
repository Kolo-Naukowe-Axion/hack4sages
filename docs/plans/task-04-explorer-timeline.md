# Task 4: Planet Explorer — Timeline + Summary

> **Pre-req:** Read `docs/plans/context.md` first. Tasks 1-3 must be complete.

## Files

- Create: `web/src/components/PlanetTimeline.tsx`
- Create: `web/src/components/PlanetSummary.tsx`
- Modify: `web/src/app/explorer/page.tsx`

## Section A: Page Header

- Heading: "Planet Explorer" — `font-display text-4xl text-heading`
- Subheading: "Select an exoplanet to analyze its atmosphere for biosignatures" — `text-muted`
- Left-aligned, `py-12 px-8`

## Section B: Timeline (min-h-[30vh])

"use client" component. Props: `planets: Planet[]`, `selectedId: string | null`, `onSelect: (id: string) => void`.

Horizontal scrollable strip:
- Darker lane background: `bg-deep/40 rounded-2xl py-8 px-6`
- Horizontal connecting line (absolute, thin, `bg-border`)
- Planet nodes along the line:
  - Circular (`w-16 h-16 rounded-full border-2`) with first letter of planet name inside
  - Default: `border-border bg-deep text-muted`
  - Hover: `border-cyan shadow-[0_0_12px_var(--color-cyan)] text-heading` + tooltip with key stats
  - Selected: `border-cyan bg-cyan/10 shadow-[0_0_20px_var(--color-cyan)] scale-110 text-cyan`
  - Discovery year above: `font-mono text-xs text-muted`
  - Planet name below: `text-sm text-text`
  - Green dot if `inHabitableZone`
- Scroll arrows: Lucide ChevronLeft/ChevronRight in frosted glass circles at edges
- `useRef` + smooth scroll by 300px on click
- Hide scrollbar: `scrollbar-width: none` / `::-webkit-scrollbar { display: none }`

## Section C: Planet Summary (appears on selection, min-h-[40vh])

Frosted glass card, asymmetric two-column (55/45):
- Animate in: slide down + fade (`transition-all duration-500`)
- **Left (55%)**:
  - Planet name: `font-display text-3xl text-heading`
  - Star system: `text-muted text-sm`
  - Data rows (NOT generic HTML table — use 2-col grid or dl/dt/dd):
    - Mass: `0.69 M⊕` | Radius: `0.92 R⊕` | Temperature: `251 K` | Period: `6.1 days` | Distance: `39.6 ly`
    - Labels: `text-muted text-sm`, Values: `font-mono text-cyan`
    - Null values → `—`
- **Right (45%)**:
  - Recharts `AreaChart` with spectrum data
  - X-axis: wavelength (μm), Y-axis: transit depth, styled with `stroke: var(--color-muted)`, mono tick font
  - Area: `stroke: var(--color-cyan)`, `fill: rgba(0,229,255,0.08)`
  - Custom tooltip in frosted glass style
  - Badge below: Lucide Satellite icon + "Real JWST Data" (green) or "Synthetic Spectrum" (muted outlined)

## Wiring

Explorer page (`"use client"`):
- Import planets from `data/planets`
- `useState` for `selectedPlanetId`
- Render PlanetTimeline → PlanetSummary (conditional)

## Review

Playwright screenshot. Select a planet, verify:
- Summary slides in smoothly
- Chart renders with styled axes
- Hover/selected states visible on timeline nodes
- Layout doesn't break with long planet names

## Commit

```bash
git add web/src/ && git commit -m "feat: planet timeline and summary panel with spectrum chart"
```
