# Task 5: Planet Explorer — Analysis + Results

> **Pre-req:** Read `docs/plans/context.md` first. Tasks 1-4 must be complete.

## Files

- Create: `web/src/components/AnalyzeButton.tsx`
- Create: `web/src/components/ResultCard.tsx`
- Create: `web/src/components/ResultsPanel.tsx`
- Create: `web/src/components/BridgePanel.tsx`
- Modify: `web/src/app/explorer/page.tsx`

## Section D: Submit Button

Centered, generous vertical padding (`py-12`):
- Disabled: `bg-muted/20 text-muted cursor-not-allowed opacity-50`
- Ready: `bg-cyan text-void font-display font-semibold text-lg px-10 py-4 rounded-full shadow-[0_0_20px_rgba(0,229,255,0.3)] hover:scale-105 hover:shadow-[0_0_30px_rgba(0,229,255,0.5)] transition-all focus:ring-2 focus:ring-cyan/50 focus:outline-none`
- Loading: text → "Analyzing..." + Lucide Loader2 icon with `animate-spin`

## Section E: Results (3 columns, min-h-[60vh])

3-column grid (`grid-cols-1 lg:grid-cols-3 gap-6 px-8`). Each ResultCard:

- Frosted glass card, `p-6` minimum
- **Header**: model name + type badge
  - Quantum: Lucide Atom + "Quantum" pill `bg-cyan/10 text-cyan`
  - Classical: Lucide Brain + "Classical" pill `bg-muted/20 text-muted`
- **Verdict** (prominent):
  - "BIOSIGNATURE DETECTED" → `text-green font-display font-bold text-xl` + `text-shadow: 0 0 20px rgba(16,185,129,0.3)`
  - "NO BIOSIGNATURE" → `text-red`
  - "UNCERTAIN" → `text-amber`
- **Confidence ring**: CSS `conic-gradient` circle (~100px)
  - Track: `var(--color-border)`, fill: color matching verdict
  - Center: `font-mono text-2xl` percentage (e.g. "94.2%")
- **Detected gases**: flex-wrap pills (CH₄, O₃, H₂O, CO₂)
  - `bg-surface border border-border text-text text-sm px-3 py-1 rounded-full`
  - `hover:border-cyan/30 transition-colors`
- **Spectrum mini-chart**: Nivo ResponsiveLine (~150px) with area fill and colored markers at absorption wavelengths
- **Processing time**: `font-mono text-sm text-muted` right-aligned, e.g. "2.34s"

Cards stagger in: `@keyframes fadeSlideUp` with `animation-delay: 0ms, 150ms, 300ms`.

## Section F: Bridge Panel (appears after results)

Asymmetric layout (not centered text):
- Left: heading "Under the Hood" + "Three models, three approaches, one question: is there life?" — `text-muted`
- Right: 3 descriptions stacked, each with colored dot (cyan/cyan/muted) + model name + one-liner
- Bottom right: "Explore Models →" Lucide ArrowRight — outlined cyan button `hover:bg-cyan/10`
- Link to `/models`

## Wiring

Full Explorer page flow:
1. `useState` for `selectedPlanetId`, `results: PlanetResults | null`, `loading: boolean`
2. Select planet → show summary, reset results
3. Click Analyze → loading → 2-3s delay via `getMockResults()` → show results + bridge
4. Sections flow top-to-bottom with `py-12` between them

## Review

Playwright screenshot of full flow (select planet → analyze → results). Check:
- [ ] Result cards show varied data (not all identical)
- [ ] Confidence rings display correctly
- [ ] Spectrum charts show absorption bands
- [ ] Stagger animation works
- [ ] Bridge panel connects visually

## Commit

```bash
git add web/src/ && git commit -m "feat: analysis results with confidence rings, spectra, bridge panel"
```
