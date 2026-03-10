# Task 3: Landing Page

> **Pre-req:** Read `docs/plans/context.md` first. Tasks 1-2 must be complete.

Build the entire landing page as one cohesive piece, then review end-to-end.

## Files

- Modify: `web/src/app/page.tsx`

## Section A: Hero (min-h-screen)

- **NOT centered-everything** — asymmetric layout: text left-aligned with generous left margin, transit visual positioned right/background
- Heading: "Detect Life Beyond Earth" — `font-display text-5xl lg:text-7xl font-bold text-heading` with subtle cyan `text-shadow: 0 0 40px rgba(0,229,255,0.15)`
- Subheading: "The first quantum machine learning system for biosignature detection in exoplanet atmospheres." — `text-lg lg:text-xl text-muted max-w-xl`
- CTA: "Explore Planets" with Lucide ArrowRight — pill button, cyan bg, void text, glow shadow, `hover:scale-105 hover:shadow-[0_0_30px_var(--color-cyan)/40] transition-all`
- Background: CSS radial gradient simulating a star transit — bright warm circle offset right with small dark transit shadow
- Hero should not dwarf content below — keep balanced

## Section B: Science Explainer (min-h-[60vh], py-20 px-8)

- **NOT 4 identical cards** — asymmetric layout:
  - Left: large feature card (~60% width) combining "Exoplanets" + "Transmission Spectra" content with small inline spectrum illustration
  - Right: two stacked smaller cards (~40% width) — "Biosignatures" and "Quantum Detection"
- All cards: shared card style from context.md
- Icons: Lucide (Globe, Waves, Dna, Atom) — not emoji
- Scroll-triggered stagger animation: CSS `@keyframes fadeSlideUp` + IntersectionObserver

## Section C: Stats + Separator (py-16)

- Gradient separator: `h-px bg-gradient-to-r from-transparent via-cyan/30 to-transparent`
- 3 stats spread with `justify-between` (not centered):
  - "3" / "Models Compared"
  - "15+" / "Exoplanets Analyzed"
  - "5" / "Qubit Quantum Hardware"
  - Numbers: `font-mono text-cyan text-5xl font-bold`
  - Labels: `text-muted text-sm tracking-wide uppercase`

## Review

Playwright screenshot of full landing page. Check:
- [ ] Hero feels balanced (not oversized)?
- [ ] Explainer layout is varied (not 4 identical cards)?
- [ ] Enough whitespace between sections?
- [ ] Looks like Linear/Vercel, not a Bootstrap template?

Fix any issues before moving on.

## Commit

```bash
git add web/src/ && git commit -m "feat: landing page — hero, explainer, stats"
```
