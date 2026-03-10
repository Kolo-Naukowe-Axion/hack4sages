# Task 8: Visual Review Cycle 2 — Interactions + Typography

> **Pre-req:** Read `docs/plans/context.md` first. Tasks 1-7 must be complete.

## Steps

### 1. Re-screenshot all pages after Task 7 fixes

### 2. Test all interactions via Playwright

- Navigate through all 3 pages via navbar links
- On Explorer: click a planet → verify summary animates in → click Analyze → verify results stagger in
- On Models: expand/collapse all cards
- Verify hover states on buttons, cards, timeline nodes, nav links

### 3. Review details

- [ ] Typography hierarchy: H1 > H2 > H3 > body — each level visually distinct
- [ ] `font-mono` only on data values (%, seconds, gas formulas) — never on labels
- [ ] `font-display` on all headings — never `font-sans`
- [ ] Animations smooth (~200-300ms, ease-out), no jank
- [ ] Confidence rings render at correct percentages
- [ ] Spectrum charts have properly styled axes
- [ ] Gas pills readable and well-spaced
- [ ] Navbar frosted glass visible against all page backgrounds

### 4. Fix all issues

### 5. Commit

```bash
git add web/src/ && git commit -m "fix: visual polish pass 2 — typography, interactions"
```
