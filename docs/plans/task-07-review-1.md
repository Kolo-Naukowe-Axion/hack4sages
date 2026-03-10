# Task 7: Visual Review Cycle 1 — Full Site Audit

> **Pre-req:** Read `docs/plans/context.md` first. Tasks 1-6 must be complete.

## Steps

### 1. Start dev server

```bash
cd /Users/michalszczesny/projects/hack4sages/web && npm run dev
```

### 2. Screenshot all pages with Playwright at 1440px width

- `/` — Landing page (full page)
- `/explorer` — with no planet selected
- `/explorer` — with a planet selected + results showing (click a planet, click Analyze, wait for results)
- `/models` — full page

### 3. Review each screenshot

For each page, evaluate:
- [ ] Premium quality? Would a Stripe/Linear designer approve this?
- [ ] Enough whitespace? Sections don't feel cramped?
- [ ] Colors harmonious? Cyan accent not overused?
- [ ] Visual hierarchy clear? Can you scan and know what's important?
- [ ] Text readable? Sufficient contrast on all backgrounds?
- [ ] No layout breaks or overflow?
- [ ] All cards consistent styling (border-radius, padding, backdrop-blur)?
- [ ] Interactive elements look clickable?
- [ ] Starfield visible but not distracting?
- [ ] Section layouts varied (not everything centered)?

### 4. Fix every issue found

Focus on: spacing, contrast, layout balance, padding consistency.

### 5. Commit

```bash
git add web/src/ && git commit -m "fix: visual polish pass 1 — spacing, contrast, layout"
```
