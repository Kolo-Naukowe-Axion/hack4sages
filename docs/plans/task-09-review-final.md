# Task 9: Visual Review Cycle 3 — Final Premium Polish

> **Pre-req:** Read `docs/plans/context.md` first. Tasks 1-8 must be complete.

## Steps

### 1. Screenshot all pages at 1440px AND 375px (mobile)

### 2. Final quality gate

- [ ] Would this win a hackathon design award?
- [ ] Is there at least one "wow" moment per page?
- [ ] Dark theme cohesive across all pages?
- [ ] No orphaned text lines or awkward breaks?
- [ ] Mobile: everything stacks properly, nothing overflows, text readable?
- [ ] Tab through entire site — focus states visible?
- [ ] Spectrum plots look scientific, not toy-like?
- [ ] Verdict color coding (green/red/amber) immediately clear?

### 3. Apply micro-fixes

Focus on: shadow depths, border opacities, easing curves, letter-spacing, padding fine-tuning.

### 4. Final verification protocol

1. Open in Playwright at 1440px width
2. Scroll through every section on every page, screenshot each section
3. Check: all nav tabs work, all hover effects work
4. Switch to 375px mobile width, repeat full scroll
5. List every remaining issue (if any)
6. Fix them all
7. Final full-page screenshot of each page at both widths

### 5. Commit

```bash
git add web/src/ && git commit -m "fix: final visual polish — premium quality"
```
