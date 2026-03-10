# Task 7: Visual Review — Full Audit + Fix Cycles

> **Pre-req:** Read `docs/plans/context.md` first. Tasks 1-6 must be complete.

This is a single thorough review task with iterative fix cycles. Keep fixing until quality is premium.

## Step 1: Start dev server

```bash
cd /Users/michalszczesny/projects/hack4sages/web && npm run dev
```

## Step 2: Screenshot all pages at 1440px width via Playwright

- `/` — Landing page (full page)
- `/explorer` — empty state (no planet selected)
- `/explorer` — with planet selected + results showing (click a planet, click Analyze, wait)
- `/models` — full page

## Step 3: Full audit checklist

**Premium quality:**
- [ ] Would a Stripe/Linear designer approve this? If not, what specifically feels off?
- [ ] Is there at least one "wow" moment per page?
- [ ] Dark theme cohesive across all 3 pages?

**Spacing & layout:**
- [ ] Enough whitespace? No cramped sections?
- [ ] Every section has `min-h-[60vh]`+?
- [ ] Cards have `p-6`+ padding?
- [ ] Section layouts varied (not everything centered)?
- [ ] No layout breaks or overflow?
- [ ] No orphaned text lines or awkward breaks?

**Colors & visual hierarchy:**
- [ ] Colors harmonious? Cyan accent not overused?
- [ ] Visual hierarchy clear? Can you scan and know what's important?
- [ ] Text readable? Sufficient contrast on all backgrounds?
- [ ] Verdict color coding (green/red/amber) immediately clear?
- [ ] Starfield visible but not distracting?

**Typography:**
- [ ] H1 > H2 > H3 > body — each level visually distinct
- [ ] `font-mono` only on data values (%, seconds, gas formulas)
- [ ] `font-display` on all headings
- [ ] Cards have consistent border-radius, padding, backdrop-blur

**Interactions:**
- [ ] Navigate all 3 pages via navbar — active states correct
- [ ] Explorer: select planet → summary animates in → Analyze → results stagger in
- [ ] Models: expand/collapse cards work
- [ ] All hover states visible (buttons, cards, timeline nodes, nav links)
- [ ] All focus states visible (tab through entire site)
- [ ] Animations smooth (~200-300ms, ease-out)

**Data & charts:**
- [ ] Confidence rings render at correct percentages
- [ ] Spectrum charts have proper styled axes, look scientific
- [ ] Gas pills readable and well-spaced
- [ ] Result cards show varied data (not all identical)

## Step 4: Fix all issues found

Apply fixes. Focus on spacing, contrast, layout balance, padding, animation timing.

## Step 5: Re-screenshot at 1440px — verify fixes worked

## Step 6: Screenshot at 375px mobile — check responsiveness

- [ ] Everything stacks properly
- [ ] Nothing overflows horizontally
- [ ] Text readable at mobile sizes
- [ ] Navbar hamburger works
- [ ] Timeline scrolls on mobile

## Step 7: Fix any mobile issues

## Step 8: Final micro-polish pass

Shadow depths, border opacities, easing curves, letter-spacing, padding fine-tuning. The details that separate premium from good-enough.

## Step 9: Final screenshots (both widths) — confirm done

## Step 10: Commit

```bash
git add web/src/ && git commit -m "fix: visual polish — premium quality pass"
```
