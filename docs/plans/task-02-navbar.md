# Task 2: Navbar + Global Layout

> **Pre-req:** Read `docs/plans/context.md` first. Task 1 must be complete.

## Files

- Create: `web/src/components/Navbar.tsx`
- Modify: `web/src/app/layout.tsx`
- Create: `web/src/app/explorer/page.tsx` (placeholder)
- Create: `web/src/app/models/page.tsx` (placeholder)

## What to Build

Fixed top navbar (64px) with frosted glass (`backdrop-blur-xl bg-void/80 border-b border-border`):

- **Logo left**: "ExoBiome" in font-display, cyan dot on the "i" in Bio (styled span)
- **Nav links right**: Home, Explorer, Models — use Lucide icons (Home, Telescope, Cpu) alongside text
- **Active state**: `usePathname()` for active link detection → cyan underline + `shadow-[0_0_8px_var(--color-cyan)]`
- **Hover/focus**: `hover:text-cyan transition-colors` + `focus:outline-none focus:ring-2 focus:ring-cyan/30`
- **Mobile**: hamburger menu (Lucide Menu/X icons), slide-down overlay with frosted glass

Integrate into `layout.tsx` with `pt-16` on main content wrapper.

Create placeholder pages for `/explorer` and `/models` (simple heading + subheading in each).

## Review

Playwright screenshot of navbar on all 3 routes. Verify:
- Active link highlights correctly per route
- Frosted glass visible against page content
- Hover states work
- Mobile hamburger toggles

## Commit

```bash
git add web/src/ && git commit -m "feat: frosted glass navbar with routing and active states"
```
