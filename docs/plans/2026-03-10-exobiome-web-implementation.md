# ExoBiome Web App — Implementation Plan (Index)

> **For Claude:** Before each task, read `context.md` + the task file. Build in main session sequentially. Use sub-agents only for research (Context7, doc lookups). Commit after each task on `feat/web` branch.

## Task Files

| # | File | What | Depends on |
|---|---|---|---|
| 1 | `task-01-scaffold.md` | Next.js project, theme, Nivo, data layer, .gitignore, feat/web branch | — |
| 2 | `task-02-navbar.md` | Navbar + global layout + route placeholders | 1 |
| 3 | `task-03-landing.md` | Landing page (hero, explainer, stats) | 2 |
| 4 | `task-04-explorer-timeline.md` | Explorer: timeline + planet summary | 3 |
| 5 | `task-05-explorer-results.md` | Explorer: analyze button, results, bridge | 4 |
| 6 | `task-06-models.md` | Models page (cards, comparison, footer) | 5 |
| 7 | `task-07-review.md` | Full visual audit + fix cycles (single thorough pass) | 6 |

## How to Execute

1. Read `context.md` (shared specs, ~120 lines)
2. Read current task file (~60-80 lines)
3. Build it, screenshot via Playwright, self-review
4. Fix issues, commit
5. Move to next task
