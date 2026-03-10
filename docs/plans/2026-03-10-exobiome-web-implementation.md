# ExoBiome Web App — Implementation Plan (Index)

> **For Claude:** Before each task, read `context.md` + the task file. After completing a task, commit and move to the next file. This keeps context small per task.

## Task Files

| # | File | What | Depends on |
|---|---|---|---|
| 1 | `task-01-scaffold.md` | Next.js project, theme, data layer | — |
| 2 | `task-02-navbar.md` | Navbar + global layout + route placeholders | 1 |
| 3 | `task-03-landing.md` | Landing page (hero, explainer, stats) | 2 |
| 4 | `task-04-explorer-timeline.md` | Explorer: timeline + planet summary | 3 |
| 5 | `task-05-explorer-results.md` | Explorer: analyze button, results, bridge | 4 |
| 6 | `task-06-models.md` | Models page (cards, comparison, footer) | 5 |
| 7 | `task-07-review-1.md` | Visual review cycle 1 — full audit | 6 |
| 8 | `task-08-review-2.md` | Visual review cycle 2 — interactions | 7 |
| 9 | `task-09-review-final.md` | Visual review cycle 3 — final polish | 8 |

## How to Execute

1. Read `context.md` (shared project specs, rules, color palette)
2. Read the current task file
3. Build what it says, using `frontend-design` skill for component code
4. Take Playwright screenshots and self-review per the task's review section
5. Commit
6. Move to next task file

Context stays small: only `context.md` (~100 lines) + current task (~60-80 lines) need to be loaded.
