# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

> **See also:** [`../backend/CLAUDE.md`](../backend/CLAUDE.md) for the Flask API layer, and the root [`../CLAUDE.md`](../CLAUDE.md) for project-wide commands and architecture.

## Structure

```
frontend/
├── templates/index.html   — Single HTML page (all UI lives here)
└── static/
    ├── js/app.js          — All client-side logic (no framework, no build step)
    └── css/style.css      — Custom styles layered on top of Tailwind
```

**No build step.** Tailwind CSS is loaded from CDN (`cdn.tailwindcss.com`). DOMPurify is also CDN-loaded. There is no `package.json`, no bundler, and no transpilation — `app.js` is served as-is.

## Key Patterns in `app.js`

**`gatherPayload()`** — single source of truth for the POST `/api/train` request body. All form field → API key mappings live here. Note: `death_threshold` is displayed as a percentage (0–100) but sent as a ratio (`/100.0`).

**`applyResult(data)`** — populates the metric cards after a successful training response. Reads `data.results.best_lifetime`, `data.results.best_episode`, `data.results.avg_lifetime_final_10`.

**`switchTab(tab)`** — toggles between the `panelCurrent` and `panelHistory` divs. Calling `switchTab("history")` always triggers a fresh `fetchHistory()` call to `/api/history`.

**`_benchmarkTasks` Map** — tracks in-flight `POST /api/evaluate` jobs. Key is `run_id`, value is `{ taskId, sectionEl }`. The polling loop updates the DOM directly via `sectionEl`.

**`data-config-mirror` attribute** — any element with this attribute and a matching `data-config-mirror="<key>"` will have its text content synced to the corresponding payload key on every form `input`/`change` event via `syncConfigDisplay()`.

## UI Layout

Two-tab right panel:
- **Current** (`panelCurrent`) — result image, metrics grid, training status
- **History** (`panelHistory`) — rendered from `GET /api/history`; each run card has a "Run Benchmark" button that calls `POST /api/evaluate` and polls for results

History cards are built entirely in JS via `buildHistoryCard()` / `buildSection()` / `buildKVRow()`. The benchmark comparison table is built by `buildBenchmarkTable()`, which sorts trained model first, then baselines by descending mean reward.

## Gotchas

- **DOMPurify is required** for any HTML rendered from API responses. The history panel uses `innerHTML` assignments — always sanitize with `DOMPurify.sanitize()` before setting.
- **Tailwind config is inline** in `index.html` (the `tailwind.config = { ... }` block). Custom semantic color tokens (e.g., `bg-surface`, `text-on-surface`) are defined there — not in `style.css`.
- **`resultImage.src` cache-busting** — the image URL gets `?t=Date.now()` appended on every new result to force the browser to re-fetch the same filename.
