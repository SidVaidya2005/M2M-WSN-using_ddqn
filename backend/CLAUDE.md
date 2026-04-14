# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

> **See also:** [`../frontend/CLAUDE.md`](../frontend/CLAUDE.md) for the UI layer, and the root [`../CLAUDE.md`](../CLAUDE.md) for project-wide commands and architecture. Authoritative API rules live in [`../.claude/rules/api-design.md`](../.claude/rules/api-design.md).

## Structure

```
backend/
├── app.py       — Flask app factory (create_app), entry point
├── routes.py    — All API route handlers, registered as blueprint `api_bp`
├── schemas.py   — Marshmallow request validation schemas
└── tasks.py     — Sync/async training execution + DDQN-vs-DQN comparison
```

## App Factory & Blueprint

`create_app()` in `app.py` is the Flask app factory. Routes are registered as a blueprint (`api_bp`) at the `/api` prefix. The blueprint import uses a try/except to handle both `python -m backend.app` (package import) and `python backend/app.py` (direct script) invocations.

The loaded config object is stored on `app.config["CONFIG"]` — retrieve it inside route handlers with `current_app.config.get("CONFIG")`. Never call `get_config()` directly inside a route handler.

## Route Overview

| Method | Path | Sync/Async | Handler |
|--------|------|-----------|---------|
| GET | `/api/health` | sync | `health()` |
| GET | `/api/config` | sync | `get_config_endpoint()` |
| POST | `/api/train` | **blocking** | `train_model()` → `run_training()` |
| POST | `/api/train/async` | non-blocking | `train_model_async()` → `submit_training_task()` |
| GET | `/api/tasks/<task_id>` | sync | `task_status()` |
| GET | `/api/history` | sync | `get_history()` |
| GET | `/api/compare?a=<run_id>&b=<run_id>` | sync | `compare_runs_endpoint()` → `compare_runs()` |
| GET | `/api/results/<path>` | sync | file serve |
| GET | `/api/visualizations/<path>` | sync | file serve |

**`/api/evaluate` does not exist** — baseline benchmarking was removed in Phase 0. The only comparison surface is `/api/compare`, which runs DDQN-vs-DQN on two existing runs.

## `tasks.py` — Execution Engine

This module contains all heavy computation. Entry points:

- **`run_training(params, config)`** — synchronous; called by `POST /api/train`. Builds env + agent (`DDQNAgent` if `model_type=="ddqn"`, else `DQNAgent`), runs `Trainer.train()`, writes `{run_id}_model.pth`, `{run_id}_plot.png`, and `{run_id}_metadata.json` (Phase 3 schema — see [`../.claude/rules/artifacts.md`](../.claude/rules/artifacts.md)).
- **`submit_training_task(params, config)`** — wraps `run_training` in a daemon thread; returns a UUID `task_id`. Background wrapper is `_run_training_background`.
- **`compare_runs(run_id_a, run_id_b, config)`** — loads both `{run_id}_metadata.json` files via an internal `_load_meta` helper, calls `plot_comparison_dashboard()`, writes `compare_{a}_vs_{b}.png`, and returns `{status, image_url, run_a, run_b}`.
- **`get_task(task_id)`** — returns `{status: ...}` from the in-memory registry, or `"not_found"`.

**Task registry** (`_tasks` dict) is in-memory and protected by `_tasks_lock`. It is lost on server restart — tasks submitted before a restart will return `"not_found"`.

## `schemas.py` — Validation

`TrainingRequestSchema` uses marshmallow. Validate with `.load(request.json or {})` — it applies defaults and raises `ValidationError` on bad input. Routes catch `ValidationError` and return 400.

Current defaults (see `schemas.py`):

| Field | Default | Notes |
|-------|---------|-------|
| `episodes` | 100 | |
| `nodes` | `None` | **Resolved at runtime** as `params.get("nodes") or config.environment.num_nodes` (50) |
| `learning_rate` | 1e-4 | |
| `gamma` | 0.99 | |
| `batch_size` | 64 | |
| `death_threshold` | 0.3 | |
| `max_steps` | 1000 | |
| `seed` | 42 | |
| `model_type` | `"ddqn"` | Only `"ddqn"` and `"dqn"` are accepted |

`EvaluationRequestSchema` no longer exists — don't reintroduce it.

## Path Helpers

`routes.py` exposes two internal helpers:
- `_project_root()` — resolves the project root regardless of CWD
- `_abs(relative_path)` — joins `_project_root()` with a config-provided relative path (e.g. `config.paths.metrics`)

Always use `_abs()` when constructing file paths from config strings — `config.paths.*` values are relative strings, not absolute paths. Without `_abs()`, routes break when the server is launched from a directory other than the project root.

`_apply_config_defaults(run, config)` is used by `GET /api/history` to fill null config fields in old metadata files with current project defaults, so the frontend never sees `N/A` for known-defaultable fields.

## Gotchas

- **Task registry is ephemeral.** Restarting the server loses all queued/running task state. The frontend's polling loop will receive `"not_found"` after a restart.
- **`POST /api/train` blocks the server** for the full training duration. For long runs, use `POST /api/train/async` and poll `/api/tasks/<id>`.
- **`compare_runs()` needs both metadata files to exist.** If either `{run_id}_metadata.json` is missing, the endpoint returns an error — it does not regenerate anything.
- **Blueprint import dual-path.** The try/except import in `app.py` (`from .routes import api_bp` / `from backend.routes import api_bp`) is intentional — do not simplify it without testing both invocation styles.
- **Two-agent rule.** Only `DDQNAgent` and `DQNAgent` exist. Any new agent must subclass `BaseAgent` and be wired into the agent-selection block in `run_training()` — do not add baseline-policy shortcuts.
