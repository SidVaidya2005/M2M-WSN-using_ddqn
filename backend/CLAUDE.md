# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

> **See also:** [`../frontend/CLAUDE.md`](../frontend/CLAUDE.md) for the UI layer, and the root [`../CLAUDE.md`](../CLAUDE.md) for project-wide commands and architecture.

## Structure

```
backend/
├── app.py       — Flask app factory (create_app), entry point
├── routes.py    — All API route handlers, registered as blueprint `api_bp`
├── schemas.py   — Marshmallow request validation schemas
└── tasks.py     — Sync/async training execution and baseline benchmark logic
```

## App Factory & Blueprint

`create_app()` in `app.py` is the Flask app factory. Routes are registered as a blueprint (`api_bp`) at the `/api` prefix. The blueprint import uses a try/except to handle both `python -m backend.app` (package import) and `python backend/app.py` (direct script) invocations.

The loaded config object is stored on `app.config["CONFIG"]` — retrieve it inside route handlers with `current_app.config.get("CONFIG")`.

## Route Overview

| Method | Path | Sync/Async | Handler |
|--------|------|-----------|---------|
| GET | `/api/health` | sync | `health()` |
| GET | `/api/config` | sync | `get_config_endpoint()` |
| POST | `/api/train` | **blocking** | `train_model()` → `run_training()` |
| POST | `/api/train/async` | non-blocking | `train_model_async()` → `submit_training_task()` |
| GET | `/api/tasks/<task_id>` | sync | `task_status()` |
| GET | `/api/history` | sync | `get_history()` |
| POST | `/api/evaluate` | non-blocking | `start_benchmark()` → `submit_benchmark_task()` |
| GET | `/api/results/<path>` | sync | file serve |
| GET | `/api/visualizations/<path>` | sync | file serve |

## `tasks.py` — Execution Engine

This module contains all heavy computation. Two entry points:

- **`run_training(params, config)`** — synchronous; called directly by `POST /api/train`. Builds env + agent, runs `Trainer.train()`, saves `{run_id}_model.pth`, `{run_id}_plot.png`, and `{run_id}_metadata.json`.
- **`submit_training_task(params, config)`** — wraps `run_training` in a daemon thread; returns a UUID `task_id`.

Similarly, `run_baseline_benchmark()` / `submit_benchmark_task()` for baseline evaluation.

**Task registry** (`_tasks` dict) is in-memory and protected by `_tasks_lock`. It is lost on server restart — tasks submitted before a restart will return `"not_found"`.

`run_baseline_benchmark()` reconstructs the WSN environment from `{run_id}_metadata.json`, so the metadata file must exist and its `config` block must accurately reflect the original training parameters. It writes `{run_id}_evaluation.json` on completion.

## `schemas.py` — Validation

`TrainingRequestSchema` and `EvaluationRequestSchema` use marshmallow. Validate with `.load(request.json or {})` — it applies defaults (e.g. `episodes=100`, `nodes=550`, `model_type="ddqn"`) and raises `ValidationError` on bad input. Routes catch `ValidationError` and return 400.

## Path Helpers

`routes.py` exposes two internal helpers:
- `_project_root()` — resolves the project root regardless of CWD
- `_abs(relative_path)` — joins `_project_root()` with a config-provided relative path (e.g. `config.paths.metrics`)

Always use `_abs()` when constructing file paths from config strings — `config.paths.*` values are relative strings, not absolute paths.

## Gotchas

- **Task registry is ephemeral.** Restarting the server loses all queued/running task state. The frontend's polling loop will receive `"not_found"` after a restart.
- **`POST /api/train` blocks the server** for the full training duration. For long runs, use `POST /api/train/async` and poll.
- **`run_baseline_benchmark` requires an exact env match.** It reads `nodes`, `max_steps`, `death_threshold` from `{run_id}_metadata.json` — if the metadata is missing or has null fields, env construction may fall back to config defaults and produce mismatched results.
- **Blueprint import dual-path.** The try/except import in `app.py` (`from .routes import api_bp` / `from backend.routes import api_bp`) is intentional — do not simplify it without testing both invocation styles.
