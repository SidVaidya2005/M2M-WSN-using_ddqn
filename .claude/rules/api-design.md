# API Design Rules

## Route Table (Current)

| Endpoint | Mode | Purpose |
|----------|------|---------|
| `GET /api/health` | sync | Health check |
| `GET /api/config` | sync | Return current config as JSON |
| `POST /api/train` | **Blocking** — waits for training to finish | Frontend default (connection must stay open) |
| `POST /api/train/async` | Non-blocking — returns `task_id` immediately | Long runs, scripted calls |
| `GET /api/tasks/<task_id>` | sync | Poll task status |
| `GET /api/history` | sync | List all training runs (newest first) |
| `GET /api/compare?a=<run_id>&b=<run_id>` | sync | Generate DDQN-vs-DQN comparison PNG |
| `GET /api/results/<path>` | sync | Serve metrics JSON files |
| `GET /api/visualizations/<path>` | sync | Serve visualization PNGs |

**Removed endpoints:** `/api/evaluate` (baseline benchmarking) was removed in Phase 0.

For any new long-running operation: add a `submit_*_task()` function to `tasks.py` using the daemon-thread pattern, and expose both sync and async routes.

## Task Lifecycle

```
POST → submit_*_task() → returns task_id (UUID)
GET /api/tasks/<task_id> → { status: "queued" | "running" | "completed" | "failed" | "not_found" }
```

- Task state is **in-memory only** — lost on server restart
- `"not_found"` is returned as a 404 with `{"status": "not_found", "task_id": "..."}`
- Completed tasks stay in the registry indefinitely (no expiry)

## Validation Pattern

Every route that accepts a body must validate through a marshmallow schema:

```python
schema = TrainingRequestSchema()
try:
    params = schema.load(request.json or {})
except ValidationError as e:
    return jsonify({"status": "error", "message": str(e.messages)}), 400
```

Schemas live in `backend/schemas.py`. `load_default` provides the default value — never set defaults in route handlers.

**`nodes` field:** defaults to `None` in the schema; resolved at runtime as `params.get("nodes") or config.environment.num_nodes` (50 from config).

## Response Shape

Success:
```json
{ "status": "success", "run_id": "run_YYYYMMDD_HHMMSS", ... }
```

Error:
```json
{ "status": "error", "message": "..." }
```

Task poll:
```json
{ "status": "completed", "result": { ... } }
{ "status": "failed",    "error": "..." }
```

Always include `"status"` as the first field. Never return bare data without a status wrapper.

## Path Resolution in Routes

Use the `_abs()` helper for all file paths derived from config:

```python
# Correct
metadata_path = _abs(config.paths.metrics) / f"{run_id}_metadata.json"

# Wrong — config.paths.* are relative strings, not Path objects
metadata_path = Path(config.paths.metrics) / f"{run_id}_metadata.json"  # breaks if CWD ≠ project root
```

`_abs()` is defined in `routes.py` and calls `_project_root()` to anchor all paths to the project root regardless of CWD.

## History Endpoint

`GET /api/history` scans `results/metrics/` for `*_metadata.json` files. Each run's `config` block is filled with project defaults for any null fields via `_apply_config_defaults()`. Runs are returned newest-first.
