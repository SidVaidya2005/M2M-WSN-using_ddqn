# API Design Rules

## Sync vs Async

| Endpoint | Mode | Use when |
|----------|------|----------|
| `POST /api/train` | **Blocking** — waits for training to finish | Frontend default (connection must stay open) |
| `POST /api/train/async` | Non-blocking — returns `task_id` immediately | Long runs, scripted calls |
| `POST /api/evaluate` | Non-blocking — returns `task_id` | Always (baselines take minutes) |

For any new long-running operation: add a `submit_*_task()` function to `tasks.py` using the daemon-thread pattern, and expose both sync and async routes.

## Task Lifecycle

```
POST → submit_*_task() → returns task_id (UUID)
GET /api/tasks/<task_id> → { status: "queued" | "running" | "completed" | "failed" | "not_found" }
```

- Task state is **in-memory only** — lost on server restart
- `"not_found"` is a valid status (not a 404) — handle it in clients the same as `"failed"`
- Completed tasks stay in the registry indefinitely (no expiry); do not rely on this for large result storage

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
