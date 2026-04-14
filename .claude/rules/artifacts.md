# Artifacts & Run ID Rules

## Run ID Format

```
run_YYYYMMDD_HHMMSS
```

Generated at the start of each training job:
```python
run_id = datetime.datetime.now().strftime("run_%Y%m%d_%H%M%S")
```

The run_id is the primary key for all output artifacts from a single training run.

## File Naming Convention

| File | Path | Written by |
|------|------|-----------|
| Model weights | `results/models/{run_id}_model.pth` | `tasks.run_training()` |
| Training metadata | `results/metrics/{run_id}_metadata.json` | `tasks.run_training()` |
| Training plot (4-panel) | `results/visualizations/{run_id}_plot.png` | `tasks.run_training()` |
| Comparison plot | `results/visualizations/compare_{id_a}_vs_{id_b}.png` | `tasks.compare_runs()` |

**Removed:**
- `{run_id}_evaluation.json` — baseline evaluation artifacts no longer exist
- Legacy `trained_model_ddqn.pth` — avoid referencing in new code

## Current Metadata Schema

`{run_id}_metadata.json` structure (Phase 3 schema — live):
```json
{
  "run_id": "run_20260414_080528",
  "timestamp": "<ISO 8601>",
  "model_used": "ddqn",
  "episodes": 100,
  "num_nodes": 50,
  "learning_rate": 0.0001,
  "gamma": 0.99,
  "death_threshold": 0.3,
  "max": 1000,
  "seed": 42,
  "metrics": {
    "mean_reward": 145.32,
    "max_reward": 180.5,
    "best_episode": 73,
    "avg_final_10": 172.4,
    "final_coverage": 0.87,
    "final_avg_soh": 0.94,
    "network_lifetime": 95
  },
  "series": {
    "episode_reward":  [...],
    "coverage":        [...],
    "avg_soh":         [...],
    "alive_fraction":  [...],
    "mean_soc":        [...],
    "step_counts":     [...]
  },
  "image_url": "/api/visualizations/{run_id}_plot.png",
  "model_path": "results/models/{run_id}_model.pth"
}
```

### Field Glossary

| Field | Description |
|-------|-------------|
| `model_used` | `"ddqn"` or `"dqn"` |
| `max` | Max steps per episode (aliased from `max_steps` in the wire protocol) |
| `final_coverage` | Grid coverage fraction at the last episode |
| `final_avg_soh` | Mean SoH across all nodes at the last episode |
| `network_lifetime` | Episode index at which `alive_fraction` first dropped below `1 - death_threshold` |
| `step_counts` | Per-episode count of steps survived |

## Config Field Handling

`config` fields may be `null` if training crashed before saving. Code that reads metadata must
handle null values. The `GET /api/history` endpoint fills null config fields with current
project defaults via `_apply_config_defaults()`.

Old runs (pre-Phase 3) use a `config` sub-object instead of top-level fields. Client code
should normalise with a helper like:
```js
const model = run.model_used ?? run.config?.model_type ?? "?";
const nodes  = run.num_nodes  ?? run.config?.nodes ?? "?";
```

## GET /api/history Behavior

Scans `results/metrics/` for `*_metadata.json` files. For each run, null config fields are
filled from `config.yaml` defaults. Runs are returned newest-first.

## Comparison Artifacts

`GET /api/compare?a=<run_id>&b=<run_id>` loads both metadata files, calls
`plot_comparison_dashboard()`, saves `compare_{a}_vs_{b}.png` to
`results/visualizations/`, and returns:
```json
{ "status": "success", "image_url": "...", "run_a": "...", "run_b": "..." }
```
