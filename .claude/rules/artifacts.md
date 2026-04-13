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
| Training plot | `results/visualizations/{run_id}_plot.png` | `tasks.run_training()` |

**Removed:**
- `{run_id}_evaluation.json` — baseline evaluation artifacts no longer exist
- Legacy `trained_model_ddqn.pth` — avoid referencing in new code

## Current Metadata Schema

`{run_id}_metadata.json` structure:
```json
{
  "run_id": "run_20260406_080528",
  "timestamp": "<ISO 8601>",
  "config": {
    "model_type": "ddqn",
    "episodes": 100,
    "nodes": 50,
    "learning_rate": 0.0001,
    "gamma": 0.99,
    "batch_size": 64,
    "death_threshold": 0.3,
    "max_steps": 1000,
    "seed": 42
  },
  "metrics": {
    "mean_reward": 145.32,
    "max_reward": 180.5,
    "best_episode": 73,
    "avg_final_10": 172.4
  },
  "image_url": "/api/visualizations/{run_id}_plot.png",
  "model_path": "<relative path to .pth>"
}
```

### Planned Schema Upgrade (Phase 3)

The metadata will be restructured to:
- Promote `model_used`, `episodes`, `num_nodes`, `learning_rate`, `gamma`, `death_threshold`, `max`, `seed` to top-level fields
- Add `series` block with per-episode arrays: `episode_reward`, `coverage`, `avg_soh`, `alive_fraction`, `mean_soc`
- Add richer `metrics`: `final_coverage`, `final_avg_soh`, `network_lifetime`

## Config Field Handling

`config` fields may be `null` if training crashed before saving. Code that reads metadata must handle null values. The `GET /api/history` endpoint fills null config fields with current project defaults via `_apply_config_defaults()`.

## GET /api/history Behavior

Scans `results/metrics/` for `*_metadata.json` files. For each run, null config fields are filled from `config.yaml` defaults. Runs are returned newest-first.
