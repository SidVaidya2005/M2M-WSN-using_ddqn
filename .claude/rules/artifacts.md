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
| Baseline evaluation | `results/metrics/{run_id}_evaluation.json` | `tasks.run_baseline_benchmark()` |
| Training plot | `results/visualizations/{run_id}_plot.png` | `tasks.run_training()` |

**Legacy**: `results/models/trained_model_ddqn.pth` exists for CLI scripts only. Never reference this in backend code — use the `run_id` pattern.

## Metadata Schema

`{run_id}_metadata.json` structure:
```json
{
  "run_id": "run_20260406_080528",
  "timestamp": "<ISO 8601>",
  "config": {
    "model_type": "ddqn",
    "episodes": 100,
    "nodes": 550,
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
  "model_path": "<absolute path to .pth>"
}
```

`config` fields may be `null` if training crashed before saving. Code that reads metadata must handle null values.

## GET /api/history Behavior

Scans `results/metrics/` for `*_metadata.json` files. For each run, if a matching `{run_id}_evaluation.json` exists, it is inlined as `run["evaluation"]`. Runs are returned newest-first.

## Benchmark Reconstruction Rule

`run_baseline_benchmark()` rebuilds `WSNEnv` from the metadata `config` block. If any config field is `null`, it falls back to the current `config.yaml` defaults — which may not match the original training environment. Always ensure training runs complete fully before benchmarking.

## migrate_legacy_runs.py

Copies pre-run_id artifacts (e.g. `trained_model_ddqn.pth`) into the `run_{timestamp}_*` naming scheme. Safe to run multiple times (idempotent, copies only — never moves or deletes). Run this once after pulling from a branch that predates the run_id system.
