# Training Output Spec

Defines the exact contracts for artifacts produced by a training run.

## Artifacts Per Run

Every training run (CLI or API) produces exactly three files:

```
results/models/{run_id}_model.pth           # PyTorch model weights
results/metrics/{run_id}_metadata.json      # Run config + summary metrics
results/visualizations/{run_id}_plot.png    # Training visualization plot
```

## Current Metadata Schema (Phase 0-1)

```json
{
  "run_id": "run_20260413_153000",
  "timestamp": "2026-04-13T15:30:00.123456",
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
  "image_url": "/api/visualizations/run_20260413_153000_plot.png",
  "model_path": "results/models/run_20260413_153000_model.pth"
}
```

## Target Metadata Schema (Phase 3)

```json
{
  "run_id": "run_20260413_153000",
  "timestamp": "2026-04-13T15:30:00.123456",
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
    "final_coverage": 0.85,
    "final_avg_soh": 0.97,
    "network_lifetime": 87
  },
  "series": {
    "episode_reward": [12.5, 14.2, ...],
    "coverage": [0.6, 0.65, ...],
    "avg_soh": [1.0, 0.999, ...],
    "alive_fraction": [1.0, 1.0, ...],
    "mean_soc": [0.95, 0.92, ...]
  },
  "image_url": "/api/visualizations/run_20260413_153000_plot.png",
  "model_path": "results/models/run_20260413_153000_model.pth"
}
```

Key changes in Phase 3:
- `model_used`, `episodes`, `num_nodes`, `learning_rate`, `gamma`, `death_threshold`, `max` (alias for `max_steps`), `seed` promoted to top level
- `series` block added with per-episode arrays
- `metrics` extended with `final_coverage`, `final_avg_soh`, `network_lifetime`

## Current Visualization (Phase 0-1)

Single plot: reward over episodes with a smoothed curve.

## Target Visualization (Phase 3)

4-panel 2×2 figure:

| Panel | Content |
|-------|---------|
| Top-left | **Network Coverage** over episodes (line + 10-ep moving average) |
| Top-right | **Battery Health (avg SoH)** over episodes |
| Bottom-left | **Network Lifetime** — bar of per-episode steps-until-first-death + horizontal line at overall lifetime |
| Bottom-right | **Reward & Mean SoC** — dual-axis (reward left, mean SoC right) |

## API Response (POST /api/train)

```json
{
  "status": "success",
  "message": "Training completed with DDQN.",
  "run_id": "run_20260413_153000",
  "episodes": 100,
  "nodes": 50,
  "model_type": "ddqn",
  "mean_reward": 145.32,
  "max_reward": 180.5,
  "results": {
    "best_lifetime": 180.5,
    "best_episode": 73,
    "avg_lifetime_final_10": 172.4
  },
  "model_path": "results/models/run_20260413_153000_model.pth",
  "image_url": "/api/visualizations/run_20260413_153000_plot.png"
}
```
