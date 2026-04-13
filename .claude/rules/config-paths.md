# Config & Paths Rules

## Accessing Config

```python
from config.settings import get_config
config = get_config()               # returns singleton; safe to call multiple times
config = get_config("custom.yaml")  # first call sets the YAML source; subsequent calls ignore the argument
```

Never instantiate the config class directly. The singleton is reset between test sessions via `settings_module._config = None` in `conftest.py` — replicate this in any new test module that imports config.

## Config Structure (YAML keys → attribute access)

### Training
```
config.training.episodes          config.training.batch_size
config.training.learning_rate     config.training.gamma
config.training.epsilon_start     config.training.epsilon_end
config.training.epsilon_decay     config.training.target_update_frequency
config.training.replay_buffer_size config.training.min_replay_size
```

### Environment
```
config.environment.num_nodes       # default: 50
config.environment.arena_size      # [500, 500]
config.environment.sink_position   # [250, 250]
config.environment.max_steps       # 1000
config.environment.death_threshold # 0.3
config.environment.seed            # 42
config.environment.timestep_energy_awake  # 1.0
config.environment.energy_sleep    # 0.01
```

### Environment — Reward Weights (Phase 1+)
```
config.environment.reward_weights.coverage   # 10.0
config.environment.reward_weights.energy     # 5.0
config.environment.reward_weights.soh        # 1.0
config.environment.reward_weights.balance    # 2.0
```

### Environment — Charging (configured, wired in Phase 2)
```
config.environment.charging.enabled    # true
config.environment.charging.rate       # 0.05
config.environment.charging.threshold  # 0.2
```

### Environment — Cooperative Wake-Up (configured, wired in Phase 2)
```
config.environment.wake_cooperation.low_battery_soc  # 0.5
```

### Paths
```
config.paths.models               → str, e.g. "results/models"
config.paths.metrics              → str, e.g. "results/metrics"
config.paths.visualizations       → str, e.g. "results/visualizations"
config.paths.logs                 → str, e.g. "logs"
```

### Visualization
```
config.visualization.save_plots        # true
config.visualization.plot_dpi          # 150
config.visualization.animation_interval # 100
```

`config.paths.create_all()` creates all result directories — call this once at app startup (already done in `create_app()`).

## Dataclass Hierarchy

```
Config
├── TrainingConfig
├── EnvironmentConfig
│   ├── RewardWeightsConfig
│   ├── ChargingConfig
│   └── WakeCooperationConfig
├── PathConfig
└── VisualizationConfig
```

All are dataclasses in `config/settings.py`. `Config.load()` constructs nested sub-configs from YAML dicts. `Config.to_dict()` serialises them all.

## Path Handling Rules

- `config.paths.*` values are **relative strings**, not `Path` objects
- Always wrap with `Path()` before using `/` for joining
- In `backend/` code, use `_abs(config.paths.*)` to anchor to project root
- In `src/` and `scripts/` code, use `Path(config.paths.models) / filename` — these run from project root by convention

```python
# Correct in backend/
model_path = _abs(config.paths.models) / f"{run_id}_model.pth"

# Correct in scripts/
model_path = Path(config.paths.models) / f"{run_id}_model.pth"

# Wrong everywhere
model_path = config.paths.models + "/" + filename   # string concat, not portable
```

## Scripts sys.path Convention

All scripts in `scripts/` insert the project root via:
```python
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
```
Replicate this in any new script, or run scripts only from the project root with `python scripts/myscript.py`.

## Flask App Config

The loaded config object is stored on `app.config["CONFIG"]`. Access inside route handlers:
```python
config = current_app.config.get("CONFIG")
if not config:
    return jsonify({"error": "Configuration not loaded"}), 500
```
Never call `get_config()` directly inside route handlers — use `current_app.config`.
