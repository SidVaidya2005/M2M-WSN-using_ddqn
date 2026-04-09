# Config & Paths Rules

## Accessing Config

```python
from config.settings import get_config
config = get_config()               # returns singleton; safe to call multiple times
config = get_config("custom.yaml")  # first call sets the YAML source; subsequent calls ignore the argument
```

Never instantiate the config class directly. The singleton is reset between test sessions via `settings_module._config = None` in `conftest.py` — replicate this in any new test module that imports config.

## Config Structure (YAML keys → attribute access)

```
config.training.episodes          config.training.batch_size
config.training.learning_rate     config.training.gamma

config.environment.num_nodes      config.environment.max_steps
config.environment.arena_size     config.environment.sink_position
config.environment.death_threshold

config.paths.models               → str, e.g. "results/models"
config.paths.metrics              → str, e.g. "results/metrics"
config.paths.visualizations       → str, e.g. "results/visualizations"
config.paths.logs                 → str, e.g. "logs"
```

`config.paths.create_all()` creates all result directories — call this once at app startup (already done in `create_app()`).

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
