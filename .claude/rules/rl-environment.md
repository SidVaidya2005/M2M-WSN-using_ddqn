# RL Environment Rules

## WSNEnv Critical Rules

- **`reset()` returns `(obs, info)` tuple** — Gymnasium-compliant. Use `state, info = env.reset()`.
- **`step()` returns 4 values**: `(next_state, reward, done, info)` — not 5. No truncation flag.
- Always call `env.close()` after evaluation loops (releases simulation resources).

## Observation Space

Currently 5 features per node, flat array of shape `(N * 5,)`:

| Index (per node) | Feature | Range |
|-----------------|---------|-------|
| 0 | State of Charge (SoC) | [0, 1] |
| 1 | State of Health (SoH) | [0, 1] |
| 2 | last_action | {0, 1} |
| 3 | distance_to_sink (normalized) | [0, 1] |
| 4 | activity_ratio (EMA) | [0, 1] |

**Planned (Phase 2):** 6th feature `charging_flag` ({0, 1}) will be added — `state_dim` becomes `N * 6`.

`state_dim = env.observation_space.shape[0]` — always derive this from the env, never hardcode.

## Action Space

Per-node binary: `{SLEEP=0, AWAKE=1}`. The agent outputs an action vector of length `N`.

## Reward Function

```
reward = w_cov * r_coverage + w_eng * r_energy + w_soh * r_soh + w_bal * r_balance
```

Weights are configured in `config.yaml` under `environment.reward_weights`:
- `coverage: 10.0` — fraction of nodes awake (more awake = better coverage)
- `energy: 5.0` — normalized energy efficiency (lower drain = higher score)
- `soh: 1.0` — average battery health across nodes
- `balance: 2.0` — fairness (low std of charge levels)

> **Note:** The current `wsn_env.py` still uses hardcoded weights `(10, 5, 1, 2)`.
> Phase 2 will wire these from `config.environment.reward_weights`.

## BatteryModel

Located in `src/envs/battery_model.py`. Tracks SoC and SoH per node.

- SoH degrades via cycle-based + calendar degradation — it does **not** recover
- A node is dead when `SoC <= soc_threshold` (default `0.01`) or `SoH <= soh_threshold` (default `0.05`)
- Episode ends when `dead_nodes > death_threshold * N` (default `death_threshold=0.3`)

### Charging (Phase 2 — configured but not yet wired)

Config section `environment.charging`:
- `enabled: true` — master toggle
- `rate: 0.05` — SoC fraction recovered per step while charging
- `threshold: 0.2` — SoC below which a node enters charging state

### Cooperative Wake-Up (Phase 2 — configured but not yet wired)

Config section `environment.wake_cooperation`:
- `low_battery_soc: 0.5` — when an AWAKE node's SoC drops to this level, its nearest SLEEP neighbor is forced AWAKE for the next step

## WSNEnv Constructor

```python
WSNEnv(
    N=50,                         # number of sensor nodes (default from config)
    arena_size=(500, 500),        # physical arena in meters
    sink=(250, 250),              # sink node position
    max_steps=1000,               # steps per episode
    death_threshold=0.3,          # fraction of dead nodes triggering episode end
)
```

Default node count is **50** (from `config.yaml`), not 550. Always reconstruct with the **same parameters used during training** when evaluating.

## Info Dict (per step)

```python
info = {
    "total_energy": float,      # total energy consumed this step
    "coverage_ratio": float,    # fraction of awake nodes
    "avg_soh": float,           # mean SoH across all nodes
    "dead_nodes": int,          # count of dead nodes
    "step_count": int,          # current step number
}
```
