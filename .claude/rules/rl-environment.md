# RL Environment Rules

## WSNEnv Critical Rules

- **`reset()` returns `(obs, info)` tuple** — Gymnasium-compliant. Use `state, info = env.reset()`.
- **`step()` returns 4 values**: `(next_state, reward, done, info)` — not 5. No truncation flag.
- Always call `env.close()` after evaluation loops (releases simulation resources).

## Observation Space

**6 features per node**, flat array of shape `(N * 6,)`:

| Index (per node) | Feature | Range |
|-----------------|---------|-------|
| 0 | State of Charge (SoC, normalized) | [0, 1] |
| 1 | State of Health (SoH) | [0, 1] |
| 2 | last_action | {0, 1} |
| 3 | distance_to_sink (normalized) | [0, 1] |
| 4 | activity_ratio (EMA) | [0, 1] |
| 5 | charging_flag | {0, 1} |

`state_dim = env.observation_space.shape[0]` — always derive this from the env, never hardcode.
For N=50: `state_dim = 300`. For tests with N=10: `state_dim = 60`.

## Action Space

Per-node binary: `{SLEEP=0, AWAKE=1}`. The agent outputs an action vector of length `N`.

## Reward Function

```
reward = w_cov * r_coverage + w_eng * r_energy + w_soh * r_soh + w_bal * r_balance
```

Weights are pulled from the `WSNEnv` constructor (which reads from `config.yaml` under
`environment.reward_weights`):
- `coverage: 10.0` — grid coverage fraction (awake nodes within sensing radius of grid cells)
- `energy: 5.0` — normalized energy efficiency (lower drain = higher score)
- `soh: 1.0` — average battery health across nodes
- `balance: 2.0` — fairness (low std of charge levels)

## BatteryModel

Located in `src/envs/battery_model.py`. Tracks SoC and SoH per node.

- SoH degrades via cycle-based + calendar degradation — it does **not** recover
- A node is dead when `SoC <= soc_threshold` (default `0.01`) or `SoH <= soh_threshold` (default `0.05`)
- Episode ends when `dead_nodes > death_threshold * N` (default `death_threshold=0.3`)

### Charging

Config section `environment.charging` (wired since Phase 2):
- `enabled: true` — master toggle
- `rate: 0.05` — SoC fraction recovered per step while charging
- `threshold: 0.2` — SoC below which a node enters charging state (exits at SoC ≥ 0.95)

A charging node is forced SLEEP and its `charging_flag` (obs index 5) is set to `1`.

### Cooperative Wake-Up

Config section `environment.wake_cooperation` (wired since Phase 2):
- `low_battery_soc: 0.5` — when an AWAKE node's SoC drops to ≤ this fraction (of E_max),
  its nearest SLEEP neighbor is forced AWAKE for the next step

Triggered after the agent's action but before physics; affected node IDs logged in
`info["cooperative_wakes"]`.

## WSNEnv Constructor

```python
WSNEnv(
    N=50,                              # number of sensor nodes
    arena_size=(500, 500),             # physical arena in metres
    sink=(250, 250),                   # sink node position
    max_steps=1000,                    # steps per episode
    death_threshold=0.3,               # fraction of dead nodes that ends the episode
    seed=42,                           # RNG seed
    reward_weights=(10.0, 5.0, 1.0, 2.0),  # (coverage, energy, soh, balance)
    charging_enabled=True,             # enable charging state machine
    charging_rate=0.05,                # SoC fraction recovered per step
    charging_threshold=0.2,            # SoC below which a node enters charging
    wake_cooperation_soc=0.5,          # SoC fraction threshold for cooperative wake
    sensing_radius=50.0,               # node sensing radius in metres (for coverage grid)
)
```

Default node count is **50** (from `config.yaml`), not 550.

## Info Dict (per step)

```python
info = {
    "coverage":           float,  # grid coverage fraction (0–1)
    "avg_soh":            float,  # mean SoH across all nodes
    "alive_fraction":     float,  # fraction of nodes still alive
    "dead_count":         int,    # count of dead nodes this step
    "mean_soc":           float,  # mean SoC (normalized) across all nodes
    "cooperative_wakes":  list,   # node IDs woken by cooperative rule this step
    "charging_count":     int,    # number of nodes currently charging
    "step_count":         int,    # current step number within the episode
}
```
