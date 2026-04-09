# RL Environment Rules

## WSNEnv Critical Rules

- **`reset()` returns a plain `np.ndarray`**, not `(obs, info)`. Never unpack: `state, _ = env.reset()` crashes because the array has `N*5` elements (e.g. 2750 for 550 nodes).
- **`step()` returns 4 values**: `(next_state, reward, done, info)` — not 5. No truncation flag.
- Always call `env.close()` after evaluation loops (releases simulation resources).

## Observation Space

5 features per node, flat array of shape `(N * 5,)`:

| Index (per node) | Feature | Range |
|-----------------|---------|-------|
| 0 | State of Charge (SoC) | [0, 1] |
| 1 | State of Health (SoH) | [0, 1] |
| 2 | last_action | {0, 1} |
| 3 | distance_to_sink (normalized) | [0, 1] |
| 4 | activity_ratio | [0, 1] |

`state_dim = env.observation_space.shape[0]` — always derive this from the env, never hardcode.

## Action Space

Per-node binary: `{SLEEP=0, AWAKE=1}`. The agent outputs an action vector of length `N`.

## Reward Function

```
reward = 10 * r_coverage + 5 * r_energy + 1 * r_soh + 2 * r_balance
```

- `r_coverage`: fraction of nodes awake (more awake = better coverage)
- `r_energy`: normalized energy efficiency (lower drain = higher score)
- `r_soh`: average battery health across nodes
- `r_balance`: fairness (low std of charge levels)

To adjust tradeoffs, change weights in `src/envs/wsn_env.py step()`. Retrain after any weight change.

## BatteryModel

Located in `src/envs/battery_model.py`. Tracks SoC and SoH per node.

- SoH degrades via cycle-based + calendar degradation — it does **not** recover
- A node is dead when `SoC < death_threshold` (default `0.3`)
- Episode ends when enough nodes die (controlled by `death_threshold` passed to `WSNEnv`)

## WSNEnv Constructor

```python
WSNEnv(
    N=550,                        # number of sensor nodes
    arena_size=(500, 500),        # physical arena in meters
    sink=(250, 250),              # sink node position
    max_steps=1000,               # steps per episode
    death_threshold=0.3,          # SoC at which a node is considered dead
)
```

Always reconstruct with the **same parameters used during training** when evaluating — mismatch produces invalid comparison results.
