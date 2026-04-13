# M2M Environment Spec

Defines the Machine-to-Machine specific behaviors that distinguish this from a generic WSN RL environment.

## Current State (Phase 1 complete)

The environment (`src/envs/wsn_env.py`) currently implements:
- 5-feature observation per node: SoC, SoH, last_action, distance_to_sink, activity_ratio
- Binary per-node actions: SLEEP (0) / AWAKE (1)
- Weighted reward: coverage, energy, SoH, balance
- Battery degradation (cycle + calendar) via BatteryModel
- Episode termination on death fraction exceeding threshold

Config sections `reward_weights`, `charging`, and `wake_cooperation` exist in `config.yaml` and `settings.py` but are **not yet wired** into the environment.

## Phase 2 Target — What Needs To Be Built

### 1. Observation Expansion (5 → 6 features)

Add `charging_flag` (index 5) per node:

| Idx | Feature | Range | Status |
|-----|---------|-------|--------|
| 0 | SoC (normalized) | [0,1] | ✅ Exists |
| 1 | SoH | [0,1] | ✅ Exists |
| 2 | last_action | {0,1} | ✅ Exists |
| 3 | distance_to_sink (norm) | [0,1] | ✅ Exists |
| 4 | activity_ratio (EMA) | [0,1] | ✅ Exists |
| 5 | **charging_flag** | {0,1} | 🔲 New |

`state_dim` changes from `N*5` to `N*6`. Agent networks auto-derive dimensions from env, so no hardcoded changes needed in agents.

### 2. Charging System

**Config:** `environment.charging`
```yaml
charging:
  enabled: true
  rate: 0.05         # SoC fraction recovered per step
  threshold: 0.2     # SoC below which node enters charging
```

**Behavior:**
1. When `SoC < charging.threshold`: node enters `charging = True`
2. While charging:
   - Node is forced SLEEP (regardless of agent action)
   - Battery recovers at `charging.rate` per step
   - Small calendar SoH penalty still applies per charge cycle
3. Exits charging when `SoC ≥ 0.95`

**BatteryModel changes needed:**
- Add `charging: bool` state attribute
- Add `charge(rate)` method: increases SoC up to E_max, applies calendar SoH penalty
- Add `is_charging` and `needs_charge(threshold)` helper properties

### 3. Coverage Metric Upgrade

Replace simple "fraction awake" with proper spatial coverage:
```
coverage = |covered_cells| / |grid_cells|
```
A node at position `p_i` covers a grid cell `c` if `||p_i - c|| ≤ sensing_radius`.

Fall back to cheap "fraction awake" if `environment.coverage_mode: simple`.

### 4. Cooperative Wake-Up (M2M Core Feature)

**Config:** `environment.wake_cooperation`
```yaml
wake_cooperation:
  low_battery_soc: 0.5
```

**Behavior (applied post-agent-action, pre-physics):**
1. After the agent emits its action vector
2. For each AWAKE node with `SoC ≤ low_battery_soc`:
   - Find its nearest SLEEP neighbor
   - Force that neighbor AWAKE for this step
3. Deduplicate (each node woken at most once per step)
4. Record affected IDs in `info["cooperative_wakes"]`

This is a **deterministic environment-side rule**, not learned by the agent. It preserves coverage during handoff without the agent having to learn the cooperation from scratch.

### 5. Reward Weights From Config

Wire `config.environment.reward_weights` into the step function:
```python
rw = config.environment.reward_weights
reward = rw.coverage * r_coverage + rw.energy * r_energy + rw.soh * r_soh + rw.balance * r_balance
```

Currently these are hardcoded as `10.0, 5.0, 1.0, 2.0` in `wsn_env.py`.

### 6. Enriched Info Dict

Each step returns:
```python
info = {
    "coverage": float,           # proper coverage metric
    "avg_soh": float,            # mean SoH across all nodes
    "alive_fraction": float,     # fraction of non-dead nodes
    "dead_count": int,           # count of dead nodes
    "mean_soc": float,           # mean normalized SoC
    "cooperative_wakes": list,   # node IDs woken by M2M rule
    "charging_count": int,       # nodes currently charging
}
```

The trainer uses these to build per-episode metric series for the Phase 3 visualization.
