# Plan: Expand Per-Node Observation From 6 → 9 Features

## Context

`paper.md` (Section 4.1, Section 1.3 contribution #2, Abstract, Conclusion §7) now describes a **9-feature centralized observation** that adds three geometry-aware features per node:

- `x_norm = x_i / W`
- `y_norm = y_i / H`
- `awake_density = |{ j ≠ i : ã_j = 1 ∧ alive(j) ∧ ‖p_i − p_j‖ ≤ r_s }| / (N − 1)`

The codebase still emits **6 features** (`src/envs/wsn_env.py` line 101: `shape=(N * 6,)`). The aim of this change is to bring the implementation in line with the paper so reviewer-facing claims about the agent observing spatial geometry — and specifically the local awake-neighbour density that motivates the coverage objective — are backed by what the model actually receives. After this change, `state_dim = N * 9` (450 for N=50, 90 for tests with N=10).

This is a single mechanical expansion; reward, action space, training loop, agent class, and the constraint-aware execution layer all stay as-is. Existing 6-feature `.pth` checkpoints will be incompatible and are archived rather than reused.

## Feature Ordering (Paper-Aligned)

Each per-node block (`9i + k`) is laid out exactly as in `paper.md` §4.1:

| k | Feature                      | Range      | Status     |
|---|------------------------------|------------|------------|
| 0 | `SoC_i / E_max`              | [0, 1]     | unchanged  |
| 1 | `SoH_i`                      | [0, 1]     | unchanged  |
| 2 | `last_action_i` (executed)   | {0, 1}     | unchanged  |
| 3 | `dist_norm_i` (to sink)      | [0, 1]     | unchanged  |
| 4 | `x_i / W`                    | [0, 1]     | **NEW**    |
| 5 | `y_i / H`                    | [0, 1]     | **NEW**    |
| 6 | `awake_density_i`            | [0, 1]     | **NEW**    |
| 7 | `activity_ratio_i` (EMA)     | [0, 1]     | shifted 4→7 |
| 8 | `charging_flag_i`            | {0, 1}     | shifted 5→8 |

**Reordering implication**: existing test `tests/test_env.py::test_charging_node_forced_sleep_in_obs` reads `obs[5]` for the charging flag — this becomes `obs[8]`.

## Files to Modify

### Code (1 file)

- **`src/envs/wsn_env.py`**
  - Line 101: `shape=(N * 6,)` → `shape=(N * 9,)`
  - `_get_obs()` (lines 267–278): replace 6-element extend with 9-element extend in paper order. Insert `x_norm`, `y_norm`, `awake_density` between `dist_norm` and `activity_ratio`.
  - Compute `awake_density` from the post-step state: iterate alive nodes, use `self.last_action[j] == 1` and `np.linalg.norm(self.positions[i] - self.positions[j]) <= self.sensing_radius`, divide by `N - 1`. Guard `N == 1` (return 0.0).
  - At `reset()`, `last_action` is zero-initialised (line 127), so `awake_density` is correctly 0 in the first observation.

### Tests (3 files)

- **`tests/conftest.py`** — Line 13: `STATE_DIM = N_NODES * 6` → `STATE_DIM = N_NODES * 9`. The downstream `sample_state` fixture (line 33 area) inherits this automatically.
- **`tests/test_env.py`**
  - Lines 14–15, 25, 38: change every `node_count * 6` to `node_count * 9`.
  - Lines 216–228 (`test_charging_node_forced_sleep_in_obs`): change comment "6-feature block" → "9-feature block" and `obs[5]` → `obs[8]`.
  - **Add 3 new focused tests** in `tests/test_env.py`:
    1. `test_obs_xy_normalised_coordinates` — set known `positions[0]` after `reset()` (e.g. `(125, 250)` with arena `(500, 500)`); step and assert `obs[4] == 0.25`, `obs[5] == 0.5`.
    2. `test_obs_awake_density_count` — construct a 4-node env with controlled positions so node 0 has exactly 2 neighbours within `sensing_radius`; force the executed-action vector via charging/cooperation rules to leave both neighbours AWAKE; assert `obs[6] == 2 / (N − 1) == 2/3` for node 0.
    3. `test_obs_awake_density_zero_at_reset` — after `env.reset()` (no step yet), every block's `obs[9*i + 6]` is 0.0 because `last_action` is zero-initialised.
- **`tests/test_agent.py`** — Re-read to confirm no hardcoded `6` / `60` / `300` literals; update only if found. (Map said none exist; verify.)

### Documentation (5 files)

- **`CLAUDE.md`**
  - Line 21: "6-feature observation space" → "9-feature observation space"
  - Line 95: feature list → "9 features per node (SoC, SoH, last_action, distance_to_sink, x_norm, y_norm, awake_density, activity_ratio, charging_flag)"
- **`.claude/rules/rl-environment.md`**
  - Lines 11–26: replace the 6-row table with the 9-row paper-aligned table above. Update "For N=50: `state_dim = 300`. For tests with N=10: `state_dim = 60`." → 450 / 90.
- **`.claude/rules/agents-training.md`**
  - Line 62: "With 6 features per node: `state_dim = N * 6` (e.g. 300 for N=50, 60 for tests with N=10)." → "With 9 features per node: `state_dim = N * 9` (e.g. 450 for N=50, 90 for tests with N=10)."
- **`README.md`** (lines 282–291)
  - Header: "6 features per node, flattened into a shape `(N * 6,)` array (`state_dim = 300` for 50 nodes)" → "9 features per node, flattened into a shape `(N * 9,)` array (`state_dim = 450` for 50 nodes)"
  - Add 4 rows to match paper order (insert x_norm, y_norm, awake_density between dist_norm and activity_ratio; charging_flag becomes index 8).
- **`workflow.md`**
  - Lines 19, 37, 40: every `\mathbb{R}^{6N}` and "300-dimensional" → `\mathbb{R}^{9N}` and "450-dimensional".
  - Line 126: `\mathbb{R}^{300} \xrightarrow{\text{FC}_{512}}` → `\mathbb{R}^{450}`.
  - Line 354: `Online Q-Net\n(300→512→256→100)` → `(450→512→256→100)`.

### Artifact Housekeeping (no code change, but in plan scope)

- Create `results/models/legacy_6feature/` (do not commit a `.gitkeep` — directory only needs to exist).
- Move all 10 existing `.pth` files from `results/models/` into `results/models/legacy_6feature/`:
  ```
  run_20260320_162545_model.pth     run_20260418_115020_model.pth
  run_20260320_162546_model.pth     run_20260418_115451_model.pth
  run_20260324_175842_model.pth     run_20260418_124132_model.pth
  run_20260406_080528_model.pth     run_20260418_124353_model.pth
  trained_model_ddqn.pth            run_20260418_134826_model.pth
  ```
- **Do not move** `results/metrics/*_metadata.json` or `results/visualizations/*.png` — `GET /api/history` still reads metadata, and the comparison and visualisation flows continue to work for legacy runs.

## Functions / Utilities Already in Place — Reuse, Don't Rebuild

- `self.positions` (`src/envs/wsn_env.py:90, 119`) — `(N, 2)` float array; reuse for `x_norm`, `y_norm`.
- `self.arena_size` (line 74) — tuple `(W, H)`; divisor for normalisation.
- `self.sensing_radius` (line 88) — coverage sensing radius; reuse as the `r_s` for awake-neighbour search.
- `self.last_action` (line 93, written back on lines 192/197/201/203) — already holds the **executed** action `ã_t` after charging override and cooperative wake-up. Reuse directly; do not introduce a parallel tracking array.
- `self.batteries[j].is_dead()` — reuse for the alive mask in `awake_density`.
- `state_dim = env.observation_space.shape[0]` is already derived dynamically in `backend/tasks.py:70` and in `Trainer`. **No change needed in agent constructors, `backend/tasks.py`, `scripts/train.py`, or `scripts/compare.py`** — the `(N * 9,)` shape propagates automatically.

## What Is Out of Scope

- No change to reward function, weights, or `r_s = 100m` value.
- No change to `BaseAgent` / `DDQNAgent` / `DQNAgent` class signatures or constructors.
- No change to charging or cooperative wake-up rules — only the *observation* of them changes.
- No change to `paper.md` — already updated to 9 features.
- No retraining or migration of legacy `.pth` checkpoints.

## Verification

After implementation, run in order:

1. **Unit tests pass**
   ```bash
   pytest tests/ -v
   ```
   Expect: all existing tests pass with `9 * N_NODES` shape; the 3 new feature tests pass; `test_charging_node_forced_sleep_in_obs` passes with `obs[8]`.

2. **Shape sanity check from REPL**
   ```bash
   python -c "from src.envs.wsn_env import WSNEnv; e = WSNEnv(N=50); o, _ = e.reset(); print(o.shape, o.shape[0] == 50*9)"
   ```
   Expect: `(450,) True`.

3. **Smoke training** (short)
   ```bash
   python scripts/train.py --episodes 5 --nodes 50 --model-type ddqn --seed 42
   ```
   Expect: completes; new `results/models/run_*_model.pth` written with input-layer dim 450; metadata JSON valid.

4. **Backend smoke** (sync route)
   ```bash
   python -m backend.app &  # localhost:5001
   curl -s -X POST http://localhost:5001/api/train \
     -H 'Content-Type: application/json' \
     -d '{"episodes": 3, "nodes": 50, "model_type": "ddqn"}'
   ```
   Expect: `status: success`, new run id, no shape errors.

5. **History/compare still works for legacy runs**
   ```bash
   curl -s http://localhost:5001/api/history | jq '.[0]'
   ```
   Expect: legacy metadata still surfaces (PNGs still served), since only `.pth` files moved.

6. **Confirm legacy `.pth` files cleanly fail to load (expected)**
   ```bash
   python -c "
   import torch
   sd = torch.load('results/models/legacy_6feature/trained_model_ddqn.pth', map_location='cpu')
   print('first layer in shape:', next(iter(sd.values())).shape)
   "
   ```
   Expect: shape includes `300` (i.e. 6×50), confirming why a 9-feature agent cannot consume it.

## Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| Reordering features (charging shifts 5→8) silently breaks downstream code that reads obs by index. | Only one such read exists (`tests/test_env.py:227`); explicitly updated. Step-3 grep over `src/`, `backend/`, `frontend/`, `scripts/` for `obs\[\s*[0-9]` to confirm no other index reads. |
| `awake_density` uses `last_action`, which is zero at reset — first-observation feature is 0. | Documented in §4.1 of the paper (it's the EMA-style cold start); a dedicated test (`test_obs_awake_density_zero_at_reset`) pins the behaviour. |
| Loading any of the 10 archived `.pth` checkpoints into a 9-feature agent will fail. | Files moved to `legacy_6feature/`; metadata stays in place so `/api/history` and comparison plots still work. New runs after this change are forward-compatible. |
| Tests in CI that depend on saved checkpoints (none currently). | Confirmed `tests/` does not load any `.pth`; greenfield test runs only. |

## Critical Files — Quick Reference

| File | Why it matters |
|------|----------------|
| `src/envs/wsn_env.py` | Only code file with semantic changes (`_get_obs`, `observation_space`). |
| `tests/conftest.py:13` | `STATE_DIM` constant; shape ripple. |
| `tests/test_env.py` | Shape assertions + reordered charging-flag index test + 3 new tests. |
| `backend/tasks.py:70` | Already derives `state_dim` from env — no edit, but verify after changes. |
| `CLAUDE.md`, `.claude/rules/rl-environment.md`, `.claude/rules/agents-training.md` | Documentation must match code or future agents are misled. |
| `README.md`, `workflow.md` | User-facing / submission docs cited by the paper context. |
| `results/models/` | 10 legacy `.pth` files relocated to `legacy_6feature/` subfolder. |
