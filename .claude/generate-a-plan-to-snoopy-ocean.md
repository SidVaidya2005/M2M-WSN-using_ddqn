# 9-Feature Observation Space Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expand the WSN environment's per-node observation from 6 → 9 features (adding `x_norm`, `y_norm`, `awake_density`) so the implementation matches paper.md §4.1 and Section 1.3 contribution #2.

**Architecture:** Single mechanical expansion in `WSNEnv._get_obs()`. Reorder features into paper-aligned layout (charging_flag shifts from index 5 → 8). `state_dim` propagates automatically through `Trainer` and `backend/tasks.py` because it's already derived from `env.observation_space.shape[0]`. Existing 6-feature `.pth` checkpoints are archived rather than migrated.

**Tech Stack:** Python 3, NumPy, Gymnasium, PyTorch (no version bumps).

## Context

`paper.md` (now committed) describes a 9-feature centralized observation that adds three geometry-aware features per node: `x_norm`, `y_norm`, and `awake_density` (count of awake live neighbours within `sensing_radius`, normalised by `N-1`). The codebase still emits 6 features (`src/envs/wsn_env.py:101` declares `shape=(N * 6,)`). Reviewer-facing claims about the agent observing spatial geometry — and specifically the local awake-neighbour density that motivates the coverage objective — must be backed by what the model actually receives. After this change `state_dim = N * 9` (450 for N=50, 90 for tests with N=10).

The full design rationale, paper-aligned feature ordering table, scope boundaries, risks, and verification matrix are documented in `.claude/plan.md`. **That document is the source of truth**; this plan only sequences the work into executable TDD steps.

## Final 9-Feature Layout (per-node block at indices `9i + k`)

| k | Feature              | Source                                            |
|---|----------------------|---------------------------------------------------|
| 0 | `SoC / E_max`        | `batt.soc / batt.E_max`                           |
| 1 | `SoH`                | `batt.soh`                                        |
| 2 | `last_action`        | `self.last_action[i]`                             |
| 3 | `dist_norm`          | `self.dist_norm[i]`                               |
| 4 | `x_norm` **NEW**     | `self.positions[i, 0] / self.arena_size[0]`       |
| 5 | `y_norm` **NEW**     | `self.positions[i, 1] / self.arena_size[1]`       |
| 6 | `awake_density` **NEW** | live neighbours within `sensing_radius` with `last_action == 1`, ÷ `(N-1)` |
| 7 | `activity_ratio`     | `self.recent_activity[i]` (was index 4)           |
| 8 | `charging_flag`      | `float(batt.charging)` (was index 5)              |

## File Structure

**Modify:**
- `src/envs/wsn_env.py:101, 266-279` — observation space shape + `_get_obs()` body
- `tests/conftest.py:15` — `STATE_DIM` constant
- `tests/test_env.py:17, 28, 46, 137-138` — shape assertions + reordered charging-flag index
- `CLAUDE.md:21, 95` — feature count + feature list
- `.claude/rules/rl-environment.md:11-26` — observation table + `state_dim` examples
- `.claude/rules/agents-training.md:62` — `state_dim` example
- `README.md:282-292` — observation header + table
- `workflow.md:19, 37, 40, 126, 354` — `\mathbb{R}^{6N}` → `\mathbb{R}^{9N}`, `300` → `450`

**Create (test cases inserted into `tests/test_env.py`):**
- `test_obs_xy_normalised_coordinates`
- `test_obs_awake_density_count`
- `test_obs_awake_density_zero_at_reset`

**Relocate (no code change):**
- `results/models/*.pth` (10 files) → `results/models/legacy_6feature/`

**Reuse (do NOT rebuild):**
- `self.positions` (`wsn_env.py:90, 119`), `self.arena_size` (`wsn_env.py:74`), `self.sensing_radius` (`wsn_env.py:88`), `self.last_action` (`wsn_env.py:93, 192, 203`), `BatteryModel.is_dead()` (`battery_model.py:91-103`).

---

### Task 1: Add the three new feature-specific tests (failing)

**Files:**
- Modify: `tests/test_env.py` — append 3 new test functions at end of file

- [ ] **Step 1: Add `test_obs_xy_normalised_coordinates`**

```python
def test_obs_xy_normalised_coordinates():
    """obs[9*i + 4] == x_i / W and obs[9*i + 5] == y_i / H after a step."""
    import numpy as np
    from src.envs.wsn_env import WSNEnv

    env = WSNEnv(N=4, arena_size=(500, 500), max_steps=10, seed=0)
    env.reset()
    env.positions[0] = np.array([125.0, 250.0])
    obs, _, _, _ = env.step(np.array([1, 1, 1, 1]))

    assert obs[9 * 0 + 4] == pytest.approx(0.25)
    assert obs[9 * 0 + 5] == pytest.approx(0.5)
```

- [ ] **Step 2: Add `test_obs_awake_density_count`**

```python
def test_obs_awake_density_count():
    """Node 0 with exactly 2 awake live neighbours within sensing_radius
    reports awake_density = 2 / (N - 1)."""
    import numpy as np
    from src.envs.wsn_env import WSNEnv

    env = WSNEnv(N=4, arena_size=(500, 500), sensing_radius=100.0,
                 max_steps=10, seed=0)
    env.reset()
    # Node 0 at origin; nodes 1, 2 within 100m; node 3 well outside.
    env.positions[0] = np.array([250.0, 250.0])
    env.positions[1] = np.array([300.0, 250.0])  # 50m from node 0
    env.positions[2] = np.array([250.0, 320.0])  # 70m from node 0
    env.positions[3] = np.array([10.0, 10.0])    # ~340m from node 0

    obs, _, _, _ = env.step(np.array([1, 1, 1, 1]))

    assert obs[9 * 0 + 6] == pytest.approx(2.0 / (4 - 1))
```

- [ ] **Step 3: Add `test_obs_awake_density_zero_at_reset`**

```python
def test_obs_awake_density_zero_at_reset():
    """At reset(), last_action is zero-initialised so awake_density is 0
    for every node."""
    from src.envs.wsn_env import WSNEnv

    env = WSNEnv(N=10, max_steps=10, seed=0)
    obs, _ = env.reset()

    for i in range(10):
        assert obs[9 * i + 6] == 0.0
```

- [ ] **Step 4: Run new tests; expect failures**

Run: `pytest tests/test_env.py::test_obs_xy_normalised_coordinates tests/test_env.py::test_obs_awake_density_count tests/test_env.py::test_obs_awake_density_zero_at_reset -v`

Expected: All three FAIL — `xy_normalised` and `awake_density_count` fail with `IndexError` (obs only has shape `(N*6,)` so `9*0+5` reaches into next node's block; assertion values won't match); `density_zero_at_reset` fails for the same indexing reason.

- [ ] **Step 5: Commit failing tests**

```bash
git add tests/test_env.py
git commit -m "test: add failing tests for 9-feature observation (x_norm, y_norm, awake_density)"
```

---

### Task 2: Update env observation shape, feature ordering, and dependent shape constants atomically

This task is one atomic commit because changing the observation shape breaks `STATE_DIM` and the `obs[5]` charging-flag read simultaneously.

**Files:**
- Modify: `src/envs/wsn_env.py:101` (shape) and `src/envs/wsn_env.py:266-279` (`_get_obs`)
- Modify: `tests/conftest.py:15`
- Modify: `tests/test_env.py:17, 28, 46, 137-138`

- [ ] **Step 1: Update `observation_space` shape in `src/envs/wsn_env.py`**

Replace `wsn_env.py:101-103`:

```python
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(N * 9,), dtype=np.float32
        )
```

- [ ] **Step 2: Rewrite `_get_obs()` in `src/envs/wsn_env.py:266-279`**

Replace the entire method body with the paper-aligned 9-feature layout including `awake_density`:

```python
    def _get_obs(self) -> np.ndarray:
        """Construct 9-feature-per-node observation vector (paper §4.1).

        Layout per node i:
            0: SoC / E_max
            1: SoH
            2: last_action (executed)
            3: dist_norm (to sink)
            4: x_norm = x_i / W
            5: y_norm = y_i / H
            6: awake_density (live awake neighbours within sensing_radius / (N-1))
            7: activity_ratio (EMA)
            8: charging_flag
        """
        W, H = self.arena_size[0], self.arena_size[1]
        obs = []
        for i in range(self.N):
            batt = self.batteries[i]

            if self.N == 1:
                awake_density = 0.0
            else:
                count = 0
                for j in range(self.N):
                    if j == i:
                        continue
                    if self.batteries[j].is_dead():
                        continue
                    if self.last_action[j] != 1:
                        continue
                    if (np.linalg.norm(self.positions[i] - self.positions[j])
                            <= self.sensing_radius):
                        count += 1
                awake_density = count / (self.N - 1)

            obs.extend([
                batt.soc / batt.E_max,
                batt.soh,
                float(self.last_action[i]),
                self.dist_norm[i],
                self.positions[i, 0] / W,
                self.positions[i, 1] / H,
                awake_density,
                self.recent_activity[i],
                float(batt.charging),
            ])
        return np.array(obs, dtype=np.float32)
```

- [ ] **Step 3: Update `STATE_DIM` in `tests/conftest.py:15`**

Replace:
```python
STATE_DIM = N_NODES * 6  # 6 features per node (Phase 2: added charging_flag)
```
with:
```python
STATE_DIM = N_NODES * 9  # 9 features per node (paper §4.1: + x_norm, y_norm, awake_density)
```

- [ ] **Step 4: Update shape assertions in `tests/test_env.py`**

Find every `* 6` shape literal (lines 17, 28, 46) and change to `* 9`. Specifically:

- Line ~17: `assert wsn_env.observation_space.shape == (node_count * 9,)`
- Line ~28: `assert obs.shape == (node_count * 9,)`
- Line ~46: `assert obs.shape == (node_count * 9,)`

- [ ] **Step 5: Update `test_charging_node_forced_sleep_in_obs` in `tests/test_env.py:137-138`**

Replace the comment and the index:

```python
    # charging_flag is index 8 (0-based) in each 9-feature block (paper §4.1)
    charging_flag_node0 = obs[8]
```

- [ ] **Step 6: Run the full test suite**

Run: `pytest tests/ -v`

Expected: ALL tests pass — the 3 new feature tests from Task 1 now succeed, all shape assertions succeed, `test_charging_node_forced_sleep_in_obs` succeeds with the new `obs[8]` index.

If any test fails, do NOT proceed. Re-read the failure and fix in this task before committing.

- [ ] **Step 7: Shape sanity check from REPL**

Run:
```bash
python -c "from src.envs.wsn_env import WSNEnv; e = WSNEnv(N=50); o, _ = e.reset(); print(o.shape, o.shape[0] == 50*9)"
```

Expected output: `(450,) True`

- [ ] **Step 8: Commit**

```bash
git add src/envs/wsn_env.py tests/conftest.py tests/test_env.py
git commit -m "feat(env): expand observation to 9 features per node (paper §4.1)

Adds x_norm, y_norm, and awake_density to the per-node observation
block, reorders to match paper-aligned layout. state_dim is now N*9
(450 for N=50, 90 for N=10).

charging_flag moved from index 5 -> 8; activity_ratio from 4 -> 7."
```

---

### Task 3: Update documentation files to reflect 9-feature observation

No tests are involved — these are documentation-only edits. Group into one commit.

**Files:**
- Modify: `CLAUDE.md:21, 95`
- Modify: `.claude/rules/rl-environment.md:11-26`
- Modify: `.claude/rules/agents-training.md:62`
- Modify: `README.md:282-292`
- Modify: `workflow.md:19, 37, 40, 126, 354`

- [ ] **Step 1: Update `CLAUDE.md`**

- Line 21: `6-feature observation space` → `9-feature observation space`
- Line 95: `Observation: 6 features per node (SoC, SoH, last_action, distance_to_sink, activity_ratio, charging_flag).` → `Observation: 9 features per node (SoC, SoH, last_action, distance_to_sink, x_norm, y_norm, awake_density, activity_ratio, charging_flag).`

- [ ] **Step 2: Update `.claude/rules/rl-environment.md` lines 11–26**

Replace the 6-row observation table with this 9-row paper-aligned table:

```markdown
**9 features per node**, flat array of shape `(N * 9,)`:

| Index (per node) | Feature | Range |
|-----------------|---------|-------|
| 0 | State of Charge (SoC, normalized) | [0, 1] |
| 1 | State of Health (SoH) | [0, 1] |
| 2 | last_action | {0, 1} |
| 3 | distance_to_sink (normalized) | [0, 1] |
| 4 | x_norm (x / W) | [0, 1] |
| 5 | y_norm (y / H) | [0, 1] |
| 6 | awake_density | [0, 1] |
| 7 | activity_ratio (EMA) | [0, 1] |
| 8 | charging_flag | {0, 1} |
```

Then update the line stating: `For N=50: state_dim = 300. For tests with N=10: state_dim = 60.` → `For N=50: state_dim = 450. For tests with N=10: state_dim = 90.`

- [ ] **Step 3: Update `.claude/rules/agents-training.md:62`**

Replace:
```
With 6 features per node: `state_dim = N * 6` (e.g. 300 for N=50, 60 for tests with N=10).
```
with:
```
With 9 features per node: `state_dim = N * 9` (e.g. 450 for N=50, 90 for tests with N=10).
```

- [ ] **Step 4: Update `README.md` lines 282–292**

Replace the heading line:
```
6 features per node, flattened into a shape `(N * 6,)` array (`state_dim = 300` for 50 nodes)
```
with:
```
9 features per node, flattened into a shape `(N * 9,)` array (`state_dim = 450` for 50 nodes)
```

Replace the 6-row table with the 9-row table from Step 2 (insert `x_norm`, `y_norm`, `awake_density` at indices 4, 5, 6; shift `activity_ratio` to 7 and `charging_flag` to 8).

- [ ] **Step 5: Update `workflow.md`**

- Line 19: `\mathbb{R}^{6N}` → `\mathbb{R}^{9N}`
- Line 37: `\mathbb{R}^{6N}` → `\mathbb{R}^{9N}` (and update the per-node feature list to include x_norm, y_norm, awake_density)
- Line 40: `300-dimensional` → `450-dimensional`
- Line 126: `\mathbb{R}^{300} \xrightarrow{\text{FC}_{512}}` → `\mathbb{R}^{450} \xrightarrow{\text{FC}_{512}}`
- Line 354: `Online Q-Net\n(300→512→256→100)` → `Online Q-Net\n(450→512→256→100)`

- [ ] **Step 6: Verify with grep — no stale 6-feature references remain**

Run:
```bash
grep -rnE "6[- ]feature|N \* 6|state_dim = 300|state_dim = 60|6N|\\\\mathbb\\{R\\}\\^\\{6" \
  CLAUDE.md README.md workflow.md .claude/rules/ src/CLAUDE.md backend/CLAUDE.md frontend/CLAUDE.md 2>/dev/null
```

Expected: no matches (or only matches inside historical change-log / paper.md citation contexts, if any — review each).

- [ ] **Step 7: Commit**

```bash
git add CLAUDE.md .claude/rules/rl-environment.md .claude/rules/agents-training.md README.md workflow.md
git commit -m "docs: update observation space references to 9 features (paper §4.1)"
```

---

### Task 4: Archive legacy 6-feature `.pth` checkpoints

The 10 existing checkpoints were trained against the 6-feature observation and cannot load into a 9-feature agent. Move them to a clearly named subfolder. **Do not move metadata or visualisation PNGs** — `/api/history` and the comparison endpoint must continue to work for legacy runs.

**Files:**
- Create: `results/models/legacy_6feature/` directory
- Move: 10 `.pth` files from `results/models/` to `results/models/legacy_6feature/`

- [ ] **Step 1: Verify the 10 `.pth` files are still present**

Run: `ls results/models/*.pth | wc -l`
Expected: `10`

- [ ] **Step 2: Create the legacy directory and move the files**

Run:
```bash
mkdir -p results/models/legacy_6feature
mv results/models/*.pth results/models/legacy_6feature/
ls results/models/legacy_6feature/ | wc -l
```
Expected last line: `10`

- [ ] **Step 3: Verify metadata and visualisations are untouched**

Run:
```bash
ls results/metrics/*_metadata.json | wc -l
ls results/visualizations/*.png | wc -l
```
Both counts should match what they were before this task (no files deleted/moved).

- [ ] **Step 4: Confirm a legacy `.pth` file has 6-feature first-layer shape**

Run:
```bash
python -c "
import torch
sd = torch.load('results/models/legacy_6feature/trained_model_ddqn.pth', map_location='cpu')
first_weight = next(iter(sd.values()))
print('first layer in shape:', first_weight.shape)
"
```
Expected: shape includes `300` (i.e. `[..., 300]`), confirming the 6-feature input dimension and why a 9-feature agent cannot consume it.

- [ ] **Step 5: Commit the relocation**

```bash
git add results/models/
git commit -m "chore(artifacts): archive legacy 6-feature .pth checkpoints to legacy_6feature/

These 10 checkpoints were trained against the 6-feature observation
and cannot load into the 9-feature agent. Metadata JSON and
visualisation PNGs are intentionally left in place so /api/history
and comparison plots still work for legacy runs."
```

---

### Task 5: End-to-end verification

Smoke-test the pipeline end-to-end. No code changes; this task only runs the system and confirms expected behaviour.

- [ ] **Step 1: Full unit test suite**

Run: `pytest tests/ -v`
Expected: every test passes, including the 3 new feature tests and the reordered `test_charging_node_forced_sleep_in_obs`.

- [ ] **Step 2: Smoke training (5 episodes)**

Run:
```bash
python scripts/train.py --episodes 5 --nodes 50 --model-type ddqn --seed 42
```
Expected: completes without shape errors; a new `results/models/run_*_model.pth` is written.

- [ ] **Step 3: Confirm new checkpoint has 9-feature first-layer shape**

Run:
```bash
python -c "
import torch, glob
latest = sorted(glob.glob('results/models/run_*_model.pth'))[-1]
sd = torch.load(latest, map_location='cpu')
print(latest, '->', next(iter(sd.values())).shape)
"
```
Expected: `... -> torch.Size([..., 450])` — confirms the new 9-feature input dimension.

- [ ] **Step 4: Backend sync route smoke test**

Run (in one terminal):
```bash
python -m backend.app
```

In another terminal:
```bash
curl -s -X POST http://localhost:5001/api/train \
  -H 'Content-Type: application/json' \
  -d '{"episodes": 3, "nodes": 50, "model_type": "ddqn"}'
```

Expected: response contains `"status": "success"` and a new `run_id`. No shape errors in the server log.

Stop the server when done.

- [ ] **Step 5: History endpoint still serves legacy runs**

With the server running:
```bash
curl -s http://localhost:5001/api/history | python -m json.tool | head -50
```

Expected: legacy runs still appear; their metadata JSON is parsed normally even though the underlying `.pth` files moved.

- [ ] **Step 6: Final acceptance — no commit; this is a verification task**

If all 5 verification steps passed, the implementation is complete. If any failed, file an issue against the failing area and fix BEFORE merging — do not merge a partially-working observation expansion.

---

## Critical Files Quick-Reference

| File | Why it matters |
|------|----------------|
| `src/envs/wsn_env.py` | Only code file with semantic changes (`_get_obs`, `observation_space`). |
| `tests/conftest.py:15` | `STATE_DIM` constant; shape ripples to every test fixture. |
| `tests/test_env.py` | Shape assertions + reordered charging-flag index test + 3 new feature tests. |
| `backend/tasks.py:70` | Already derives `state_dim` from env — verified, no edit needed. |
| `CLAUDE.md`, `.claude/rules/rl-environment.md`, `.claude/rules/agents-training.md` | Documentation must match code or future agents are misled. |
| `README.md`, `workflow.md` | User-facing / submission documents cited by the paper context. |
| `results/models/legacy_6feature/` | New directory holding 10 archived 6-feature checkpoints. |

## Out of Scope (per `.claude/plan.md`)

- Reward function, weights, or `r_s = 100m` value — unchanged.
- `BaseAgent` / `DDQNAgent` / `DQNAgent` class signatures or constructors — unchanged.
- Charging or cooperative wake-up rules — only the *observation* of them changes.
- `paper.md` — already updated to 9 features upstream.
- Retraining or migration of legacy `.pth` checkpoints — explicitly archived rather than re-trained.
