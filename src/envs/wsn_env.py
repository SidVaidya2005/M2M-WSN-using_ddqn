"""
Wireless Sensor Network (WSN) simulation environment.

Gym-compatible environment for training agents to optimize sleep/awake scheduling
while maintaining network coverage and maximizing network lifetime.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, List, Optional, Tuple, Any

from .battery_model import BatteryModel


class WSNEnv(gym.Env):
    """
    Gym environment for WSN sleep/awake scheduling with battery degradation.

    Observation (6 features per node, flat array of shape N*6):
      [soc_norm, soh, last_action, dist_to_sink_norm, activity_ratio, charging_flag]

    Action: per-node binary (0=SLEEP, 1=AWAKE) — MultiDiscrete of length N.

    M2M behaviours (Phase 2):
      - Charging: nodes below charging_threshold are forced SLEEP and recover SoC.
      - Cooperative wake-up: an AWAKE node whose SoC drops to ≤ wake_cooperation_soc
        causes its nearest non-charging SLEEP neighbour to be forcibly woken.
      - Proper grid-based coverage metric (sensing_radius).
    """

    metadata = {"render.modes": ["human"]}

    # Grid resolution for coverage calculation (20×20 sample points)
    _GRID_RES = 20

    def __init__(
        self,
        N: int = 10,
        arena_size: Tuple[int, int] = (500, 500),
        sink: Tuple[int, int] = (250, 250),
        timestep_energy_awake: float = 1.0,
        energy_sleep: float = 0.01,
        max_steps: int = 10000,
        seed: Optional[int] = None,
        death_threshold: float = 0.3,
        # M2M / Phase-2 parameters (defaults match config.yaml)
        reward_weights: Tuple[float, float, float, float] = (10.0, 5.0, 1.0, 2.0),
        charging_enabled: bool = True,
        charging_rate: float = 0.05,
        charging_threshold: float = 0.2,
        wake_cooperation_soc: float = 0.5,
        sensing_radius: float = 100.0,
    ):
        """Initialize WSN environment.

        Args:
            N: Number of sensor nodes.
            arena_size: Physical area (width, height) in metres.
            sink: Sink node position (x, y).
            timestep_energy_awake: Energy consumed per step when awake.
            energy_sleep: Leakage energy per step when sleeping.
            max_steps: Maximum steps per episode.
            seed: Random seed.
            death_threshold: Episode ends when dead_nodes > death_threshold * N.
            reward_weights: (w_coverage, w_energy, w_soh, w_balance).
            charging_enabled: Enable the charging subsystem.
            charging_rate: SoC fraction of E_max recovered per step while charging.
            charging_threshold: Normalised SoC below which a node enters charging.
            wake_cooperation_soc: Normalised SoC threshold that triggers cooperative wake.
            sensing_radius: Coverage sensing radius in metres.
        """
        super().__init__()
        self.N = N
        self.arena_size = arena_size
        self.sink = np.array(sink, dtype=float)
        self.timestep_energy_awake = timestep_energy_awake
        self.energy_sleep = energy_sleep
        self.max_steps = max_steps
        self.step_count = 0
        self.rng = np.random.RandomState(seed)
        self.death_threshold = death_threshold

        # M2M params
        self.reward_weights = reward_weights
        self.charging_enabled = charging_enabled
        self.charging_rate = charging_rate
        self.charging_threshold = charging_threshold
        self.wake_cooperation_soc = wake_cooperation_soc
        self.sensing_radius = sensing_radius

        # Node positions (initialised in reset, seeded now for reproducibility)
        self.positions = self.rng.rand(N, 2) * np.array(arena_size)
        dists = np.linalg.norm(self.positions - self.sink, axis=1)
        self.dist_norm = dists / np.sqrt(arena_size[0] ** 2 + arena_size[1] ** 2)

        # Batteries
        self.batteries = [
            BatteryModel(E_max=100.0, soh_init=1.0, k_cycle=5e-5, alpha=1.2, calendar_decay=5e-7)
            for _ in range(N)
        ]

        # Per-node running state
        self.last_action = np.zeros(N, dtype=int)
        self.recent_activity = np.zeros(N, dtype=float)

        # Precompute grid sample points for coverage calculation
        gx = np.linspace(0, arena_size[0], self._GRID_RES)
        gy = np.linspace(0, arena_size[1], self._GRID_RES)
        grid_x, grid_y = np.meshgrid(gx, gy)
        self.grid_points = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1)  # (400, 2)

        # Observation & action spaces — 6 features per node
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(N * 6,), dtype=np.float32
        )
        self.action_space = spaces.MultiDiscrete([2] * N)

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(self) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment to initial state.

        Returns:
            (observation, info) — Gymnasium-compliant.
        """
        self.step_count = 0
        self.positions = self.rng.rand(self.N, 2) * np.array(self.arena_size)
        dists = np.linalg.norm(self.positions - self.sink, axis=1)
        self.dist_norm = dists / np.sqrt(self.arena_size[0] ** 2 + self.arena_size[1] ** 2)

        self.batteries = [
            BatteryModel(E_max=100.0, soh_init=1.0, k_cycle=5e-5, alpha=1.2, calendar_decay=5e-7)
            for _ in range(self.N)
        ]
        self.last_action = np.zeros(self.N, dtype=int)
        self.recent_activity = np.zeros(self.N, dtype=float)

        return self._get_obs(), {}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """Execute one environment step.

        Args:
            action: Array of length N with values 0 (SLEEP) or 1 (AWAKE).

        Returns:
            (observation, reward, done, info)
        """
        assert len(action) == self.N
        self.step_count += 1

        effective_action = np.array(action, dtype=int)

        # ── 1. Charging override ──────────────────────────────────────────
        charging_set: set = set()
        if self.charging_enabled:
            for i, batt in enumerate(self.batteries):
                if batt.is_dead():
                    batt.charging = False
                    continue
                soc_frac = batt.soc / batt.E_max
                # Exit charging when SoC reaches 95 %
                if batt.charging and soc_frac >= 0.95:
                    batt.charging = False
                # Enter charging when SoC drops below threshold
                elif not batt.charging and batt.needs_charge(self.charging_threshold):
                    batt.charging = True

                if batt.charging:
                    effective_action[i] = 0  # forced SLEEP
                    batt.charge(self.charging_rate)
                    charging_set.add(i)

        # ── 2. Cooperative wake-up ────────────────────────────────────────
        cooperative_wakes: List[int] = []
        woken_set: set = set()
        for i, batt in enumerate(self.batteries):
            if batt.is_dead() or effective_action[i] != 1:
                continue
            soc_frac = batt.soc / batt.E_max
            if soc_frac > self.wake_cooperation_soc:
                continue
            # Find nearest non-charging, non-dead, SLEEP neighbour
            best_j, best_dist = -1, float("inf")
            for j in range(self.N):
                if (
                    j != i
                    and effective_action[j] == 0
                    and not self.batteries[j].is_dead()
                    and j not in charging_set
                    and j not in woken_set
                ):
                    d = float(np.linalg.norm(self.positions[i] - self.positions[j]))
                    if d < best_dist:
                        best_dist, best_j = d, j
            if best_j >= 0:
                effective_action[best_j] = 1
                woken_set.add(best_j)
                cooperative_wakes.append(best_j)

        # ── 3. Physics: energy draw for non-charging nodes ────────────────
        per_node_energy = np.zeros(self.N, dtype=np.float32)
        for i, batt in enumerate(self.batteries):
            if i in charging_set:
                # Charger already handled SoC; no discharge call
                self.recent_activity[i] = 0.9 * self.recent_activity[i]
                self.last_action[i] = 0
                continue
            if effective_action[i] == 1:  # AWAKE
                energy_draw = self.timestep_energy_awake * (1.0 + 0.1 * self.dist_norm[i])
                batt.discharge(energy_draw)
                self.recent_activity[i] = 0.9 * self.recent_activity[i] + 0.1
                per_node_energy[i] = energy_draw
            else:  # SLEEP
                batt.discharge(self.energy_sleep)
                self.recent_activity[i] = 0.9 * self.recent_activity[i]
                per_node_energy[i] = self.energy_sleep
            self.last_action[i] = int(effective_action[i])

        total_energy_used = float(per_node_energy.sum())

        # ── 4. Derived statistics ─────────────────────────────────────────
        soc_fracs = np.array([b.soc / b.E_max for b in self.batteries], dtype=np.float32)
        avg_soh = float(np.mean([b.soh for b in self.batteries]))
        dead_count = sum(1 for b in self.batteries if b.is_dead())
        alive_fraction = (self.N - dead_count) / self.N
        mean_soc = float(np.mean(soc_fracs))
        coverage = self._compute_coverage(effective_action)

        # ── 5. Reward ─────────────────────────────────────────────────────
        w_cov, w_eng, w_soh, w_bal = self.reward_weights

        r_coverage = float(np.clip(coverage, 0.0, 1.0))

        weighted_energy = float(np.sum(per_node_energy * (1.0 - soc_fracs)))
        r_energy = -float(
            np.clip(weighted_energy / (self.N * self.timestep_energy_awake * 2.0), 0.0, 1.0)
        )

        r_soh = float(np.clip(avg_soh - 0.99, -1.0, 1.0))

        soc_std = float(np.std(soc_fracs))
        r_balance = float(np.clip(-soc_std, -1.0, 0.0))

        reward = (
            w_cov * r_coverage
            + w_eng * r_energy
            + w_soh * r_soh
            + w_bal * r_balance
        )

        # ── 6. Terminal condition ─────────────────────────────────────────
        done = False
        if dead_count > self.death_threshold * self.N:
            done = True
            reward -= 10.0
        if self.step_count >= self.max_steps:
            done = True

        info: Dict[str, Any] = {
            "total_energy": total_energy_used,
            "coverage": coverage,
            "coverage_ratio": coverage,        # backward-compat alias
            "avg_soh": avg_soh,
            "alive_fraction": alive_fraction,
            "dead_count": dead_count,
            "dead_nodes": dead_count,           # backward-compat alias
            "mean_soc": mean_soc,
            "cooperative_wakes": cooperative_wakes,
            "charging_count": len(charging_set),
            "step_count": self.step_count,
        }

        return self._get_obs(), float(reward), done, info

    def render(self, mode: str = "human") -> None:
        """Render simple text output of environment state."""
        socs = [round(b.soc, 1) for b in self.batteries]
        sohs = [round(b.soh, 3) for b in self.batteries]
        print(f"Step {self.step_count}: socs={socs}, sohs={sohs}")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_obs(self) -> np.ndarray:
        """Construct 6-feature-per-node observation vector."""
        obs = []
        for i in range(self.N):
            batt = self.batteries[i]
            obs.extend([
                batt.soc / batt.E_max,          # 0: SoC normalised
                batt.soh,                        # 1: SoH
                float(self.last_action[i]),      # 2: last action
                self.dist_norm[i],               # 3: distance to sink (norm)
                self.recent_activity[i],         # 4: activity EMA
                float(batt.charging),            # 5: charging flag (NEW)
            ])
        return np.array(obs, dtype=np.float32)

    def _compute_coverage(self, effective_action: np.ndarray) -> float:
        """Grid-based coverage: fraction of arena covered by awake, alive nodes.

        A node covers a grid point if its Euclidean distance to that point is
        within sensing_radius.  Only non-dead AWAKE nodes contribute.

        Args:
            effective_action: Per-node action after all overrides.

        Returns:
            Coverage fraction in [0, 1].
        """
        covered = np.zeros(len(self.grid_points), dtype=bool)
        for i in range(self.N):
            if effective_action[i] == 1 and not self.batteries[i].is_dead():
                dists = np.linalg.norm(self.grid_points - self.positions[i], axis=1)
                covered |= dists <= self.sensing_radius
        return float(covered.sum()) / len(self.grid_points)
