"""
Wireless Sensor Network (WSN) simulation environment.

Gym-compatible environment for training agents to optimize sleep/awake scheduling
while maintaining network coverage and maximizing network lifetime.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Tuple, Dict, Any

from .battery_model import BatteryModel


class WSNEnv(gym.Env):
    """
    Gym environment for WSN sleep/awake scheduling with battery degradation.
    
    Observation: Concatenated per-node features:
    - soc_normalized (0..1)
    - soh (0..1)
    - last_action (0 or 1)
    - distance_to_sink_normalized (0..1)
    - recent_activity_ratio (0..1)
    
    Action: Discrete agent outputs for each node (0=SLEEP, 1=AWAKE)
    """

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        N: int = 10,
        arena_size: Tuple[int, int] = (500, 500),
        sink: Tuple[int, int] = (250, 250),
        timestep_energy_awake: float = 1.0,
        energy_sleep: float = 0.01,
        max_steps: int = 10000,
        seed: int = None,
        death_threshold: float = 0.3,
    ):
        """Initialize WSN environment.
        
        Args:
            N: Number of sensor nodes
            arena_size: Physical area dimensions (width, height)
            sink: Sink node position (x, y)
            timestep_energy_awake: Energy consumed per timestep when awake
            energy_sleep: Energy consumed per timestep when sleeping (leakage)
            max_steps: Maximum episode length
            seed: Random seed
            death_threshold: Episode ends if > this fraction of nodes are dead
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

        # Initialize node positions
        self.positions = self.rng.rand(N, 2) * np.array(arena_size)

        # Compute normalized distances to sink
        dists = np.linalg.norm(self.positions - self.sink, axis=1)
        self.dist_norm = dists / np.sqrt(arena_size[0] ** 2 + arena_size[1] ** 2)

        # Battery models for each node
        self.batteries = [
            BatteryModel(
                E_max=100.0, soh_init=1.0, k_cycle=5e-5, alpha=1.2, calendar_decay=5e-7
            )
            for _ in range(N)
        ]

        # Track per-node state
        self.last_action = np.zeros(N, dtype=int)
        self.recent_activity = np.zeros(N, dtype=float)  # Exponential moving average

        # Define observation and action spaces
        obs_dim_per_node = 5
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(N * obs_dim_per_node,),
            dtype=np.float32,
        )
        self.action_space = spaces.MultiDiscrete([2] * N)  # 0 or 1 per node

    def reset(self) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment to initial state.

        Returns:
            Tuple of (observation, info) — Gymnasium-compliant.
        """
        self.step_count = 0
        self.positions = self.rng.rand(self.N, 2) * np.array(self.arena_size)
        dists = np.linalg.norm(self.positions - self.sink, axis=1)
        self.dist_norm = dists / np.sqrt(self.arena_size[0] ** 2 + self.arena_size[1] ** 2)

        self.batteries = [
            BatteryModel(
                E_max=100.0,
                soh_init=1.0,
                k_cycle=5e-5,
                alpha=1.2,
                calendar_decay=5e-7,
            )
            for _ in range(self.N)
        ]
        self.last_action = np.zeros(self.N, dtype=int)
        self.recent_activity = np.zeros(self.N, dtype=float)

        return self._get_obs(), {}

    def _get_obs(self) -> np.ndarray:
        """Construct observation vector from current state."""
        obs = []
        for i in range(self.N):
            soc = self.batteries[i].soc / self.batteries[i].E_max
            soh = self.batteries[i].soh
            la = float(self.last_action[i])
            d = self.dist_norm[i]
            ra = self.recent_activity[i]
            obs.extend([soc, soh, la, d, ra])
        return np.array(obs, dtype=np.float32)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """Execute one environment step.
        
        Args:
            action: Array of length N with values 0 (SLEEP) or 1 (AWAKE)
            
        Returns:
            - observation: Updated observation
            - reward: Step reward
            - done: Episode finished flag
            - info: Debug info dict
        """
        assert len(action) == self.N
        self.step_count += 1

        # Simulate energy draw & battery update
        per_node_energy = np.zeros(self.N, dtype=np.float32)
        coverage_active = 0

        for i, a in enumerate(action):
            if a == 1:  # AWAKE
                # Energy draw with distance penalty
                energy_draw = self.timestep_energy_awake * (1 + 0.1 * self.dist_norm[i])
                self.batteries[i].discharge(energy_draw)
                self.recent_activity[i] = 0.9 * self.recent_activity[i] + 0.1 * 1.0
                coverage_active += 1
                per_node_energy[i] = energy_draw
            else:  # SLEEP
                # Small leakage current
                self.batteries[i].discharge(self.energy_sleep)
                self.recent_activity[i] = 0.9 * self.recent_activity[i] + 0.1 * 0.0
                per_node_energy[i] = self.energy_sleep

            self.last_action[i] = int(a)

        total_energy_used = float(per_node_energy.sum())

        # Post-discharge SoC fractions (normalized to [0, 1])
        soc_fracs = np.array([b.soc / b.E_max for b in self.batteries], dtype=np.float32)

        # Compute reward components — each normalized to a bounded range before summing

        # Coverage: fraction of awake nodes, in [0, 1]
        coverage_ratio = coverage_active / self.N
        r_coverage = np.clip(coverage_ratio, 0.0, 1.0)

        # Energy penalty: base penalty scaled inversely by node SoC so depleted
        # nodes are penalized more for drawing power; clipped to [-1, 0]
        weighted_energy = float(np.sum(per_node_energy * (1.0 - soc_fracs)))
        r_energy = -np.clip(
            weighted_energy / (self.N * self.timestep_energy_awake * 2.0), 0.0, 1.0
        )

        # SoH reward: penalize health degradation, clipped to [-1, 1]
        avg_soh = float(np.mean([b.soh for b in self.batteries]))
        r_soh = np.clip(avg_soh - 0.99, -1.0, 1.0)

        # Balance reward: penalize uneven charge distribution, clipped to [-1, 0]
        soc_std = float(np.std(soc_fracs))
        r_balance = np.clip(-soc_std, -1.0, 0.0)

        # Combined reward with balanced weights
        reward = 10.0 * r_coverage + 5.0 * r_energy + 1.0 * r_soh + 2.0 * r_balance

        # Terminal condition: too many nodes dead
        dead_nodes = sum(1 for b in self.batteries if b.is_dead())
        done = False
        if dead_nodes > self.death_threshold * self.N:
            done = True
            reward -= 10.0  # Heavy penalty for network failure

        # Episode timeout
        if self.step_count >= self.max_steps:
            done = True

        info = {
            "total_energy": total_energy_used,
            "coverage_ratio": coverage_ratio,
            "avg_soh": avg_soh,
            "dead_nodes": dead_nodes,
            "step_count": self.step_count,
        }

        return self._get_obs(), float(reward), done, info

    def render(self, mode: str = "human") -> None:
        """Render simple text output of environment state."""
        socs = [round(b.soc, 1) for b in self.batteries]
        sohs = [round(b.soh, 3) for b in self.batteries]
        print(f"Step {self.step_count}: socs={socs}, sohs={sohs}")
