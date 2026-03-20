"""
Baseline policies for benchmarking.

These policies represent different WSN scheduling strategies to compare against
the learned DDQN policy.
"""

import numpy as np
from src.agents.base_agent import BaseAgent


class RandomPolicy(BaseAgent):
    """
    Baseline: Random sleep/awake decisions for each node.
    
    This is the weakest baseline - purely random decisions.
    """

    def select_action(self, state: np.ndarray, eval_mode: bool = False) -> np.ndarray:
        """Return random action for each node.
        
        Args:
            state: Current observation
            eval_mode: Ignored for random policy
            
        Returns:
            Random action array
        """
        return np.random.randint(0, self.action_dim, size=self.node_count)

    def store_transition(self, state, action, reward, next_state, done) -> None:
        """No-op for random policy."""
        pass

    def learn_step(self) -> None:
        """No-op for random policy."""
        return None

    def save_model(self, path: str) -> None:
        """No-op for random policy."""
        pass

    def load_model(self, path: str) -> None:
        """No-op for random policy."""
        pass


class GreedyPolicy(BaseAgent):
    """
    Baseline: Greedy heuristic.
    
    Wakes nodes with highest combined SoC*SoH score, keeps ~50% awake.
    Balances energy and health with distance penalty.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        node_count: int,
        awake_ratio: float = 0.5,
    ):
        """Initialize greedy policy.
        
        Args:
            state_dim: State dimension
            action_dim: Action dimension
            node_count: Number of nodes
            awake_ratio: Fraction of nodes to keep awake
        """
        super().__init__(state_dim, action_dim, node_count)
        self.awake_ratio = awake_ratio

    def select_action(self, state: np.ndarray, eval_mode: bool = False) -> np.ndarray:
        """Wake nodes with highest (SoC * SoH) score.
        
        Args:
            state: Current observation
            eval_mode: Ignored for greedy policy
            
        Returns:
            Action with top-scoring nodes awake
        """
        obs_dim_per_node = 5
        scores = []

        for i in range(self.node_count):
            idx = i * obs_dim_per_node
            soc_norm = state[idx]
            soh = state[idx + 1]
            dist_norm = state[idx + 3]

            # Score: prioritize healthy nodes, penalize distance
            score = soc_norm * soh * (1.0 - 0.3 * dist_norm)
            scores.append(score)

        scores = np.array(scores)

        # Wake top nodes by score
        num_awake = max(1, int(self.node_count * self.awake_ratio))
        top_indices = np.argsort(scores)[-num_awake:]

        action = np.zeros(self.node_count, dtype=int)
        action[top_indices] = 1

        return action

    def store_transition(self, state, action, reward, next_state, done) -> None:
        """No-op for greedy policy."""
        pass

    def learn_step(self) -> None:
        """No-op for greedy policy."""
        return None

    def save_model(self, path: str) -> None:
        """No-op for greedy policy."""
        pass

    def load_model(self, path: str) -> None:
        """No-op for greedy policy."""
        pass


class EnergyConservativePolicy(BaseAgent):
    """
    Baseline: Energy-conservative policy.
    
    Minimizes power consumption by keeping only the healthiest nodes awake (~20%).
    Sacrifices coverage for maximum network lifetime.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        node_count: int,
        awake_ratio: float = 0.2,
    ):
        """Initialize conservative policy.
        
        Args:
            state_dim: State dimension
            action_dim: Action dimension
            node_count: Number of nodes
            awake_ratio: Fraction of nodes to keep awake (low for conservation)
        """
        super().__init__(state_dim, action_dim, node_count)
        self.awake_ratio = awake_ratio

    def select_action(self, state: np.ndarray, eval_mode: bool = False) -> np.ndarray:
        """Wake only the healthiest nodes to maximize lifetime.
        
        Args:
            state: Current observation
            eval_mode: Ignored for conservative policy
            
        Returns:
            Action with only healthiest nodes awake
        """
        obs_dim_per_node = 5
        health_scores = []

        for i in range(self.node_count):
            idx = i * obs_dim_per_node
            soc_norm = state[idx]
            soh = state[idx + 1]

            # Only score by health, ignore distance
            health_score = soc_norm * soh
            health_scores.append(health_score)

        health_scores = np.array(health_scores)

        # Wake only the healthiest nodes
        num_awake = max(1, int(self.node_count * self.awake_ratio))
        top_indices = np.argsort(health_scores)[-num_awake:]

        action = np.zeros(self.node_count, dtype=int)
        action[top_indices] = 1

        return action

    def store_transition(self, state, action, reward, next_state, done) -> None:
        """No-op for conservative policy."""
        pass

    def learn_step(self) -> None:
        """No-op for conservative policy."""
        return None

    def save_model(self, path: str) -> None:
        """No-op for conservative policy."""
        pass

    def load_model(self, path: str) -> None:
        """No-op for conservative policy."""
        pass


class BalancedRotationPolicy(BaseAgent):
    """
    Baseline: Balanced rotation policy.
    
    Distributes energy burden fairly by rotating which nodes are awake.
    Keeps ~50% awake but changes the set periodically.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        node_count: int,
        awake_ratio: float = 0.5,
        rotation_period: int = 10,
    ):
        """Initialize balanced rotation policy.
        
        Args:
            state_dim: State dimension
            action_dim: Action dimension
            node_count: Number of nodes
            awake_ratio: Fraction of nodes to keep awake
            rotation_period: Steps between rotations
        """
        super().__init__(state_dim, action_dim, node_count)
        self.awake_ratio = awake_ratio
        self.rotation_period = rotation_period
        self.step_count = 0

    def select_action(self, state: np.ndarray, eval_mode: bool = False) -> np.ndarray:
        """Rotate which nodes are awake to distribute load.
        
        Args:
            state: Current observation
            eval_mode: Ignored for rotation policy
            
        Returns:
            Action with rotated set of nodes awake
        """
        num_awake = max(1, int(self.node_count * self.awake_ratio))

        # Circular rotation based on step count
        offset = self.step_count // self.rotation_period
        start_idx = offset % self.node_count

        action = np.zeros(self.node_count, dtype=int)
        indices = np.arange(start_idx, start_idx + num_awake) % self.node_count
        action[indices] = 1

        self.step_count += 1
        return action

    def store_transition(self, state, action, reward, next_state, done) -> None:
        """No-op for rotation policy."""
        pass

    def learn_step(self) -> None:
        """No-op for rotation policy."""
        return None

    def save_model(self, path: str) -> None:
        """No-op for rotation policy."""
        pass

    def load_model(self, path: str) -> None:
        """No-op for rotation policy."""
        pass

    def reset(self) -> None:
        """Reset step counter on episode reset."""
        self.step_count = 0
