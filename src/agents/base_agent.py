"""
Abstract base class for all RL agents.

Defines the common interface that all agent implementations must follow.
"""

from abc import ABC, abstractmethod
from typing import Tuple, Optional
import numpy as np


class BaseAgent(ABC):
    """Abstract base class for reinforcement learning agents."""

    def __init__(self, state_dim: int, action_dim: int, node_count: int):
        """Initialize agent.

        Args:
            state_dim: Dimension of state/observation space
            action_dim: Number of discrete actions per node
            node_count: Number of nodes in the network
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.node_count = node_count
        self._is_training: bool = True

    @abstractmethod
    def select_action(self, state: np.ndarray, eval_mode: bool = False) -> np.ndarray:
        """Select action for current state.
        
        Args:
            state: Current observation
            eval_mode: If True, use deterministic policy (no exploration)
            
        Returns:
            Action array of shape (node_count,)
        """
        pass

    @abstractmethod
    def learn_step(self) -> Optional[float]:
        """Perform one learning update from replay buffer.
        
        Returns:
            Loss value or None if no update performed
        """
        pass

    @abstractmethod
    def store_transition(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Store experience in replay buffer.
        
        Args:
            state: Current observation
            action: Action taken
            reward: Reward received
            next_state: Resulting observation
            done: Episode termination flag
        """
        pass

    @abstractmethod
    def save_model(self, path: str) -> None:
        """Save agent model and parameters.
        
        Args:
            path: File path to save model
        """
        pass

    @abstractmethod
    def load_model(self, path: str) -> None:
        """Load agent model and parameters.
        
        Args:
            path: File path to load model from
        """
        pass

    def eval(self) -> None:
        """Switch agent to evaluation mode — disables stochastic exploration."""
        self._is_training = False

    def train(self) -> None:
        """Switch agent back to training mode — re-enables exploration."""
        self._is_training = True

    def reset(self) -> None:
        """Reset any episodic state (e.g., for certain exploration strategies)."""
        pass
