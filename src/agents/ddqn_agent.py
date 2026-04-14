"""
Double Deep Q-Network (DDQN) agent implementation.

Uses two neural networks (policy and target) with decoupled action selection and evaluation
to reduce overestimation bias in Q-learning.
"""

import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, namedtuple
from typing import Optional

from .base_agent import BaseAgent

Transition = namedtuple(
    "Transition", ("state", "action", "reward", "next_state", "done")
)


class ReplayBuffer:
    """Experience replay buffer for storing and sampling transitions."""

    def __init__(self, capacity: int = 100000):
        """Initialize replay buffer.
        
        Args:
            capacity: Maximum number of transitions to store
        """
        self.buffer = deque(maxlen=capacity)

    def push(self, *args) -> None:
        """Store a transition."""
        self.buffer.append(Transition(*args))

    def sample(self, batch_size: int) -> Transition:
        """Sample a random batch of transitions.
        
        Args:
            batch_size: Number of transitions to sample
            
        Returns:
            Batch of transitions
        """
        batch = random.sample(self.buffer, batch_size)
        return Transition(*zip(*batch))

    def __len__(self) -> int:
        return len(self.buffer)


class QNetwork(nn.Module):
    """Neural network for Q-value approximation."""

    def __init__(
        self, input_dim: int, output_dim: int, hidden_dims: list = None
    ):
        """Initialize Q-network with specified architecture.
        
        Args:
            input_dim: Input dimension (state space size)
            output_dim: Output dimension (action space size)
            hidden_dims: List of hidden layer sizes
        """
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [512, 256]

        layers = []
        last_dim = input_dim

        for h in hidden_dims:
            layers.append(nn.Linear(last_dim, h))
            layers.append(nn.ReLU())
            last_dim = h

        layers.append(nn.Linear(last_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DDQNAgent(BaseAgent):
    """
    Double Deep Q-Network (DDQN) Agent for continuous action scheduling.
    
    Uses two networks:
    - Policy network (q_net): Updated every step
    - Target network (target_net): Updated periodically
    
    This decoupling reduces overestimation bias in value estimates.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        node_count: int,
        lr: float = 1e-4,
        gamma: float = 0.99,
        batch_size: int = 64,
        buffer_size: int = 200000,
        min_replay_size: int = 500,
        update_target_every: int = 500,
        hidden_dims: list = None,
        device: str = None,
    ):
        """Initialize DDQN agent.
        
        Args:
            state_dim: State space dimension
            action_dim: Number of discrete actions per node (typically 2)
            node_count: Number of nodes in the network
            lr: Learning rate
            gamma: Discount factor
            batch_size: Training batch size
            buffer_size: Replay buffer capacity
            min_replay_size: Minimum buffer size before training
            update_target_every: Update target network every N steps
            hidden_dims: Hidden layer dimensions
            device: PyTorch device (cuda/cpu)
        """
        super().__init__(state_dim, action_dim, node_count)

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        if hidden_dims is None:
            hidden_dims = [512, 256]

        self.output_dim = node_count * action_dim

        self.q_net = QNetwork(state_dim, self.output_dim, hidden_dims).to(
            self.device
        )
        self.target_net = QNetwork(state_dim, self.output_dim, hidden_dims).to(
            self.device
        )
        self.target_net.load_state_dict(self.q_net.state_dict())

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)

        self.gamma = gamma
        self.batch_size = batch_size
        self.update_target_every = update_target_every
        self.learn_steps = 0

        self.replay = ReplayBuffer(capacity=buffer_size)
        self.min_replay_size = min_replay_size

        self.eps_start = 1.0
        self.eps_end = 0.05
        self.eps_decay = 5e4
        self._eval_epsilon: Optional[float] = None

    def eval(self) -> None:
        """Switch to evaluation mode — forces epsilon to 0.0 (fully greedy)."""
        super().eval()
        self._eval_epsilon = 0.0

    def train(self) -> None:
        """Restore training mode — epsilon reverts to the decay schedule."""
        super().train()
        self._eval_epsilon = None

    def select_action(
        self, state: np.ndarray, eval_mode: bool = False
    ) -> np.ndarray:
        """Select actions using epsilon-greedy policy.

        Args:
            state: Current observation
            eval_mode: If True, use greedy policy (no exploration).
                       set_eval_mode() also forces greedy regardless of this flag.

        Returns:
            Action array of shape (node_count,)
        """
        is_greedy = eval_mode or not self._is_training

        if self._eval_epsilon is not None:
            eps = self._eval_epsilon
        elif is_greedy:
            eps = self.eps_end
        else:
            eps = self.eps_end + (self.eps_start - self.eps_end) * max(
                0, 1 - self.learn_steps / self.eps_decay
            )

        if random.random() < eps and not is_greedy:
            return np.random.randint(0, self.action_dim, size=self.node_count)

        s = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_vals = (
                self.q_net(s)
                .cpu()
                .numpy()
                .reshape(self.node_count, self.action_dim)
            )
        actions = q_vals.argmax(axis=1)
        return actions

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
        self.replay.push(
            state.astype(np.float32),
            action.astype(np.int64),
            np.float32(reward),
            next_state.astype(np.float32),
            bool(done),
        )

    def _compute_next_q(self, next_state: torch.Tensor) -> torch.Tensor:
        """Compute the bootstrapped per-sample Q-value for next_state.

        Double-DQN: actions are selected by the online network and evaluated
        by the target network. Subclasses may override to change this rule.
        Returns a (batch, 1) tensor — already mean-reduced over nodes.
        """
        next_q_online = self.q_net(next_state).view(
            self.batch_size, self.node_count, self.action_dim
        )
        next_actions = next_q_online.argmax(dim=2, keepdim=True)
        next_q_target = self.target_net(next_state).view(
            self.batch_size, self.node_count, self.action_dim
        )
        next_q_selected = torch.gather(next_q_target, 2, next_actions).squeeze(-1)
        return next_q_selected.mean(dim=1, keepdim=True)

    def learn_step(self) -> Optional[float]:
        """Run one training step. Returns loss, or None until the buffer warms up."""
        if len(self.replay) < self.min_replay_size:
            return None

        transitions = self.replay.sample(self.batch_size)
        state = torch.from_numpy(np.stack(transitions.state)).to(self.device)
        next_state = torch.from_numpy(np.stack(transitions.next_state)).to(self.device)
        reward = torch.from_numpy(
            np.asarray(transitions.reward, dtype=np.float32)
        ).unsqueeze(1).to(self.device)
        done = torch.from_numpy(
            np.asarray(transitions.done, dtype=np.float32)
        ).unsqueeze(1).to(self.device)
        actions = torch.from_numpy(np.stack(transitions.action)).to(self.device)

        q_values = self.q_net(state).view(
            self.batch_size, self.node_count, self.action_dim
        )
        q_selected = torch.gather(q_values, 2, actions.unsqueeze(-1)).squeeze(-1)
        q_selected_mean = q_selected.mean(dim=1, keepdim=True)

        with torch.no_grad():
            target = reward + (1.0 - done) * self.gamma * self._compute_next_q(next_state)

        loss = torch.nn.functional.mse_loss(q_selected_mean, target)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), 10.0)
        self.optimizer.step()

        self.learn_steps += 1
        if self.learn_steps % self.update_target_every == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        return loss.item()

    def save_model(self, path: str) -> None:
        """Save agent model.
        
        Args:
            path: File path to save model
        """
        checkpoint = {
            "q_net": self.q_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "learn_steps": self.learn_steps,
        }
        torch.save(checkpoint, path)

    def load_model(self, path: str) -> None:
        """Load agent model.
        
        Args:
            path: File path to load model from
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.q_net.load_state_dict(checkpoint["q_net"])
        self.target_net.load_state_dict(checkpoint["target_net"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.learn_steps = checkpoint.get("learn_steps", 0)
