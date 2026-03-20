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
from typing import Optional, Tuple

from .base_agent import BaseAgent

# Experience tuple for replay buffer
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
        update_target_every: int = 1000,
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

        # Output dimension: flatten all node-action pairs
        self.output_dim = node_count * action_dim

        # Networks
        self.q_net = QNetwork(state_dim, self.output_dim, hidden_dims).to(
            self.device
        )
        self.target_net = QNetwork(state_dim, self.output_dim, hidden_dims).to(
            self.device
        )
        self.target_net.load_state_dict(self.q_net.state_dict())

        # Optimizer
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)

        # Hyperparameters
        self.gamma = gamma
        self.batch_size = batch_size
        self.update_target_every = update_target_every
        self.learn_steps = 0

        # Replay buffer
        self.replay = ReplayBuffer(capacity=buffer_size)
        self.min_replay_size = min_replay_size

        # Epsilon-greedy exploration schedule
        self.eps_start = 1.0
        self.eps_end = 0.05
        self.eps_decay = 1e5

    def select_action(
        self, state: np.ndarray, eval_mode: bool = False
    ) -> np.ndarray:
        """Select actions using epsilon-greedy policy.
        
        Args:
            state: Current observation
            eval_mode: If True, use greedy policy (no exploration)
            
        Returns:
            Action array of shape (node_count,)
        """
        # Compute epsilon decay
        eps = (
            self.eps_end
            if eval_mode
            else self.eps_end
            + (self.eps_start - self.eps_end)
            * max(0, 1 - self.learn_steps / self.eps_decay)
        )

        # Explore
        if random.random() < eps and not eval_mode:
            return np.random.randint(0, self.action_dim, size=self.node_count)

        # Exploit
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

    def learn_step(self) -> Optional[float]:
        """Perform one training step using experience replay.
        
        Returns:
            Loss value or None if insufficient data in buffer
        """
        if len(self.replay) < self.min_replay_size:
            return None

        # Sample batch from replay buffer
        transitions = self.replay.sample(self.batch_size)
        state = torch.FloatTensor(np.stack(transitions.state)).to(self.device)
        next_state = torch.FloatTensor(np.stack(transitions.next_state)).to(
            self.device
        )
        reward = (
            torch.FloatTensor(np.array(transitions.reward))
            .unsqueeze(1)
            .to(self.device)
        )
        done = (
            torch.FloatTensor(np.array(transitions.done).astype(np.float32))
            .unsqueeze(1)
            .to(self.device)
        )
        actions = torch.LongTensor(np.stack(transitions.action)).to(self.device)

        # Compute Q-values for current states
        q_values = self.q_net(state)  # (B, node_count * action_dim)
        q_values = q_values.view(
            self.batch_size, self.node_count, self.action_dim
        )  # (B, N, A)

        # Gather Q-values for executed actions
        actions_unsq = actions.unsqueeze(-1)  # (B, N, 1)
        q_selected = torch.gather(q_values, dim=2, index=actions_unsq).squeeze(
            -1
        )  # (B, N)
        q_selected_mean = q_selected.mean(dim=1, keepdim=True)  # (B, 1)

        # Double DQN: Decouple action selection and evaluation
        with torch.no_grad():
            # Use policy network to SELECT actions in next state
            next_q_vals_online = self.q_net(next_state).view(
                self.batch_size, self.node_count, self.action_dim
            )
            next_actions_online = next_q_vals_online.argmax(dim=2)  # (B, N)

            # Use target network to EVALUATE those actions
            next_q_vals_target = self.target_net(next_state).view(
                self.batch_size, self.node_count, self.action_dim
            )
            idx = next_actions_online.unsqueeze(-1)
            next_q_selected = torch.gather(
                next_q_vals_target, dim=2, index=idx
            ).squeeze(-1)  # (B, N)
            next_q_selected_mean = next_q_selected.mean(dim=1, keepdim=True)  # (B, 1)

            # Compute target
            target = reward + (1.0 - done) * self.gamma * next_q_selected_mean

        # MSE Loss
        loss = nn.MSELoss()(q_selected_mean, target)

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), 10.0)
        self.optimizer.step()

        # Update target network periodically
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
