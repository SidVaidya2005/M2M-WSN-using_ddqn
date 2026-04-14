"""Deep Q-Network (DQN) agent — vanilla max-target variant of DDQN."""

import torch

from .ddqn_agent import DDQNAgent


class DQNAgent(DDQNAgent):
    """DQN agent: target network both selects and evaluates next-state actions."""

    def _compute_next_q(self, next_state: torch.Tensor) -> torch.Tensor:
        next_q_target = self.target_net(next_state).view(
            self.batch_size, self.node_count, self.action_dim
        )
        return next_q_target.max(dim=2).values.mean(dim=1, keepdim=True)
