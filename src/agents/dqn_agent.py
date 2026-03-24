"""Deep Q-Network (DQN) agent implementation."""

from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from .ddqn_agent import DDQNAgent


class DQNAgent(DDQNAgent):
    """DQN agent using max target values from target network."""

    def learn_step(self) -> Optional[float]:
        """Perform one DQN learning step using experience replay."""
        if len(self.replay) < self.min_replay_size:
            return None

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

        q_values = self.q_net(state)
        q_values = q_values.view(self.batch_size, self.node_count, self.action_dim)

        actions_unsq = actions.unsqueeze(-1)
        q_selected = torch.gather(q_values, dim=2, index=actions_unsq).squeeze(-1)
        q_selected_mean = q_selected.mean(dim=1, keepdim=True)

        with torch.no_grad():
            next_q_vals_target = self.target_net(next_state).view(
                self.batch_size, self.node_count, self.action_dim
            )
            next_q_max = next_q_vals_target.max(dim=2).values
            next_q_max_mean = next_q_max.mean(dim=1, keepdim=True)
            target = reward + (1.0 - done) * self.gamma * next_q_max_mean

        loss = nn.MSELoss()(q_selected_mean, target)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), 10.0)
        self.optimizer.step()

        self.learn_steps += 1
        if self.learn_steps % self.update_target_every == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        return loss.item()
