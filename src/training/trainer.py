"""Generic training loop for RL agents."""

from typing import Callable, Dict, List, Optional, Tuple
import numpy as np
from pathlib import Path

try:
    from tqdm import tqdm as _tqdm
    _TQDM_AVAILABLE = True
except ImportError:
    _TQDM_AVAILABLE = False

from ..agents.base_agent import BaseAgent
from ..envs.wsn_env import WSNEnv
from ..utils.logger import get_logger

logger = get_logger(__name__)


class Trainer:
    """Drives the per-episode loop and records aggregate series for plotting."""

    def __init__(self, agent: BaseAgent, env: WSNEnv, seed: int = 42):
        self.agent = agent
        self.env = env
        self.seed = seed
        np.random.seed(seed)

        self.episode_rewards: List[float] = []
        self.episode_series: Dict[str, List] = {
            "coverage": [],
            "avg_soh": [],
            "alive_fraction": [],   # retained for network_lifetime property
            "mean_soc": [],         # retained for energy_consumption computation
            "energy_consumption": [],
            "throughput": [],
        }

    @property
    def network_lifetime(self) -> int:
        """Episode index where alive_fraction first crossed (1 - death_threshold).

        Returns total episodes if the network never dropped below the threshold.
        """
        death_threshold = getattr(self.env, "death_threshold", 0.3)
        alive_threshold = 1.0 - death_threshold
        for i, af in enumerate(self.episode_series["alive_fraction"]):
            if af < alive_threshold:
                return i + 1
        return len(self.episode_series["alive_fraction"])

    def train(
        self,
        episodes: int,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> List[float]:
        """Train the agent and return per-episode total rewards.

        Args:
            episodes: Number of training episodes.
            progress_callback: Optional callable invoked after each episode as
                ``progress_callback(current_episode, total_episodes)``.
        """
        logger.info(f"Training for {episodes} episodes")

        episode_iter = (
            _tqdm(range(episodes), desc="Training", unit="ep")
            if _TQDM_AVAILABLE
            else range(episodes)
        )

        for episode in episode_iter:
            episode_reward, summary = self._run_episode()

            energy_consumption = summary["start_mean_soc"] - summary["end_mean_soc"]
            throughput = summary["coverage"] * summary["alive_fraction"]

            self.episode_rewards.append(episode_reward)
            self.episode_series["coverage"].append(summary["coverage"])
            self.episode_series["avg_soh"].append(summary["avg_soh"])
            self.episode_series["alive_fraction"].append(summary["alive_fraction"])
            self.episode_series["mean_soc"].append(summary["end_mean_soc"])
            self.episode_series["energy_consumption"].append(float(energy_consumption))
            self.episode_series["throughput"].append(float(throughput))

            if (episode + 1) % 10 == 0:
                mean_reward = np.mean(self.episode_rewards[-10:])
                logger.info(
                    f"Episode {episode+1}/{episodes} - "
                    f"Reward: {episode_reward:.2f}, 10-ep MA: {mean_reward:.2f}"
                )

            if progress_callback is not None:
                progress_callback(episode + 1, episodes)

        self.env.close()
        return self.episode_rewards

    def _run_episode(self) -> Tuple[float, Dict]:
        """Run one training episode; returns (total_reward, summary)."""
        state, _ = self.env.reset()

        episode_reward = 0.0
        done = False

        step_coverages: List[float] = []
        step_soh: List[float] = []
        step_alive: List[float] = []
        step_soc: List[float] = []
        step_count = 0

        while not done:
            action = self.agent.select_action(state)
            next_state, reward, done, info = self.env.step(action)

            episode_reward += reward
            step_coverages.append(float(info.get("coverage", 0.0)))
            step_soh.append(float(info.get("avg_soh", 0.0)))
            step_alive.append(float(info.get("alive_fraction", 1.0)))
            step_soc.append(float(info.get("mean_soc", 0.0)))
            step_count += 1

            self.agent.store_transition(state, action, reward, next_state, done)
            self.agent.learn_step()

            state = next_state

        return episode_reward, {
            "coverage": float(np.mean(step_coverages)) if step_coverages else 0.0,
            "avg_soh": float(np.mean(step_soh)) if step_soh else 0.0,
            # alive_fraction reported as final-step value (end-state of the network)
            "alive_fraction": step_alive[-1] if step_alive else 0.0,
            # SoC at first recorded step (proxy for start-of-episode charge)
            "start_mean_soc": step_soc[0] if step_soc else 1.0,
            # SoC at last recorded step (end-of-episode charge)
            "end_mean_soc": step_soc[-1] if step_soc else 0.0,
        }

    def save_checkpoint(self, path: str) -> None:
        """Save agent weights to disk, creating parent dirs as needed."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.agent.save_model(str(path))
        logger.info(f"Saved checkpoint to {path}")
