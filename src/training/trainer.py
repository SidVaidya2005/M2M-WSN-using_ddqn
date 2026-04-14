"""Generic training loop for RL agents."""

from typing import Dict, List, Tuple, Optional, Callable
import numpy as np
from pathlib import Path

from ..agents.base_agent import BaseAgent
from ..envs.wsn_env import WSNEnv
from ..utils.logger import get_logger
from ..utils.metrics import compute_episode_metrics

logger = get_logger(__name__)


class Trainer:
    """Generic trainer for RL agents."""

    def __init__(
        self,
        agent: BaseAgent,
        env: WSNEnv,
        logger_obj=None,
        seed: int = 42,
    ):
        self.agent = agent
        self.env = env
        self.logger = logger_obj or logger
        self.seed = seed

        np.random.seed(seed)

        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = []
        self.episode_metrics: List[Dict] = []

        # Per-episode series — one value per episode, populated during train()
        self.episode_series: Dict[str, List] = {
            "episode_reward": [],
            "coverage": [],
            "avg_soh": [],
            "alive_fraction": [],
            "mean_soc": [],
            "step_counts": [],
        }

    @property
    def network_lifetime(self) -> int:
        """Episode number at which alive_fraction first fell below (1 - death_threshold).

        Returns total trained episodes if the network never crossed the threshold.
        """
        death_threshold = getattr(self.env, "death_threshold", 0.3)
        alive_threshold = 1.0 - death_threshold
        for i, af in enumerate(self.episode_series["alive_fraction"]):
            if af < alive_threshold:
                return i + 1  # 1-indexed
        return len(self.episode_series["alive_fraction"])

    def train(
        self,
        episodes: int,
        callbacks: Optional[List[Callable]] = None,
    ) -> Tuple[List[float], List[Dict]]:
        """Train agent for specified number of episodes.

        Returns:
            Tuple of (episode_rewards, episode_metrics)
        """
        if callbacks is None:
            callbacks = []

        self.logger.info(f"Training for {episodes} episodes")

        for episode in range(episodes):
            episode_reward, episode_info, episode_summary = self._run_episode(training=True)

            self.episode_rewards.append(episode_reward)
            self.episode_series["episode_reward"].append(episode_reward)
            self.episode_series["coverage"].append(episode_summary["coverage"])
            self.episode_series["avg_soh"].append(episode_summary["avg_soh"])
            self.episode_series["alive_fraction"].append(episode_summary["alive_fraction"])
            self.episode_series["mean_soc"].append(episode_summary["mean_soc"])
            self.episode_series["step_counts"].append(episode_summary["step_count"])

            metrics = compute_episode_metrics([episode_reward], [episode_info])
            self.episode_metrics.append(metrics)

            if (episode + 1) % 10 == 0:
                mean_reward = np.mean(self.episode_rewards[-10:])
                self.logger.info(
                    f"Episode {episode+1}/{episodes} - "
                    f"Reward: {episode_reward:.2f}, "
                    f"10-ep MA: {mean_reward:.2f}"
                )

            for callback in callbacks:
                callback(episode, episode_reward, metrics)

        self.env.close()
        return self.episode_rewards, self.episode_metrics

    def evaluate(
        self,
        episodes: int = 10,
    ) -> Tuple[List[float], List[Dict]]:
        """Evaluate agent without training.

        Returns:
            Tuple of (episode_rewards, episode_metrics)
        """
        self.logger.info(f"Evaluating for {episodes} episodes")

        eval_rewards = []
        eval_metrics = []

        for episode in range(episodes):
            episode_reward, episode_info, _ = self._run_episode(training=False)
            eval_rewards.append(episode_reward)

            metrics = compute_episode_metrics([episode_reward], [episode_info])
            eval_metrics.append(metrics)

            if (episode + 1) % 5 == 0:
                mean_reward = np.mean(eval_rewards[-5:])
                self.logger.info(
                    f"Eval Episode {episode+1}/{episodes} - "
                    f"Reward: {episode_reward:.2f}, "
                    f"5-ep MA: {mean_reward:.2f}"
                )

        return eval_rewards, eval_metrics

    def _run_episode(
        self, training: bool = True
    ) -> Tuple[float, Dict, Dict]:
        """Run a single episode.

        Returns:
            Tuple of (total_reward, final_info_dict, episode_summary)
            episode_summary contains per-episode aggregates:
                coverage, avg_soh, alive_fraction (final), mean_soc, step_count
        """
        state, info = self.env.reset()

        episode_reward = 0.0
        episode_info: Dict = {}
        done = False

        step_coverages: List[float] = []
        step_soh: List[float] = []
        step_alive: List[float] = []
        step_soc: List[float] = []
        step_count = 0

        while not done:
            action = self.agent.select_action(state, eval_mode=not training)
            next_state, reward, done, info = self.env.step(action)

            episode_reward += reward
            episode_info = info

            step_coverages.append(float(info.get("coverage", 0.0)))
            step_soh.append(float(info.get("avg_soh", 0.0)))
            step_alive.append(float(info.get("alive_fraction", 1.0)))
            step_soc.append(float(info.get("mean_soc", 0.0)))
            step_count += 1

            if training:
                self.agent.store_transition(state, action, reward, next_state, done)
                self.agent.learn_step()

            state = next_state

        episode_summary = {
            "coverage": float(np.mean(step_coverages)) if step_coverages else 0.0,
            "avg_soh": float(np.mean(step_soh)) if step_soh else 0.0,
            # alive_fraction: use final-step value (reflects end-state of network)
            "alive_fraction": step_alive[-1] if step_alive else 0.0,
            "mean_soc": float(np.mean(step_soc)) if step_soc else 0.0,
            "step_count": step_count,
        }

        return episode_reward, episode_info, episode_summary

    def save_checkpoint(self, path: str) -> None:
        """Save agent weights to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.agent.save_model(str(path))
        self.logger.info(f"Saved checkpoint to {path}")

    def load_checkpoint(self, path: str) -> None:
        """Load agent weights from disk."""
        self.agent.load_model(path)
        self.logger.info(f"Loaded checkpoint from {path}")
