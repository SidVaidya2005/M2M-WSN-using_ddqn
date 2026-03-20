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
        """Initialize trainer.
        
        Args:
            agent: RL agent to train
            env: Environment to train in
            logger_obj: Logger instance
            seed: Random seed
        """
        self.agent = agent
        self.env = env
        self.logger = logger_obj or logger
        self.seed = seed
        
        np.random.seed(seed)
        
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_metrics = []

    def train(
        self,
        episodes: int,
        callbacks: Optional[List[Callable]] = None,
    ) -> Tuple[List[float], List[Dict]]:
        """Train agent for specified number of episodes.
        
        Args:
            episodes: Number of training episodes
            callbacks: List of callback functions to call after each episode
            
        Returns:
            Tuple of (episode_rewards, episode_metrics)
        """
        if callbacks is None:
            callbacks = []
        
        self.logger.info(f"Training for {episodes} episodes")
        
        for episode in range(episodes):
            episode_reward, episode_info = self._run_episode(training=True)
            
            self.episode_rewards.append(episode_reward)
            metrics = compute_episode_metrics(
                [episode_reward], 
                [episode_info]
            )
            self.episode_metrics.append(metrics)
            
            # Log progress
            if (episode + 1) % 10 == 0:
                mean_reward = np.mean(self.episode_rewards[-10:])
                self.logger.info(
                    f"Episode {episode+1}/{episodes} - "
                    f"Reward: {episode_reward:.2f}, "
                    f"10-ep MA: {mean_reward:.2f}"
                )
            
            # Run callbacks
            for callback in callbacks:
                callback(episode, episode_reward, metrics)
        
        return self.episode_rewards, self.episode_metrics

    def evaluate(
        self,
        episodes: int = 10,
    ) -> Tuple[List[float], List[Dict]]:
        """Evaluate agent without training.
        
        Args:
            episodes: Number of evaluation episodes
            
        Returns:
            Tuple of (episode_rewards, episode_metrics)
        """
        self.logger.info(f"Evaluating for {episodes} episodes")
        
        eval_rewards = []
        eval_metrics = []
        
        for episode in range(episodes):
            episode_reward, episode_info = self._run_episode(training=False)
            eval_rewards.append(episode_reward)
            
            metrics = compute_episode_metrics(
                [episode_reward],
                [episode_info]
            )
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
    ) -> Tuple[float, Dict]:
        """Run a single episode.
        
        Args:
            training: If True, update agent. If False, just evaluate.
            
        Returns:
            Tuple of (total_episode_reward, final_info_dict)
        """
        state, info = self.env.reset()
        episode_reward = 0.0
        episode_info = None
        done = False
        
        while not done:
            # Select action
            action = self.agent.select_action(state, eval_mode=not training)
            
            # Step environment
            next_state, reward, done, info = self.env.step(action)
            episode_reward += reward
            episode_info = info
            
            # Store transition and learn
            if training:
                self.agent.store_transition(
                    state, action, reward, next_state, done
                )
                self.agent.learn_step()
            
            state = next_state
        
        return episode_reward, episode_info or {}

    def save_checkpoint(self, path: str) -> None:
        """Save agent and training history.
        
        Args:
            path: Path to save checkpoint
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        self.agent.save_model(str(path))
        self.logger.info(f"Saved checkpoint to {path}")

    def load_checkpoint(self, path: str) -> None:
        """Load agent and training history.
        
        Args:
            path: Path to load checkpoint
        """
        self.agent.load_model(path)
        self.logger.info(f"Loaded checkpoint from {path}")
