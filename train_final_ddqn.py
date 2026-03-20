#!/usr/bin/env python3
"""
Legacy training script for backward compatibility.

⚠️  DEPRECATED: Use scripts/train_model.py instead
    python scripts/train_model.py --episodes 100

This module is maintained to support existing code that imports train_final_ddqn().
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import torch
from typing import Tuple, Dict

from config.settings import get_config
from src.agents.ddqn_agent import DDQNAgent
from src.envs.wsn_env import WSNEnv
from src.training.trainer import Trainer
from src.utils.logger import get_logger
from src.utils.visualization import plot_training_curve, save_metrics_json

logger = get_logger(__name__)


def train_final_ddqn(
    episodes: int = 100,
    seed: int = 42,
    N: int = 550,
    lr: float = 1e-4,
    gamma: float = 0.99,
    batch_size: int = 64,
    death_threshold: float = 0.3,
) -> Tuple[DDQNAgent, Dict]:
    """
    Legacy training function for backward compatibility.
    
    Trains DDQN agent to maximize network lifetime while maintaining coverage.
    
    Args:
        episodes: Training episodes
        seed: Random seed
        N: Number of nodes
        lr: Learning rate
        gamma: Discount factor
        batch_size: Training batch size
        death_threshold: Network death threshold
        
    Returns:
        Tuple of (trained_agent, results_dict)
    """
    config = get_config()
    config.paths.create_all()
    
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Create environment
    env = WSNEnv(
        N=N,
        arena_size=tuple(config.environment.arena_size),
        sink=tuple(config.environment.sink_position),
        max_steps=config.environment.max_steps,
        death_threshold=death_threshold,
        seed=seed,
    )
    
    # Create agent
    state_dim = env.observation_space.shape[0]
    action_dim = 2
    agent = DDQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        node_count=N,
        lr=lr,
        gamma=gamma,
        batch_size=batch_size,
    )
    
    # Create trainer
    trainer = Trainer(agent, env, logger_obj=logger, seed=seed)
    
    # Print header
    print(f"\n{'='*80}")
    print(f"DDQN TRAINING - NETWORK LIFETIME OPTIMIZATION")
    print(f"{'='*80}")
    print(f"Episodes: {episodes}")
    print(f"Nodes: {N}")
    print(f"Learning Rate: {lr}")
    print(f"Gamma: {gamma}")
    print(f"{'='*80}\n")
    
    # Train
    rewards, metrics = trainer.train(episodes=episodes)
    
    # Print summary
    print(f"\n{'='*80}")
    print(f"TRAINING COMPLETE")
    print(f"{'='*80}")
    mean_reward = np.mean(rewards[-10:])
    print(f"Final 10-ep Mean Reward: {mean_reward:.2f}")
    print(f"Final 10-ep Mean Coverage: {np.mean([m.get('mean_coverage', 0) for m in metrics[-10:]]):.1f}%")
    print(f"{'='*80}\n")
    
    # Save model
    model_path = os.path.join(config.paths.models, "final_ddqn_latest.pth")
    trainer.save_checkpoint(model_path)
    
    # Plot training curve
    plot_path = os.path.join(config.paths.visualizations, "final_ddqn_training.png")
    plot_training_curve(rewards, output_path=plot_path)
    
    # Save metrics
    metrics_data = {
        "rewards": rewards,
        "metrics": metrics,
        "config": {
            "episodes": episodes,
            "nodes": N,
            "lr": lr,
            "gamma": gamma,
            "batch_size": batch_size,
        },
    }
    metrics_path = os.path.join(config.paths.metrics, "final_ddqn_results.json")
    save_metrics_json(metrics_data, metrics_path)
    
    return agent, metrics_data


if __name__ == '__main__':
    train_final_ddqn(episodes=100)
