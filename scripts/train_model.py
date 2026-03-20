"""
Standalone training script for WSN DDQN model.

This script can be run from command line to train models without using the web interface.

Usage:
    python scripts/train_model.py --episodes 100 --nodes 550 --lr 1e-4
"""

import argparse
import json
from pathlib import Path
from typing import Dict

from config.settings import get_config
from src.agents.ddqn_agent import DDQNAgent
from src.envs.wsn_env import WSNEnv
from src.training.trainer import Trainer
from src.utils.logger import get_logger
from src.utils.metrics import aggregate_metrics
from src.utils.visualization import save_metrics_json, plot_training_curve

logger = get_logger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train DDQN agent for WSN scheduling"
    )
    
    parser.add_argument(
        "--episodes",
        type=int,
        default=100,
        help="Number of training episodes",
    )
    parser.add_argument(
        "--nodes",
        type=int,
        default=550,
        help="Number of network nodes",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help="Discount factor",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Training batch size",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Output directory for models and metrics",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=10,
        help="Number of evaluation episodes after training",
    )
    
    return parser.parse_args()


def main():
    """Main training script."""
    args = parse_args()
    
    # Load config
    config = get_config()
    config.paths.create_all()
    
    logger.info(f"Training configuration: {args}")
    
    # Create environment
    logger.info(f"Creating WSN environment with {args.nodes} nodes...")
    env = WSNEnv(
        N=args.nodes,
        arena_size=tuple(config.environment.arena_size),
        sink=tuple(config.environment.sink_position),
        max_steps=config.environment.max_steps,
        death_threshold=config.environment.death_threshold,
        seed=args.seed,
    )
    
    # Create agent
    logger.info("Initializing DDQN agent...")
    state_dim = env.observation_space.shape[0]
    action_dim = 2
    
    agent = DDQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        node_count=args.nodes,
        lr=args.lr,
        gamma=args.gamma,
        batch_size=args.batch_size,
    )
    
    # Create trainer
    trainer = Trainer(agent, env, logger_obj=logger, seed=args.seed)
    
    # Training phase
    logger.info(f"Starting training for {args.episodes} episodes...")
    train_rewards, train_metrics = trainer.train(episodes=args.episodes)
    
    logger.info("Training completed!")
    logger.info(f"Mean reward: {sum(train_rewards) / len(train_rewards):.2f}")
    logger.info(f"Max reward: {max(train_rewards):.2f}")
    
    # Save trained model
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = output_dir / "trained_model.pth"
    trainer.save_checkpoint(str(model_path))
    logger.info(f"Model saved to {model_path}")
    
    # Evaluation phase
    logger.info(f"Starting evaluation for {args.eval_episodes} episodes...")
    eval_rewards, eval_metrics = trainer.evaluate(episodes=args.eval_episodes)
    logger.info(f"Evaluation mean reward: {sum(eval_rewards) / len(eval_rewards):.2f}")
    
    # Save metrics
    metrics_data = {
        "training": {
            "rewards": train_rewards,
            "metrics": train_metrics,
        },
        "evaluation": {
            "rewards": eval_rewards,
            "metrics": eval_metrics,
        },
        "config": {
            "episodes": args.episodes,
            "nodes": args.nodes,
            "lr": args.lr,
            "gamma": args.gamma,
            "batch_size": args.batch_size,
        },
    }
    
    metrics_path = output_dir / "training_metrics.json"
    save_metrics_json(metrics_data, str(metrics_path))
    logger.info(f"Metrics saved to {metrics_path}")
    
    # Plot training curve
    plot_path = output_dir / "training_curve.png"
    plot_training_curve(train_rewards, output_path=str(plot_path))
    logger.info(f"Training curve saved to {plot_path}")
    
    logger.info("Training script completed successfully!")
    
    return metrics_data


if __name__ == "__main__":
    main()
