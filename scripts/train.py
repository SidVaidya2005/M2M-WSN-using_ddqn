"""
Standalone training script for WSN DDQN/DQN model.

This script can be run from command line to train models without using the web interface.

Usage:
    python scripts/train_model.py --episodes 500 --nodes 550 --lr 1e-4 --model-type ddqn
"""

import argparse
import sys
from pathlib import Path

# Ensure project root is on sys.path when run from any directory
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from config.settings import get_config
from src.agents.ddqn_agent import DDQNAgent
from src.agents.dqn_agent import DQNAgent
from src.envs.wsn_env import WSNEnv
from src.training.trainer import Trainer
from src.utils.logger import get_logger
from src.utils.visualization import save_metrics_json, plot_training_curve

logger = get_logger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train a DDQN or DQN agent for WSN scheduling"
    )

    parser.add_argument(
        "--episodes",
        type=int,
        default=100,
        help="Number of training episodes (default: 100)",
    )
    parser.add_argument(
        "--nodes",
        type=int,
        default=50,
        help="Number of sensor nodes (default: 50)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate (default: 1e-4)",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help="Discount factor (default: 0.99)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Training batch size (default: 64)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="ddqn",
        choices=["dqn", "ddqn"],
        help="Agent type: dqn or ddqn (default: ddqn)",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=10,
        help="Number of evaluation episodes after training (default: 10)",
    )

    return parser.parse_args()


def main():
    """Main training script."""
    args = parse_args()

    config = get_config()
    config.paths.create_all()

    logger.info(
        f"Training {args.model_type.upper()}: episodes={args.episodes}, "
        f"nodes={args.nodes}, lr={args.lr}, gamma={args.gamma}, "
        f"batch_size={args.batch_size}, seed={args.seed}"
    )

    env = WSNEnv(
        N=args.nodes,
        arena_size=tuple(config.environment.arena_size),
        sink=tuple(config.environment.sink_position),
        max_steps=config.environment.max_steps,
        death_threshold=config.environment.death_threshold,
        seed=args.seed,
    )

    if env.observation_space.shape is None:
        raise ValueError("env.observation_space.shape is None — cannot determine state dimension")
    state_dim = env.observation_space.shape[0]
    agent_class = DDQNAgent if args.model_type == "ddqn" else DQNAgent
    agent = agent_class(
        state_dim=state_dim,
        action_dim=2,
        node_count=args.nodes,
        lr=args.lr,
        gamma=args.gamma,
        batch_size=args.batch_size,
    )
    logger.info(f"Initialized {agent_class.__name__} (state_dim={state_dim})")

    trainer = Trainer(agent, env, logger_obj=logger, seed=args.seed)
    train_rewards, train_metrics = trainer.train(episodes=args.episodes)

    mean_reward = sum(train_rewards) / len(train_rewards)
    logger.info(f"Training complete — mean: {mean_reward:.2f}, max: {max(train_rewards):.2f}")

    model_path = Path(config.paths.models) / f"trained_model_{args.model_type}.pth"
    trainer.save_checkpoint(str(model_path))

    eval_rewards, eval_metrics = trainer.evaluate(episodes=args.eval_episodes)
    eval_mean = sum(eval_rewards) / len(eval_rewards)
    logger.info(f"Evaluation complete — mean: {eval_mean:.2f}")

    metrics_data = {
        "training": {"rewards": train_rewards, "metrics": train_metrics},
        "evaluation": {"rewards": eval_rewards, "metrics": eval_metrics},
        "config": {
            "episodes": args.episodes,
            "nodes": args.nodes,
            "lr": args.lr,
            "gamma": args.gamma,
            "batch_size": args.batch_size,
            "model_type": args.model_type,
            "seed": args.seed,
        },
    }

    metrics_path = Path(config.paths.metrics) / f"training_metrics_{args.model_type}.json"
    save_metrics_json(metrics_data, str(metrics_path))

    plot_path = Path(config.paths.visualizations) / f"{args.model_type}_training_curve.png"
    plot_training_curve(train_rewards, output_path=str(plot_path))

    logger.info(f"Metrics → {metrics_path}")
    logger.info(f"Plot    → {plot_path}")
    logger.info(f"Model   → {model_path}")

    return metrics_data


if __name__ == "__main__":
    main()
