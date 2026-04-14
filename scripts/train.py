"""
CLI wrapper for WSN DDQN/DQN training.

Produces identical artifacts to the web API (run_YYYYMMDD_HHMMSS naming).

Usage:
    python scripts/train.py --episodes 500 --nodes 50 --lr 1e-4 --model-type ddqn
    python scripts/train.py --model-type dqn --episodes 100 --seed 42
"""

import argparse
import sys
from pathlib import Path

# Ensure project root is on sys.path when run from any directory
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from config.settings import get_config
from backend.tasks import run_training
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _parse_args():
    config = get_config()
    env = config.environment
    tr = config.training

    parser = argparse.ArgumentParser(
        description="Train a DDQN or DQN agent for WSN scheduling"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=tr.episodes,
        help=f"Number of training episodes (default: {tr.episodes})",
    )
    parser.add_argument(
        "--nodes",
        type=int,
        default=None,
        help=f"Number of sensor nodes (default: {env.num_nodes} from config)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=tr.learning_rate,
        help=f"Learning rate (default: {tr.learning_rate})",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=tr.gamma,
        help=f"Discount factor (default: {tr.gamma})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=tr.batch_size,
        help=f"Training batch size (default: {tr.batch_size})",
    )
    parser.add_argument(
        "--death-threshold",
        type=float,
        default=env.death_threshold,
        help=f"Fraction of dead nodes ending an episode (default: {env.death_threshold})",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=env.max_steps,
        help=f"Max steps per episode (default: {env.max_steps})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=env.seed,
        help=f"Random seed (default: {env.seed})",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="ddqn",
        choices=["dqn", "ddqn"],
        help="Agent type: dqn or ddqn (default: ddqn)",
    )
    return parser.parse_args()


def main():
    args = _parse_args()
    config = get_config()
    config.paths.create_all()

    params = {
        "episodes": args.episodes,
        "nodes": args.nodes,            # None → resolved inside run_training
        "learning_rate": args.lr,
        "gamma": args.gamma,
        "batch_size": args.batch_size,
        "death_threshold": args.death_threshold,
        "max_steps": args.max_steps,
        "seed": args.seed,
        "model_type": args.model_type,
    }

    logger.info(
        f"Training {args.model_type.upper()}: episodes={args.episodes}, "
        f"nodes={args.nodes or config.environment.num_nodes}, lr={args.lr}, "
        f"gamma={args.gamma}, batch_size={args.batch_size}, seed={args.seed}"
    )

    result = run_training(params, config)

    m = result.get("metrics", {})
    print(f"\nTraining complete — {result.get('model_used', args.model_type).upper()}")
    print(f"  Run ID:           {result['run_id']}")
    print(f"  Mean reward:      {m.get('mean_reward', 0):.4f}")
    print(f"  Final coverage:   {m.get('final_coverage', 0):.4f}")
    print(f"  Final avg SoH:    {m.get('final_avg_soh', 0):.4f}")
    print(f"  Network lifetime: {m.get('network_lifetime', '?')} episodes")
    print(f"  Plot:             {result['image_url']}")
    print(f"  Model:            {result['model_path']}")

    return result


if __name__ == "__main__":
    main()
