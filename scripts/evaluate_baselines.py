"""
Baseline evaluation script.

Compares DDQN against baseline policies to benchmark performance.

Usage:
    python scripts/evaluate_baselines.py
    python scripts/evaluate_baselines.py --model results/models/trained_model_ddqn.pth --episodes 10
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Ensure project root is on sys.path when run from any directory
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from config.settings import get_config
from src.agents.base_agent import BaseAgent
from src.agents.ddqn_agent import DDQNAgent
from src.baselines import (
    RandomPolicy,
    GreedyPolicy,
    EnergyConservativePolicy,
    BalancedRotationPolicy,
)
from src.envs.wsn_env import WSNEnv
from src.training.trainer import Trainer
from src.utils.logger import get_logger
from src.utils.metrics import compute_episode_metrics
from src.utils.visualization import save_metrics_json

logger = get_logger(__name__)


def evaluate_policy(
    policy: BaseAgent,
    env: WSNEnv,
    episodes: int = 10,
) -> Tuple[List[float], List[Dict]]:
    """Evaluate a policy for a fixed number of episodes.

    Args:
        policy: Any BaseAgent subclass (baseline or trained agent)
        env: WSN environment instance
        episodes: Number of evaluation episodes

    Returns:
        Tuple of (episode_rewards, episode_metrics)
    """
    rewards = []
    metrics = []

    for _ in range(episodes):
        # WSNEnv.reset() returns a plain numpy array (not a tuple)
        state = env.reset()
        episode_reward = 0.0
        episode_info = {}
        done = False

        while not done:
            action = policy.select_action(state, eval_mode=True)
            next_state, reward, done, info = env.step(action)
            episode_reward += reward
            episode_info = info
            state = next_state

        rewards.append(episode_reward)
        metrics.append(compute_episode_metrics([episode_reward], [episode_info]))

    return rewards, metrics


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate baseline policies against a trained DDQN agent"
    )

    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to trained DDQN model (.pth). Skips DDQN eval if not provided.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=10,
        help="Number of evaluation episodes per policy (default: 10)",
    )
    parser.add_argument(
        "--nodes",
        type=int,
        default=550,
        help="Number of sensor nodes (default: 550)",
    )

    return parser.parse_args()


def main():
    """Main evaluation script."""
    args = parse_args()

    config = get_config()
    config.paths.create_all()

    logger.info(f"Evaluating policies — {args.episodes} episodes, {args.nodes} nodes")

    # Shared environment
    env = WSNEnv(
        N=args.nodes,
        arena_size=tuple(config.environment.arena_size),
        sink=tuple(config.environment.sink_position),
        max_steps=config.environment.max_steps,
    )

    state_dim = env.observation_space.shape[0]
    action_dim = 2

    # Baseline policies to compare
    policies = {
        "Random": RandomPolicy(state_dim, action_dim, args.nodes),
        "Greedy": GreedyPolicy(state_dim, action_dim, args.nodes, awake_ratio=0.5),
        "EnergyConservative": EnergyConservativePolicy(
            state_dim, action_dim, args.nodes, awake_ratio=0.2
        ),
        "BalancedRotation": BalancedRotationPolicy(
            state_dim, action_dim, args.nodes, awake_ratio=0.5
        ),
    }

    results = {}

    # Evaluate each baseline
    for name, policy in policies.items():
        logger.info(f"Evaluating {name}...")
        rewards, metrics = evaluate_policy(policy, env, episodes=args.episodes)
        mean = sum(rewards) / len(rewards)
        results[name] = {
            "rewards": rewards,
            "mean_reward": mean,
            "metrics": metrics,
        }
        logger.info(f"  {name}: mean reward = {mean:.2f}")

    # Evaluate trained DDQN if a model path was given
    model_path = Path(args.model) if args.model else Path(config.paths.models) / "trained_model_ddqn.pth"
    if model_path.exists():
        logger.info(f"Evaluating DDQN from {model_path}...")
        agent = DDQNAgent(state_dim=state_dim, action_dim=action_dim, node_count=args.nodes)
        agent.load_model(str(model_path))
        trainer = Trainer(agent, env, logger_obj=logger)
        rewards, metrics = trainer.evaluate(episodes=args.episodes)
        mean = sum(rewards) / len(rewards)
        results["DDQN"] = {
            "rewards": rewards,
            "mean_reward": mean,
            "metrics": metrics,
        }
        logger.info(f"  DDQN: mean reward = {mean:.2f}")
    else:
        logger.warning(f"No model found at {model_path} — skipping DDQN evaluation")

    # Save comparison results
    comparison_path = Path(config.paths.metrics) / "baseline_comparison.json"
    save_metrics_json(results, str(comparison_path))
    logger.info(f"Results saved to {comparison_path}")

    # Print summary table
    print("\n" + "=" * 50)
    print("BASELINE COMPARISON SUMMARY")
    print("=" * 50)
    for name, result in results.items():
        print(f"  {name:<22s}  {result['mean_reward']:>8.2f}")
    print("=" * 50)


if __name__ == "__main__":
    main()
