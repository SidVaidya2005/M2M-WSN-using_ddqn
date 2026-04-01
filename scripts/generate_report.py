"""
Report generation script.

Loads saved metrics JSON files and prints a structured research summary.

Usage:
    python scripts/generate_report.py
    python scripts/generate_report.py --training results/metrics/training_metrics_ddqn.json \
                                      --comparison results/metrics/baseline_comparison.json
"""

import argparse
import sys
from pathlib import Path

# Ensure project root is on sys.path when run from any directory
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from config.settings import get_config
from src.utils.visualization import load_metrics_json
from src.utils.logger import get_logger

logger = get_logger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate a research report from saved metrics files"
    )
    parser.add_argument(
        "--training",
        type=str,
        default=None,
        help="Path to training metrics JSON (default: results/metrics/training_metrics_ddqn.json)",
    )
    parser.add_argument(
        "--comparison",
        type=str,
        default=None,
        help="Path to baseline comparison JSON (default: results/metrics/baseline_comparison.json)",
    )
    return parser.parse_args()


def _section(title: str) -> None:
    width = 60
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


def report_training(metrics_path: Path) -> None:
    """Print training run summary."""
    if not metrics_path.exists():
        print(f"  [not found: {metrics_path}]")
        return

    data = load_metrics_json(str(metrics_path))
    cfg = data.get("config", {})
    train = data.get("training", {})
    ev = data.get("evaluation", {})

    rewards = train.get("rewards", [])
    eval_rewards = ev.get("rewards", [])

    _section("TRAINING CONFIGURATION")
    for key, val in cfg.items():
        print(f"  {key:<20s}  {val}")

    _section("TRAINING RESULTS")
    if rewards:
        mean_r = sum(rewards) / len(rewards)
        max_r = max(rewards)
        min_r = min(rewards)
        trailing = min(10, len(rewards))
        final_ma = sum(rewards[-trailing:]) / trailing
        print(f"  Episodes          {len(rewards)}")
        print(f"  Mean reward       {mean_r:>10.4f}")
        print(f"  Max  reward       {max_r:>10.4f}")
        print(f"  Min  reward       {min_r:>10.4f}")
        print(f"  Final {trailing}-ep MA      {final_ma:>10.4f}")
    else:
        print("  No training reward data found.")

    _section("EVALUATION RESULTS")
    if eval_rewards:
        eval_mean = sum(eval_rewards) / len(eval_rewards)
        eval_max = max(eval_rewards)
        print(f"  Episodes          {len(eval_rewards)}")
        print(f"  Mean reward       {eval_mean:>10.4f}")
        print(f"  Max  reward       {eval_max:>10.4f}")
    else:
        print("  No evaluation data found.")


def report_comparison(comparison_path: Path) -> None:
    """Print baseline comparison summary."""
    if not comparison_path.exists():
        print(f"  [not found: {comparison_path}]")
        return

    data = load_metrics_json(str(comparison_path))

    _section("BASELINE COMPARISON")
    print(f"  {'Policy':<22s}  {'Mean Reward':>12s}")
    print(f"  {'-'*22}  {'-'*12}")

    sorted_policies = sorted(
        data.items(),
        key=lambda kv: kv[1].get("mean_reward", float("-inf")),
        reverse=True,
    )
    for name, result in sorted_policies:
        mean = result.get("mean_reward", float("nan"))
        print(f"  {name:<22s}  {mean:>12.4f}")

    # Compute DDQN improvement over best baseline
    if "DDQN" in data:
        ddqn_mean = data["DDQN"].get("mean_reward", 0)
        baseline_means = [
            v.get("mean_reward", 0)
            for k, v in data.items()
            if k != "DDQN"
        ]
        if baseline_means:
            best_baseline = max(baseline_means)
            improvement = ddqn_mean - best_baseline
            pct = (improvement / abs(best_baseline) * 100) if best_baseline != 0 else float("inf")
            print(f"\n  DDQN vs best baseline: {improvement:+.4f} ({pct:+.1f}%)")


def main():
    """Generate and print the research report."""
    args = parse_args()
    config = get_config()

    training_path = Path(args.training) if args.training else (
        Path(config.paths.metrics) / "training_metrics_ddqn.json"
    )
    comparison_path = Path(args.comparison) if args.comparison else (
        Path(config.paths.metrics) / "baseline_comparison.json"
    )

    print("\n" + "#" * 60)
    print("  WSN DDQN TRAINING PLATFORM — RESEARCH REPORT")
    print("#" * 60)

    report_training(training_path)
    report_comparison(comparison_path)

    print("\n" + "=" * 60)
    print("  END OF REPORT")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
