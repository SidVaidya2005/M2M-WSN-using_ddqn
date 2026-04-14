"""
DDQN vs DQN comparison CLI tool.

Loads two saved training runs and produces a 2x2 side-by-side comparison plot
(Network Coverage, Battery Health, Network Lifetime, Reward) saved to
results/visualizations/compare_<run_a>_vs_<run_b>.png.

Usage:
    # Auto-pick most recent DDQN and DQN runs:
    python scripts/compare.py

    # Specify runs explicitly:
    python scripts/compare.py --run-a run_20260414_080000 --run-b run_20260414_090000
"""

import argparse
import json
import sys
from pathlib import Path

# Ensure project root is on sys.path when run from any directory
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from config.settings import get_config
from src.utils.visualization import plot_comparison_dashboard
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _load_meta(metrics_dir: Path, run_id: str) -> dict:
    path = metrics_dir / f"{run_id}_metadata.json"
    if not path.exists():
        raise FileNotFoundError(f"Metadata file not found: {path}")
    with open(path) as f:
        return json.load(f)


def _find_latest(metrics_dir: Path, model_type: str) -> str:
    """Return the run_id of the most recent completed run for the given model type."""
    for meta_file in sorted(metrics_dir.glob("*_metadata.json"), reverse=True):
        try:
            with open(meta_file) as f:
                data = json.load(f)
            # Support both old schema (config.model_type) and new schema (model_used)
            mtype = data.get("model_used") or data.get("config", {}).get("model_type", "")
            if mtype == model_type:
                return data["run_id"]
        except Exception:
            continue
    raise ValueError(
        f"No completed {model_type.upper()} runs found in {metrics_dir}.\n"
        f"Run 'python scripts/train.py --model-type {model_type}' first."
    )


def _get_label(meta: dict, run_id: str) -> str:
    mtype = (
        meta.get("model_used")
        or meta.get("config", {}).get("model_type", "?")
    ).upper()
    return f"{mtype} ({run_id[-8:]})"


def _print_summary(label: str, meta: dict) -> None:
    m = meta.get("metrics", {})
    cfg = meta.get("config", {})
    episodes = meta.get("episodes") or cfg.get("episodes", "?")
    nodes = meta.get("num_nodes") or cfg.get("nodes", "?")
    print(f"\n  [{label}]")
    print(f"    Run ID:           {meta.get('run_id', '?')}")
    print(f"    Episodes:         {episodes}  |  Nodes: {nodes}")
    print(f"    Mean reward:      {m.get('mean_reward', 0):.4f}")
    print(f"    Final coverage:   {m.get('final_coverage', 0):.4f}")
    print(f"    Final avg SoH:    {m.get('final_avg_soh', 0):.4f}")
    print(f"    Network lifetime: {m.get('network_lifetime', '?')} episodes")


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Compare two DDQN/DQN training runs and produce a side-by-side plot"
    )
    parser.add_argument(
        "--run-a",
        type=str,
        default=None,
        help="Run ID of the first run (default: most recent DDQN run)",
    )
    parser.add_argument(
        "--run-b",
        type=str,
        default=None,
        help="Run ID of the second run (default: most recent DQN run)",
    )
    return parser.parse_args()


def main():
    args = _parse_args()
    config = get_config()
    config.paths.create_all()

    metrics_dir = Path(config.paths.metrics)
    vis_dir = Path(config.paths.visualizations)

    run_id_a = args.run_a
    run_id_b = args.run_b

    if run_id_a is None:
        run_id_a = _find_latest(metrics_dir, "ddqn")
        logger.info(f"Auto-selected DDQN run: {run_id_a}")
        print(f"Auto-selected DDQN run: {run_id_a}")

    if run_id_b is None:
        run_id_b = _find_latest(metrics_dir, "dqn")
        logger.info(f"Auto-selected DQN run: {run_id_b}")
        print(f"Auto-selected DQN run:  {run_id_b}")

    meta_a = _load_meta(metrics_dir, run_id_a)
    meta_b = _load_meta(metrics_dir, run_id_b)

    series_a = meta_a.get("series", {})
    series_b = meta_b.get("series", {})

    label_a = _get_label(meta_a, run_id_a)
    label_b = _get_label(meta_b, run_id_b)

    filename = f"compare_{run_id_a}_vs_{run_id_b}.png"
    plot_path = vis_dir / filename
    plot_comparison_dashboard(series_a, series_b, label_a, label_b,
                              output_path=str(plot_path))

    print(f"\nComparison complete:")
    _print_summary(label_a, meta_a)
    _print_summary(label_b, meta_b)
    print(f"\n  Plot saved: {plot_path}")

    return str(plot_path)


if __name__ == "__main__":
    main()
