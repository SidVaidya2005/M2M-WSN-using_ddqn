"""Visualization utilities for plots and animations."""

from typing import Dict, List, Optional
import numpy as np
import json
from pathlib import Path


def save_metrics_json(metrics: Dict, output_path: str) -> None:
    """Save metrics to JSON file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        if isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [convert_to_serializable(v) for v in obj]
        return obj

    with open(output_path, "w") as f:
        json.dump(convert_to_serializable(metrics), f, indent=2)


def load_metrics_json(input_path: str) -> Dict:
    """Load metrics from JSON file."""
    with open(input_path, "r") as f:
        return json.load(f)


def _moving_avg(values: List[float], window: int) -> Optional[List[float]]:
    """Return a moving average series, or None if there are fewer points than the window."""
    if len(values) < window:
        return None
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid").tolist()


def plot_training_dashboard(
    rewards: List[float],
    series: Optional[Dict[str, List]] = None,
    output_path: Optional[str] = None,
    window_size: int = 10,
) -> None:
    """Save a 2×2 training dashboard PNG.

    Panels
    ------
    Top-left  : Network Coverage over episodes (line + moving average)
    Top-right : Battery Health (avg SoH) over episodes (line + moving average)
    Bottom-left : Network Lifetime — steps-per-episode bar + mean horizontal line
    Bottom-right: Episode Reward (left axis) & Mean SoC (right axis), dual-axis

    Args:
        rewards: Per-episode total rewards.
        series: Dict with keys ``coverage``, ``avg_soh``, ``alive_fraction``,
                ``mean_soc``, ``step_counts`` (each a list aligned with ``rewards``).
                When None the function falls back to a single reward-curve panel.
        output_path: Where to write the PNG. Parent directories are created.
        window_size: Window for moving-average overlays.
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
    except ImportError:
        print("Warning: matplotlib not installed, skipping plot generation")
        return

    if series is None:
        # Fallback: single reward curve (keeps backward compat for callers without series)
        _plot_single_reward_curve(rewards, output_path, window_size)
        return

    episodes = list(range(1, len(rewards) + 1))
    coverage = series.get("coverage", [])
    avg_soh = series.get("avg_soh", [])
    mean_soc = series.get("mean_soc", [])
    step_counts = series.get("step_counts", [])

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("WSN Training Dashboard", fontsize=14, fontweight="bold", y=0.98)

    ORANGE = "#fe964a"
    BLUE = "#0077b6"
    GREEN = "#2a9d8f"
    PURPLE = "#9b5de5"
    GREY_BG = "#fdfdfd"

    # ── Panel 1: Network Coverage ────────────────────────────────────────────
    ax = axes[0, 0]
    ax.set_facecolor(GREY_BG)
    if coverage:
        ax.plot(episodes, coverage, color=ORANGE, alpha=0.8, linewidth=1.2,
                label="Coverage")
        ma = _moving_avg(coverage, window_size)
        if ma is not None:
            ax.plot(
                range(window_size, len(coverage) + 1),
                ma,
                color=BLUE, linewidth=2.2,
                label=f"{window_size}-ep MA",
            )
    ax.set_xlabel("Episode")
    ax.set_ylabel("Coverage fraction")
    ax.set_title("Network Coverage")
    ax.set_ylim(0.0, 1.05)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── Panel 2: Battery Health (avg SoH) ───────────────────────────────────
    ax = axes[0, 1]
    ax.set_facecolor(GREY_BG)
    if avg_soh:
        ax.plot(episodes, avg_soh, color=GREEN, alpha=0.8, linewidth=1.2,
                label="Avg SoH")
        ma = _moving_avg(avg_soh, window_size)
        if ma is not None:
            ax.plot(
                range(window_size, len(avg_soh) + 1),
                ma,
                color=BLUE, linewidth=2.2,
                label=f"{window_size}-ep MA",
            )
    ax.set_xlabel("Episode")
    ax.set_ylabel("State of Health")
    ax.set_title("Battery Health (avg SoH)")
    ax.set_ylim(0.0, 1.05)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── Panel 3: Network Lifetime (steps per episode) ────────────────────────
    ax = axes[1, 0]
    ax.set_facecolor(GREY_BG)
    if step_counts:
        ax.bar(episodes, step_counts, color=PURPLE, alpha=0.7, width=0.8,
               label="Steps / episode")
        mean_steps = float(np.mean(step_counts))
        ax.axhline(mean_steps, color=ORANGE, linewidth=2.0, linestyle="--",
                   label=f"Mean: {mean_steps:.0f}")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Steps survived")
    ax.set_title("Network Lifetime (steps per episode)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    # ── Panel 4: Reward & Mean SoC (dual axis) ──────────────────────────────
    ax = axes[1, 1]
    ax.set_facecolor(GREY_BG)
    ax.plot(episodes, rewards, color=ORANGE, alpha=0.85, linewidth=1.2,
            label="Episode Reward")
    ma_r = _moving_avg(rewards, window_size)
    if ma_r is not None:
        ax.plot(
            range(window_size, len(rewards) + 1),
            ma_r,
            color=BLUE, linewidth=2.2,
            label=f"{window_size}-ep MA (reward)",
        )
    ax.set_xlabel("Episode")
    ax.set_ylabel("Episode Reward", color=ORANGE)
    ax.tick_params(axis="y", labelcolor=ORANGE)
    ax.set_title("Reward & Mean SoC")
    ax.grid(True, alpha=0.3)

    if mean_soc:
        ax2 = ax.twinx()
        ax2.plot(episodes, mean_soc, color=GREEN, alpha=0.7, linewidth=1.5,
                 linestyle=":", label="Mean SoC")
        ax2.set_ylabel("Mean SoC (0–1)", color=GREEN)
        ax2.tick_params(axis="y", labelcolor=GREEN)
        ax2.set_ylim(0.0, 1.05)
        # Combined legend
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="lower right")
    else:
        ax.legend(fontsize=8)

    fig.tight_layout(rect=[0, 0, 1, 0.97])

    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150, bbox_inches="tight")

    plt.close(fig)


def _plot_single_reward_curve(
    episode_rewards: List[float],
    output_path: Optional[str],
    window_size: int,
) -> None:
    """Internal fallback: single-panel reward curve (used when series is None)."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    episodes = range(1, len(episode_rewards) + 1)
    ax.plot(list(episodes), episode_rewards, color="#fe964a", alpha=0.9,
            label="Episode Reward", linewidth=1.5)
    ma = _moving_avg(episode_rewards, window_size)
    if ma is not None:
        ax.plot(
            range(window_size, len(episode_rewards) + 1),
            ma,
            color="#0077b6",
            label=f"{window_size}-Episode MA",
            linewidth=2.5,
        )
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.set_title("Training Progress")
    ax.legend()
    ax.set_facecolor("#fdfdfd")
    ax.grid(True, alpha=0.3)

    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150, bbox_inches="tight")

    plt.close(fig)


def plot_comparison_dashboard(
    series_a: Dict[str, List],
    series_b: Dict[str, List],
    label_a: str = "Run A",
    label_b: str = "Run B",
    output_path: Optional[str] = None,
) -> None:
    """Save a 2×2 DDQN-vs-DQN comparison PNG with both agents overlaid.

    Panels
    ------
    Top-left  : Network Coverage — both agents overlaid
    Top-right : Battery Health (avg SoH) — both agents overlaid
    Bottom-left : Network Lifetime (steps/episode) — both agents overlaid
    Bottom-right: Episode Reward — both agents overlaid

    Args:
        series_a: Per-episode series dict for run A (keys: episode_reward, coverage,
                  avg_soh, step_counts, …).
        series_b: Same for run B.
        label_a: Legend label for run A.
        label_b: Legend label for run B.
        output_path: Where to write the PNG.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Warning: matplotlib not installed, skipping comparison plot")
        return

    COLOR_A = "#7bd0ff"   # primary blue — typically DDQN
    COLOR_B = "#fe964a"   # orange — typically DQN
    GREY_BG = "#fdfdfd"

    def _ep(values):
        return list(range(1, len(values) + 1))

    def _overlay(ax, key, ylabel, title, ylim=None):
        va = series_a.get(key, [])
        vb = series_b.get(key, [])
        ax.set_facecolor(GREY_BG)
        if va:
            ax.plot(_ep(va), va, color=COLOR_A, alpha=0.8, linewidth=1.5, label=label_a)
        if vb:
            ax.plot(_ep(vb), vb, color=COLOR_B, alpha=0.8, linewidth=1.5, label=label_b)
        ax.set_xlabel("Episode")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        if ylim:
            ax.set_ylim(*ylim)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f"DDQN vs DQN Comparison\n{label_a}  ·  {label_b}",
        fontsize=13, fontweight="bold", y=0.98,
    )

    _overlay(axes[0, 0], "coverage",       "Coverage fraction", "Network Coverage",        ylim=(0, 1.05))
    _overlay(axes[0, 1], "avg_soh",        "Avg SoH",           "Battery Health (avg SoH)", ylim=(0, 1.05))
    _overlay(axes[1, 0], "step_counts",    "Steps survived",    "Network Lifetime (steps/ep)")
    _overlay(axes[1, 1], "episode_reward", "Episode Reward",    "Reward")

    fig.tight_layout(rect=[0, 0, 1, 0.95])

    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150, bbox_inches="tight")

    plt.close(fig)


# Backward-compat alias — callers that only have rewards still work.
def plot_training_curve(
    episode_rewards: List[float],
    output_path: Optional[str] = None,
    window_size: int = 10,
) -> None:
    """Deprecated alias for plot_training_dashboard without series data."""
    plot_training_dashboard(episode_rewards, series=None,
                            output_path=output_path, window_size=window_size)
