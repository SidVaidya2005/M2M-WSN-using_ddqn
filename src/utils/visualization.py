"""Visualization utilities for plots and animations."""

import matplotlib
matplotlib.use('Agg')  # non-interactive backend safe for background threads

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


# ── Shared style constants ────────────────────────────────────────────────────

ORANGE  = "#fe964a"
BLUE    = "#0077b6"
GREEN   = "#2a9d8f"
PURPLE  = "#9b5de5"
TEAL    = "#06d6a0"
GREY_BG = "#fdfdfd"

# Canonical 4-metric panel config: (series_key, ylabel, title, color)
_PANELS = [
    ("coverage",           "Coverage fraction",       "Network Coverage",     ORANGE),
    ("avg_soh",            "State of Health",          "Battery Health",       GREEN),
    ("energy_consumption", "SoC drain (start − end)",  "Energy Consumption",   PURPLE),
    ("throughput",         "Coverage × Alive fraction","Throughput",           TEAL),
]


def _draw_metric_panel(ax, values, label, ylabel, title, color, window_size=50):
    """Draw a single metric line + moving-average overlay onto ax."""
    ax.set_facecolor(GREY_BG)
    if values:
        episodes = list(range(len(values)))
        ax.plot(episodes, values, color=color, alpha=0.8, linewidth=1.2, label=label)
        ma = _moving_avg(values, window_size)
        if ma is not None:
            ax.plot(
                range(window_size - 1, len(values)),
                ma,
                color=BLUE, linewidth=2.2,
                label=f"{window_size}-ep MA",
            )
    ax.set_xlabel("Episode")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xlim(left=0)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)


def plot_training_dashboard(
    rewards: List[float],
    series: Optional[Dict[str, List]] = None,
    output_path: Optional[str] = None,
    window_size: int = 50,
) -> None:
    """Save a 2×2 training dashboard PNG with the 4 canonical metrics.

    Panels
    ------
    Top-left     : Network Coverage
    Top-right    : Battery Health (avg SoH)
    Bottom-left  : Energy Consumption (SoC drain per episode)
    Bottom-right : Throughput (coverage × alive_fraction)

    Args:
        rewards: Per-episode total rewards (kept for API compatibility).
        series: Dict with keys ``coverage``, ``avg_soh``, ``energy_consumption``,
                ``throughput`` (each a list aligned with ``rewards``).
                When None the function falls back to a single reward-curve panel.
        output_path: Where to write the PNG. Parent directories are created.
        window_size: Window for moving-average overlays.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Warning: matplotlib not installed, skipping plot generation")
        return

    if series is None:
        _plot_single_reward_curve(rewards, output_path, window_size)
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("WSN Training Dashboard", fontsize=14, fontweight="bold", y=0.98)

    for ax, (key, ylabel, title, color) in zip(axes.flat, _PANELS):
        _draw_metric_panel(
            ax,
            series.get(key, []),
            label=title,
            ylabel=ylabel,
            title=title,
            color=color,
            window_size=window_size,
        )

    fig.tight_layout(rect=[0, 0, 1, 0.97])

    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150, bbox_inches="tight")

    plt.close(fig)


def plot_individual_metrics(
    series: Dict[str, List],
    output_dir: str,
    window_size: int = 50,
) -> Dict[str, str]:
    """Save one PNG per canonical metric into output_dir.

    Returns a dict mapping metric name → absolute file path.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Warning: matplotlib not installed, skipping individual metric plots")
        return {}

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    filenames = {
        "coverage":           "coverage.png",
        "avg_soh":            "battery_health.png",
        "energy_consumption": "energy_consumption.png",
        "throughput":         "throughput.png",
    }

    saved: Dict[str, str] = {}
    for key, ylabel, title, color in _PANELS:
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        fig.suptitle(title, fontsize=12, fontweight="bold")
        _draw_metric_panel(
            ax,
            series.get(key, []),
            label=title,
            ylabel=ylabel,
            title=title,
            color=color,
            window_size=window_size,
        )
        fig.tight_layout()
        dest = out_dir / filenames[key]
        fig.savefig(dest, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved[key] = str(dest)

    return saved


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
    episodes = range(len(episode_rewards))
    ax.plot(list(episodes), episode_rewards, color=ORANGE, alpha=0.9,
            label="Episode Reward", linewidth=1.5)
    ma = _moving_avg(episode_rewards, window_size)
    if ma is not None:
        ax.plot(
            range(window_size - 1, len(episode_rewards)),
            ma,
            color=BLUE,
            label=f"{window_size}-Episode MA",
            linewidth=2.5,
        )
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.set_title("Training Progress")
    ax.legend()
    ax.set_facecolor(GREY_BG)
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
    individual_output_dir: Optional[str] = None,
) -> Dict[str, str]:
    """Save a 2×2 DDQN-vs-DQN comparison PNG with both agents overlaid,
    and optionally save each panel as a separate PNG.

    Panels
    ------
    Top-left     : Network Coverage — both agents overlaid
    Top-right    : Battery Health (avg SoH) — both agents overlaid
    Bottom-left  : Energy Consumption — both agents overlaid
    Bottom-right : Throughput — both agents overlaid

    Args:
        series_a: Per-episode series dict for run A.
        series_b: Same for run B.
        label_a: Legend label for run A.
        label_b: Legend label for run B.
        output_path: Where to write the combined PNG.
        individual_output_dir: Directory to write one PNG per panel. If None,
            individual plots are derived from output_path's stem directory.

    Returns:
        Dict mapping ``"combined"`` and each panel key to their saved file paths.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Warning: matplotlib not installed, skipping comparison plot")
        return {}

    COLOR_A = "#7bd0ff"
    COLOR_B = "#fe964a"

    PANELS_CMP = [
        ("coverage",           "Coverage fraction",        "Network Coverage",      (0, 1.05)),
        ("mean_soc",           "Mean State of Charge",     "Battery Charge Level",  (0, 1.05)),
        ("energy_consumption", "SoC drain (start − end)",  "Energy Consumption",    None),
        ("throughput",         "Coverage × Alive fraction","Throughput",            None),
    ]

    FILENAMES_CMP = {
        "coverage":           "coverage.png",
        "mean_soc":           "battery_charge.png",
        "energy_consumption": "energy_consumption.png",
        "throughput":         "throughput.png",
    }

    def _ep(values):
        return list(range(len(values)))

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
        ax.set_xlim(left=0)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        if ylim:
            ax.set_ylim(*ylim)

    saved: Dict[str, str] = {}

    # Combined 2×2 dashboard
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f"DDQN vs DQN Comparison\n{label_a}  ·  {label_b}",
        fontsize=13, fontweight="bold", y=0.98,
    )

    _overlay(axes[0, 0], "coverage",           "Coverage fraction",        "Network Coverage",    ylim=(0, 1.05))
    _overlay(axes[0, 1], "avg_soh",            "Avg SoH",                  "Battery Health",      ylim=(0, 1.05))
    _overlay(axes[1, 0], "energy_consumption", "SoC drain (start − end)",  "Energy Consumption")
    _overlay(axes[1, 1], "throughput",         "Coverage × Alive fraction","Throughput")

    fig.tight_layout(rect=[0, 0, 1, 0.95])

    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150, bbox_inches="tight")
        saved["combined"] = str(out)

    plt.close(fig)

    # Individual panels
    ind_dir = Path(individual_output_dir) if individual_output_dir else (
        Path(output_path).parent / Path(output_path).stem if output_path else None
    )
    if ind_dir:
        ind_dir.mkdir(parents=True, exist_ok=True)
        for key, ylabel, title, ylim in PANELS_CMP:
            fig, ax = plt.subplots(1, 1, figsize=(8, 5))
            fig.suptitle(
                f"{title}\n{label_a}  ·  {label_b}",
                fontsize=12, fontweight="bold",
            )
            _overlay(ax, key, ylabel, title, ylim=ylim)
            fig.tight_layout()
            dest = ind_dir / FILENAMES_CMP[key]
            fig.savefig(dest, dpi=150, bbox_inches="tight")
            plt.close(fig)
            saved[key] = str(dest)

    return saved


# Backward-compat alias — callers that only have rewards still work.
def plot_training_curve(
    episode_rewards: List[float],
    output_path: Optional[str] = None,
    window_size: int = 50,
) -> None:
    """Deprecated alias for plot_training_dashboard without series data."""
    plot_training_dashboard(episode_rewards, series=None,
                            output_path=output_path, window_size=window_size)
