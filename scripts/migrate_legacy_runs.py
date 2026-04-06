"""
Migrate pre-Phase-2 training artifacts into the new run_id format.

Running this script is safe and idempotent:
  - Files are COPIED, never moved or deleted.
  - Metadata files that already exist are skipped.
  - Source artifacts that don't exist are skipped with a warning.

Usage:
    python scripts/migrate_legacy_runs.py
"""

import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

# ── Project root ─────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent

METRICS_DIR = ROOT / "results" / "metrics"
MODELS_DIR  = ROOT / "results" / "models"
VIZ_DIR     = ROOT / "results" / "visualizations"

ARCHIVE_METRICS = ROOT / "results" / "archive" / "metrics"
ARCHIVE_MODELS  = ROOT / "results" / "archive" / "models"
ARCHIVE_VIZ     = ROOT / "results" / "archive" / "visualizations"

# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)

def _save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  [saved]  {path.relative_to(ROOT)}")

def _copy(src: Path, dst: Path) -> bool:
    """Copy src → dst; returns False if src is missing."""
    if not src.exists():
        print(f"  [skip]   source not found: {src.relative_to(ROOT)}")
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    print(f"  [copied] {src.relative_to(ROOT)} → {dst.relative_to(ROOT)}")
    return True

def _rewards_stats(rewards: list) -> dict:
    if not rewards:
        return {"mean_reward": None, "max_reward": None,
                "best_episode": None, "avg_final_10": None}
    mean  = sum(rewards) / len(rewards)
    mx    = max(rewards)
    best  = rewards.index(mx) + 1
    trail = rewards[-min(10, len(rewards)):]
    avg10 = sum(trail) / len(trail)
    return {
        "mean_reward":  round(mean, 4),
        "max_reward":   round(mx,   4),
        "best_episode": best,
        "avg_final_10": round(avg10, 4),
    }

def _mtime_iso(path: Path) -> str:
    ts = path.stat().st_mtime
    return datetime.fromtimestamp(ts).isoformat()

def _skip_if_exists(meta_path: Path) -> bool:
    if meta_path.exists():
        print(f"  [exists] {meta_path.relative_to(ROOT)} — skipping")
        return True
    return False

# ── Migration entries ─────────────────────────────────────────────────────────

def migrate_improved_ddqn():
    """500-episode improved DDQN training run (archive)."""
    run_id     = "run_20260320_162545"
    meta_path  = METRICS_DIR / f"{run_id}_metadata.json"
    model_dst  = MODELS_DIR  / f"{run_id}_model.pth"
    plot_dst   = VIZ_DIR     / f"{run_id}_plot.png"
    image_url  = f"/api/visualizations/{run_id}_plot.png"

    print(f"\n── {run_id} (improved 500-ep DDQN) ──────────────────────────")
    if _skip_if_exists(meta_path):
        return

    # Metrics
    src_metrics = ARCHIVE_METRICS / "improved_ddqn_results.json"
    raw = _load_json(src_metrics) if src_metrics.exists() else {}
    rewards  = raw.get("rewards", [])
    episodes = raw.get("episodes", len(rewards)) or len(rewards)
    stats    = _rewards_stats(rewards)

    # Artifacts
    _copy(ARCHIVE_MODELS / "final_ddqn_best_lifetime_ep1.pth", model_dst)
    has_plot = _copy(ARCHIVE_VIZ / "improved_ddqn_training_curves.png", plot_dst)

    metadata = {
        "run_id":    run_id,
        "timestamp": _mtime_iso(src_metrics) if src_metrics.exists() else datetime.now().isoformat(),
        "config": {
            "model_type":      "ddqn",
            "episodes":        episodes,
            "nodes":           50,
            "learning_rate":   0.0001,
            "gamma":           0.99,
            "batch_size":      64,
            "death_threshold": 0.3,
            "max_steps":       1000,
            "seed":            42,
        },
        "metrics":   stats,
        "image_url": image_url if has_plot else None,
        "model_path": str(model_dst),
        "_note": "Migrated from archive/metrics/improved_ddqn_results.json",
    }
    _save_json(meta_path, metadata)


def migrate_final_ddqn():
    """10-episode final evaluation run (archive)."""
    run_id    = "run_20260320_162546"
    meta_path = METRICS_DIR / f"{run_id}_metadata.json"
    model_dst = MODELS_DIR  / f"{run_id}_model.pth"
    plot_dst  = VIZ_DIR     / f"{run_id}_plot.png"
    image_url = f"/api/visualizations/{run_id}_plot.png"

    print(f"\n── {run_id} (final 10-ep evaluation) ───────────────────────")
    if _skip_if_exists(meta_path):
        return

    src_metrics = ARCHIVE_METRICS / "final_ddqn_results.json"
    raw = _load_json(src_metrics) if src_metrics.exists() else {}
    rewards = raw.get("rewards", [])
    stats   = _rewards_stats(rewards)

    # Prefer richer fields from the raw file when available
    if raw.get("best_lifetime"):
        stats["max_reward"]   = float(raw["best_lifetime"])
    if raw.get("avg_lifetime_final_10"):
        stats["avg_final_10"] = float(raw["avg_lifetime_final_10"])

    _copy(ARCHIVE_MODELS / "final_ddqn_latest.pth", model_dst)
    has_plot = _copy(ARCHIVE_VIZ / "final_ddqn_training.png", plot_dst)

    metadata = {
        "run_id":    run_id,
        "timestamp": _mtime_iso(src_metrics) if src_metrics.exists() else datetime.now().isoformat(),
        "config": {
            "model_type":      "ddqn",
            "episodes":        len(rewards) or 10,
            "nodes":           50,
            "learning_rate":   0.0001,
            "gamma":           0.99,
            "batch_size":      64,
            "death_threshold": 0.3,
            "max_steps":       1000,
            "seed":            42,
        },
        "metrics":   stats,
        "image_url": image_url if has_plot else None,
        "model_path": str(model_dst),
        "_note": "Migrated from archive/metrics/final_ddqn_results.json",
    }
    _save_json(meta_path, metadata)

    # Also migrate the realistic_comparison → evaluation JSON
    _migrate_realistic_comparison(run_id)


def _migrate_realistic_comparison(run_id: str):
    """Adapt realistic_comparison.json to the evaluation format for a run."""
    src = ARCHIVE_METRICS / "realistic_comparison.json"
    if not src.exists():
        return

    raw  = _load_json(src)
    # realistic_comparison uses 'energy_efficiency' as a performance proxy
    name_map = {
        "DDQN (Trained)":      ("DDQN",               "trained"),
        "Random":              ("Random",              "baseline"),
        "Greedy":              ("Greedy",              "baseline"),
        "Energy Conservative": ("EnergyConservative",  "baseline"),
        "Balanced":            ("BalancedRotation",    "baseline"),
    }
    results = {}
    for src_name, (dst_name, ptype) in name_map.items():
        entry = raw.get(src_name, {})
        # energy_efficiency ≈ service_time × 10 — use it as reward proxy
        mean_reward = entry.get("energy_efficiency", entry.get("service_time", 0))
        results[dst_name] = {
            "mean_reward":  float(mean_reward),
            "policy_type":  ptype,
        }

    eval_path = METRICS_DIR / f"{run_id}_evaluation.json"
    if eval_path.exists():
        print(f"  [exists] {eval_path.relative_to(ROOT)} — skipping")
        return

    eval_data = {
        "run_id":             run_id,
        "benchmark_episodes": 10,
        "timestamp":          _mtime_iso(src),
        "results":            results,
        "_note": "Migrated from archive/metrics/realistic_comparison.json",
    }
    _save_json(eval_path, eval_data)


def migrate_trained_model_ddqn():
    """The old-style trained_model_ddqn.pth in results/models/ (no metrics)."""
    src_model = MODELS_DIR / "trained_model_ddqn.pth"
    if not src_model.exists():
        print(f"\n── trained_model_ddqn.pth not found, skipping ──────────")
        return

    run_id    = "run_20260324_175842"
    meta_path = METRICS_DIR / f"{run_id}_metadata.json"
    model_dst = MODELS_DIR  / f"{run_id}_model.pth"

    print(f"\n── {run_id} (trained_model_ddqn.pth — no metrics) ──────────")
    if _skip_if_exists(meta_path):
        return

    _copy(src_model, model_dst)

    metadata = {
        "run_id":    run_id,
        "timestamp": _mtime_iso(src_model),
        "config": {
            "model_type":      "ddqn",
            "episodes":        None,
            "nodes":           None,
            "learning_rate":   None,
            "gamma":           None,
            "batch_size":      None,
            "death_threshold": None,
            "max_steps":       None,
            "seed":            None,
        },
        "metrics": {
            "mean_reward":  None,
            "max_reward":   None,
            "best_episode": None,
            "avg_final_10": None,
        },
        "image_url":  None,
        "model_path": str(model_dst),
        "_note": "Migrated from results/models/trained_model_ddqn.pth — no training metrics available",
    }
    _save_json(meta_path, metadata)


def migrate_run_20260406():
    """Phase-2 model whose metadata was never saved (crash or early exit)."""
    src_model = MODELS_DIR / "run_20260406_080528_model.pth"
    if not src_model.exists():
        print(f"\n── run_20260406_080528_model.pth not found, skipping ───")
        return

    run_id    = "run_20260406_080528"
    meta_path = METRICS_DIR / f"{run_id}_metadata.json"

    print(f"\n── {run_id} (Phase-2 model — metrics lost) ──────────────────")
    if _skip_if_exists(meta_path):
        return

    metadata = {
        "run_id":    run_id,
        "timestamp": _mtime_iso(src_model),
        "config": {
            "model_type":      "ddqn",
            "episodes":        None,
            "nodes":           None,
            "learning_rate":   None,
            "gamma":           None,
            "batch_size":      None,
            "death_threshold": None,
            "max_steps":       None,
            "seed":            None,
        },
        "metrics": {
            "mean_reward":  None,
            "max_reward":   None,
            "best_episode": None,
            "avg_final_10": None,
        },
        "image_url":  None,
        "model_path": str(src_model),
        "_note": "Model file found but training metrics were not saved (likely a crash or early exit).",
    }
    _save_json(meta_path, metadata)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("═" * 60)
    print("  WSN DDQN — Legacy Run Migration")
    print("═" * 60)
    print(f"  Results root: {ROOT / 'results'}")

    migrate_improved_ddqn()
    migrate_final_ddqn()
    migrate_trained_model_ddqn()
    migrate_run_20260406()

    print("\n═" * 60)
    print("  Migration complete.")
    print("  Refresh the browser (http://127.0.0.1:5001) to see")
    print("  the legacy runs in the Training History tab.")
    print("═" * 60)


if __name__ == "__main__":
    # Must run from project root so relative paths resolve correctly
    import os
    os.chdir(ROOT)
    main()
