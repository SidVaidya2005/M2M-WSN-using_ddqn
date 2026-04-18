"""Training execution engine.

``run_training`` runs synchronously and is wrapped by ``submit_training_task``
to provide a UUID-keyed background-thread queue polled via
``GET /api/tasks/<task_id>``. ``compare_runs`` produces the DDQN-vs-DQN plot.
"""

import datetime
import json
import threading
import uuid
from pathlib import Path
from typing import Any, Dict

from src.agents.ddqn_agent import DDQNAgent
from src.agents.dqn_agent import DQNAgent
from src.envs.wsn_env import WSNEnv
from src.training.trainer import Trainer
from src.utils.logger import get_logger
from src.utils.visualization import (
    plot_training_dashboard,
    plot_individual_metrics,
    plot_comparison_dashboard,
)

logger = get_logger(__name__)

_tasks: Dict[str, Dict[str, Any]] = {}
_tasks_lock = threading.Lock()


def get_task(task_id: str) -> Dict[str, Any]:
    """Return current state of a task.

    Args:
        task_id: UUID returned by submit_training_task

    Returns:
        Dict with keys: status, result, error
        status is one of: queued | running | completed | failed | not_found
    """
    with _tasks_lock:
        return dict(_tasks.get(task_id, {"status": "not_found"}))


def run_training(params: dict, config, progress_callback=None) -> Dict[str, Any]:
    """Execute training synchronously and return the result dict.

    Used by the /api/train route for synchronous (blocking) invocation.

    Args:
        params: Validated request parameters from TrainingRequestSchema
        config: Application Config object

    Returns:
        Result dictionary ready to be returned as JSON
    """
    run_id = datetime.datetime.now().strftime("run_%Y%m%d_%H%M%S")
    episodes = params["episodes"]
    nodes = params.get("nodes") or config.environment.num_nodes
    lr = params["learning_rate"]
    gamma = params["gamma"]
    batch_size = params["batch_size"]
    death_threshold = params["death_threshold"]
    max_steps = params.get("max_steps", config.environment.max_steps)
    seed = params["seed"]
    model_type = params["model_type"]

    env = WSNEnv(
        N=nodes,
        arena_size=tuple(config.environment.arena_size),
        sink=tuple(config.environment.sink_position),
        max_steps=max_steps,
        death_threshold=death_threshold,
        reward_weights=(
            config.environment.reward_weights.coverage,
            config.environment.reward_weights.energy,
            config.environment.reward_weights.soh,
            config.environment.reward_weights.balance,
        ),
        charging_enabled=config.environment.charging.enabled,
        charging_rate=config.environment.charging.rate,
        charging_threshold=config.environment.charging.threshold,
        wake_cooperation_soc=config.environment.wake_cooperation.low_battery_soc,
        sensing_radius=config.environment.sensing_radius,
    )

    state_dim = env.observation_space.shape[0]
    agent_class = DDQNAgent if model_type == "ddqn" else DQNAgent
    agent = agent_class(
        state_dim=state_dim,
        action_dim=2,
        node_count=nodes,
        lr=lr,
        gamma=gamma,
        batch_size=batch_size,
    )

    trainer = Trainer(agent, env, seed=seed)
    rewards = trainer.train(episodes=episodes, progress_callback=progress_callback)

    model_path = Path(config.paths.models) / f"{run_id}_model.pth"
    trainer.save_checkpoint(str(model_path))

    ep_series = trainer.episode_series

    # Combined 4-panel dashboard (served as the primary image)
    plot_filename = f"{run_id}_plot.png"
    plot_path = Path(config.paths.visualizations) / plot_filename
    plot_training_dashboard(rewards, series=ep_series, output_path=str(plot_path))
    image_url = f"/api/visualizations/{plot_filename}"

    # Individual metric PNGs saved under results/visualizations/<datetime>/
    datetime_suffix = run_id[len("run_"):]   # "YYYYMMDD_HHMMSS"
    individual_dir = Path(config.paths.visualizations) / datetime_suffix
    individual_paths = plot_individual_metrics(ep_series, output_dir=str(individual_dir))
    individual_image_urls = {
        key: f"/api/visualizations/{datetime_suffix}/{Path(path).name}"
        for key, path in individual_paths.items()
    }

    mean_reward = float(sum(rewards) / len(rewards)) if rewards else 0.0
    max_reward = float(max(rewards)) if rewards else 0.0
    best_episode = (int(rewards.index(max(rewards))) + 1) if rewards else 0
    trailing = min(10, len(rewards))
    avg_final_10 = float(sum(rewards[-trailing:]) / trailing) if rewards else 0.0

    coverage_series = ep_series.get("coverage", [])
    soh_series = ep_series.get("avg_soh", [])
    final_coverage = float(coverage_series[-1]) if coverage_series else 0.0
    final_avg_soh = float(soh_series[-1]) if soh_series else 0.0
    network_lifetime = trainer.network_lifetime

    metadata = {
        "run_id": run_id,
        "timestamp": datetime.datetime.now().isoformat(),
        "model_used": model_type,
        "episodes": episodes,
        "num_nodes": nodes,
        "learning_rate": lr,
        "gamma": gamma,
        "death_threshold": death_threshold,
        "max": max_steps,
        "seed": seed,
        "metrics": {
            "mean_reward": mean_reward,
            "max_reward": max_reward,
            "best_episode": best_episode,
            "avg_final_10": avg_final_10,
            "final_coverage": final_coverage,
            "final_avg_soh": final_avg_soh,
            "network_lifetime": network_lifetime,
        },
        "series": {
            "coverage":           [float(v) for v in coverage_series],
            "avg_soh":            [float(v) for v in soh_series],
            "energy_consumption": [float(v) for v in ep_series.get("energy_consumption", [])],
            "throughput":         [float(v) for v in ep_series.get("throughput", [])],
        },
        "image_url": image_url,
        "individual_image_urls": individual_image_urls,
        "model_path": str(model_path),
    }
    metadata_path = Path(config.paths.metrics) / f"{run_id}_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    return {
        "status": "success",
        "message": f"Training completed with {model_type.upper()}.",
        "run_id": run_id,
        "model_used": model_type,
        "num_nodes": nodes,
        "episodes": episodes,
        "mean_reward": mean_reward,
        "metrics": metadata["metrics"],
        "model_path": str(model_path),
        "image_url": image_url,
    }


def _run_training_background(task_id: str, params: dict, config) -> None:
    """Thread target: run training and update the task registry."""
    with _tasks_lock:
        _tasks[task_id]["status"] = "running"
        _tasks[task_id]["progress"] = 0

    def _progress(current: int, total: int) -> None:
        pct = round(current / total * 100) if total else 0
        with _tasks_lock:
            _tasks[task_id]["progress"] = pct

    try:
        result = run_training(params, config, progress_callback=_progress)
        with _tasks_lock:
            _tasks[task_id]["status"] = "completed"
            _tasks[task_id]["progress"] = 100
            _tasks[task_id]["result"] = result
        logger.info(f"Task {task_id} completed successfully")
    except Exception as exc:
        logger.error(f"Task {task_id} failed: {exc}", exc_info=True)
        with _tasks_lock:
            _tasks[task_id]["status"] = "failed"
            _tasks[task_id]["error"] = str(exc)


def submit_training_task(params: dict, config) -> str:
    """Submit a training job to run asynchronously in a background thread.

    Args:
        params: Validated request parameters from TrainingRequestSchema
        config: Application Config object

    Returns:
        task_id: UUID string — poll GET /api/tasks/<task_id> for status
    """
    task_id = str(uuid.uuid4())
    with _tasks_lock:
        _tasks[task_id] = {"status": "queued", "progress": 0, "result": None, "error": None}

    thread = threading.Thread(
        target=_run_training_background,
        args=(task_id, params, config),
        daemon=True,
    )
    thread.start()
    logger.info(f"Submitted background task {task_id}")
    return task_id


def compare_runs(run_id_a: str, run_id_b: str, config) -> Dict[str, Any]:
    """Generate a 2×2 DDQN-vs-DQN comparison PNG from two saved training runs.

    Loads the metadata JSON for each run_id, extracts per-episode series, and
    produces a side-by-side overlay plot saved to results/visualizations/.

    Args:
        run_id_a: First run ID (e.g. "run_20260414_080000")
        run_id_b: Second run ID
        config: Application Config object

    Returns:
        Dict with ``image_url``, ``run_a``, ``run_b`` keys.

    Raises:
        FileNotFoundError: If either metadata file does not exist.
    """
    metrics_dir = Path(config.paths.metrics)
    vis_dir = Path(config.paths.visualizations)

    def _load_meta(run_id: str) -> dict:
        path = metrics_dir / f"{run_id}_metadata.json"
        if not path.exists():
            raise FileNotFoundError(f"{run_id}_metadata.json not found in {metrics_dir}")
        with open(path) as f:
            return json.load(f)

    meta_a = _load_meta(run_id_a)
    meta_b = _load_meta(run_id_b)

    series_a = meta_a.get("series", {})
    series_b = meta_b.get("series", {})

    label_a = f"{(meta_a.get('model_used') or meta_a.get('config', {}).get('model_type', '?')).upper()} ({run_id_a[-8:]})"
    label_b = f"{(meta_b.get('model_used') or meta_b.get('config', {}).get('model_type', '?')).upper()} ({run_id_b[-8:]})"

    filename = f"compare_{run_id_a}_vs_{run_id_b}.png"
    plot_path = vis_dir / filename
    plot_comparison_dashboard(series_a, series_b, label_a, label_b,
                              output_path=str(plot_path))

    logger.info(f"Comparison plot saved: {plot_path}")
    return {
        "image_url": f"/api/visualizations/{filename}",
        "run_a": run_id_a,
        "run_b": run_id_b,
    }
