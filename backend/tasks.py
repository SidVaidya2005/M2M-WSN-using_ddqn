"""Background task execution for training jobs.

Provides a thread-based task queue for running training jobs asynchronously.
Each job is assigned a UUID task_id that can be polled via GET /api/tasks/<task_id>.

The /api/train endpoint currently runs synchronously (for frontend compatibility).
To switch to full async: call submit_training_task() and return the task_id immediately,
then let the client poll /api/tasks/<task_id> for status.
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
from src.utils.visualization import plot_training_dashboard

logger = get_logger(__name__)

# In-memory task registry: {task_id -> {status, result, error}}
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


def run_training(params: dict, config) -> Dict[str, Any]:
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
    rewards, _ = trainer.train(episodes=episodes)

    model_path = Path(config.paths.models) / f"{run_id}_model.pth"
    trainer.save_checkpoint(str(model_path))

    # ── Build series from trainer ─────────────────────────────────────────
    ep_series = trainer.episode_series  # populated during train()

    plot_filename = f"{run_id}_plot.png"
    plot_path = Path(config.paths.visualizations) / plot_filename
    plot_training_dashboard(rewards, series=ep_series, output_path=str(plot_path))
    image_url = f"/api/visualizations/{plot_filename}"

    # ── Scalar summary metrics ────────────────────────────────────────────
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

    # ── New metadata schema (Phase 3) ────────────────────────────────────
    metadata = {
        # Top-level fields match user spec exactly
        "run_id": run_id,
        "timestamp": datetime.datetime.now().isoformat(),
        "model_used": model_type,
        "episodes": episodes,
        "num_nodes": nodes,
        "learning_rate": lr,
        "gamma": gamma,
        "death_threshold": death_threshold,
        "max": max_steps,          # user spec uses "max" not "max_steps"
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
            "episode_reward": [float(r) for r in rewards],
            "coverage": [float(v) for v in coverage_series],
            "avg_soh": [float(v) for v in soh_series],
            "alive_fraction": [float(v) for v in ep_series.get("alive_fraction", [])],
            "mean_soc": [float(v) for v in ep_series.get("mean_soc", [])],
        },
        "image_url": image_url,
        "model_path": str(model_path),
    }
    metadata_path = Path(config.paths.metrics) / f"{run_id}_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    return {
        "status": "success",
        "message": f"Training completed with {model_type.upper()}.",
        "run_id": run_id,
        # New top-level fields
        "model_used": model_type,
        "num_nodes": nodes,
        # Backward-compat aliases for the frontend (Phase 4 will clean these up)
        "model_type": model_type,
        "nodes": nodes,
        "episodes": episodes,
        "mean_reward": mean_reward,
        "max_reward": max_reward,
        "metrics": metadata["metrics"],
        "series": metadata["series"],
        "results": {
            "best_lifetime": max_reward,
            "best_episode": best_episode,
            "avg_lifetime_final_10": avg_final_10,
        },
        "model_path": str(model_path),
        "image_url": image_url,
    }


def _run_training_background(task_id: str, params: dict, config) -> None:
    """Thread target: run training and update the task registry."""
    with _tasks_lock:
        _tasks[task_id]["status"] = "running"

    try:
        result = run_training(params, config)
        with _tasks_lock:
            _tasks[task_id]["status"] = "completed"
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
        _tasks[task_id] = {"status": "queued", "result": None, "error": None}

    thread = threading.Thread(
        target=_run_training_background,
        args=(task_id, params, config),
        daemon=True,
    )
    thread.start()
    logger.info(f"Submitted background task {task_id}")
    return task_id
