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
from src.utils.visualization import plot_training_curve

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
    nodes = params["nodes"]
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

    plot_filename = f"{run_id}_plot.png"
    plot_path = Path(config.paths.visualizations) / plot_filename
    plot_training_curve(rewards, output_path=str(plot_path))

    mean_reward = float(sum(rewards) / len(rewards)) if rewards else 0.0
    max_reward = float(max(rewards)) if rewards else 0.0
    best_episode = (int(rewards.index(max(rewards))) + 1) if rewards else 0
    trailing = min(10, len(rewards))
    avg_final_10 = float(sum(rewards[-trailing:]) / trailing) if rewards else 0.0
    image_url = f"/api/visualizations/{plot_filename}"

    metadata = {
        "run_id": run_id,
        "timestamp": datetime.datetime.now().isoformat(),
        "config": {
            "model_type": model_type,
            "episodes": episodes,
            "nodes": nodes,
            "learning_rate": lr,
            "gamma": gamma,
            "batch_size": batch_size,
            "death_threshold": death_threshold,
            "max_steps": max_steps,
            "seed": seed,
        },
        "metrics": {
            "mean_reward": mean_reward,
            "max_reward": max_reward,
            "best_episode": best_episode,
            "avg_final_10": avg_final_10,
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
        "episodes": episodes,
        "nodes": nodes,
        "model_type": model_type,
        "mean_reward": mean_reward,
        "max_reward": max_reward,
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
