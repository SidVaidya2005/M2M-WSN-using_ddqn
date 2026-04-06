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
from src.baselines import (
    RandomPolicy,
    GreedyPolicy,
    EnergyConservativePolicy,
    BalancedRotationPolicy,
)
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


# ── Baseline benchmark ───────────────────────────────────────────────────────

def _run_policy_episodes(policy, env: WSNEnv, episodes: int) -> List[float]:
    """Run `episodes` episodes with policy and return total rewards per episode."""
    rewards = []
    for _ in range(episodes):
        state, _ = env.reset()
        episode_reward = 0.0
        done = False
        while not done:
            action = policy.select_action(state, eval_mode=True)
            next_state, reward, done, _info = env.step(action)
            episode_reward += reward
            state = next_state
        rewards.append(episode_reward)
    return rewards


def run_baseline_benchmark(params: dict, config) -> Dict[str, Any]:
    """Benchmark a trained model against baseline policies and save results.

    Reads the run metadata to reconstruct the original environment, evaluates
    four baseline policies plus the trained model, and writes
    results/metrics/{run_id}_evaluation.json.

    Args:
        params: Must contain 'run_id' (str) and 'episodes' (int).
        config: Application Config object.

    Returns:
        Benchmark result dict ready to be returned as JSON.

    Raises:
        FileNotFoundError: If the run metadata file cannot be found.
    """
    run_id = params["run_id"]
    episodes = params["episodes"]

    metadata_path = Path(config.paths.metrics) / f"{run_id}_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"No metadata found for run '{run_id}'")

    with open(metadata_path) as f:
        metadata = json.load(f)

    run_cfg = metadata["config"]
    nodes = run_cfg["nodes"]
    model_type = run_cfg.get("model_type", "ddqn")
    max_steps = run_cfg.get("max_steps", config.environment.max_steps)
    death_threshold = run_cfg.get("death_threshold", 0.3)

    env = WSNEnv(
        N=nodes,
        arena_size=tuple(config.environment.arena_size),
        sink=tuple(config.environment.sink_position),
        max_steps=max_steps,
        death_threshold=death_threshold,
    )
    state_dim = env.observation_space.shape[0]
    action_dim = 2

    baselines = {
        "Random": RandomPolicy(state_dim, action_dim, nodes),
        "Greedy": GreedyPolicy(state_dim, action_dim, nodes, awake_ratio=0.5),
        "EnergyConservative": EnergyConservativePolicy(
            state_dim, action_dim, nodes, awake_ratio=0.2
        ),
        "BalancedRotation": BalancedRotationPolicy(
            state_dim, action_dim, nodes, awake_ratio=0.5
        ),
    }

    benchmark_results: Dict[str, Any] = {}

    for name, policy in baselines.items():
        logger.info(f"Benchmarking baseline '{name}' for run {run_id}…")
        rewards = _run_policy_episodes(policy, env, episodes)
        benchmark_results[name] = {
            "mean_reward": float(sum(rewards) / len(rewards)),
            "policy_type": "baseline",
        }

    model_path = Path(config.paths.models) / f"{run_id}_model.pth"
    if model_path.exists():
        agent_class = DDQNAgent if model_type == "ddqn" else DQNAgent
        agent = agent_class(state_dim=state_dim, action_dim=action_dim, node_count=nodes)
        agent.load_model(str(model_path))
        # Put agent in inference mode (PyTorch method, not Python built-in)
        set_inference_mode = getattr(agent, "eval", None)
        if set_inference_mode:
            set_inference_mode()
        logger.info(f"Benchmarking trained {model_type.upper()} for run {run_id}…")
        rewards = _run_policy_episodes(agent, env, episodes)
        benchmark_results[model_type.upper()] = {
            "mean_reward": float(sum(rewards) / len(rewards)),
            "policy_type": "trained",
        }
    else:
        logger.warning(f"Model not found at {model_path} — skipping trained model")

    env.close()

    output: Dict[str, Any] = {
        "run_id": run_id,
        "benchmark_episodes": episodes,
        "timestamp": datetime.datetime.now().isoformat(),
        "results": benchmark_results,
    }

    benchmark_path = Path(config.paths.metrics) / f"{run_id}_evaluation.json"
    with open(benchmark_path, "w") as f:
        json.dump(output, f, indent=2)

    logger.info(f"Benchmark results saved to {benchmark_path}")
    return output


def _run_benchmark_background(task_id: str, params: dict, config) -> None:
    """Thread target: run baseline benchmark and update the task registry."""
    with _tasks_lock:
        _tasks[task_id]["status"] = "running"

    try:
        result = run_baseline_benchmark(params, config)
        with _tasks_lock:
            _tasks[task_id]["status"] = "completed"
            _tasks[task_id]["result"] = result
        logger.info(f"Benchmark task {task_id} completed successfully")
    except Exception as exc:
        logger.error(f"Benchmark task {task_id} failed: {exc}", exc_info=True)
        with _tasks_lock:
            _tasks[task_id]["status"] = "failed"
            _tasks[task_id]["error"] = str(exc)


def submit_benchmark_task(params: dict, config) -> str:
    """Submit a baseline benchmark job to run in a background thread.

    Args:
        params: Validated parameters from EvaluationRequestSchema.
        config: Application Config object.

    Returns:
        task_id: UUID string — poll GET /api/tasks/<task_id> for status.
    """
    task_id = str(uuid.uuid4())
    with _tasks_lock:
        _tasks[task_id] = {"status": "queued", "result": None, "error": None}

    thread = threading.Thread(
        target=_run_benchmark_background,
        args=(task_id, params, config),
        daemon=True,
    )
    thread.start()
    logger.info(f"Submitted benchmark task {task_id} for run '{params['run_id']}'")
    return task_id
