"""API routes for WSN DDQN training platform."""

import json

from flask import Blueprint, request, jsonify, current_app, send_from_directory
from marshmallow import ValidationError
from pathlib import Path


def _project_root() -> Path:
    """Return the project root directory (one level above backend/)."""
    return Path(current_app.root_path).parent


def _abs(relative_path: str) -> Path:
    """Resolve a config-relative path to an absolute path from the project root."""
    p = Path(relative_path)
    return p if p.is_absolute() else _project_root() / p


def _apply_config_defaults(run: dict, config) -> None:
    """Fill null config fields in a run dict with current config defaults.

    Runs migrated from legacy artifacts may have null for all hyperparameters.
    Rather than showing 'N/A', we substitute the project defaults so the
    History cards look complete.  Metrics remain null when genuinely unknown.
    """
    cfg = run.get("config")
    if not isinstance(cfg, dict):
        return
    env = config.environment
    tr  = config.training
    defaults = {
        "nodes":           env.num_nodes,
        "learning_rate":   tr.learning_rate,
        "gamma":           tr.gamma,
        "batch_size":      tr.batch_size,
        "death_threshold": env.death_threshold,
        "max_steps":       env.max_steps,
        "seed":            env.seed,
    }
    for key, val in defaults.items():
        if cfg.get(key) is None:
            cfg[key] = val

from src.utils.logger import get_logger
from .schemas import TrainingRequestSchema
from .tasks import run_training, submit_training_task, get_task, compare_runs

api_bp = Blueprint("api", __name__)
logger = get_logger(__name__)

_training_schema = TrainingRequestSchema()


@api_bp.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({"status": "healthy"}), 200


@api_bp.route("/config", methods=["GET"])
def get_config_endpoint():
    """Get current configuration."""
    config = current_app.config.get("CONFIG")
    if config:
        return jsonify(config.to_dict()), 200
    return jsonify({"error": "Configuration not loaded"}), 500


@api_bp.route("/train", methods=["POST"])
def train_model():
    """Start a synchronous training run and return results.

    Request body (all fields optional — defaults from config/config.yaml):
        episodes      int   1-10000
        nodes         int   10-10000
        learning_rate float 1e-6 to 0.1
        gamma         float 0.0 to 1.0
        batch_size    int   8-512
        death_threshold float 0.0-1.0
        seed          int
        model_type    str   "dqn" | "ddqn"

    Returns:
        200  Training result dict (status, mean_reward, image_url, …)
        400  Validation errors
        500  Training exception
    """
    data = request.get_json() or {}

    try:
        params = _training_schema.load(data)
    except ValidationError as exc:
        return jsonify({"status": "error", "errors": exc.messages}), 400

    config = current_app.config.get("CONFIG")

    logger.info(
        f"Starting training: episodes={params['episodes']}, nodes={params['nodes']}, "
        f"lr={params['learning_rate']}, batch_size={params['batch_size']}, "
        f"model_type={params['model_type']}, seed={params['seed']}"
    )

    try:
        result = run_training(params, config)
        logger.info(f"Training completed: mean_reward={result['mean_reward']:.4f}")
        return jsonify(result), 200
    except Exception as exc:
        logger.error(f"Training failed: {exc}", exc_info=True)
        return jsonify({"status": "error", "message": str(exc)}), 500


@api_bp.route("/train/async", methods=["POST"])
def train_model_async():
    """Submit a training job and return a task_id immediately.

    The client can poll GET /api/tasks/<task_id> for status.

    Returns:
        202  {"status": "queued", "task_id": "<uuid>"}
        400  Validation errors
    """
    data = request.get_json() or {}

    try:
        params = _training_schema.load(data)
    except ValidationError as exc:
        return jsonify({"status": "error", "errors": exc.messages}), 400

    config = current_app.config.get("CONFIG")
    task_id = submit_training_task(params, config)
    return jsonify({"status": "queued", "task_id": task_id}), 202


@api_bp.route("/tasks/<task_id>", methods=["GET"])
def task_status(task_id: str):
    """Poll the status of a background training task.

    Returns:
        200  {"status": "queued"|"running"|"completed"|"failed", "result": …, "error": …}
        404  Task not found
    """
    task = get_task(task_id)
    if task["status"] == "not_found":
        return jsonify({"status": "not_found", "task_id": task_id}), 404
    return jsonify(task), 200


@api_bp.route("/history", methods=["GET"])
def get_history():
    """Return all training run metadata, newest first.

    Scans results/metrics/ for *_metadata.json files and returns them as a
    sorted JSON list so the frontend can render the Training History panel.

    Returns:
        200  List of run metadata dicts
        500  Configuration not loaded
    """
    config = current_app.config.get("CONFIG")
    if not config:
        return jsonify({"error": "Configuration not loaded"}), 500

    metrics_dir = _abs(config.paths.metrics)
    runs = []
    for meta_file in sorted(metrics_dir.glob("*_metadata.json"), reverse=True):
        try:
            with open(meta_file) as f:
                run = json.load(f)
            # Fill null config fields with project defaults
            _apply_config_defaults(run, config)
            runs.append(run)
        except Exception as exc:
            logger.warning(f"Skipping corrupt metadata file {meta_file}: {exc}")
    return jsonify(runs), 200


@api_bp.route("/compare", methods=["GET"])
def compare_runs_endpoint():
    """Generate and serve a 2×2 DDQN-vs-DQN comparison plot.

    Query params:
        a  Run ID of the first training run
        b  Run ID of the second training run

    Returns:
        200  {"status": "success", "image_url": "...", "run_a": "...", "run_b": "..."}
        400  Missing run IDs
        404  Metadata file not found for one of the run IDs
        500  Plot generation failed
    """
    run_id_a = request.args.get("a", "").strip()
    run_id_b = request.args.get("b", "").strip()
    if not run_id_a or not run_id_b:
        return jsonify({"status": "error",
                        "message": "Provide ?a=<run_id>&b=<run_id>"}), 400

    config = current_app.config.get("CONFIG")
    if not config:
        return jsonify({"error": "Configuration not loaded"}), 500

    try:
        result = compare_runs(run_id_a, run_id_b, config)
        return jsonify({"status": "success", **result}), 200
    except FileNotFoundError as exc:
        return jsonify({"status": "error", "message": str(exc)}), 404
    except Exception as exc:
        logger.error(f"Comparison failed: {exc}", exc_info=True)
        return jsonify({"status": "error", "message": str(exc)}), 500


@api_bp.route("/results/<path:filename>", methods=["GET"])
def serve_results(filename):
    """Serve metrics JSON files."""
    config = current_app.config.get("CONFIG")
    if not config:
        return jsonify({"error": "Configuration not loaded"}), 500
    try:
        return send_from_directory(_abs(config.paths.metrics), filename)
    except Exception as exc:
        logger.error(f"Failed to serve results/{filename}: {exc}")
        return jsonify({"error": "File not found"}), 404


@api_bp.route("/visualizations/<path:filename>", methods=["GET"])
def serve_visualizations(filename):
    """Serve visualization plots."""
    config = current_app.config.get("CONFIG")
    if not config:
        return jsonify({"error": "Configuration not loaded"}), 500
    try:
        return send_from_directory(_abs(config.paths.visualizations), filename)
    except Exception as exc:
        logger.error(f"Failed to serve visualizations/{filename}: {exc}")
        return jsonify({"error": "File not found"}), 404
