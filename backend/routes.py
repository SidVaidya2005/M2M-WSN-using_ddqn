"""API routes for WSN DDQN training platform."""

import json

from flask import Blueprint, request, jsonify, current_app, send_from_directory
from marshmallow import ValidationError
from pathlib import Path

from src.utils.logger import get_logger
from .schemas import TrainingRequestSchema, EvaluationRequestSchema
from .tasks import run_training, submit_training_task, get_task, submit_benchmark_task

api_bp = Blueprint("api", __name__)
logger = get_logger(__name__)

_training_schema = TrainingRequestSchema()
_benchmark_schema = EvaluationRequestSchema()


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

    metrics_dir = Path(config.paths.metrics)
    runs = []
    for meta_file in sorted(metrics_dir.glob("*_metadata.json"), reverse=True):
        try:
            with open(meta_file) as f:
                run = json.load(f)
            # Inline benchmark results if they exist
            bench_file = metrics_dir / f"{run['run_id']}_evaluation.json"
            if bench_file.exists():
                try:
                    with open(bench_file) as bf:
                        run["evaluation"] = json.load(bf)
                except Exception as exc:
                    logger.warning(f"Could not read benchmark file {bench_file}: {exc}")
            runs.append(run)
        except Exception as exc:
            logger.warning(f"Skipping corrupt metadata file {meta_file}: {exc}")
    return jsonify(runs), 200


@api_bp.route("/evaluate", methods=["POST"])
def start_benchmark():
    """Submit an async baseline benchmark job for a completed training run.

    Request body:
        run_id    str  required — the run_id from GET /api/history
        episodes  int  optional — evaluation episodes per policy (default 10, max 100)

    Returns:
        202  {"status": "queued", "task_id": "<uuid>"}
        400  Validation errors
        404  run_id metadata not found
    """
    data = request.get_json() or {}

    try:
        params = _benchmark_schema.load(data)
    except ValidationError as exc:
        return jsonify({"status": "error", "errors": exc.messages}), 400

    config = current_app.config.get("CONFIG")
    metadata_path = Path(config.paths.metrics) / f"{params['run_id']}_metadata.json"
    if not metadata_path.exists():
        return jsonify({"status": "error", "message": f"Run '{params['run_id']}' not found"}), 404

    task_id = submit_benchmark_task(params, config)
    return jsonify({"status": "queued", "task_id": task_id}), 202


@api_bp.route("/results/<path:filename>", methods=["GET"])
def serve_results(filename):
    """Serve metrics JSON files."""
    config = current_app.config.get("CONFIG")
    if not config:
        return jsonify({"error": "Configuration not loaded"}), 500
    try:
        return send_from_directory(Path(config.paths.metrics), filename)
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
        return send_from_directory(Path(config.paths.visualizations), filename)
    except Exception as exc:
        logger.error(f"Failed to serve visualizations/{filename}: {exc}")
        return jsonify({"error": "File not found"}), 404
