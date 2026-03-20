"""API routes for WSN DDQN training platform."""

from flask import Blueprint, request, jsonify, current_app, send_from_directory
from pathlib import Path

from config.settings import get_config
from src.agents.ddqn_agent import DDQNAgent
from src.envs.wsn_env import WSNEnv
from src.training.trainer import Trainer
from src.utils.logger import get_logger

api_bp = Blueprint("api", __name__)
logger = get_logger(__name__)


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
    """
    Start training endpoint.
    
    Request body:
    {
        "episodes": int (default: 100),
        "nodes": int (default: 550),
        "learning_rate": float (default: 1e-4),
        "gamma": float (default: 0.99),
        "batch_size": int (default: 64)
    }
    
    Returns: Training results
    """
    try:
        config = current_app.config.get("CONFIG")
        
        # Get request parameters
        data = request.get_json()
        episodes = int(data.get("episodes", config.training.episodes))
        nodes = int(data.get("nodes", config.environment.num_nodes))
        lr = float(data.get("learning_rate", config.training.learning_rate))
        gamma = float(data.get("gamma", config.training.gamma))
        batch_size = int(data.get("batch_size", config.training.batch_size))
        
        logger.info(
            f"Starting training: episodes={episodes}, nodes={nodes}, "
            f"lr={lr}, batch_size={batch_size}"
        )
        
        # Create environment
        env = WSNEnv(
            N=nodes,
            arena_size=tuple(config.environment.arena_size),
            sink=tuple(config.environment.sink_position),
            max_steps=config.environment.max_steps,
            death_threshold=config.environment.death_threshold,
        )
        
        # Create agent
        state_dim = env.observation_space.shape[0]
        action_dim = 2
        agent = DDQNAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            node_count=nodes,
            lr=lr,
            gamma=gamma,
            batch_size=batch_size,
        )
        
        # Create trainer
        trainer = Trainer(agent, env, logger_obj=logger)
        
        # Train
        rewards, metrics = trainer.train(episodes=episodes)
        
        # Save model
        model_path = Path(config.paths.models) / "trained_model.pth"
        trainer.save_checkpoint(str(model_path))
        
        # Prepare results
        results = {
            "status": "success",
            "episodes": episodes,
            "nodes": nodes,
            "mean_reward": float(sum(rewards) / len(rewards)),
            "max_reward": float(max(rewards)),
            "model_path": str(model_path),
        }
        
        logger.info(f"Training completed: {results}")
        return jsonify(results), 200
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}", exc_info=True)
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


@api_bp.route("/results/<path:filename>", methods=["GET"])
def serve_results(filename):
    """Serve results files."""
    try:
        config = current_app.config.get("CONFIG")
        results_dir = Path(config.paths.metrics)
        return send_from_directory(results_dir, filename)
    except Exception as e:
        logger.error(f"Failed to serve results: {str(e)}")
        return jsonify({"error": "File not found"}), 404
