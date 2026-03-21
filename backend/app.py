"""Flask web server for WSN DDQN training platform."""

from flask import Flask, send_from_directory
from flask_cors import CORS
from pathlib import Path

from config.settings import get_config


def create_app(config_path: str = "config/config.yaml") -> Flask:
    """Create and configure Flask application.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configured Flask application
    """
    app = Flask(
        __name__,
        template_folder="../frontend/templates",
        static_folder="../frontend/static",
    )
    
    # Load configuration
    config = get_config(config_path)
    app.config["CONFIG"] = config
    
    # Enable CORS
    CORS(app)
    
    # Create required directories
    config.paths.create_all()
    
    # Register blueprints
    from .routes import api_bp
    app.register_blueprint(api_bp, url_prefix="/api")
    
    # Register error handlers
    @app.errorhandler(400)
    def bad_request(error):
        return {"error": "Bad request", "message": str(error)}, 400
    
    @app.errorhandler(500)
    def internal_error(error):
        return {"error": "Internal server error", "message": str(error)}, 500

    @app.route("/index.css")
    def serve_index_css():
        """Serve template stylesheet used by index page."""
        templates_dir = Path(__file__).resolve().parent.parent / "templates"
        return send_from_directory(templates_dir, "index.css")
    
    return app


if __name__ == "__main__":
    app = create_app()
    app.run(debug=True, host="0.0.0.0", port=5000)
