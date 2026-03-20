"""Structured logging for the application."""

import logging
import logging.config
from pathlib import Path
from typing import Optional

import yaml


def setup_logging(config_path: str = "config/logging_config.yaml") -> logging.Logger:
    """
    Setup logging from YAML configuration file.
    
    Args:
        config_path: Path to logging config YAML file
        
    Returns:
        Logger instance
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        # Fallback to basic config if file not found
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
        return logging.getLogger(__name__)
    
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    logging.config.dictConfig(config)
    return logging.getLogger(__name__)


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Get a logger instance.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


# Initialize logging
logger = setup_logging()
