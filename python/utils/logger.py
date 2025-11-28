"""
Logger initialization and configuration utilities.

Author: s Bostan
Created on: Nov, 2025
"""

import logging
import logging.config
import os
import yaml
from pathlib import Path


def setup_logging(config_path: str = None, env: str = None) -> None:
    """
    Setup logging configuration from YAML file.
    
    Args:
        config_path: Path to logging configuration YAML file
        env: Environment name (dev/prod/test)
    """
    if config_path is None:
        # Default to logging_config.yaml in project root
        config_path = Path(__file__).parent.parent / "logging_config.yaml"
    
    if env is None:
        env = os.getenv("ENV", "dev")
    
    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Load configuration
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Adjust log levels based on environment
        if env == "prod":
            config['root']['level'] = 'INFO'
            for logger_name, logger_config in config.get('loggers', {}).items():
                if isinstance(logger_config, dict) and 'level' in logger_config:
                    if logger_config['level'] == 'DEBUG':
                        logger_config['level'] = 'INFO'
        elif env == "test":
            config['root']['level'] = 'WARNING'
        
        logging.config.dictConfig(config)
    else:
        # Fallback to basic configuration
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(name)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    
    # Add correlation ID filter to root logger
    from python.utils.logging_formatter import CorrelationIDFilter
    root_logger = logging.getLogger()
    for handler in root_logger.handlers:
        handler.addFilter(CorrelationIDFilter())


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the given name.
    
    Args:
        name: Logger name (typically __name__)
    
    Returns:
        Logger instance
    """
    return logging.getLogger(name)

