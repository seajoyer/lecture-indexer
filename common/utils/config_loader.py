"""
Configuration loader for the Lecture Video Content Indexer.
Loads configuration from YAML files.
"""

import os
import yaml
import logging
from typing import Dict, Any

# Configure logging
logger = logging.getLogger(__name__)

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from a YAML file.

    Args:
        config_path: Path to the configuration file

    Returns:
        Configuration dictionary
    """
    try:
        # Check if file exists
        if not os.path.exists(config_path):
            logger.warning(f"Configuration file not found: {config_path}")
            return {}

        # Load YAML file
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        if not config:
            logger.warning(f"Empty configuration file: {config_path}")
            return {}

        logger.info(f"Loaded configuration from {config_path}")
        return config

    except Exception as e:
        logger.error(f"Error loading configuration from {config_path}: {e}")
        return {}
