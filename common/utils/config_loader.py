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
        # Print current working directory and absolute path for debugging
        cwd = os.getcwd()
        abs_path = os.path.abspath(config_path)
        logger.info(f"Loading config from '{config_path}'")
        logger.info(f"Current working directory: {cwd}")
        logger.info(f"Absolute path: {abs_path}")

        # Check if file exists
        if not os.path.exists(config_path):
            logger.warning(f"Configuration file not found: {config_path}")

            # Try to find the file in common locations
            potential_locations = [
                # Current directory
                config_path,
                # Parent directory
                os.path.join("..", config_path),
                # Absolute path from project root
                os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), config_path)
            ]

            found = False
            for loc in potential_locations:
                if os.path.exists(loc):
                    logger.info(f"Found configuration at alternative location: {loc}")
                    config_path = loc
                    found = True
                    break

            if not found:
                logger.error("Configuration file not found in any location")
                return {}

        # Load YAML file
        with open(config_path, 'r') as f:
            content = f.read()
            logger.debug(f"Config file content (first 100 chars): {content[:100]}...")

            config = yaml.safe_load(content)

        if not config:
            logger.warning(f"Empty configuration file: {config_path}")
            return {}

        # Log key names from the loaded config for verification
        logger.info(f"Loaded configuration from {config_path} with keys: {list(config.keys())}")

        # Special case for checking API key
        if 'youtube_api_key' in config:
            key = config['youtube_api_key']
            # Mask the key for logging (show only first few and last few characters)
            if key and len(key) > 8:
                masked_key = f"{key[:4]}...{key[-4:]}"
                logger.info(f"Found YouTube API key in config: {masked_key}")
            else:
                logger.warning("YouTube API key is missing or too short")
        else:
            logger.warning("No YouTube API key in loaded configuration")

        return config

    except yaml.YAMLError as e:
        logger.error(f"YAML parsing error in {config_path}: {e}")
        return {}
    except Exception as e:
        logger.error(f"Error loading configuration from {config_path}: {e}")
        return {}
