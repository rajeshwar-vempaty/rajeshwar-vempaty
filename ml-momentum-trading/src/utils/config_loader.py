"""
Configuration loader utility
"""

import yaml
from pathlib import Path
from typing import Dict


def load_config(config_path: str = "config/config.yaml") -> Dict:
    """
    Load YAML configuration file

    Args:
        config_path: Path to config file

    Returns:
        Configuration dict
    """
    config_file = Path(config_path)

    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    return config
