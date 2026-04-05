import logging
import os
import random
from pathlib import Path

import numpy as np
import torch
import yaml


def setup_logger(name: str = "yolo-tmr", log_file: str | None = None) -> logging.Logger:
    """Setup logger with both console and optional file handlers"""
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(device: str = "cuda") -> torch.device:
    if device == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    elif device == "cuda":
        print("CUDA not available, using CPU")
        return torch.device("cpu")
    return torch.device("cpu")


def create_dirs(paths: list) -> None:
    for path in paths:
        os.makedirs(path, exist_ok=True)


def get_traffic_sign_classes(config_yaml: str | None = None) -> dict:
    """Load traffic sign classes from YAML config file or use defaults

    Args:
        config_yaml: Path to data.yaml config file. If None, uses default location.
                    Falls back to hardcoded GTSRB classes if file not found.

    Returns:
        Dictionary mapping class index to class name
    """
    # Determine config file path
    if config_yaml is None:
        # Try to find configs/data.yaml relative to project root
        possible_paths = [
            Path("configs/data.yaml"),
            Path(__file__).parent.parent / "configs" / "data.yaml",
            Path.cwd() / "configs" / "data.yaml",
        ]

        for path in possible_paths:
            if path.exists():
                config_yaml = str(path)
                break

    # Try to load from YAML file
    if config_yaml and Path(config_yaml).exists():
        try:
            with open(config_yaml, encoding="utf-8") as f:
                config = yaml.safe_load(f)

            if config and "names" in config:
                names = config["names"]
                # Convert to proper dict with int keys if needed
                if isinstance(names, dict):
                    return {int(k): v for k, v in names.items()}
                elif isinstance(names, list):
                    return dict(enumerate(names))
        except Exception as e:
            logging.warning(
                f"Failed to load classes from {config_yaml}: {e}. "
                f"Falling back to default GTSRB classes."
            )

    # Default GTSRB classes (German traffic signs) - 43 classes
    # This is used as fallback when YAML file is not available
    default_classes = {
        0: "Speed limit 20km/h",
        1: "Speed limit 30km/h",
        2: "Speed limit 50km/h",
        3: "Speed limit 60km/h",
        4: "Speed limit 70km/h",
        5: "Speed limit 80km/h",
        6: "End of speed limit 80km/h",
        7: "Speed limit 100km/h",
        8: "Speed limit 120km/h",
        9: "No passing",
        10: "No passing for trucks",
        11: "Right of way at next intersection",
        12: "Priority road",
        13: "Yield",
        14: "Stop",
        15: "No entry",
        16: "Prohibited for all vehicles",
        17: "Prohibited for trucks",
        18: "No entry for bicycles",
        19: "Speed limit warning",
        20: "Dangerous curve left",
        21: "Dangerous curve right",
        22: "Double dangerous curve",
        23: "Bumpy road",
        24: "Slippery road",
        25: "Road narrows on right",
        26: "Construction",
        27: "Traffic signals",
        28: "Pedestrian crossing",
        29: "Children crossing",
        30: "Bicycle crossing",
        31: "Beware of ice/snow",
        32: "Wild animals crossing",
        33: "End of speed and passing limits",
        34: "Turn right ahead",
        35: "Turn left ahead",
        36: "Ahead only",
        37: "Go straight or turn right",
        38: "Go straight or turn left",
        39: "Keep right",
        40: "Keep left",
        41: "Roundabout mandatory",
        42: "End of no passing zone",
    }

    return default_classes


def print_config(config) -> None:
    """Print configuration"""
    print("=" * 60)
    print("PROJECT CONFIGURATION")
    print("=" * 60)
    print(str(config))
    print("=" * 60)
