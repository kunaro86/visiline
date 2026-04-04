"""
Configuration management for YOLO-TMR project
"""

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class TrainConfig:
    """Training configuration"""

    epochs: int = 100
    batch_size: int = 16
    imgsz: int = 640
    device: str = "cuda"  # cuda or cpu
    workers: int = 4
    patience: int = 20  # EarlyStopping patience
    optimizer: str = "SGD"
    momentum: float = 0.937
    weight_decay: float = 0.0005
    lr0: float = 0.01  # initial learning rate
    lrf: float = 0.1  # final learning rate (lr0 * lrf)
    warmup_epochs: float = 3.0
    warmup_momentum: float = 0.8
    seed: int = 42
    save_period: int = 10  # Save checkpoint every n epochs
    val_split: float = 0.1
    test_split: float = 0.1
    augment: bool = True
    mosaic: float = 1.0
    mixup: float = 0.0
    hsv_h: float = 0.015  # HSV-Hue augmentation
    hsv_s: float = 0.7  # HSV-Saturation augmentation
    hsv_v: float = 0.4  # HSV-Value augmentation
    degrees: float = 10.0  # Rotation
    translate: float = 0.1  # Translation
    scale: float = 0.5  # Scaling
    flipud: float = 0.0  # Vertical flip
    fliplr: float = 0.5  # Horizontal flip


@dataclass
class ModelConfig:
    """Model configuration"""

    model_name: str = "yolov8n"  # nano, small, medium, large, xlarge
    num_classes: int = 43  # Number of traffic sign classes
    pretrained: bool = True
    task: str = "detect"  # detect or classify


@dataclass
class DataConfig:
    """Data configuration"""

    data_dir: str = "./data"
    train_dir: str = "train"
    val_dir: str = "val"
    test_dir: str = "test"
    format: str = "coco"  # or yolo, coco, etc.
    cache: bool = True
    shuffle: bool = True


@dataclass
class Config:
    """Main configuration class"""

    project_root: Path = field(default_factory=lambda: Path(__file__).parent.parent)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    data: DataConfig = field(default_factory=DataConfig)

    # Output directories
    output_dir: str | None = None
    weights_dir: str | None = None
    logs_dir: str | None = None

    def __post_init__(self):
        """Setup directory paths after initialization"""
        if self.output_dir is None:
            self.output_dir = str(self.project_root / "outputs")
        if self.weights_dir is None:
            self.weights_dir = str(self.project_root / "outputs" / "weights")
        if self.logs_dir is None:
            self.logs_dir = str(self.project_root / "outputs" / "logs")

        # Create directories if they don't exist
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.weights_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "Config":
        """Load configuration from YAML file"""
        with open(yaml_path, encoding="utf-8") as f:
            config_dict = yaml.safe_load(f)

        return cls.from_dict(config_dict)

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "Config":
        """Create config from dictionary"""
        model_config = ModelConfig(**config_dict.get("model", {}))
        train_config = TrainConfig(**config_dict.get("train", {}))
        data_config = DataConfig(**config_dict.get("data", {}))

        other_args = {
            k: v for k, v in config_dict.items() if k not in ["model", "train", "data"]
        }

        return cls(
            model=model_config, train=train_config, data=data_config, **other_args
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary"""
        return {
            "model": asdict(self.model),
            "train": asdict(self.train),
            "data": asdict(self.data),
            "output_dir": self.output_dir,
            "weights_dir": self.weights_dir,
            "logs_dir": self.logs_dir,
        }

    def save(self, yaml_path: str) -> None:
        """Save configuration to YAML file"""
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, allow_unicode=True)

    def __str__(self) -> str:
        """String representation"""
        return yaml.dump(self.to_dict(), default_flow_style=False, allow_unicode=True)
