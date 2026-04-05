"""
Configuration management based on dataclasses, with YAML support
支持灵活的YAML配置加载(单文件或多文件合并)
"""

import os
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class TrainConfig:
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
    """模型配置: 指定模型类型、权重位置和加载策略

    - weight_path: 自定义权重存放目录(相对路径)
    - official_path: 官方预训练权重存放目录(相对路径)
    - pretrained: True 时从 official_path 寻找并在需要时下载: False 时从 weight_path 寻找, 缺失则报错

    注意: num_classes 已移除, 应从 configs/data.yaml 读取以避免冲突
    """

    model_name: str = "yolov8n"  # nano, small, medium, large, xlarge
    weight_path: str = "raw/weights"  # 自定义权重目录
    official_path: str = "./models"  # 官方预训练权重目录
    pretrained: bool = (
        True  # True: 从 official_path 寻找/下载; False: 从 weight_path 寻找
    )
    task: str = "detect"  # detect or classify


@dataclass
class DataConfig:
    """默认的数据集分割和加载配置
    可通过from_yaml方法读取YAML配置覆盖
    """

    data_dir: str = "./data"
    train_dir: str = "train"
    val_dir: str = "val"
    test_dir: str = "test"
    format: str = "yolo"  # 暂时未支持其他格式
    cache: bool = True
    shuffle: bool = True


@dataclass
class Config:
    """全局配置类, 提供关于项目结构和默认配置的管理, 支持从YAML文件加载"""

    project_root: Path = field(default_factory=lambda: Path(__file__).parent.parent)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    data: DataConfig = field(default_factory=DataConfig)

    # 输出目录
    output_dir: str | None = None
    weights_dir: str | None = None
    logs_dir: str | None = None

    def __post_init__(self):
        if self.output_dir is None:
            self.output_dir = str(self.project_root / "outputs")
        if self.weights_dir is None:
            self.weights_dir = str(self.project_root / "outputs" / "weights")
        if self.logs_dir is None:
            self.logs_dir = str(self.project_root / "outputs" / "logs")

        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.weights_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "Config":
        """用于读取configs目录下的配置文件, 以及用户指定的自定义配置文件"""
        with open(yaml_path, encoding="utf-8") as f:
            config_dict = yaml.safe_load(f)

        return cls.from_dict(config_dict)

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "Config":
        """核心的配置解析方法, 从字典创建Config实例, 处理嵌套结构和必需字段验证

        注意: num_classes 不再从 model 字段读取, 应从 configs/data.yaml 读取
        """

        model_dict = config_dict.get("model", {})

        # 删除模型配置中的 num_classes(如果存在), 避免与 data.yaml 冲突
        model_dict.pop("num_classes", None)

        model_config = ModelConfig(**model_dict)
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


class ConfigManager:
    """
    灵活的配置管理器, 支持多个YAML文件的加载和合并
    用于支持新的数据处理流程配置
    """

    def __init__(self, project_root: Path | None = None):
        """
        初始化配置管理器

        Args:
            project_root: 项目根目录
        """
        self.project_root = project_root or Path(__file__).parent.parent
        self.configs_dir = self.project_root / "configs"
        self.config_dict = {}

    def load_yaml(self, yaml_path: str | Path) -> dict:
        """
        加载单个YAML文件

        Args:
            yaml_path: YAML文件路径

        Returns:
            解析后的配置字典
        """
        filepath = Path(yaml_path)
        if not filepath.is_absolute():
            filepath = self.configs_dir / filepath

        if not filepath.exists():
            raise FileNotFoundError(f"Configuration file not found: {filepath}")

        with open(filepath, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    def load_multiple(self, config_files: Sequence[str | Path]) -> dict:
        """
        加载多个YAML文件并合并

        Args:
            config_files: YAML文件路径列表

        Returns:
            合并后的配置字典
        """
        merged_config = {}

        for config_file in config_files:
            config = self.load_yaml(config_file)
            merged_config.update(config)

        return merged_config

    def load_pipeline_config(self, base_config: str = "pipeline.yaml") -> dict:
        """
        加载完整的流程配置(合并多个文件)

        Args:
            base_config: 基础配置文件(指定要加载的其他模块配置)

        Returns:
            完整的合并配置
        """
        # 加载基础配置
        base = self.load_yaml(base_config)

        # 根据base中的enabled_steps, 加载对应的模块配置
        configs_to_load = [base_config]  # 必须加载基础配置

        pipeline = base.get("pipeline", {})
        enabled_steps = pipeline.get("enabled_steps", [])

        # 映射步骤到配置文件
        step_to_file = {
            "video_extraction": "video_extraction.yaml",
            "image_cleaning": "image_cleaning.yaml",
            "image_normalization": "normalization.yaml",
            "prepare_dataset": "data_preparation.yaml",
        }

        for step in enabled_steps:
            if step in step_to_file:
                config_file = step_to_file[step]
                if (self.configs_dir / config_file).exists():
                    configs_to_load.append(config_file)

        # 加载并合并所有配置
        return self.load_multiple(configs_to_load)

    def get(self, key: str, default: Any = None) -> Any:
        """
        获取配置值(支持点号访问, 如 'train.epochs')

        Args:
            key: 配置键(支持嵌套)
            default: 默认值

        Returns:
            配置值
        """
        keys = key.split(".")
        value = self.config_dict

        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default

        return value

    def __getitem__(self, key: str) -> Any:
        """支持字典索引访问"""
        return self.config_dict.get(key)
