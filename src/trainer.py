import logging
import os
from dataclasses import asdict

from .config import Config
from .model import Model
from .utils import print_config, set_seed

logger = logging.getLogger(__name__)


class Trainer:
    def __init__(self, config: Config):
        self.config = config
        self.model = None
        self.results = None
        self._setup()

    def _setup(self) -> None:
        """设置随机种子, 初始化模型"""
        set_seed(self.config.train.seed)
        self.model = Model(self.config.model, self.config.train.device)
        logger.info("Trainer initialized")

    def train(self, data_yaml: str, output_dir: str | None = None) -> dict:

        if self.model is None:
            raise RuntimeError("Model not initialized")

        if output_dir is None:
            output_dir = self.config.output_dir

        os.makedirs(output_dir, exist_ok=True)  # type: ignore

        logger.info("=" * 60)
        logger.info("STARTING MODEL TRAINING")
        logger.info("=" * 60)
        print_config(self.config)

        try:
            # collect core training args into a single dict for clarity
            train_kwargs = {
                "data_yaml": data_yaml,
                "project": output_dir,
                "name": "traffic_sign_model",
                "exist_ok": True,
                "save": True,
                "device": "0" if self.config.train.device == "cuda" else "cpu",
            }

            # 从配置类中提取训练相关的参数, 但排除掉不适用于YOLO训练的参数
            train_dict = asdict(self.config.train)
            train_dict.pop("val_split", None)
            train_dict.pop("test_split", None)

            train_kwargs.update(train_dict)

            results = self.model.train(**train_kwargs)

            self.results = results
            logger.info("=" * 60)
            logger.info("TRAINING COMPLETED SUCCESSFULLY")
            logger.info("=" * 60)

            return {"success": True, "results": results}

        except Exception as e:
            logger.error(f"Training failed: {e!s}", exc_info=True)
            return {"success": False, "error": str(e)}

    def validate(self, data_yaml: str) -> dict:

        if self.model is None:
            raise RuntimeError("Model not initialized")

        logger.info("Validating model...")

        try:
            results = self.model.validate(data_yaml=data_yaml)
            logger.info("Validation completed")
            return {"success": True, "results": results}
        except Exception as e:
            logger.error(f"Validation failed: {e!s}", exc_info=True)
            return {"success": False, "error": str(e)}

    def export_model(
        self, export_format: str = "onnx", output_path: str | None = None
    ) -> dict:
        """
        Returns:
            结果字典, 包含成功标志和导出路径或错误信息
        """
        if self.model is None:
            raise RuntimeError("Model not initialized")

        logger.info(f"Exporting model to {export_format}...")

        try:
            export_path = self.model.export(export_format=export_format)
            logger.info(f"Model exported successfully: {export_path}")
            return {"success": True, "export_path": str(export_path)}
        except Exception as e:
            logger.error(f"Export failed: {e!s}", exc_info=True)
            return {"success": False, "error": str(e)}
