"""
Training module for YOLO-TMR project
"""

import logging
import os

from .config import Config
from .model import TrafficSignModel
from .utils import print_config, set_seed

logger = logging.getLogger(__name__)


class Trainer:
    """Model trainer"""

    def __init__(self, config: Config):
        """
        Initialize trainer

        Args:
            config: Config instance
        """
        self.config = config
        self.model = None
        self.results = None
        self._setup()

    def _setup(self) -> None:
        """Setup training environment"""
        # Set random seed
        set_seed(self.config.train.seed)

        # Initialize model
        self.model = TrafficSignModel(self.config.model, self.config.train.device)

        logger.info("Trainer initialized")

    def train(self, data_yaml: str, output_dir: str | None = None) -> dict:
        """
        Train the model

        Args:
            data_yaml: Path to data.yaml file
            output_dir: Output directory for results

        Returns:
            Training results dictionary
        """
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
            # Train model
            results = self.model.train(
                data_yaml=data_yaml,
                epochs=self.config.train.epochs,
                batch_size=self.config.train.batch_size,
                imgsz=self.config.train.imgsz,
                patience=self.config.train.patience,
                seed=self.config.train.seed,
                device="0" if self.config.train.device == "cuda" else "cpu",
                project=output_dir,
                name="traffic_sign_model",
                exist_ok=True,
                save=True,
                save_period=self.config.train.save_period,
                optimizer=self.config.train.optimizer,
                momentum=self.config.train.momentum,
                weight_decay=self.config.train.weight_decay,
                lr0=self.config.train.lr0,
                lrf=self.config.train.lrf,
                warmup_epochs=self.config.train.warmup_epochs,
                warmup_momentum=self.config.train.warmup_momentum,
                augment=self.config.train.augment,
                mosaic=self.config.train.mosaic,
                mixup=self.config.train.mixup,
                hsv_h=self.config.train.hsv_h,
                hsv_s=self.config.train.hsv_s,
                hsv_v=self.config.train.hsv_v,
                degrees=self.config.train.degrees,
                translate=self.config.train.translate,
                scale=self.config.train.scale,
                flipud=self.config.train.flipud,
                fliplr=self.config.train.fliplr,
            )

            self.results = results
            logger.info("=" * 60)
            logger.info("TRAINING COMPLETED SUCCESSFULLY")
            logger.info("=" * 60)

            return {"success": True, "results": results}

        except Exception as e:
            logger.error(f"Training failed: {e!s}", exc_info=True)
            return {"success": False, "error": str(e)}

    def validate(self, data_yaml: str) -> dict:
        """
        Validate the model

        Args:
            data_yaml: Path to data.yaml file

        Returns:
            Validation results dictionary
        """
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
        Export trained model

        Args:
            export_format: Export format (onnx, torchscript, tflite, etc.)
            output_path: Output path for exported model

        Returns:
            Export result dictionary
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
