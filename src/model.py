"""
Model wrapper for YOLO-TMR project
"""

import torch
from ultralytics import YOLO

from .config import ModelConfig


class TrafficSignModel:
    """Traffic sign detection model wrapper"""

    def __init__(self, config: ModelConfig, device: str = "cuda"):
        """
        Initialize traffic sign model

        Args:
            config: ModelConfig instance
            device: Device to use (cuda or cpu)
        """
        self.config = config
        self.device = device
        self.model = None
        self._load_model()

    def _load_model(self) -> None:
        """Load YOLOv8 model"""
        # Load pretrained model
        model_name = f"{self.config.model_name}.pt"
        self.model = YOLO(model_name)

        # Move to device
        if torch.cuda.is_available() and self.device == "cuda":
            self.model.to("cuda")

    def train(
        self,
        data_yaml: str,
        epochs: int,
        batch_size: int,
        imgsz: int = 640,
        patience: int = 20,
        seed: int = 42,
        device: str = "0",
        **kwargs,
    ) -> None:
        """
        Train the model

        Args:
            data_yaml: Path to data.yaml file
            epochs: Number of epochs
            batch_size: Batch size
            imgsz: Image size
            patience: Early stopping patience
            seed: Random seed
            device: Device ID (0 for first GPU)
            **kwargs: Additional arguments to pass to YOLO train
        """
        results = self.model.train(  # type: ignore
            data=data_yaml,
            epochs=epochs,
            batch=batch_size,
            imgsz=imgsz,
            patience=patience,
            seed=seed,
            device=device,
            save=True,
            cache=True,
            augment=True,
            **kwargs,
        )
        return results

    def validate(self, data_yaml: str, imgsz: int = 640) -> dict:
        """
        Validate the model

        Args:
            data_yaml: Path to data.yaml file
            imgsz: Image size

        Returns:
            Validation results
        """
        results = self.model.val(data=data_yaml, imgsz=imgsz)  # type: ignore
        return results

    def predict(
        self, source, conf: float = 0.25, iou: float = 0.45, imgsz: int = 640, **kwargs
    ):
        """
        Run inference

        Args:
            source: Image, video, or directory path
            conf: Confidence threshold
            iou: IOU threshold
            imgsz: Image size
            **kwargs: Additional arguments

        Returns:
            Prediction results
        """
        results = self.model.predict(  # type: ignore
            source=source, conf=conf, iou=iou, imgsz=imgsz, **kwargs
        )
        return results

    def export(self, export_format: str = "onnx", **kwargs) -> str:
        """
        Export model to different formats

        Args:
            export_format: Export format (onnx, torchscript, tflite, etc.)
            **kwargs: Additional export arguments

        Returns:
            Path to exported model
        """
        path = self.model.export(format=export_format, **kwargs)  # type: ignore
        return path

    def get_model_info(self) -> dict:
        """Get model information"""
        return {
            "model_name": self.config.model_name,
            "num_classes": self.config.num_classes,
            "task": self.config.task,
            "device": self.device,
        }
