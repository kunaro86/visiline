from pathlib import Path

import torch
from ultralytics import YOLO

from .config import ModelConfig


class Model:
    def __init__(self, config: ModelConfig, device: str = "cuda"):
        self.config = config
        self.device = device
        self.model = None
        self._load_model()

    def _load_model(self) -> None:
        """Load model supporting custom weights or official pre-trained weights.

        When pretrained=True:
          - Look for corresponding .pt file in official_path
          - If not found, let ultralytics auto-download and save to official_path

        When pretrained=False:
          - Look for corresponding .pt file in weight_path
          - Raise FileNotFoundError if not found
        """
        model_filename = f"{self.config.model_name}.pt"

        if self.config.pretrained:
            # Load from official pre-trained weights
            official_dir = Path(self.config.official_path)
            official_path = official_dir / model_filename

            if official_path.exists():
                # Load existing model
                self.model = YOLO(str(official_path))
            else:
                # Create directory and let ultralytics auto-download
                official_dir.mkdir(parents=True, exist_ok=True)
                self.model = YOLO(self.config.model_name)
                # Model will be cached by ultralytics in its default location
        else:
            # Load from custom weights (strict mode)
            weight_dir = Path(self.config.weight_path)
            weight_path = weight_dir / model_filename

            if not weight_path.exists():
                raise FileNotFoundError(
                    f"Model weight not found at {weight_path} (pretrained=False).\n"
                    f"Please ensure the model file exists or set pretrained=True "
                    f"to download from official resource."
                )

            self.model = YOLO(str(weight_path))

        # Move to device if available
        if torch.cuda.is_available() and self.device == "cuda":
            self.model.to("cuda")

    def train(self, data_yaml: str, **kwargs) -> None:
        results = self.model.train(  # type: ignore
            data=data_yaml,
            cache=True,
            **kwargs,
        )
        return results

    def validate(self, data_yaml: str, imgsz: int = 640) -> dict:
        results = self.model.val(data=data_yaml, imgsz=imgsz)  # type: ignore
        return results

    def predict(
        self, source, conf: float = 0.25, iou: float = 0.45, imgsz: int = 640, **kwargs
    ):
        results = self.model.predict(  # type: ignore
            source=source, conf=conf, iou=iou, imgsz=imgsz, **kwargs
        )
        return results

    def export(self, export_format: str = "onnx", **kwargs) -> str:
        path = self.model.export(format=export_format, **kwargs)  # type: ignore
        return path

    def get_model_info(self) -> dict:
        """Get model configuration information."""
        return {
            "model_name": self.config.model_name,
            "weight_path": self.config.weight_path,
            "official_path": self.config.official_path,
            "pretrained": self.config.pretrained,
            "task": self.config.task,
            "device": self.device,
        }
