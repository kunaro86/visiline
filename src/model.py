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
        model_name = f"{self.config.model_name}.pt"
        self.model = YOLO(model_name)

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
        return {
            "model_name": self.config.model_name,
            "num_classes": self.config.num_classes,
            "task": self.config.task,
            "device": self.device,
        }
