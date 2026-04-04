"""
Inference module for YOLO-TMR project
"""

import json
import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .config import Config
from .model import TrafficSignModel
from .utils import get_traffic_sign_classes

logger = logging.getLogger(__name__)


class Predictor:
    """Traffic sign predictor"""

    def __init__(self, config: Config, model_path: str | None = None):
        """
        Initialize predictor

        Args:
            config: Config instance
            model_path: Path to trained model weights (optional)
        """
        self.config = config
        self.model = TrafficSignModel(config.model, config.train.device)

        # Load traffic sign classes from configs/data.yaml
        # Try to find data.yaml in configs directory
        data_yaml_path = Path(config.project_root) / "configs" / "data.yaml"
        self.classes = get_traffic_sign_classes(
            str(data_yaml_path) if data_yaml_path.exists() else None
        )

        if model_path:
            self.load_model(model_path)

        logger.info("Predictor initialized")

    def load_model(self, model_path: str) -> None:
        """Load trained model weights"""
        try:
            self.model.model.load(model_path)  # type: ignore
            logger.info(f"Model loaded from {model_path}")
        except Exception as e:
            logger.error(f"Failed to load model: {e!s}")
            raise

    def predict(
        self,
        source: str | Path | np.ndarray,
        conf: float = 0.25,
        iou: float = 0.45,
        visualize: bool = False,
    ) -> dict[str, Any]:
        """
        Run inference on image(s)

        Args:
            source: Image path, directory, or numpy array
            conf: Confidence threshold
            iou: IOU threshold
            visualize: Whether to visualize results

        Returns:
            Predictions dictionary
        """
        try:
            results = self.model.predict(
                source=source,
                conf=conf,
                iou=iou,
                imgsz=self.config.train.imgsz,
                verbose=False,
            )

            predictions = self._parse_results(results)

            if visualize and isinstance(source, (str, Path)):
                self._visualize_predictions(source, predictions)

            return {"success": True, "predictions": predictions}

        except Exception as e:
            logger.error(f"Prediction failed: {e!s}", exc_info=True)
            return {"success": False, "error": str(e)}

    def predict_image(
        self, image_path: str, conf: float = 0.25, iou: float = 0.45
    ) -> dict[str, Any]:
        """
        Predict single image

        Args:
            image_path: Path to image
            conf: Confidence threshold
            iou: IOU threshold

        Returns:
            Prediction results
        """
        return self.predict(image_path, conf=conf, iou=iou, visualize=False)

    def predict_batch(
        self, image_dir: str, conf: float = 0.25, iou: float = 0.45
    ) -> dict[str, Any]:
        """
        Predict batch of images

        Args:
            image_dir: Directory containing images
            conf: Confidence threshold
            iou: IOU threshold

        Returns:
            Batch prediction results
        """
        return self.predict(image_dir, conf=conf, iou=iou, visualize=False)

    def predict_video(
        self,
        video_path: str,
        conf: float = 0.25,
        iou: float = 0.45,
        output_path: str | None = None,
    ) -> dict[str, Any]:
        """
        Predict video

        Args:
            video_path: Path to video
            conf: Confidence threshold
            iou: IOU threshold
            output_path: Path to save output video

        Returns:
            Video prediction results
        """
        try:
            results = self.model.predict(
                source=video_path,
                conf=conf,
                iou=iou,
                imgsz=self.config.train.imgsz,
                save=output_path is not None,
                project=str(Path(output_path).parent) if output_path else None,
                name="predictions",
                exist_ok=True,
            )

            logger.info(f"Video prediction completed: {video_path}")
            return {"success": True, "results": results}

        except Exception as e:
            logger.error(f"Video prediction failed: {e!s}", exc_info=True)
            return {"success": False, "error": str(e)}

    def _parse_results(self, results) -> list[dict]:
        """Parse YOLO results to dictionary format"""
        predictions = []

        for result in results:
            if result.boxes is None or len(result.boxes) == 0:
                predictions.append(
                    {
                        "image_path": str(result.path)
                        if hasattr(result, "path")
                        else None,
                        "detections": [],
                    }
                )
                continue

            detections = []
            for box in result.boxes:
                detection = {
                    "class_id": int(box.cls[0].item()),
                    "class_name": self.classes.get(int(box.cls[0].item()), "Unknown"),
                    "confidence": float(box.conf[0].item()),
                    "bbox": {
                        "x1": float(box.xyxy[0][0].item()),
                        "y1": float(box.xyxy[0][1].item()),
                        "x2": float(box.xyxy[0][2].item()),
                        "y2": float(box.xyxy[0][3].item()),
                    },
                }
                detections.append(detection)

            predictions.append(
                {
                    "image_path": str(result.path) if hasattr(result, "path") else None,
                    "detections": detections,
                }
            )

        return predictions

    def _visualize_predictions(
        self, source: str | Path, predictions: list[dict]
    ) -> None:
        """Visualize predictions"""
        source_path = Path(source)

        if source_path.is_file():
            image_files = [source_path]
        else:
            image_files = list(source_path.glob("*.[jJ][pP][gG]")) + list(
                source_path.glob("*.[pP][nN][gG]")
            )

        for i, img_file in enumerate(image_files):
            if i >= len(predictions):
                break

            img = cv2.imread(str(img_file))
            if img is None:
                continue

            for det in predictions[i]["detections"]:
                bbox = det["bbox"]
                x1, y1 = int(bbox["x1"]), int(bbox["y1"])
                x2, y2 = int(bbox["x2"]), int(bbox["y2"])

                # Draw bounding box
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Draw label
                label = f"{det['class_name']} ({det['confidence']:.2f})"
                cv2.putText(
                    img,
                    label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )

            # Show image
            cv2.imshow("Predictions", img)
            if cv2.waitKey(0) == 27:  # ESC key
                break

        cv2.destroyAllWindows()

    def save_predictions(self, predictions: list[dict], output_path: str) -> None:
        """Save predictions to JSON file"""
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(predictions, f, indent=2, ensure_ascii=False)

        logger.info(f"Predictions saved to {output_path}")
