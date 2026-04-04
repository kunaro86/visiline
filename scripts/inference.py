"""
Inference script for YOLO traffic sign recognition
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import Config
from src.inference import Predictor
from src.utils import setup_logger


def main():
    parser = argparse.ArgumentParser(description="Run inference on traffic sign images")
    parser.add_argument(
        "--model-path", type=str, required=True, help="Path to trained model weights"
    )
    parser.add_argument(
        "--source", type=str, required=True, help="Image, video, or directory path"
    )
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="IOU threshold")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path to save predictions (JSON)",
    )
    parser.add_argument(
        "--visualize", action="store_true", help="Visualize predictions"
    )
    parser.add_argument(
        "--config", type=str, default=None, help="Path to configuration YAML file"
    )

    args = parser.parse_args()

    # Setup logger
    logger = setup_logger(log_file="outputs/logs/inference.log")
    logger.info("=" * 60)
    logger.info("YOLO Traffic Sign Recognition Inference")
    logger.info("=" * 60)

    # Load configuration
    if args.config:
        config = Config.from_yaml(args.config)
    else:
        config = Config()
        config.train.device = args.device

    # Initialize predictor
    logger.info(f"Loading model from {args.model_path}...")
    predictor = Predictor(config, model_path=args.model_path)

    # Run inference
    logger.info(f"Running inference on {args.source}...")
    result = predictor.predict(
        source=args.source, conf=args.conf, iou=args.iou, visualize=args.visualize
    )

    if not result["success"]:
        logger.error(f"Inference failed: {result.get('error')}")
        return 1

    predictions = result["predictions"]

    # Print results
    logger.info("\nDetections found:")
    for i, pred in enumerate(predictions):
        logger.info(f"\nImage {i + 1}: {pred['image_path']}")
        for det in pred["detections"]:
            logger.info(
                f"  - {det['class_name']}: "
                f"confidence={det['confidence']:.3f}, "
                f"bbox={det['bbox']}"
            )

    # Save results if specified
    if args.output:
        predictor.save_predictions(predictions, args.output)
        logger.info(f"Predictions saved to {args.output}")

    logger.info("Inference completed successfully!")
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
