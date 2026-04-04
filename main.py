"""
YOLO Traffic Sign Recognition - Main entry point
Unified CLI for training, inference, and data preparation
"""

import argparse
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.config import Config
from src.data import DatasetManager
from src.inference import Predictor
from src.trainer import Trainer
from src.utils import print_config, setup_logger


def setup_parser():
    """Setup command line argument parser"""
    parser = argparse.ArgumentParser(
        prog="yolo-tmr",
        description="YOLO Traffic Mark Recognition - Training and Deployment Acceleration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Prepare dataset
  python main.py prepare-data --data-dir data --split --images-dir raw/images --labels-dir raw/labels

  # Train model
  python main.py train --config configs/default.yaml --epochs 100

  # Run inference
  python main.py predict --model-path outputs/weights/best.pt --source test.jpg

  # Quick validation
  python main.py train --config configs/quick.yaml
        """,
    )

    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # ==================== Train Command ====================
    train_parser = subparsers.add_parser("train", help="Train YOLO model")
    train_parser.add_argument(
        "--config",
        "-c",
        type=str,
        default="configs/default.yaml",
        help="Path to configuration file (default: configs/default.yaml)",
    )
    train_parser.add_argument(
        "--data-yaml",
        type=str,
        default="configs/data.yaml",
        help="Path to data.yaml file",
    )
    train_parser.add_argument(
        "--epochs", type=int, default=None, help="Override number of epochs in config"
    )
    train_parser.add_argument(
        "--batch-size", type=int, default=None, help="Override batch size in config"
    )
    train_parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "cpu"],
        default=None,
        help="Override device in config",
    )
    train_parser.add_argument(
        "--model",
        type=str,
        choices=["yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x"],
        default=None,
        help="Override model in config",
    )

    # ==================== Predict Command ====================
    predict_parser = subparsers.add_parser("predict", help="Run inference")
    predict_parser.add_argument(
        "--model-path",
        "-m",
        type=str,
        required=True,
        help="Path to trained model weights",
    )
    predict_parser.add_argument(
        "--source", "-s", type=str, required=True, help="Image, video or directory path"
    )
    predict_parser.add_argument(
        "--conf", type=float, default=0.25, help="Confidence threshold"
    )
    predict_parser.add_argument("--iou", type=float, default=0.45, help="IOU threshold")
    predict_parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output path for predictions JSON",
    )
    predict_parser.add_argument(
        "--visualize", action="store_true", help="Visualize predictions"
    )

    # ==================== Prepare Data Command ====================
    data_parser = subparsers.add_parser("prepare-data", help="Prepare dataset")
    data_parser.add_argument(
        "--data-dir", type=str, default="data", help="Root data directory"
    )
    data_parser.add_argument(
        "--num-classes", type=int, default=43, help="Number of classes"
    )
    data_parser.add_argument(
        "--split", action="store_true", help="Split dataset into train/val/test"
    )
    data_parser.add_argument(
        "--images-dir",
        type=str,
        default=None,
        help="Source images directory (for --split)",
    )
    data_parser.add_argument(
        "--labels-dir",
        type=str,
        default=None,
        help="Source labels directory (for --split)",
    )
    data_parser.add_argument(
        "--train-ratio", type=float, default=0.8, help="Training set ratio"
    )
    data_parser.add_argument(
        "--val-ratio", type=float, default=0.1, help="Validation set ratio"
    )

    # ==================== Validate Command ====================
    validate_parser = subparsers.add_parser("validate", help="Validate model")
    validate_parser.add_argument(
        "--model-path", "-m", type=str, required=True, help="Path to model weights"
    )
    validate_parser.add_argument(
        "--data-yaml",
        type=str,
        default="configs/data.yaml",
        help="Path to data.yaml file",
    )

    # ==================== Export Command ====================
    export_parser = subparsers.add_parser("export", help="Export model")
    export_parser.add_argument(
        "--model-path", "-m", type=str, required=True, help="Path to model weights"
    )
    export_parser.add_argument(
        "--format",
        "-f",
        type=str,
        choices=["onnx", "torchscript", "tflite", "pb", "saved_model"],
        default="onnx",
        help="Export format",
    )

    # ==================== Info Command ====================
    info_parser = subparsers.add_parser("info", help="Show configuration")
    info_parser.add_argument(
        "--config",
        "-c",
        type=str,
        default="configs/default.yaml",
        help="Path to configuration file",
    )

    return parser


def cmd_train(args, logger):
    """Train command handler"""
    # Load configuration
    if os.path.exists(args.config):
        config = Config.from_yaml(args.config)
        logger.info(f"Configuration loaded from {args.config}")
    else:
        config = Config()
        logger.info("Using default configuration")

    # Override config with command line arguments
    if args.epochs:
        config.train.epochs = args.epochs
    if args.batch_size:
        config.train.batch_size = args.batch_size
    if args.device:
        config.train.device = args.device
    if args.model:
        config.model.model_name = args.model

    print_config(config)

    # Verify dataset
    logger.info("\nVerifying dataset...")
    dataset_manager = DatasetManager(
        data_dir=str(project_root / "data"), num_classes=config.model.num_classes
    )
    train_count, val_count, _test_count = dataset_manager.verify_dataset()

    if train_count == 0 or val_count == 0:
        logger.error("Dataset not properly set up!")
        logger.error("Please run: python main.py prepare-data --help")
        return 1

    # Train
    trainer = Trainer(config)
    result = trainer.train(data_yaml=args.data_yaml, output_dir=config.output_dir)

    return 0 if result["success"] else 1


def cmd_predict(args, logger):
    """Predict command handler"""
    config = Config()

    logger.info(f"Loading model from {args.model_path}...")
    predictor = Predictor(config, model_path=args.model_path)

    logger.info(f"Running inference on {args.source}...")
    result = predictor.predict(
        source=args.source, conf=args.conf, iou=args.iou, visualize=args.visualize
    )

    if not result["success"]:
        logger.error(f"Inference failed: {result.get('error')}")
        return 1

    predictions = result["predictions"]
    logger.info(f"\nFound {sum(len(p['detections']) for p in predictions)} detections")

    # Save results if specified
    if args.output:
        predictor.save_predictions(predictions, args.output)
        logger.info(f"Predictions saved to {args.output}")

    return 0


def cmd_prepare_data(args, logger):
    """Prepare data command handler"""
    dataset_manager = DatasetManager(
        data_dir=args.data_dir, num_classes=args.num_classes
    )

    logger.info("Creating YOLO dataset structure...")
    dataset_manager.create_yolo_structure()

    if args.split:
        if not args.images_dir or not args.labels_dir:
            logger.error("--images-dir and --labels-dir are required for --split")
            return 1

        logger.info("Splitting dataset...")
        dataset_manager.split_dataset(
            images_dir=args.images_dir,
            labels_dir=args.labels_dir,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=1.0 - args.train_ratio - args.val_ratio,
        )

    logger.info("Verifying dataset...")
    train_count, val_count, test_count = dataset_manager.verify_dataset()

    logger.info("\nDataset prepared successfully!")
    logger.info(f"  Training: {train_count}")
    logger.info(f"  Validation: {val_count}")
    logger.info(f"  Test: {test_count}")

    return 0


def cmd_validate(args, logger):
    """Validate command handler"""
    config = Config()
    trainer = Trainer(config)

    logger.info(f"Validating model {args.model_path}...")
    result = trainer.validate(data_yaml=args.data_yaml)

    if not result["success"]:
        logger.error(f"Validation failed: {result.get('error')}")
        return 1

    logger.info("Validation completed!")
    return 0


def cmd_export(args, logger):
    """Export command handler"""
    config = Config()
    trainer = Trainer(config)
    trainer.model.model.load(args.model_path)  # type: ignore

    logger.info(f"Exporting model to {args.format}...")
    result = trainer.export_model(
        export_format=args.format, output_path=f"outputs/exported_model.{args.format}"
    )

    if not result["success"]:
        logger.error(f"Export failed: {result.get('error')}")
        return 1

    logger.info(f"Model exported to {result['export_path']}")
    return 0


def cmd_info(args, logger):
    """Info command handler"""
    config = Config.from_yaml(args.config) if os.path.exists(args.config) else Config()

    print_config(config)
    return 0


def main():
    """Main entry point"""
    # Setup parser
    parser = setup_parser()
    args = parser.parse_args()

    # Create output directories
    os.makedirs("outputs/logs", exist_ok=True)

    # Setup logger
    logger = setup_logger(name="yolo-tmr", log_file="outputs/logs/yolo-tmr.log")

    logger.info("=" * 60)
    logger.info("YOLO Traffic Sign Recognition")
    logger.info("=" * 60)

    # Handle commands
    if args.command == "train":
        return cmd_train(args, logger)
    elif args.command == "predict":
        return cmd_predict(args, logger)
    elif args.command == "prepare-data":
        return cmd_prepare_data(args, logger)
    elif args.command == "validate":
        return cmd_validate(args, logger)
    elif args.command == "export":
        return cmd_export(args, logger)
    elif args.command == "info":
        return cmd_info(args, logger)
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
