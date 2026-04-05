"""
Training script for YOLO traffic sign recognition
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import Config
from src.data import DatasetManager
from src.trainer import Trainer
from src.utils import print_config, setup_logger


def main():
    parser = argparse.ArgumentParser(
        description="Train YOLO model for traffic sign recognition"
    )
    parser.add_argument(
        "--config", type=str, default=None, help="Path to configuration YAML file"
    )
    parser.add_argument(
        "--data-yaml", type=str, default="data/data.yaml", help="Path to data.yaml file"
    )
    parser.add_argument(
        "--epochs", type=int, default=100, help="Number of training epochs"
    )
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use (cuda or cpu)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n",
        choices=["yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x"],
        help="YOLOv8 model size",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--output-dir", type=str, default="outputs", help="Output directory"
    )

    args = parser.parse_args()

    # Setup logger
    logger = setup_logger(log_file="outputs/logs/train.log")
    logger.info("=" * 60)
    logger.info("YOLO Traffic Sign Recognition Training")
    logger.info("=" * 60)

    # Load or create configuration
    if args.config:
        config = Config.from_yaml(args.config)
        logger.info(f"Configuration loaded from {args.config}")
    else:
        config = Config()
        config.train.epochs = args.epochs
        config.train.batch_size = args.batch_size
        config.train.device = args.device
        config.train.seed = args.seed
        config.model.model_name = args.model
        config.output_dir = args.output_dir
        logger.info("Using default configuration")

    print_config(config)

    # Verify data
    logger.info("\nVerifying dataset...")
    dataset_manager = DatasetManager(
        data_dir=str(project_root / "dataset"),
        num_classes=config.model.num_classes,  # type: ignore
    )
    train_count, val_count, _test_count = dataset_manager.verify_dataset()

    if train_count == 0 or val_count == 0:
        logger.error("Dataset not properly set up. Please prepare the dataset first.")
        logger.error("Please create data structure with:")
        logger.error("  - data/train/images/ and data/train/labels/")
        logger.error("  - data/val/images/ and data/val/labels/")
        return 1

    # Initialize trainer
    trainer = Trainer(config)

    # Train model
    logger.info("\nStarting model training...")
    result = trainer.train(data_yaml=args.data_yaml, output_dir=args.output_dir)

    if result["success"]:
        logger.info("Training completed successfully!")

        # Validation
        logger.info("\nValidating model...")
        val_result = trainer.validate(data_yaml=args.data_yaml)

        if val_result["success"]:
            logger.info("Validation completed!")
        else:
            logger.warning(f"Validation failed: {val_result.get('error')}")

        return 0
    else:
        logger.error(f"Training failed: {result.get('error')}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
