"""
Data preparation script for YOLO traffic sign recognition
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data import DatasetManager
from src.utils import setup_logger


def main():
    parser = argparse.ArgumentParser(description="Prepare dataset for training")
    parser.add_argument(
        "--data-dir", type=str, default="data", help="Root data directory"
    )
    parser.add_argument("--num-classes", type=int, default=43, help="Number of classes")
    parser.add_argument(
        "--split", action="store_true", help="Split dataset into train/val/test"
    )
    parser.add_argument(
        "--images-dir",
        type=str,
        default=None,
        help="Directory containing images (required if --split is used)",
    )
    parser.add_argument(
        "--labels-dir",
        type=str,
        default=None,
        help="Directory containing labels (required if --split is used)",
    )
    parser.add_argument(
        "--train-ratio", type=float, default=0.8, help="Training set ratio"
    )
    parser.add_argument(
        "--val-ratio", type=float, default=0.1, help="Validation set ratio"
    )

    args = parser.parse_args()

    # Setup logger
    logger = setup_logger(log_file="outputs/logs/data_prep.log")
    logger.info("=" * 60)
    logger.info("Data Preparation for YOLO Traffic Sign Recognition")
    logger.info("=" * 60)

    # Initialize dataset manager
    dataset_manager = DatasetManager(
        data_dir=args.data_dir, num_classes=args.num_classes
    )

    # Create YOLO dataset structure
    logger.info("Creating YOLO dataset structure...")
    data_yaml = dataset_manager.create_yolo_structure()

    # Split dataset if requested
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

    # Verify dataset
    logger.info("Verifying dataset...")
    train_count, val_count, test_count = dataset_manager.verify_dataset()

    logger.info("\nDataset preparation completed!")
    logger.info(f"  Training samples: {train_count}")
    logger.info(f"  Validation samples: {val_count}")
    logger.info(f"  Test samples: {test_count}")
    logger.info(f"\ndata.yaml location: {data_yaml}")

    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
