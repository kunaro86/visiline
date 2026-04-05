"""
Data Pre-Processing Tool for YOLO-TMR
处理视频、清洗图像、标准化名称、分割数据集的独立工具
"""

import argparse
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.config import ConfigManager
from src.data import DatasetManager
from src.image_cleaner import ImageCleaner
from src.image_normalizer import ImageNormalizer
from src.pipeline import PipelineManager
from src.utils import setup_logger
from src.video_processor import VideoProcessor


def setup_parser():
    """Setup argument parser for preprocessing tool"""
    parser = argparse.ArgumentParser(
        prog="preprocess",
        description="Data pre-processing tool: video extraction, image cleaning, normalization, dataset splitting",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Complete pipeline
  %(prog)s pipeline --config configs/pipeline.yaml

  # Extract video frames
  %(prog)s extract-video --config configs/pre-processing.yaml

  # Clean images
  %(prog)s clean-images --config configs/pre-processing.yaml

  # Normalize image names
  %(prog)s normalize-images --config configs/pre-processing.yaml

  # Split dataset
  %(prog)s split-dataset --images-dir raw/images --data-dir data --train-ratio 0.8

  # Or use config file
  %(prog)s split-dataset --config configs/pre-processing.yaml
        """,
    )

    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )

    subparsers = parser.add_subparsers(dest="command", help="Pre-processing commands")

    # ==================== Pipeline Command ====================
    pipeline_parser = subparsers.add_parser(
        "pipeline", help="Run complete pre-processing pipeline"
    )
    pipeline_parser.add_argument(
        "--config",
        "-c",
        type=str,
        default="configs/pipeline.yaml",
        help="Pipeline configuration file",
    )

    # ==================== Extract Video Command ====================
    extract_parser = subparsers.add_parser(
        "extract-video", help="Extract keyframes from videos"
    )
    extract_parser.add_argument(
        "--config",
        "-c",
        type=str,
        default="configs/pre-processing.yaml",
        help="Configuration file",
    )
    extract_parser.add_argument(
        "--video-dir",
        type=str,
        default=None,
        help="Video directory (overrides config)",
    )
    extract_parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for extracted frames",
    )

    # ==================== Clean Images Command ====================
    clean_parser = subparsers.add_parser(
        "clean-images", help="Clean and validate images"
    )
    clean_parser.add_argument(
        "--config",
        "-c",
        type=str,
        default="configs/pre-processing.yaml",
        help="Configuration file",
    )
    clean_parser.add_argument(
        "--image-dir",
        type=str,
        default=None,
        help="Image directory to clean (overrides config)",
    )

    # ==================== Normalize Images Command ====================
    normalize_parser = subparsers.add_parser(
        "normalize-images", help="Normalize image names"
    )
    normalize_parser.add_argument(
        "--config",
        "-c",
        type=str,
        default="configs/pre-processing.yaml",
        help="Configuration file",
    )
    normalize_parser.add_argument(
        "--image-dir",
        type=str,
        default=None,
        help="Image directory to normalize (overrides config)",
    )

    # ==================== Split Dataset Command ====================
    split_parser = subparsers.add_parser(
        "split-dataset", help="Split dataset into train/val/test"
    )
    split_parser.add_argument(
        "--config",
        "-c",
        type=str,
        default="configs/pre-processing.yaml",
        help="Split configuration file",
    )
    split_parser.add_argument(
        "--images-dir",
        type=str,
        default=None,
        help="Source images directory (overrides config)",
    )
    split_parser.add_argument(
        "--labels-dir",
        type=str,
        default=None,
        help="Source labels directory (overrides config)",
    )
    split_parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Output data directory (overrides config)",
    )
    split_parser.add_argument(
        "--train-ratio",
        type=float,
        default=None,
        help="Train ratio (overrides config)",
    )
    split_parser.add_argument(
        "--val-ratio",
        type=float,
        default=None,
        help="Val ratio (overrides config)",
    )

    return parser


def cmd_pipeline(args, logger):
    """Pipeline command handler - run complete preprocessing"""
    logger.info("Loading pipeline configuration...")
    config_manager = ConfigManager(project_root)
    config = config_manager.load_pipeline_config(args.config)

    logger.info("Initializing pipeline manager...")
    pipeline = PipelineManager(config)

    logger.info("Starting pre-processing pipeline...")
    results = pipeline.run_pipeline()

    return 0 if not any("error" in r for r in results.values()) else 1


def cmd_extract_video(args, logger):
    """Extract video command handler"""
    logger.info("Loading configuration...")
    config_manager = ConfigManager(project_root)
    config = config_manager.load_yaml(args.config)

    # Override with command line arguments
    if args.video_dir:
        config["video_extraction"]["video_dir"] = args.video_dir
    if args.output_dir:
        config["video_extraction"]["image_output_dir"] = args.output_dir

    logger.info("Extracting keyframes from videos...")
    processor = VideoProcessor(config.get("video_extraction", {}))
    processor.extract_all_keyframes()

    return 0


def cmd_clean_images(args, logger):
    """Clean images command handler"""
    logger.info("Loading configuration...")
    config_manager = ConfigManager(project_root)
    config = config_manager.load_yaml(args.config)

    # Override with command line arguments
    if args.image_dir:
        config["image_cleaning"]["image_dir"] = args.image_dir

    logger.info("Cleaning images...")
    cleaner = ImageCleaner(config.get("image_cleaning", {}))
    cleaner.clean_all()

    return 0


def cmd_normalize_images(args, logger):
    """Normalize images command handler"""
    logger.info("Loading configuration...")
    config_manager = ConfigManager(project_root)
    config = config_manager.load_yaml(args.config)

    # Override with command line arguments
    if args.image_dir:
        config["normalization"]["image_dir"] = args.image_dir

    logger.info("Normalizing image names...")
    normalizer = ImageNormalizer(config.get("normalization", {}))
    normalizer.normalize_names()

    return 0


def cmd_split_dataset(args, logger):
    """Split dataset command handler"""
    logger.info("Loading configuration...")
    config_manager = ConfigManager(project_root)
    config = config_manager.load_yaml(args.config)

    split_config = config.get("dataset_split", {})
    if not split_config:
        logger.error("[dataset_split] section not found in configuration file")
        return 1

    # Override with command line arguments
    if args.images_dir:
        split_config["raw_images_dir"] = args.images_dir
    if args.labels_dir:
        split_config["raw_labels_dir"] = args.labels_dir
    if args.data_dir:
        split_config["data_dir"] = args.data_dir
    if args.train_ratio:
        split_config["train_ratio"] = args.train_ratio
    if args.val_ratio:
        split_config["val_ratio"] = args.val_ratio

    # Get data directory and resolve num_classes from data.yaml
    data_dir = split_config.get("data_dir")
    if not data_dir:
        logger.error("[dataset_split.data_dir] not specified in config or CLI")
        return 1

    # Read num_classes from existing data.yaml or fail
    data_yaml_path = os.path.join(data_dir, "data.yaml")
    if not os.path.exists(data_yaml_path):
        logger.error(f"data.yaml not found at {data_yaml_path}")
        logger.error("Please run: python preprocess.py split-dataset after data preparation")
        logger.error("or ensure data directory structure is correct")
        return 1

    data_yaml_config = config_manager.load_yaml(data_yaml_path)
    num_classes = data_yaml_config.get("nc")
    if num_classes is None:
        logger.error(f"[nc] (number of classes) not found in {data_yaml_path}")
        return 1

    logger.info(f"Using {num_classes} classes from {data_yaml_path}")
    dataset_manager = DatasetManager(data_dir=data_dir, num_classes=num_classes)

    logger.info("Creating YOLO dataset structure...")
    force = True
    dataset_manager.create_yolo_structure(force=force)

    # Get split parameters (require explicit configuration, no defaults)
    images_dir = split_config.get("raw_images_dir")
    labels_dir = split_config.get("raw_labels_dir")
    train_ratio = split_config.get("train_ratio")
    val_ratio = split_config.get("val_ratio")

    # Validate required parameters
    if not images_dir:
        logger.error("[dataset_split.raw_images_dir] not specified in config or CLI")
        return 1
    if not train_ratio or not val_ratio:
        logger.error("[dataset_split.train_ratio] and [dataset_split.val_ratio] are required")
        return 1

    # Calculate test_ratio and validate
    test_ratio = 1.0 - train_ratio - val_ratio
    if test_ratio < 0 or train_ratio < 0 or val_ratio < 0:
        logger.error(f"Invalid split ratios: train={train_ratio}, val={val_ratio}, test={test_ratio}")
        logger.error("Ensure: train_ratio + val_ratio <= 1.0 and all >= 0")
        return 1
    if test_ratio < 0.01:
        logger.warning(f"Test set ratio is very small ({test_ratio:.1%}), consider adjusting")

    # Check if images directory exists
    if not os.path.exists(images_dir):
        logger.error(f"Images directory not found: {images_dir}")
        logger.error(
            "Please prepare images first using: python preprocess.py extract-video"
        )
        return 1

    logger.info(f"Splitting dataset from {images_dir}...")
    dataset_manager.split_dataset(
        images_dir=images_dir,
        labels_dir=labels_dir if os.path.exists(labels_dir) else None,  # type: ignore
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
    )

    logger.info("Verifying dataset...")
    train_count, val_count, test_count = dataset_manager.verify_dataset()

    logger.info("\n" + "=" * 60)
    logger.info("✅ Dataset split successfully!")
    logger.info("=" * 60)
    logger.info(f"  Training samples:   {train_count}")
    logger.info(f"  Validation samples: {val_count}")
    logger.info(f"  Test samples:       {test_count}")
    logger.info(f"  Output directory:   {data_dir}")
    logger.info("\n  Next step: python main.py train --config configs/default.yaml")
    logger.info("=" * 60)

    return 0


def main():
    """Main entry point"""
    parser = setup_parser()
    args = parser.parse_args()

    # Create output directories
    os.makedirs("outputs/logs", exist_ok=True)

    # Setup logger
    logger = setup_logger(name="preprocess", log_file="outputs/logs/preprocess.log")

    logger.info("=" * 60)
    logger.info("YOLO-TMR Pre-Processing Tool")
    logger.info("=" * 60)

    # Handle commands
    if args.command == "pipeline":
        return cmd_pipeline(args, logger)
    elif args.command == "extract-video":
        return cmd_extract_video(args, logger)
    elif args.command == "clean-images":
        return cmd_clean_images(args, logger)
    elif args.command == "normalize-images":
        return cmd_normalize_images(args, logger)
    elif args.command == "split-dataset":
        return cmd_split_dataset(args, logger)
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
