"""
Data loading and preprocessing for YOLO-TMR project
"""

import logging
import shutil
from pathlib import Path

import yaml

from .utils import get_traffic_sign_classes

logger = logging.getLogger(__name__)


class DatasetManager:
    """Manage traffic sign dataset"""

    def __init__(self, data_dir: str, num_classes: int = 43):
        """
        Initialize dataset manager

        Args:
            data_dir: Root directory of dataset
            num_classes: Number of traffic sign classes
        """
        self.data_dir = Path(data_dir)
        self.num_classes = num_classes
        self.train_dir = self.data_dir / "train"
        self.val_dir = self.data_dir / "val"
        self.test_dir = self.data_dir / "test"

    def create_yolo_structure(self, force: bool = False) -> str:
        """
        Create YOLO dataset structure and return path to data.yaml

        Expected structure:
        dataset/
        ├── train/
        │   ├── images/
        │   └── labels/
        ├── val/
        │   ├── images/
        │   └── labels/
        └── test/
            ├── images/
            └── labels/

        Args:
            force: If True, overwrite existing data.yaml. Default is False (safe mode).

        Returns:
            Path to data.yaml file
        """
        # Create directories
        dirs = [
            self.train_dir / "images",
            self.train_dir / "labels",
            self.val_dir / "images",
            self.val_dir / "labels",
            self.test_dir / "images",
            self.test_dir / "labels",
        ]

        for dir_path in dirs:
            dir_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"✅ YOLO dataset directories created at {self.data_dir}")

        # Handle data.yaml - protect against overwriting custom configs
        data_yaml_path = self.data_dir / "data.yaml"
        
        if data_yaml_path.exists() and not force:
            # data.yaml already exists - preserve user's custom configuration
            logger.warning(
                f"⚠️  {data_yaml_path} already exists. Skipping overwrite to preserve custom config."
            )
            logger.info(
                "   If you need to regenerate it, use --force flag or manually delete:"
            )
            logger.info(f"   rm {data_yaml_path}")
            logger.info("")
            return str(data_yaml_path)

        # Create or overwrite data.yaml
        data_yaml_content = {
            "path": str(self.data_dir.absolute()),
            "train": "train/images",
            "val": "val/images",
            "test": "test/images",
            "nc": self.num_classes,
            "names": self._get_class_names(),
        }

        with open(data_yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(
                data_yaml_content, f, default_flow_style=False, allow_unicode=True
            )

        if force:
            logger.warning(f"⚠️  data.yaml regenerated (--force flag used)")
        else:
            logger.info(f"✅ data.yaml created at {data_yaml_path}")

        logger.debug(f"   nc: {self.num_classes} classes")

        return str(data_yaml_path)

    def split_dataset(
        self,
        images_dir: str,
        labels_dir: str,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
    ) -> None:
        """
        Split dataset into train/val/test

        Args:
            images_dir: Directory containing images
            labels_dir: Directory containing labels (YOLO format)
            train_ratio: Training ratio (default 0.8)
            val_ratio: Validation ratio (default 0.1)
            test_ratio: Test ratio (default 0.1)
        """
        import random

        images_path = Path(images_dir)
        labels_path = Path(labels_dir)

        # Get all image files
        image_files = sorted(
            [
                f
                for f in images_path.iterdir()
                if f.suffix.lower() in [".jpg", ".jpeg", ".png"]
            ]
        )

        if not image_files:
            logger.warning(f"No images found in {images_dir}")
            return

        # Shuffle and split
        random.shuffle(image_files)
        total = len(image_files)
        train_count = int(total * train_ratio)
        val_count = int(total * val_ratio)

        train_files = image_files[:train_count]
        val_files = image_files[train_count : train_count + val_count]
        test_files = image_files[train_count + val_count :]

        # Copy files
        self._copy_dataset_split(
            train_files, labels_path, self.train_dir, "train", len(train_files)
        )
        self._copy_dataset_split(
            val_files, labels_path, self.val_dir, "val", len(val_files)
        )
        self._copy_dataset_split(
            test_files, labels_path, self.test_dir, "test", len(test_files)
        )

        logger.info(
            f"Dataset split - Train: {len(train_files)}, "
            f"Val: {len(val_files)}, Test: {len(test_files)}"
        )

    def _copy_dataset_split(
        self,
        image_files: list[Path],
        labels_dir: Path,
        target_dir: Path,
        split_name: str,
        count: int,
    ) -> None:
        """Helper to copy dataset split"""
        target_images_dir = target_dir / "images"
        target_labels_dir = target_dir / "labels"

        target_images_dir.mkdir(parents=True, exist_ok=True)
        target_labels_dir.mkdir(parents=True, exist_ok=True)

        for img_file in image_files:
            # Copy image
            shutil.copy2(img_file, target_images_dir / img_file.name)

            # Copy corresponding label
            label_file = labels_dir / (img_file.stem + ".txt")
            if label_file.exists():
                shutil.copy2(label_file, target_labels_dir / label_file.name)

        logger.debug(f"Copied {count} files to {split_name}")

    def _get_class_names(self) -> dict:
        """Get class names for traffic signs from centralized utils

        Returns:
            Dictionary mapping class index to class name
        """
        # Load from utils to ensure single source of truth
        # This will use configs/data.yaml if available, otherwise falls back to defaults
        return get_traffic_sign_classes()

    def verify_dataset(self) -> tuple[int, int, int]:
        """
        Verify dataset integrity

        Returns:
            Tuple of (train_count, val_count, test_count)
        """
        train_images = list((self.train_dir / "images").glob("*.[jJ][pP][gG]")) + list(
            (self.train_dir / "images").glob("*.[pP][nN][gG]")
        )
        val_images = list((self.val_dir / "images").glob("*.[jJ][pP][gG]")) + list(
            (self.val_dir / "images").glob("*.[pP][nN][gG]")
        )
        test_images = list((self.test_dir / "images").glob("*.[jJ][pP][gG]")) + list(
            (self.test_dir / "images").glob("*.[pP][nN][gG]")
        )

        logger.info("Dataset verification:")
        logger.info(f"  Train: {len(train_images)} images")
        logger.info(f"  Val: {len(val_images)} images")
        logger.info(f"  Test: {len(test_images)} images")

        return len(train_images), len(val_images), len(test_images)
