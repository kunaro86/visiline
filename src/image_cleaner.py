"""
实现图片清洗、去重和验证功能
"""

import json
import logging
import shutil
from pathlib import Path

import cv2

logger = logging.getLogger(__name__)


class ImageCleaner:
    """清洗和验证图片质量"""

    def __init__(self, config: dict):
        """
        初始化图片清洁器

        Args:
            config: 来自image_cleaning.yaml的配置字典
        """
        self.image_dir = Path(config.get("image_dir", "./raw/images"))
        self.cleaned_dir = Path(config.get("cleaned_dir", "./raw/images_cleaned"))
        self.backup_dir = Path(config.get("backup_dir", "./raw/images_removed"))
        self.backup_removed = config.get("backup_removed", False)

        self.strategies = config.get("strategies", [])
        self.allowed_formats = config.get("allowed_formats", ["jpg", "jpeg", "png"])
        self.min_width = config.get("min_width", 320)
        self.min_height = config.get("min_height", 320)
        self.blur_threshold = config.get("blur_threshold", 100)
        self.duplicate_detection = config.get("duplicate_detection_enabled", True)
        self.similarity_threshold = config.get("similarity_threshold", 0.95)

        self.generate_report = config.get("generate_report", True)
        self.report_file = Path(
            config.get("report_file", "./outputs/cleaning_report.json")
        )

        # 创建必要目录
        self.image_dir.mkdir(parents=True, exist_ok=True)
        self.report_file.parent.mkdir(parents=True, exist_ok=True)

        # 统计信息
        self.stats = {
            "total_images": 0,
            "valid": 0,
            "removed_corrupt": 0,
            "removed_small": 0,
            "removed_blur": 0,
            "removed_duplicate": 0,
        }

    def validate_image_format(self, image_path: Path) -> bool:
        """
        验证图片格式和完整性

        Args:
            image_path: 图片路径

        Returns:
            图片是否有效
        """
        try:
            # 检查扩展名
            if image_path.suffix.lower().lstrip(".") not in self.allowed_formats:
                return False

            # 尝试读取图片
            img = cv2.imread(str(image_path))
            if img is None:
                logger.warning(f"Corrupt image: {image_path.name}")
                return False

            # 检查形状
            if img.shape[0] == 0 or img.shape[1] == 0:
                logger.warning(f"Invalid image shape: {image_path.name}")
                return False

            return True
        except Exception as e:
            logger.warning(f"Error validating {image_path.name}: {e}")
            return False

    def validate_image_size(self, image_path: Path) -> bool:
        """
        验证图片尺寸是否符合要求

        Args:
            image_path: 图片路径

        Returns:
            尺寸是否有效
        """
        try:
            img = cv2.imread(str(image_path))
            if img is None:
                return False

            height, width = img.shape[:2]
            if width < self.min_width or height < self.min_height:
                logger.debug(
                    f"Image too small: {image_path.name} "
                    f"({width}x{height} < {self.min_width}x{self.min_height})"
                )
                return False

            return True
        except Exception as e:
            logger.warning(f"Error checking size of {image_path.name}: {e}")
            return False

    def detect_blur(self, image_path: Path) -> float:
        """
        使用Laplacian方差检测图片模糊程度
        值越高越清晰

        Args:
            image_path: 图片路径

        Returns:
            Laplacian方差值
        """
        try:
            img = cv2.imread(str(image_path))
            if img is None:
                return 0.0

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            return float(laplacian_var)
        except Exception as e:
            logger.warning(f"Error detecting blur in {image_path.name}: {e}")
            return 0.0

    def compute_perceptual_hash(self, image_path: Path) -> str | None:
        """
        计算图片的感知哈希(用于去重)

        Args:
            image_path: 图片路径

        Returns:
            64位二进制哈希字符串
        """
        try:
            img = cv2.imread(str(image_path))
            if img is None:
                return None

            # 调整大小为8x8
            img_resized = cv2.resize(img, (8, 8))
            gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)

            # 计算平均灰度值
            avg = gray.mean()

            # 生成哈希
            phash = "".join(
                ["1" if pixel > avg else "0" for row in gray for pixel in row]
            )
            return phash
        except Exception as e:
            logger.warning(f"Error computing hash for {image_path.name}: {e}")
            return None

    def hamming_distance(self, hash1: str, hash2: str) -> int:
        """
        计算两个哈希的汉明距离

        Args:
            hash1: 第一个哈希
            hash2: 第二个哈希

        Returns:
            汉明距离
        """
        return sum(c1 != c2 for c1, c2 in zip(hash1, hash2, strict=True))

    def find_duplicates(self) -> list[tuple[str, str]]:
        """
        找到重复的图片对

        Returns:
            重复图片对列表 [(file1, file2), ...]
        """
        logger.info("Detecting duplicate images...")

        image_files = sorted(self.image_dir.glob("*.*"))
        image_files = [f for f in image_files if self.validate_image_format(f)]

        # 计算所有图片的感知哈希
        hashes = {}
        for image_file in image_files:
            phash = self.compute_perceptual_hash(image_file)
            if phash:
                hashes[image_file.name] = phash

        # 找到相似的图片对
        duplicates = []
        examined = set()

        for file1_name, hash1 in hashes.items():
            if file1_name in examined:
                continue

            for file2_name, hash2 in hashes.items():
                if file1_name == file2_name or file2_name in examined:
                    continue

                # 计算相似度(基于汉明距离)
                distance = self.hamming_distance(hash1, hash2)
                similarity = 1 - (distance / 64)  # 64位哈希

                if similarity >= self.similarity_threshold:
                    logger.debug(
                        f"Found duplicate pair: {file1_name} <-> {file2_name} "
                        f"(similarity: {similarity:.2%})"
                    )
                    duplicates.append((file1_name, file2_name))
                    examined.add(file2_name)

            examined.add(file1_name)

        return duplicates

    def remove_image(self, image_file: Path, reason: str):
        """
        删除或备份图片

        Args:
            image_file: 图片路径
            reason: 删除原因
        """
        if self.backup_removed:
            backup_dir = self.backup_dir / reason
            backup_dir.mkdir(parents=True, exist_ok=True)
            shutil.move(str(image_file), str(backup_dir / image_file.name))
            logger.debug(f"Backed up: {image_file.name} → {reason}/")
        else:
            image_file.unlink()
            logger.debug(f"Deleted: {image_file.name}")

    def clean_all(self) -> dict:
        """
        执行所有清洗策略

        Returns:
            清洗统计信息
        """
        logger.info("=" * 60)
        logger.info("Starting image cleaning process")
        logger.info("=" * 60)

        # 获取所有图片
        image_files = sorted(self.image_dir.glob("*.*"))
        self.stats["total_images"] = len(image_files)

        logger.info(f"Found {len(image_files)} image files")

        # 第一遍: 验证格式和尺寸
        if "validate_format" in self.strategies:
            logger.info("\nPhase 1: Validating image format...")
            for image_file in list(self.image_dir.glob("*.*")):
                if not self.validate_image_format(image_file):
                    self.remove_image(image_file, "invalid_format")
                    self.stats["removed_corrupt"] += 1

        if "validate_size" in self.strategies:
            logger.info("Phase 2: Validating image size...")
            for image_file in list(self.image_dir.glob("*.*")):
                if image_file.exists() and not self.validate_image_size(image_file):
                    self.remove_image(image_file, "too_small")
                    self.stats["removed_small"] += 1

        # 第二遍: 模糊检测
        if "detect_blur" in self.strategies:
            logger.info("Phase 3: Detecting blurry images...")
            for image_file in sorted(self.image_dir.glob("*.*")):
                if image_file.exists():
                    blur_score = self.detect_blur(image_file)
                    if blur_score < self.blur_threshold:
                        logger.debug(
                            f"Blurry image: {image_file.name} (score: {blur_score:.2f})"
                        )
                        self.remove_image(image_file, "blur")
                        self.stats["removed_blur"] += 1

        # 第三遍: 去重
        if "remove_duplicates" in self.strategies:
            logger.info("Phase 4: Removing duplicate images...")
            duplicates = self.find_duplicates()
            for _, duplicate_name in duplicates:
                dup_file = self.image_dir / duplicate_name
                if dup_file.exists():
                    self.remove_image(dup_file, "duplicate")
                    self.stats["removed_duplicate"] += 1

        # 统计有效图片
        valid_images = list(self.image_dir.glob("*.*"))
        valid_images = [f for f in valid_images if self.validate_image_format(f)]
        self.stats["valid"] = len(valid_images)

        # 生成报告
        if self.generate_report:
            self._generate_report()

        # 打印统计
        logger.info("\n" + "=" * 60)
        logger.info("✅ Image cleaning completed!")
        logger.info("=" * 60)
        logger.info(f"  Total images:           {self.stats['total_images']}")
        logger.info(f"  Valid images:           {self.stats['valid']}")
        logger.info(f"  Removed (corrupt):      {self.stats['removed_corrupt']}")
        logger.info(f"  Removed (too small):    {self.stats['removed_small']}")
        logger.info(f"  Removed (blur):         {self.stats['removed_blur']}")
        logger.info(f"  Removed (duplicate):    {self.stats['removed_duplicate']}")
        logger.info("=" * 60)

        return self.stats

    def _generate_report(self):
        """生成清洗报告"""
        report = {
            "summary": self.stats,
            "config": {
                "blur_threshold": self.blur_threshold,
                "min_width": self.min_width,
                "min_height": self.min_height,
                "similarity_threshold": self.similarity_threshold,
            },
        }

        with open(self.report_file, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        logger.info(f"Report saved to {self.report_file}")
