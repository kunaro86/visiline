"""
将图片重命名为标准格式 (0001.jpg, 0002.jpg, ...)
"""

import json
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class ImageNormalizer:
    """标准化图片命名"""

    def __init__(self, config: dict):
        """
        初始化图片标准化器

        Args:
            config: 来自normalization.yaml的配置字典
        """
        self.image_dir = Path(config.get("image_dir", "./raw/images"))
        self.rename_in_place = config.get("rename_in_place", True)
        self.output_dir = Path(config.get("output_dir", "./raw/images_normalized"))

        self.prefix = config.get("prefix", "")
        self.start_index = config.get("start_index", 1)
        self.padding = config.get("padding", 4)
        self.output_format = config.get("output_format", "jpg")
        self.sort_order = config.get("sort_order", "name")

        self.create_mapping = config.get("create_mapping", True)
        self.mapping_file = Path(config.get("mapping_file", "./raw/image_mapping.json"))

        # 创建必要目录
        self.image_dir.mkdir(parents=True, exist_ok=True)
        self.mapping_file.parent.mkdir(parents=True, exist_ok=True)

    def get_sorted_images(self) -> list[Path]:
        """
        获取排序后的图片列表

        Returns:
            排序后的图片路径列表
        """
        image_files = list(self.image_dir.glob("*.*"))
        image_files = [
            f
            for f in image_files
            if f.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]
        ]

        # 根据配置排序
        if self.sort_order == "name":
            image_files.sort(key=lambda x: x.name)
        elif self.sort_order == "size":
            image_files.sort(key=lambda x: x.stat().st_size)
        elif self.sort_order == "date":
            image_files.sort(key=lambda x: x.stat().st_mtime)

        logger.info(f"Found {len(image_files)} image files to normalize")
        return image_files

    def generate_new_name(self, index: int, extension: str | None = None) -> str:
        """
        生成标准化的新文件名

        Args:
            index: 图片索引(从start_index开始)
            extension: 文件扩展名(如果为None则使用output_format)

        Returns:
            新的文件名
        """
        if extension is None:
            extension = self.output_format

        # 格式化编号
        number = str(index).zfill(self.padding)

        # 构建文件名
        if self.prefix:
            new_name = f"{self.prefix}_{number}.{extension}"
        else:
            new_name = f"{number}.{extension}"

        return new_name

    def normalize_names(self) -> dict[str, Any]:
        """
        重命名所有图片为标准格式

        Returns:
            处理统计 {
                total: int,
                renamed: int,
                failed: int,
                mapping: dict
            }
        """
        logger.info("=" * 60)
        logger.info("Starting image normalization")
        logger.info("=" * 60)

        image_files = self.get_sorted_images()
        if not image_files:
            logger.warning("No images found to normalize")
            return {"total": 0, "renamed": 0, "failed": 0, "mapping": {}}

        mapping = {}
        stats: dict[str, Any] = {"total": len(image_files), "renamed": 0, "failed": 0}

        # 重命名每个图片
        for idx, original_file in enumerate(image_files, self.start_index):
            try:
                # 生成新文件名
                new_name = self.generate_new_name(idx)
                new_path = self.image_dir / new_name

                # 检查新名称是否已存在
                if new_path.exists() and new_path != original_file:
                    logger.warning(f"Target name already exists: {new_name}")
                    stats["failed"] += 1
                    continue

                # 重命名
                if self.rename_in_place:
                    original_file.rename(new_path)
                    logger.debug(f"Renamed: {original_file.name} → {new_name}")
                else:
                    # 复制到output_dir
                    self.output_dir.mkdir(parents=True, exist_ok=True)
                    output_path = self.output_dir / new_name
                    shutil.copy2(original_file, output_path)
                    logger.debug(f"Copied: {original_file.name} → {new_name}")

                # 记录映射
                mapping[new_name] = {
                    "original": original_file.name,
                    "timestamp": datetime.now().isoformat(),
                    "index": idx - self.start_index + 1,
                }

                stats["renamed"] += 1

            except Exception as e:
                logger.error(f"Error renaming {original_file.name}: {e}")
                stats["failed"] += 1

        # 保存映射文件
        if self.create_mapping:
            self._save_mapping(mapping)

        # 打印统计
        logger.info("\n" + "=" * 60)
        logger.info("✅ Image normalization completed!")
        logger.info("=" * 60)
        logger.info(f"  Total images:      {stats['total']}")
        logger.info(f"  Renamed:           {stats['renamed']}")
        logger.info(f"  Failed:            {stats['failed']}")
        logger.info(f"  Mapping file:      {self.mapping_file}")
        logger.info("=" * 60)

        stats["mapping"] = mapping
        return stats

    def _save_mapping(self, mapping: dict):
        """
        保存映射文件

        Args:
            mapping: 映射字典
        """
        try:
            with open(self.mapping_file, "w", encoding="utf-8") as f:
                json.dump(mapping, f, indent=2, ensure_ascii=False)
            logger.info(f"Mapping file saved: {self.mapping_file}")
        except Exception as e:
            logger.error(f"Failed to save mapping: {e}")

    def restore_original_names(self) -> bool:
        """
        从映射文件恢复原始文件名(用于回滚)

        Returns:
            是否成功
        """
        if not self.mapping_file.exists():
            logger.error(f"Mapping file not found: {self.mapping_file}")
            return False

        try:
            with open(self.mapping_file, encoding="utf-8") as f:
                mapping = json.load(f)

            logger.warning("Attempting to restore original names...")

            for new_name, info in mapping.items():
                original_name = info["original"]
                current_path = self.image_dir / new_name

                if current_path.exists():
                    original_path = self.image_dir / original_name
                    current_path.rename(original_path)
                    logger.debug(f"Restored: {new_name} → {original_name}")

            logger.info("✅ Original names restored")
            return True

        except Exception as e:
            logger.error(f"Error restoring names: {e}")
            return False
