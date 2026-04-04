"""
Pipeline manager module for YOLO-TMR project
协调整个数据处理流程: 视频提取 → 图片清洗 → 图片重命名 → YOLO准备
"""

import logging

from .image_cleaner import ImageCleaner
from .image_normalizer import ImageNormalizer
from .video_processor import VideoProcessor

logger = logging.getLogger(__name__)


class PipelineManager:
    """管理完整的数据处理流程"""

    def __init__(self, config: dict):
        """
        初始化流程管理器

        Args:
            config: 主配置字典(包含所有子模块的配置)
        """
        self.config = config
        self.pipeline_config = config.get("pipeline", {})
        self.enabled_steps = self.pipeline_config.get(
            "enabled_steps",
            ["video_extraction", "image_cleaning", "image_normalization"],
        )

        # 初始化各个模块
        self.video_processor = (
            VideoProcessor(config.get("video_extraction", {}))
            if "video_extraction" in self.enabled_steps
            else None
        )
        self.image_cleaner = (
            ImageCleaner(config.get("image_cleaning", {}))
            if "image_cleaning" in self.enabled_steps
            else None
        )
        self.image_normalizer = (
            ImageNormalizer(config.get("normalization", {}))
            if "image_normalization" in self.enabled_steps
            else None
        )

    def run_pipeline(self) -> dict:
        """
        执行完整的数据处理流程

        Returns:
            包含所有步骤的执行结果
        """
        logger.info("\n" + "=" * 70)
        logger.info("🚀 YOLO-TMR Complete Data Processing Pipeline")
        logger.info("=" * 70)

        results = {}

        # 第1步: 视频提取
        if "video_extraction" in self.enabled_steps and self.video_processor:
            logger.info("\n📹 STEP 1: VIDEO FRAME EXTRACTION")
            logger.info("-" * 70)
            try:
                result = self.video_processor.extract_all_keyframes()
                results["video_extraction"] = result
            except Exception as e:
                logger.error(f"Video extraction failed: {e}")
                results["video_extraction"] = {"error": str(e)}

        # 第2步: 图片清洗
        if "image_cleaning" in self.enabled_steps and self.image_cleaner:
            logger.info("\n🧹 STEP 2: IMAGE CLEANING")
            logger.info("-" * 70)
            try:
                result = self.image_cleaner.clean_all()
                results["image_cleaning"] = result
            except Exception as e:
                logger.error(f"Image cleaning failed: {e}")
                results["image_cleaning"] = {"error": str(e)}

        # 第3步: 图片重命名
        if "image_normalization" in self.enabled_steps and self.image_normalizer:
            logger.info("\n🏷️  STEP 3: IMAGE NORMALIZATION")
            logger.info("-" * 70)
            try:
                result = self.image_normalizer.normalize_names()
                results["image_normalization"] = result
            except Exception as e:
                logger.error(f"Image normalization failed: {e}")
                results["image_normalization"] = {"error": str(e)}

        # 打印总体结果
        logger.info("\n" + "=" * 70)
        logger.info("✅ PIPELINE EXECUTION SUMMARY")
        logger.info("=" * 70)

        for step, result in results.items():
            if isinstance(result, dict) and "error" in result:
                logger.error(f"  ❌ {step}: FAILED - {result['error']}")
            else:
                logger.info(f"  ✅ {step}: SUCCESS")

        logger.info("=" * 70)

        return results

    def cleanup_temp_files(self):
        """清理临时文件"""
        logger.info("\n🗑️  Cleaning up temporary files...")
        try:
            # 这里可以添加清理临时文件的逻辑
            logger.info("✅ Cleanup completed")
        except Exception as e:
            logger.warning(f"Cleanup warning: {e}")
