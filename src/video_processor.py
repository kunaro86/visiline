"""
Video frame extraction module for YOLO-TMR project
从视频提取关键帧, 支持智能缓存机制
"""

import hashlib
import json
import logging
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


class VideoProcessor:
    """处理视频文件和提取关键帧"""

    def __init__(self, config: dict):
        """
        初始化视频处理器

        Args:
            config: 来自video_extraction.yaml的配置字典
        """
        self.video_dir = Path(config.get("video_dir", "./raw/video"))
        self.image_output_dir = Path(config.get("image_output_dir", "./raw/images"))
        self.extraction_script = config.get(
            "extraction_script", "./scripts/extract_keyframes.sh"
        )
        self.fps = config.get("frame_rate", 2)
        self.quality = config.get("quality", 85)
        self.format = config.get("format", "jpg")
        self.cache_enabled = config.get("cache_enabled", True)
        self.cache_file = Path(config.get("cache_file", "./raw/temp.json"))
        self.verify_output = config.get("verify_output", True)
        self.output_pattern = config.get("output_pattern", "extracted_{:06d}.jpg")
        self.video_formats = config.get("video_formats", [".mp4", ".avi", ".mov"])

        # 创建输出目录
        self.image_output_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)

    def compute_file_hash(self, file_path: str) -> str:
        """
        计算文件的MD5哈希值

        Args:
            file_path: 文件路径

        Returns:
            MD5哈希值
        """
        md5_hash = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                md5_hash.update(chunk)
        return md5_hash.hexdigest()

    def load_cache(self) -> dict:
        """
        从temp.json加载已处理视频的缓存

        Returns:
            缓存字典 {video_hash: {video_name, processed_date, frames_extracted, ...}}
        """
        if not self.cache_file.exists():
            return {}

        try:
            with open(self.cache_file, encoding="utf-8") as f:
                cache = json.load(f)
                logger.debug(
                    f"Loaded cache from {self.cache_file} ({len(cache)} entries)"
                )
                return cache
        except Exception as e:
            logger.warning(f"Failed to load cache from {self.cache_file}: {e}")
            return {}

    def save_cache(self, cache: dict):
        """
        保存缓存到temp.json

        Args:
            cache: 缓存字典
        """
        try:
            with open(self.cache_file, "w", encoding="utf-8") as f:
                json.dump(cache, f, indent=2, ensure_ascii=False)
                logger.info(f"Cache saved to {self.cache_file} ({len(cache)} entries)")
        except Exception as e:
            logger.error(f"Failed to save cache: {e}")

    def get_video_files(self) -> list[Path]:
        """
        获取video_dir中所有支持的视频文件

        Returns:
            视频文件路径列表
        """
        if not self.video_dir.exists():
            logger.warning(f"Video directory not found: {self.video_dir}")
            return []

        video_files = []
        for fmt in self.video_formats:
            video_files.extend(self.video_dir.glob(f"*{fmt}"))
            video_files.extend(self.video_dir.glob(f"*{fmt.upper()}"))

        logger.info(f"Found {len(video_files)} video file(s) in {self.video_dir}")
        return sorted(video_files)

    def extract_keyframes_from_video(self, video_file: Path) -> bool:
        """
        从单个视频提取关键帧

        Args:
            video_file: 视频文件路径

        Returns:
            是否成功
        """
        logger.info(f"Extracting keyframes from: {video_file.name}")

        try:
            # 使用ffmpeg提取帧
            cmd = [
                "ffmpeg",
                "-i",
                str(video_file),
                "-vf",
                f"fps={self.fps}",
                "-q:v",
                str(100 - self.quality),  # ffmpeg的q参数反向(1是最高质量)
                str(self.image_output_dir / self.output_pattern),
                "-y",  # 覆盖现有文件
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

            if result.returncode != 0:
                logger.error(f"ffmpeg failed: {result.stderr}")
                return False

            # 统计提取的帧数
            output_files = list(self.image_output_dir.glob("extracted_*.jpg"))
            frames_extracted = len(output_files)
            logger.info(
                f"✅ Extracted {frames_extracted} frames from {video_file.name}"
            )

            return True

        except subprocess.TimeoutExpired:
            logger.error(f"Video processing timeout: {video_file.name}")
            return False
        except Exception as e:
            logger.error(f"Error processing video {video_file.name}: {e}")
            return False

    def extract_all_keyframes(self) -> dict:
        """
        提取所有视频的关键帧, 支持缓存机制

        Returns:
            处理统计 {
                total_videos: int,
                cached_videos: int,
                newly_extracted: int,
                total_frames: int,
                failed: int
            }
        """
        logger.info("=" * 60)
        logger.info("Starting video frame extraction")
        logger.info("=" * 60)

        # 获取所有视频文件
        video_files = self.get_video_files()
        if not video_files:
            logger.warning("No video files found to process")
            return {
                "total_videos": 0,
                "cached_videos": 0,
                "newly_extracted": 0,
                "total_frames": 0,
                "failed": 0,
            }

        # 加载缓存
        cache = self.load_cache() if self.cache_enabled else {}

        stats = {
            "total_videos": len(video_files),
            "cached_videos": 0,
            "newly_extracted": 0,
            "total_frames": 0,
            "failed": 0,
        }

        # 处理每个视频
        for idx, video_file in enumerate(video_files, 1):
            logger.info(f"\n[{idx}/{len(video_files)}] Processing: {video_file.name}")

            # 计算视频文件哈希
            video_hash = self.compute_file_hash(str(video_file))

            # 检查缓存
            if self.cache_enabled and video_hash in cache:
                logger.info(
                    f"   ✅ Already processed (cached). "
                    f"Frames: {cache[video_hash].get('frames_extracted', '?')}"
                )
                stats["cached_videos"] += 1
                stats["total_frames"] += cache[video_hash].get("frames_extracted", 0)
                continue

            # 提取关键帧
            success = self.extract_keyframes_from_video(video_file)

            if success:
                # 统计帧数
                output_files = list(self.image_output_dir.glob("extracted_*.jpg"))
                frames_extracted = len(output_files)

                # 更新缓存
                cache[video_hash] = {
                    "video_name": video_file.name,
                    "video_size": video_file.stat().st_size,
                    "processed_date": str(Path(video_file).stat().st_mtime),
                    "frames_extracted": frames_extracted,
                    "output_dir": str(self.image_output_dir),
                }

                stats["newly_extracted"] += 1
                stats["total_frames"] += frames_extracted
            else:
                stats["failed"] += 1

        # 保存缓存
        if self.cache_enabled and (stats["newly_extracted"] > 0):
            self.save_cache(cache)

        # 打印统计
        logger.info("\n" + "=" * 60)
        logger.info("✅ Frame extraction completed!")
        logger.info("=" * 60)
        logger.info(f"  Total videos:        {stats['total_videos']}")
        logger.info(f"  From cache:          {stats['cached_videos']}")
        logger.info(f"  Newly extracted:     {stats['newly_extracted']}")
        logger.info(f"  Failed:              {stats['failed']}")
        logger.info(f"  Total frames:        {stats['total_frames']}")
        logger.info("=" * 60)

        return stats
