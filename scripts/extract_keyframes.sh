#!/bin/bash
# extract_keyframes.sh
# 从视频文件提取关键帧
# 使用 ffmpeg 根据指定的帧率提取

set -e

# 参数检查
if [ $# -lt 2 ]; then
    echo "Usage: $0 <video_file> <output_dir> [fps] [quality]"
    echo ""
    echo "Examples:"
    echo "  $0 video.mp4 ./images 2 85"
    echo "  $0 video.mp4 ./images         # defaults: fps=2, quality=85"
    exit 1
fi

VIDEO_FILE="$1"
OUTPUT_DIR="$2"
FPS="${3:-2}"
QUALITY="${4:-85}"

# 验证输入文件
if [ ! -f "$VIDEO_FILE" ]; then
    echo "❌ Error: Video file not found: $VIDEO_FILE"
    exit 1
fi

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

echo "📹 Extracting keyframes from: $(basename "$VIDEO_FILE")"
echo "   Output directory: $OUTPUT_DIR"
echo "   Frame rate: $FPS fps"
echo "   Quality: $QUALITY"
echo ""

# 计算JPEG质量 (ffmpeg uses -q:v where 1=best, 31=worst)
# 我们的quality参数是1-100，需要反向转换
# quality 85 → q 3 (good), quality 50 → q 15 (medium)
Q_VALUE=$((100 - QUALITY / 3))
[ $Q_VALUE -lt 1 ] && Q_VALUE=1
[ $Q_VALUE -gt 31 ] && Q_VALUE=31

# 使用ffmpeg提取帧
if ! command -v ffmpeg &> /dev/null; then
    echo "❌ Error: ffmpeg not found. Please install ffmpeg:"
    echo "   Ubuntu/Debian: sudo apt-get install ffmpeg"
    echo "   macOS: brew install ffmpeg"
    echo "   Windows: Download from https://ffmpeg.org/download.html"
    exit 1
fi

echo "▶️  Processing..."
ffmpeg -i "$VIDEO_FILE" \
    -vf "fps=$FPS" \
    -q:v $Q_VALUE \
    "$OUTPUT_DIR/extracted_%06d.jpg" \
    -y \
    2>&1 | grep -E "(frame=|Bitstream)"

# 统计提取的帧数
FRAME_COUNT=$(ls -1 "$OUTPUT_DIR/extracted_"*.jpg 2>/dev/null | wc -l)

echo ""
echo "✅ Frame extraction completed!"
echo "   Total frames extracted: $FRAME_COUNT"
echo "   Output directory: $OUTPUT_DIR"
