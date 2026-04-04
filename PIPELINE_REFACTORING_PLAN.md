# 完整数据处理流程 + 配置重构计划

**日期**: 2026年4月4日  
**优先级**: 🔴 核心功能  
**工作量**: 中等 (1-2天)

---

## 📊 需求分析

### 用户需求
```
原始视频 → 关键帧提取 → 图片清洗 → 标准命名 → YOLO训练
   ↓           ↓           ↓          ↓
raw/video/   raw/images  清洗后    0001.jpg
            (缓存机制)
```

### 核心诉求
1. ✅ 支持视频→图片的完整流程
2. ✅ 智能缓存机制（避免重复处理）
3. ✅ 自动图片清洗和重命名
4. ✅ **配置全在YAML中** - 命令无额外参数
5. ✅ 项目级别的配置重构

---

## 🏗️ 新的目录结构

### 数据流目录
```
raw/
├── video/                    # ← 新增：视频输入
│   ├── video1.mp4
│   └── video2.mp4
├── images/                   # 现有：提取后的原始图片
│   ├── extracted_001.jpg
│   ├── extracted_002.jpg
│   └── ...
└── temp.json                 # ← 新增：缓存文件（哈希跟踪）

data/                         # 现有
├── train/
├── val/
└── test/
```

### 配置文件结构（重构后）
```
configs/
├── default.yaml             # 现有：主配置（训练）
├── quick.yaml               # 现有：快速测试配置
├── production.yaml          # 现有：生产配置
├── data.yaml                # 现有：数据集定义
├── pipeline.yaml            # ← 新增：数据处理流程配置
├── video_extraction.yaml    # ← 新增：视频→图片配置
├── image_cleaning.yaml      # ← 新增：图片清洗配置
└── normalization.yaml       # ← 新增：标准化配置
```

---

## 🔄 新增模块和功能

### 1. 视频处理模块
**文件**: `src/video_processor.py`

#### 功能
- 管理raw/video/目录下的视频文件
- 调用scripts/extract_keyframes.sh脚本
- 构建缓存系统（基于文件哈希）
- 管理raw/temp.json

#### 核心类
```python
class VideoProcessor:
    def __init__(self, config: VideoExtractionConfig):
        self.video_dir = config.video_dir
        self.output_dir = config.image_dir
        self.cache_file = config.cache_file
        self.script_path = config.extraction_script
    
    def compute_file_hash(self, file_path: str) -> str:
        """计算视频文件的MD5哈希"""
    
    def load_cache(self) -> dict:
        """从raw/temp.json加载已处理视频的哈希"""
    
    def save_cache(self, cache: dict):
        """将处理结果保存到raw/temp.json"""
    
    def extract_keyframes(self) -> int:
        """
        提取关键帧的主方法：
        1. 扫描raw/video/中的视频
        2. 计算哈希值
        3. 检查是否在缓存中
        4. 如果不在，调用extract_keyframes.sh脚本
        5. 更新缓存
        
        Returns: 提取的关键帧总数
        """
```

### 2. 图片清洗模块
**文件**: `src/image_cleaner.py`

#### 功能
- 检验图片完整性（是否损坏）
- 删除重复的图片（基于特征向量）
- 删除模糊的图片（Laplacian variance）
- 检查图片分辨率和格式

#### 核心类
```python
class ImageCleaner:
    def __init__(self, config: ImageCleaningConfig):
        self.image_dir = config.image_dir
        self.min_size = config.min_size
        self.blur_threshold = config.blur_threshold
        self.remove_duplicates = config.remove_duplicates
    
    def validate_image(self, image_path: str) -> bool:
        """检查图片是否完整、格式正确"""
    
    def detect_blur(self, image_path: str) -> float:
        """计算图片清晰度（Laplacian variance）"""
    
    def find_duplicates(self) -> list[tuple[str, str]]:
        """找到重复的图片对"""
    
    def clean_all(self) -> dict:
        """
        清洗所有图片：
        1. 删除损坏/格式错误的图片
        2. 删除过小的图片
        3. 删除模糊的图片
        4. 删除重复的图片
        
        Returns: {
            'valid': 100,
            'removed_corrupt': 5,
            'removed_blur': 3,
            'removed_duplicate': 2,
            'removed_small': 1
        }
        """
```

### 3. 图片重命名模块
**文件**: `src/image_normalizer.py`

#### 功能
- 将清洗后的图片重命名为 0001.jpg、0002.jpg 等格式
- 创建映射表（新名称→原始名称）便于追踪

#### 核心类
```python
class ImageNormalizer:
    def __init__(self, config: NormalizationConfig):
        self.image_dir = config.image_dir
        self.prefix = config.prefix  # 如 "traffic"
        self.start_index = config.start_index
        self.mapping_file = config.mapping_file
    
    def normalize_names(self) -> dict:
        """
        重命名图片为标准格式：
        - input: [img_a.jpg, img_b.jpg, ...]
        - output: [0001.jpg, 0002.jpg, ...]
        - 创建映射文件供后续追踪
        """
```

---

## 💾 新增配置文件详解

### configs/pipeline.yaml
```yaml
# 通用流程配置
pipeline:
  # 流程步骤
  steps:
    - video_extraction      # 从视频提取关键帧
    - image_cleaning        # 清洗图片
    - image_normalization   # 重命名图片
    - prepare_dataset       # 准备为YOLO格式

  # 输入输出
  raw_dir: ./raw
  output_dir: ./data
  temp_cache_file: ./raw/temp.json
```

### configs/video_extraction.yaml
```yaml
video_extraction:
  # 输入输出
  video_dir: ./raw/video
  image_dir: ./raw/images
  
  # 关键帧提取参数
  extraction_script: ./scripts/extract_keyframes.sh
  fps: 2                          # 每秒提取2帧
  quality: 85                     # JPEG质量
  format: jpg
  
  # 缓存策略
  cache_enabled: true
  cache_file: ./raw/temp.json
  
  # 日志
  log_level: info
```

### configs/image_cleaning.yaml
```yaml
image_cleaning:
  # 输入输出
  image_dir: ./raw/images
  
  # 清洗策略
  strategies:
    - validate_format       # 检查格式完整性
    - remove_blur           # 删除模糊图片
    - remove_duplicates     # 删除重复图片
    - validate_size         # 检查最小分辨率
  
  # 参数阈值
  min_width: 320
  min_height: 320
  blur_threshold: 100         # Laplacian variance < 100 = 模糊
  
  # 输出
  report_file: ./outputs/cleaning_report.json
```

### configs/normalization.yaml
```yaml
normalization:
  # 输入输出
  image_dir: ./raw/images
  
  # 命名规则
  prefix: traffic             # 可选前缀
  start_index: 1
  padding: 4                  # 0001.jpg (4位填充)
  
  # 映射记录
  mapping_file: ./raw/image_mapping.json
  
  # 输出格式
  format: jpg
  rename_in_place: true       # 在原目录重命名
```

---

## 🔄 配置系统重构

### 现有问题
- TrainConfig, ModelConfig, DataConfig 散在代码中
- 每个参数都要在dataclass中定义
- 无法灵活扩展

### 新的设计 (配置优先)
```
configs/default.yaml
    ↓
Config.from_yaml("configs/default.yaml")  # 动态加载所有sections
    ↓
src/config.py 只定义基本类（最小化）
```

#### 新的src/config.py结构
```python
# 保留的核心dataclass（最小化）
@dataclass
class Config:
    """主配置类 - 从YAML动态加载"""
    
    # 核心路径配置
    project_root: Path
    raw_dir: str
    data_dir: str
    output_dir: str
    configs_dir: str
    
    # 模块配置（通过字典存储，灵活扩展）
    train: dict           # 训练参数
    model: dict          # 模型参数
    data: dict           # 数据参数
    pipeline: dict       # 新增：流程配置
    video_extraction: dict  # 新增：视频提取
    image_cleaning: dict    # 新增：图片清洗
    normalization: dict     # 新增：标准化
    
    @classmethod
    def from_yaml(cls, yaml_path: str):
        """从YAML文件加载所有配置"""
        # 动态加载，不需要每个字段都在dataclass中预定义
```

---

## 🎯 CLI 命令设计

### 新增命令
```bash
# 完整数据处理流程（使用小配置从configs/pipeline.yaml）
python main.py process-data

# 仅提取关键帧
python main.py extract-video

# 仅清洗图片
python main.py clean-images

# 仅重命名图片
python main.py normalize-images

# 运行完整流程后训练
python main.py run-pipeline  # = process-data + prepare-data + train
```

### 命令参数（最小化）
```bash
# 无额外参数 - 所有配置来自YAML
python main.py process-data

# 覆盖单个参数（可选）
python main.py process-data --config configs/custom.yaml

# 指定输入目录（可选）
python main.py extract-video --video-dir /path/to/videos
```

---

## 📝 shell脚本

### scripts/extract_keyframes.sh
```bash
#!/bin/bash
# 使用ffmpeg提取视频的关键帧

VIDEO_FILE=$1
OUTPUT_DIR=$2
FPS=$3
QUALITY=$4

ffmpeg -i "$VIDEO_FILE" \
    -vf "fps=$FPS" \
    -q:v $QUALITY \
    "$OUTPUT_DIR/extracted_%06d.jpg"
```

---

## 🔌 实施步骤

### Phase 1: 基础设施 (4小时)
- [ ] 创建 src/video_processor.py
- [ ] 创建 src/image_cleaner.py
- [ ] 创建 src/image_normalizer.py
- [ ] 创建 scripts/extract_keyframes.sh

### Phase 2: 配置系统 (3小时)
- [ ] 创建新的YAML配置文件
- [ ] 重构 src/config.py (字典+动态加载)
- [ ] 更新现有配置的YAML

### Phase 3: CLI集成 (2小时)
- [ ] 添加新命令到main.py
- [ ] 更新argument parser
- [ ] 集成至现有的主流程

### Phase 4: 文档和测试 (2小时)
- [ ] 更新README
- [ ] 创建完整的流程说明
- [ ] 验证端到端流程

---

## 💡 技术亮点

### 1. 智能缓存机制
```json
{
  "video_hash_123abc": {
    "video_name": "traffic_001.mp4",
    "processed_date": "2026-04-04T10:30:00",
    "frames_extracted": 150,
    "output_dir": "./raw/images"
  },
  "video_hash_456def": {...}
}
```

### 2. 图片去重
- 使用感知哈希（Perceptual Hash）
- 快速查找相似图片

### 3. 模糊检测
- 使用Laplacian方差
- 快速、可靠的清晰度评估

### 4. 配置管理
- 所有参数在YAML中
- 支持动态加载
- 易于维护和扩展

---

## ✅ 完成标志

- [ ] 所有模块实现完成
- [ ] 所有配置写入 configs/
- [ ] 无需额外 CLI 参数即可运行
- [ ] 文档完整
- [ ] 端到端流程验证通过

---

**预计完成时间**: 1-2天工作量
