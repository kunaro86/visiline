
## 🌊 从Pipeline开始


### 1. 完整的数据前处理流程（从视频到YOLO数据集）

本项目支持从原始视频文件开始的完整数据前处理管道，包括：
**视频提取** → **图像清洗** → **图像标准化** → **YOLO数据标注准备**

> 该管道流程可视具体需求从 configs/pipeline.yaml 配置文件中调整(如跳过视频提取)

#### 2.1 工作流概述

```
raw/video/*.mp4
    ↓
[提取关键帧] → raw/images/ (raw frames)
    ↓
[图像清洗] → 移除损坏/模糊/重复图像，生成清洗报告
    ↓
[标准化名称] → 0001.jpg, 0002.jpg, ... (standard naming)
    ↓
[准备数据集] → dataset/train/val/test (YOLO format)
```

#### 2.2 必要的依赖

视频关键帧提取需要在全局环境中安装 FFmpeg 额外依赖：
```bash
# Linux/Mac: brew install ffmpeg
# Windows: choco install ffmpeg
# Or download from: https://ffmpeg.org/download.html
```

#### 2.3 数据目录结构

创建以下目录结构来放置原始视频和图像：
```
raw/
├── video/              # 原始视频文件目录
│   ├── sample1.mp4
│   ├── sample2.mp4
│   └── ...
├── images/             # 提取/清洗后的图像目录
│   ├── frame_0001.jpg
│   ├── frame_0002.jpg
│   └── ...
├── images_normalized/  # 标准化后的图像（可选：rename_in_place=false时）
│   ├── 0001.jpg
│   ├── 0002.jpg
│   └── ...
├── temp.json           # 缓存文件（存储处理过的视频哈希）
└── image_mapping.json  # 映射文件（原始名→标准化名）
```

#### 2.4 完整管道执行（推荐方法）

```bash
# 执行完整的数据处理流程（读取配置一键执行）
python preprocess.py pipeline --config configs/pipeline.yaml

```

这个命令会按顺序执行：
1. 从 `raw/video/` 中提取所有视频的关键帧
2. 清洗 `raw/images/` 中的所有图像（移除损坏、模糊、重复）
3. 重命名所有图像为标准格式（0001.jpg, 0002.jpg, ...）
4. 生成详细的处理报告

处理流程会生成以下辅助文件，用于追踪处理状态：

- **raw/temp.json** - 视频处理缓存（MD5哈希追踪，避免重复处理）
- **raw/image_mapping.json** - 图像名称映射（原始名→标准化名）
- **outputs/pipeline_reports/** - 处理报告（包含清洗统计）

这些文件可用于调试和恢复操作。

#### 2.5 分步执行（灵活方法）

```bash
# 步骤1: 仅提取视频关键帧
python preprocess.py extract-video \
  --video-dir raw/video/ \
  --output-dir raw/images/

# 步骤2: 仅清洗图像（移除损坏、模糊、重复）
python preprocess.py clean-images --image-dir raw/images/

# 步骤3: 仅标准化图像名称
python preprocess.py normalize-images --image-dir raw/images/

# 完全灵活：只调用需要的步骤
```

#### 2.6 配置文件说明

管道由以下配置文件控制（均位于 `configs/` 目录）：

**configs/pipeline.yaml** - [管道编排配置](configs/pipeline.yaml)

**configs/pre-processing.yaml** - [预处理统一配置](configs/pre-processing.yaml)（视频提取、图像清洗、标准化、数据集分割）

### 3. 传统数据准备（从已有的图像和标签）

#### 3.1 数据结构说明

如果你已经有标注好的图像和标签文件，可以跳过视频处理步骤，直接使用分割命令：


#### 3.2 使用 split-dataset 命令（推荐方法）

split-dataset 命令通过配置驱动方式，将图像和标签分割为 train/val/test 集：

```bash
# 查看帮助文档
python preprocess.py split-dataset --help

# 方式1: 使用配置文件（推荐）
python preprocess.py split-dataset --config configs/pre-processing.yaml

# 方式2: 命令行覆盖配置参数
python preprocess.py split-dataset \
  --images-dir raw/images \
  --labels-dir raw/labels \
  --data-dir dataset \
  --train-ratio 0.8 \
  --val-ratio 0.1
```

#### 3.3 数据结构要求

原始数据应该放在以下结构：
```
raw/
├── images/          # 所有标注图像
│   ├── img1.jpg
│   ├── img2.jpg
│   └── ...
└── labels/          # 对应的YOLO格式标签（可选）
    ├── img1.txt
    ├── img2.txt
    └── ...
```

执行分割后，会自动创建YOLO标准的分割结构：
```
dataset/
├── train/           # 训练集 (80%)
│   ├── images/
│   └── labels/
├── val/             # 验证集 (10%)
│   ├── images/
│   └── labels/
└── test/            # 测试集 (10%)
    ├── images/
    └── labels/
```

#### 3.4 配置说明

所有 yaml 配置文件均有良好的注释，在理解 pipeline 后可进入对应的配置文件中查看配置项


### 4. 模型训练

```bash
# 使用默认配置训练（推荐）
python main.py train --config configs/default.yaml

# 快速测试（少量epoch）
python main.py train --config configs/quick.yaml

# 生产级训练（高精度）
python main.py train --config configs/production.yaml

# 自定义参数
python main.py train \
  --config configs/default.yaml \
  --epochs 200 \
  --batch-size 32 \
  --device cuda \
  --model yolov8m
```

> [!TIP]
> 以上配置文件均可以自行修改，也可以增加新的预设配置！

### 5. 模型推理

```bash
# 单张图像推理
python main.py predict \
  --model-path outputs/weights/best.pt \
  --source test.jpg

# 批量推理
python main.py predict \
  --model-path outputs/weights/best.pt \
  --source test_images/ \
  --output results.json

# 带可视化推理
python main.py predict \
  --model-path outputs/weights/best.pt \
  --source test.jpg \
  --visualize

# 视频推理
python main.py predict \
  --model-path outputs/weights/best.pt \
  --source video.mp4
```

### 6. 模型验证

```bash
python main.py validate \
  --model-path outputs/weights/best.pt \
  --data-yaml data.yaml
```

### 7. 模型导出

```bash
# 导出为ONNX格式（推荐用于部署）
python main.py export \
  --model-path outputs/weights/best.pt \
  --format onnx

# 其他格式
python main.py export \
  --model-path outputs/weights/best.pt \
  --format torchscript  # PyTorch格式

python main.py export \
  --model-path outputs/weights/best.pt \
  --format tflite  # TensorFlow Lite格式
```

## 📁 项目结构

### 初始项目结构
```
visiline/
├── main.py                # 训练推理综合入口
├── preprocess.py          # 预处理综合入口
├── pyproject.toml         # 项目依赖配置
├── README.md              # 项目文档
├── .gitignore             # Git忽略文件
│
├── src/                   # 核心源代码
│   ├── __init__.py
│   ├── config.py          # 配置管理
│   ├── model.py           # 模型封装
│   ├── trainer.py         # 训练逻辑
│   ├── inference.py       # 推理模块
│   ├── data.py            # 数据处理
│   ├── video_processor.py # 视频关键帧提取
│   ├── image_cleaner.py   # 图像质量清洗
│   ├── image_normalizer.py # 图像名称标准化
│   ├── pipeline.py        # 数据处理流程编排
│   └── utils.py           # 工具函数
│
├── scripts/               # 外部工具脚本
│   └── extract_keyframes.sh # FFmpeg关键帧提取脚本（可选）
│
├── configs/               # 配置文件
│   ├── default.yaml       # 默认配置
│   ├── quick.yaml         # 快速测试配置
│   ├── production.yaml    # 生产配置
│   ├── data.yaml          # 数据集配置
│   ├── pipeline.yaml      # 数据处理管道编排配置
│   ├── pre-processing.yaml # 预处理统一配置（视频提取、清洗、标准化）
│
├── raw/                   # 原始数据目录（待准备）
│   ├── weights/           # 预训练权重
│   ├── video/             # 原始视频文件
│   ├── images/            # 提取/清洗的图像文件
│   ├── labels/            # 原始标签文件（可选）
│   ├── temp.json          # 视频处理缓存
│   └── image_mapping.json # 图像名称映射
│
├── dataset/               # 数据集目录（分割后）
│   ├── train/             # 训练集
│   │   ├── images/
│   │   └── labels/
│   ├── val/               # 验证集
│   │   ├── images/
│   │   └── labels/
│   └── test/              # 测试集
│       ├── images/
│       └── labels/
│
└── outputs/               # 输出目录（训练后）
    ├── weights/           # 模型权重
    ├── logs/              # 日志文件
    ├── pipeline_reports/  # 预处理报告
    └── exported_model/    # 导出的模型
```

### 文件说明
- **raw/** - 放置原始的标注数据或视频文件（用户自己准备）
- **dataset/** - 执行 `preprocess.py split-dataset` 后自动生成的YOLO格式数据
- **outputs/** - 训练、推理、导出的结果文件

## 🔧 配置说明

### 默认配置 (configs/default.yaml)

标准的训练配置，适合大多数场景：
- 模型：YOLOv8-nano
- 轮次：100 epochs
- 批大小：16
- 图像大小：640×640

### 快速配置 (configs/quick.yaml)

用于快速测试和验证：
- 模型：YOLOv8-nano
- 轮次：10 epochs（快速）
- 批大小：32（更快）
- 图像大小：416×416（更小）
- 无数据增强（快速）

### 生产配置 (configs/production.yaml)

高精度生产级配置：
- 模型：YOLOv8-medium
- 轮次：300 epochs
- 批大小：16
- 图像大小：640×640
- 完整数据增强

## 📊 交通标志类别

项目默认支持43种交通标志识别（GTSRB数据集标准）：

| ID | 类别 | ID | 类别 |
|----|------|----|----|
| 0-8 | Speed limit 20-120km/h | 22-32 | 危险警告标志 |
| 9-18 | 禁行标志 | 33-42 | 指示标志 |
| 19-21 | 曲线警告标志 | - | - |

详见 [类别映射](../src/utils.py)

## 📈 训练流程

```
┌─────────────────┐
│  1. 数据准备     │
│  split-dataset  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 2. 模型训练     │
│   train         │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 3. 模型验证     │
│   validate      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 4. 推理部署     │
│   predict/      │
│   export        │
└─────────────────┘
```