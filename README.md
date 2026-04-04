# YOLO Traffic Sign Recognition (YOLO-TMR)

**YOLO Traffic Sign Recognition and Deployment Acceleration Project**

打通交通标志识别模型训练与部署的完整流程，基于YOLO进行交通标志检测任务。

## 📋 项目概要

本项目旨在构建一个完整的交通标志识别系统，包括：
- ✅ 数据处理与准备
- ✅ 模型训练与验证
- ✅ 推理与部署
- ✅ 模型加速与导出

### 主要特性

- **完整的处理流程**：数据准备 → 模型训练 → 评估验证 → 推理部署
- **灵活的配置系统**：支持YAML配置文件，快速调整训练参数
- **多种模型支持**：YOLOv8-nano/small/medium/large/xlarge多种规模
- **丰富的数据增强**：mosaic、mixup、HSV等多种增强策略
- **友好的命令行接口**：统一的CLI入口，便于各种操作
- **完整的文档**：详细的使用说明和示例代码

## 🚀 快速开始

### 0. 环境检查

> 本项目默认使用NVIDIA GPU，若要使用CPU，则在同步依赖前修改pyproject.toml文件的下载源

查看驱动版本，若CUDA驱动支持小于12.4，同样需要修改下载源
```bash
nvidia-smi
```


### 1. 环境配置

使用uv进行依赖管理
- [uv中文文档](https://uv.doczh.com/)
- [将uv与PyTorch配合使用](https://uv.doczh.com/guides/integration/pytorch/)

```bash
# 克隆项目
git clone https://github.com/kunaro86/yolo-tmr.git
cd yolo-tmr

# 一键同步依赖
uv sync
```

### 2. 数据准备

#### 数据结构说明

原始数据应该放在以下结构：
```
raw/
├── images/          # 所有标注图像
│   ├── img1.jpg
│   ├── img2.jpg
│   └── ...
└── labels/          # 对应的YOLO格式标签
    ├── img1.txt
    ├── img2.txt
    └── ...
```

执行数据准备后，会自动创建YOLO标准的分割结构：
```
data/
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

#### 数据准备命令

```bash
# 查看帮助文档
python main.py prepare-data --help

# 方式1: 从原始数据分割（推荐用于首次准备）
python main.py prepare-data \
  --data-dir data \
  --split \
  --images-dir raw/images \
  --labels-dir raw/labels \
  --train-ratio 0.8 \
  --val-ratio 0.1

# 方式2: 只创建空的YOLO数据结构（用于已有train/val/test划分的数据）
python main.py prepare-data --data-dir data
```

### 3. 模型训练

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

### 4. 模型推理

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

### 5. 模型验证

```bash
python main.py validate \
  --model-path outputs/weights/best.pt \
  --data-yaml configs/data.yaml
```

### 6. 模型导出

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
yolo-tmr/
├── main.py                 # 主程序入口
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
│   └── utils.py           # 工具函数
│
├── scripts/               # 脚本文件
│   ├── train.py           # 训练脚本
│   ├── inference.py       # 推理脚本
│   └── data_prep.py       # 数据准备脚本
│
├── configs/               # 配置文件
│   ├── default.yaml       # 默认配置
│   ├── quick.yaml         # 快速测试配置
│   ├── production.yaml    # 生产配置
│   └── data.yaml          # 数据集配置
│
├── raw/                   # 原始数据目录（待准备）
│   ├── images/            # 原始图像文件
│   └── labels/            # 原始标签文件
│
├── data/                  # 数据集目录（处理后）
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
    └── exported_model/    # 导出的模型
```

### 文件说明
- **raw/** - 放置原始的标注数据（用户需要自己准备）
- **data/** - 执行 `prepare-data` 后自动生成的YOLO格式数据
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

项目支持43种交通标志识别（GTSRB数据集标准）：

| ID | 类别 | ID | 类别 |
|----|------|----|----|
| 0-8 | Speed limit 20-120km/h | 22-32 | 危险警告标志 |
| 9-18 | 禁行标志 | 33-42 | 指示标志 |
| 19-21 | 曲线警告标志 | - | - |

详见 [类别映射](src/utils.py#get_traffic_sign_classes)

## 💻 Python API 使用

### 训练

```python
from src.config import Config
from src.trainer import Trainer

# 加载配置
config = Config.from_yaml("configs/default.yaml")

# 创建训练器
trainer = Trainer(config)

# 训练模型
result = trainer.train(
    data_yaml="configs/data.yaml",
    output_dir="outputs"
)

if result["success"]:
    print("Training completed!")
```

### 推理

```python
from src.config import Config
from src.inference import Predictor

config = Config()
predictor = Predictor(config, model_path="outputs/weights/best.pt")

# 单图推理
result = predictor.predict("test.jpg")

# 批量推理
result = predictor.predict("test_images/")

# 视频推理
result = predictor.predict_video("video.mp4", output_path="output_video.mp4")

# 保存结果
if result["success"]:
    predictor.save_predictions(result["predictions"], "predictions.json")
```

### 数据处理

```python
from src.data import DatasetManager

manager = DatasetManager(data_dir="data", num_classes=43)

# 创建YOLO数据结构
data_yaml = manager.create_yolo_structure()

# 分割数据集
manager.split_dataset(
    images_dir="raw/images",
    labels_dir="raw/labels",
    train_ratio=0.8,
    val_ratio=0.1
)

# 验证数据集
train_count, val_count, test_count = manager.verify_dataset()
```

## 📈 训练流程

```
┌─────────────────┐
│  1. 数据准备     │
│  prepare-data   │
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

## 🎯 最佳实践

### 数据准备

1. **原始数据结构**：
   ```
   raw/
   ├── images/  # 图像文件
   └── labels/  # 标签文件
   ```

2. **标签格式**：YOLO格式 (`.txt` 文件，每行一个检测框)
   ```
   # 格式: <class_id> <center_x> <center_y> <width> <height>
   # 坐标为 [0, 1] 的相对值（相对于图像宽高）
   # 示例 (图像中有2个对象):
   14 0.5 0.5 0.4 0.6
   0 0.2 0.8 0.3 0.3
   ```
   
   每个 `.txt` 文件对应一张图像：
   ```
   img1.jpg -> img1.txt
   img2.jpg -> img2.txt
   ```

3. **数据分割**：默认 8:1:1 (训练:验证:测试)

4. **图像质量**：至少 640×640 分辨率

5. **类别均衡**：确保各类别样本相对均衡

### 训练建议

1. **暖启动**：使用预训练权重进行迁移学习
2. **学习率**：初始0.01，最终衰减到0.1
3. **批大小**：根据GPU显存调整（推荐16/32/64）
4. **数据增强**：启用mosaic和mixup提高鲁棒性
5. **早停**：设置patience为20-50，避免过拟合

### 推理优化

1. **模型导出**：导出为ONNX或其他格式
2. **批处理**：使用批量推理提高吞吐量
3. **置信度阈值**：根据应用场景调整（0.25-0.5）
4. **模型蒸馏**：使用nano模型进行边缘部署

## 🐛 常见问题
### Q: 如何准备原始数据？

A: 按以下步骤准备：
1. 创建 `raw/images/` 和 `raw/labels/` 目录
2. 将图像放在 `raw/images/`
3. 为每张图像创建同名的 `.txt` 标签文件，放在 `raw/labels/`
4. 标签格式为 YOLO 格式: `<class_id> <center_x> <center_y> <width> <height>`（相对坐标）
5. 运行数据准备命令自动分割：
   ```bash
   python main.py prepare-data --split --images-dir raw/images --labels-dir raw/labels
   ```

### Q: YOLO 标签格式具体是什么？

A: 以换行符分隔，每行对应一个检测框：
```
class_id center_x center_y width height
```
- `class_id`: 类别编号（0-42，共43个交通标志类别）
- 坐标值都是相对值，范围 [0, 1]
- 坐标相对于图像的宽和高

示例 (图像中有一个停止标志):
```
14 0.5 0.5 0.4 0.6
```

### Q: 可以使用 COCO 格式的标注吗？

A: 当前版本主要支持 YOLO 格式。如需使用 COCO 格式，需要先转换为 YOLO 格式。

### Q: data.yaml 中的类别与我的数据集不符怎么办？⚠️

A: **重要**：修改 `data/data.yaml` 中的 `nc` 和 `names` 与你的数据集对应。

**问题背景**：`data.yaml` 默认包含43个GTSRB交通标志类别。如果你的数据集不同，**必须修改**！

**检查步骤**：
```bash
# 1. 统计你的标签中有多少个不同类别
grep -oE "^[0-9]+" data/train/labels/*.txt | cut -d: -f2 | sort -u | wc -l

# 2. 检查data.yaml中的nc值
grep "^nc:" data/data.yaml

# 如果不一致，需要修改！
```

**修改方法**：
```bash
# 编辑 data/data.yaml
nano data/data.yaml
```

改为你的实际类别数和名称：
```yaml
nc: 5  # 改为你实际的类别数
names:
  0: Your Class A
  1: Your Class B
  2: Your Class C
  3: Your Class D
  4: Your Class E
```

**保护机制**：从本版本起，`prepare-data` 命令已改进为：
- ✅ 如果 `data.yaml` 已存在，**不会覆盖**（保护你的修改）
- ✅ 仅在首次运行或使用 `--force` 标志时生成

**强制重新生成**（谨慎使用）：
```bash
python main.py prepare-data --force --split --images-dir raw/images --labels-dir raw/labels
# ⚠️  这会覆盖你现有的 data.yaml，请确保已备份重要配置！
```

### Q: 如何使用自己的数据集？

A: 将数据集放在 `data/` 目录，运行数据准备命令：
```bash
python main.py prepare-data --split --images-dir raw/images --labels-dir raw/labels
```

然后检查 `data/data.yaml` 的类别是否与实际数据匹配（见上问）。

### Q: 训练过程中显存不足怎么办？

A: 降低批大小或图像大小：
```bash
python main.py train --batch-size 8 --config configs/default.yaml
```

### Q: 如何加快训练速度？

A: 使用快速配置或简化模型：
```bash
python main.py train --config configs/quick.yaml --model yolov8n
```

### Q: 推理结果格式是什么？

A: JSON格式，包含每个检测框的类别、置信度和坐标，详见 [inference.py](src/inference.py)

### Q: 项目如何处理数据集的增长和变化？

A: 详见 [说明文件](docs/DATA_CHANGE_CAPABILITY.md)，包括当前能力分析、限制说明和改进方案

## 📚 参考资源

### 项目文档
- [快速开始指南](docs/QUICKSTART.md) - 5分钟上手
- [数据变化处理](docs/DATA_CHANGE_CAPABILITY.md) - 如何处理数据集增长和变化
- [示例代码](example.py) - 9个实用示例

### 外部资源
- [YOLO 官方文档](https://docs.ultralytics.com/)
- [GTSRB 数据集](http://benchmark.ini.rub.de/)
- [YOLO 目标检测](https://en.wikipedia.org/wiki/You_Only_Look_Once)

## 📝 许可证

MIT License - 详见 LICENSE 文件

## 🤝 贡献

欢迎提交Issue和Pull Request！

## ✉️ 联系方式

如有问题或建议，欢迎反馈。

---

**最后更新**: 2026/04/04

**版本**: v0.1.0

**状态**: ✅ 流程打通完成
