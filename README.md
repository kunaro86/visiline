# YOLO Traffic Sign Recognition (YOLO-TMR)

**YOLO Traffic Sign Recognition and Deployment Acceleration Project**

打通交通标志识别模型训练与部署的完整流程，基于YOLOv8进行交通标志检测任务。

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

### 1. 环境配置

```bash
# 克隆项目
git clone <project-repo>
cd yolo-tmr

# 一键同步依赖
uv sync
```

### 2. 数据准备

```bash
# 查看数据准备帮助
python main.py prepare-data --help

# 创建YOLO数据集结构
# 数据结构应该为:
# data/
# ├── images/
# └── labels/

python main.py prepare-data \
  --data-dir data \
  --split \
  --images-dir raw/images \
  --labels-dir raw/labels \
  --train-ratio 0.8 \
  --val-ratio 0.1
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
├── data/                  # 数据集目录
│   ├── train/
│   │   ├── images/
│   │   └── labels/
│   ├── val/
│   │   ├── images/
│   │   └── labels/
│   └── test/
│       ├── images/
│       └── labels/
│
└── outputs/               # 输出目录
    ├── weights/           # 模型权重
    ├── logs/              # 日志文件
    └── exported_model/    # 导出的模型

```

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

1. **数据格式**：支持COCO和YOLO标注格式
2. **数据分割**：默认8:1:1（训练:验证:测试）
3. **图像大小**：至少640×640分辨率
4. **类别均衡**：确保各类别样本均衡

### 训练建议

1. **暖启动**：使用预训练权重进行迁移学习
2. **学习率**：初始0.01，最终衰减到0.1
3. **批大小**：根据GPU显存调整（推荐16/32/64）
4. **数据增强**：启用mosaic和mixup提高鲁棒性
5. **早停**：设置patience为20-50，避免过拟合

### 推理优化

1. **模型导出**：导出为ONNX或TensorFlow Lite格式
2. **批处理**：使用批量推理提高吞吐量
3. **置信度阈值**：根据应用场景调整（0.25-0.5）
4. **模型蒸馏**：使用nano模型进行边缘部署

## 🐛 常见问题

### Q: 如何使用自己的数据集？

A: 将数据集放在 `data/` 目录，运行数据准备命令：
```bash
python main.py prepare-data --split --images-dir raw/images --labels-dir raw/labels
```

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

## 📚 参考资源

- [YOLOv8 官方文档](https://docs.ultralytics.com/)
- [GTSRB 数据集](http://benchmark.ini.rub.de/)
- [YOLO 目标检测](https://en.wikipedia.org/wiki/You_Only_Look_Once)

## 📝 许可证

MIT License - 详见 LICENSE 文件

## 🤝 贡献

欢迎提交Issue和Pull Request！

## ✉️ 联系方式

如有问题或建议，欢迎反馈。

---

**最后更新**: 2026年4月

**版本**: v0.1.0

**状态**: ✅ 流程打通完成
