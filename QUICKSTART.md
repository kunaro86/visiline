# YOLO-TMR 快速参考指南

## 📦 项目概览

```
yolo-tmr/                    # YOLO Traffic Sign Recognition 项目
├── src/                     # 核心业务逻辑
│   ├── config.py           # 配置管理系统
│   ├── model.py            # 模型包装层
│   ├── trainer.py          # 训练逻辑
│   ├── inference.py        # 推理模块
│   ├── data.py             # 数据管理
│   └── utils.py            # 工具函数
├── scripts/                # 独立脚本
│   ├── train.py            # 训练脚本
│   ├── inference.py        # 推理脚本
│   └── data_prep.py        # 数据准备脚本
├── configs/                # 配置文件
│   ├── default.yaml        # 默认配置
│   ├── quick.yaml          # 快速测试
│   ├── production.yaml     # 生产配置
│   └── data.yaml           # 数据集配置
├── data/                   # 数据目录
│   ├── train/               # 训练集
│   ├── val/                 # 验证集
│   └── test/                # 测试集
├── outputs/                # 输出目录
│   ├── weights/             # 模型权重
│   └── logs/                # 日志
├── main.py                 # 统一入口
├── example.py              # 示例代码
└── README.md               # 完整文档
```

## 🚀 常用命令

### 准备数据
```bash
# 创建YOLO数据结构
python main.py prepare-data --data-dir data

# 从原始数据分割
python main.py prepare-data \
  --split \
  --images-dir raw/images \
  --labels-dir raw/labels
```

### 训练模型
```bash
# 默认训练
python main.py train

# 快速测试
python main.py train --config configs/quick.yaml

# 自定义参数
python main.py train \
  --epochs 200 \
  --batch-size 32 \
  --model yolov8m
```

### 推理
```bash
# 单图推理
python main.py predict \
  --model-path outputs/weights/best.pt \
  --source test.jpg

# 批量推理
python main.py predict \
  --model-path outputs/weights/best.pt \
  --source test_dir/ \
  --output results.json
```

### 验证和导出
```bash
# 验证
python main.py validate --model-path outputs/weights/best.pt

# 导出
python main.py export \
  --model-path outputs/weights/best.pt \
  --format onnx
```

## 🐍 Python API 使用

### 最小示例 - 训练
```python
from src.config import Config
from src.trainer import Trainer

config = Config.from_yaml("configs/default.yaml")
trainer = Trainer(config)
result = trainer.train("configs/data.yaml")
```

### 最小示例 - 推理
```python
from src.config import Config
from src.inference import Predictor

config = Config()
predictor = Predictor(config, model_path="outputs/weights/best.pt")
result = predictor.predict("test.jpg")
```

### 完整工作流
```python
from src.config import Config
from src.trainer import Trainer
from src.inference import Predictor
from src.data import DatasetManager

# 1. 准备数据
manager = DatasetManager("data")
manager.create_yolo_structure()

# 2. 训练
config = Config.from_yaml("configs/default.yaml")
trainer = Trainer(config)
train_result = trainer.train("configs/data.yaml")

# 3. 推理
predictor = Predictor(config, "outputs/weights/best.pt")
pred_result = predictor.predict("test_images/")

# 4. 保存结果
predictor.save_predictions(
    pred_result["predictions"], 
    "results.json"
)
```

## 📊 数据格式

### 输入格式
```
data/
├── train/
│   ├── images/
│   │   ├── img1.jpg
│   │   └── img2.jpg
│   └── labels/
│       ├── img1.txt  # YOLO格式
│       └── img2.txt
├── val/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

### 标签格式 (YOLO)
```
# img1.txt
<class_id> <center_x> <center_y> <width> <height>
# 值为 [0, 1] 的相对坐标
0 0.5 0.5 0.4 0.6
```

### 推理输出格式 (JSON)
```json
[
  {
    "image_path": "test.jpg",
    "detections": [
      {
        "class_id": 14,
        "class_name": "Stop",
        "confidence": 0.95,
        "bbox": {
          "x1": 100, "y1": 150,
          "x2": 200, "y2": 250
        }
      }
    ]
  }
]
```

## ⚙️ 配置参数解释

| 参数 | 说明 | 默认值 |
|------|------|--------|
| epochs | 训练轮次 | 100 |
| batch_size | 批大小 | 16 |
| imgsz | 输入图像大小 | 640 |
| device | 设备 (cuda/cpu) | cuda |
| lr0 | 初始学习率 | 0.01 |
| patience | 早停轮次 | 20 |
| augment | 数据增强 | true |
| model_name | 模型 (nano/s/m/l/x) | nano |

## 🎯 三个配置文件对比

| 特性 | default | quick | production |
|-----|---------|-------|-----------|
| 模型 | nano | nano | medium |
| epochs | 100 | 10 | 300 |
| batch_size | 16 | 32 | 16 |
| imgsz | 640 | 416 | 640 |
| 增强 | 标准 | 无 | 完整 |
| 用处 | 常规 | 测试 | 生产 |

## 🔍 故障排查

### 显存不足
```bash
# 降低批大小
python main.py train --batch-size 8

# 降低图像大小
python main.py train --config configs/default.yaml \
  --batch-size 8
```

### 加快训练
```bash
# 使用快速配置
python main.py train --config configs/quick.yaml

# 使用nano模型
python main.py train --model yolov8n --epochs 50
```

### 推理速度慢
```python
# 降低置信度阈值
predictor.predict(source, conf=0.25)  # 默认

# 或使用较小的模型
predictor = Predictor(config, "nano_model.pt")
```

## 📝 文件对应关系

| 操作 | 主脚本 | 源代码 | 配置文件 |
|-----|--------|--------|---------|
| 准备数据 | scripts/data_prep.py | src/data.py | - |
| 训练模型 | scripts/train.py | src/trainer.py | configs/default.yaml |
| 推理 | scripts/inference.py | src/inference.py | - |
| 统一命令 | main.py | 所有src文件 | - |

## 🎓 学习路径

1. **了解配置** → 运行 `python main.py info`
2. **准备数据** → 运行 `python main.py prepare-data --help`
3. **快速测试** → 运行 `python main.py train --config configs/quick.yaml`
4. **标准训练** → 运行 `python main.py train --config configs/default.yaml`
5. **推理测试** → 运行 `python main.py predict --model-path outputs/weights/best.pt --source test.jpg`
6. **查看代码** → 查看 src/*.py 和 example.py

## 📚 相关资源

- YOLOv8 文档: https://docs.ultralytics.com/
- GTSRB 数据集: http://benchmark.ini.rub.de/
- YOLO 论文: https://arxiv.org/abs/2304.00967

## ✅ 检查清单

项目完成度检查：

- [x] 项目框架建立
- [x] 核心模块实现 (config, model, trainer, inference, data)
- [x] 命令行接口 (train, predict, prepare-data, validate, export)
- [x] 配置文件系统 (default, quick, production, data.yaml)
- [x] 数据管理模块
- [x] 训练与推理逻辑
- [x] 完整文档
- [x] 示例代码
- [x] 工具函数
- [x] 项目配置 (pyproject.toml, .gitignore)

---

**项目状态**: ✅ 全流程打通完成

**版本**: v0.1.0

**最后更新**: 2026年4月4日
