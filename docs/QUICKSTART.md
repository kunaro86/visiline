# YOLO-TMR 快速参考指南

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

### 原始数据格式（输入）
需要用户准备的格式：
```
raw/
├── images/
│   ├── img1.jpg
│   └── img2.jpg
└── labels/
    ├── img1.txt  # YOLO格式标签
    └── img2.txt
```

### 处理后的数据格式（`prepare-data` 后）
自动生成的YOLO标准格式：
```
data/
├── train/          # 训练集 (80%)
│   ├── images/
│   │   ├── img1.jpg
│   │   └── ...
│   └── labels/
│       ├── img1.txt
│       └── ...
├── val/            # 验证集 (10%)
│   ├── images/
│   └── labels/
└── test/           # 测试集 (10%)
    ├── images/
    └── labels/
```

### 标签格式 (YOLO)
`.txt` 文件中每行代表一个检测框：
```
<class_id> <center_x> <center_y> <width> <height>
```
- 坐标值为 [0, 1] 范围的相对值
- 示例：
```
# img1.txt
14 0.5 0.5 0.4 0.6    # 类别14，中心(0.5, 0.5)，宽0.4，高0.6
0 0.2 0.8 0.3 0.3     # 类别0，中心(0.2, 0.8)，宽0.3，高0.3
```

### 推理输出格式 (JSON)
模型推理结果保存为JSON格式：
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
