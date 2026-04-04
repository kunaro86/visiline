
## ⚠️ 重要注意事项

### 关于数据集配置（data.yaml）

项目新增了一个 **数据保护机制**，防止在数据准备时意外覆盖自定义的类别映射：

```bash
# 首次创建YOLO数据结构（创建新的data.yaml）
python main.py prepare-data --data-dir data --split ...

# 如果data.yaml已存在且需要覆盖，使用 --force 参数
python main.py prepare-data --data-dir data --force
```

**重要**：不使用 `--force` 参数时，如果 `data.yaml` 已存在，程序会：
1. 保留现有的类别映射（nc, names等）
2. 仅将图像和标签文件复制到 train/val/test 目录
3. 提示用户已保留原有配置

### 关于数据处理流程

数据处理流程包含以下步骤，每个步骤可独立配置或通过 pipeline.yaml 编排执行：

| 步骤 | 功能 | 配置文件 | 命令 |
|------|------|---------|------|
| 视频提取 | 从视频提取关键帧 | `configs/pre-processing.yaml` | `python main.py extract-video` |
| 图像清洗 | 移除损坏/模糊/重复的图像 | `configs/pre-processing.yaml` | `python main.py clean-images` |
| 名称标准化 | 统一重命名为 0001.jpg 格式 | `configs/pre-processing.yaml` | `python main.py normalize-images` |
| 数据分割 | 分割为 train/val/test 集 | `configs/split.yaml` | `python main.py split-dataset` |
| 流程编排 | 完整流程（上述所有步骤） | `configs/pipeline.yaml` | `python main.py process-data` |

**配置文件组织**：
- **pre-processing.yaml** - 包含视频提取、图像清洗、标准化三个部分的统一配置
- **split.yaml** - 专用于数据分割的配置（推荐方法）
- **pipeline.yaml** - 用于编排流程执行（指定启用的步骤）

### 缓存和映射文件

处理流程会生成以下辅助文件，用于追踪处理状态：

- **raw/temp.json** - 视频处理缓存（MD5哈希追踪，避免重复处理）
- **raw/image_mapping.json** - 图像名称映射（原始名→标准化名）
- **outputs/pipeline_reports/** - 处理报告（包含清洗统计）

这些文件可用于调试和恢复操作。


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
