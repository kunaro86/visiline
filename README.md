# YOLO Traffic Sign Recognition (YOLO-TMR)

**YOLO Traffic Sign Recognition and Deployment Acceleration Project**

打通YOLO训练中除数据标注以外的完整流程，提供高度灵活可配置的pipeline。

## 📋 项目概要

本项目包括两个主要工具：
- **preprocess.py** - 数据前处理工具（视频提取、图像清洗、标准化、分割）
- **main.py** - 模型训练/推理/导出工具

### 主要特性

- 🎬 **完整的数据管道**：视频 → 图像清洗 → 标准化 → 数据分割 → 训练
- 🛠️ **配置驱动设计**：所有参数在YAML中定义，支持灵活组合
- ⚡ **模块化工具**：数据处理和模型训练分离，独立调用
- 🔧 **灵活的CLI**：支持配置文件或命令行参数两种方式
- 📦 **多种模型**：基于 ultralytics 平台，模型选择灵活
- 🚀 **自动化工作流**：可进行完整端到端的自动化处理

## 🚀 快速开始

### 0. 驱动检查

> 本项目默认使用NVIDIA GPU，若要使用CPU，则在同步依赖前修改pyproject.toml文件的下载源

查看驱动版本，若CUDA驱动支持小于12.4，同样需要修改下载源
```bash
nvidia-smi
```

### 1. 项目依赖

使用uv进行依赖管理
- [uv中文文档](https://uv.doczh.com/)
- [将uv与PyTorch配合使用](https://uv.doczh.com/guides/integration/pytorch/)

```bash
# 克隆项目
git clone <repo-url>
cd yolo-tmr

# 安装依赖
uv sync

# 安装系统依赖 FFmpeg （若不使用视频关键帧提取则可跳过）
# Linux: apt install ffmpeg
# Mac: brew install ffmpeg
# Windows: choco install ffmpeg 或从 https://ffmpeg.org 下载
```

### 2. 数据处理 (使用 preprocess.py)

```bash
# 完整流程：视频 → 清洗 → 标准化 → 分割
python preprocess.py pipeline --config configs/pipeline.yaml

# 或分步执行
python preprocess.py extract-video --config configs/pre-processing.yaml
python preprocess.py clean-images --config configs/pre-processing.yaml  
python preprocess.py normalize-images --config configs/pre-processing.yaml
python preprocess.py split-dataset --config configs/pre-processing.yaml

# 或仅分割已有数据
python preprocess.py split-dataset --images-dir raw/images --data-dir data
```

### 3. 模型训练 (使用 main.py)

```bash
# 使用默认配置
python main.py train --config configs/default.yaml

# 自定义参数
python main.py train \
  --config configs/default.yaml \
  --epochs 200 \
  --batch-size 32 \
  --device cuda
```

### 4. 模型推理

```bash
# 单张图像
python main.py predict \
  --model-path outputs/weights/best.pt \
  --source test.jpg

# 批量推理
python main.py predict \
  --model-path outputs/weights/best.pt \
  --source test_images/ \
  --output results.json

# 视频推理
python main.py predict \
  --model-path outputs/weights/best.pt \
  --source video.mp4
```

### 5. 模型导出

```bash
# 导出为ONNX格式
python main.py export \
  --model-path outputs/weights/best.pt \
  --format onnx

# 其他格式
python main.py export --model-path outputs/weights/best.pt --format torchscript
python main.py export --model-path outputs/weights/best.pt --format tflite
```

## 🐛 常见问题
本项目配有详细的文档，请查看[帮助文档](docs/HELP.md)以及[常见问题汇总](docs/QnA.md)

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
