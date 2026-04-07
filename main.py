"""
模型训练、推理和数据准备的统一入口脚本

本文件侧重于命令解析与调度, 不涉及具体模型或数据处理实现
"""

import argparse
import os
import sys
from pathlib import Path

import yaml

# 将项目根目录加入模块搜索路径, 便于直接 import src 下模块
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.config import Config
from src.data import DatasetManager
from src.inference import Predictor
from src.trainer import Trainer
from src.utils import print_config, setup_logger


def _read_num_classes(data_yaml: str, logger) -> int:
    """Read class count from data.yaml; fallback to 43 when unavailable."""
    if not os.path.exists(data_yaml):
        logger.warning(f"data.yaml not found at {data_yaml}, fallback nc=43")
        return 43

    try:
        with open(data_yaml, encoding="utf-8") as f:
            data_cfg = yaml.safe_load(f) or {}

        nc = data_cfg.get("nc")
        if isinstance(nc, int) and nc > 0:
            return nc

        logger.warning(f"Invalid [nc] in {data_yaml}, fallback nc=43")
    except Exception as e:
        logger.warning(f"Failed to read nc from {data_yaml}: {e}")

    return 43


def setup_parser():
    """创建并返回命令行参数解析器

    仅负责命令与参数声明, 不执行任何业务逻辑。
    """
    parser = argparse.ArgumentParser(
        prog="yolo-tmr",
        description="YOLO Traffic Mark Recognition - Training and Deployment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train model
  python main.py train --config configs/default.yaml --epochs 100

  # Run inference
  python main.py predict --model-path outputs/weights/best.pt --source test.jpg

  # Validate model
  python main.py validate --model-path outputs/weights/best.pt

  # Export model
  python main.py export --model-path outputs/weights/best.pt --format onnx

  # Show configuration
  python main.py info --config configs/default.yaml

  # For data preprocessing, use:
  #   python preprocess.py --help
        """,
    )

    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # ==================== Train Command ====================
    train_parser = subparsers.add_parser("train", help="Train YOLO model")
    train_parser.add_argument(
        "--config",
        "-c",
        type=str,
        default="configs/default.yaml",
        help="Path to configuration file (default: configs/default.yaml)",
    )
    train_parser.add_argument(
        "--data-yaml",
        type=str,
        default="configs/data.yaml",
        help="Path to data.yaml file",
    )
    train_parser.add_argument(
        "--epochs", type=int, default=None, help="Override number of epochs in config"
    )
    train_parser.add_argument(
        "--batch-size", type=int, default=None, help="Override batch size in config"
    )
    train_parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "cpu"],
        default=None,
        help="Override device in config",
    )
    train_parser.add_argument(
        "--model",
        type=str,
        choices=["yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x"],
        default=None,
        help="Override model in config",
    )

    # ==================== Predict Command ====================
    predict_parser = subparsers.add_parser("predict", help="Run inference")
    predict_parser.add_argument(
        "--model-path",
        "-m",
        type=str,
        required=True,
        help="Path to trained model weights",
    )
    predict_parser.add_argument(
        "--source", "-s", type=str, required=True, help="Image, video or directory path"
    )
    predict_parser.add_argument(
        "--conf", type=float, default=0.25, help="Confidence threshold"
    )
    predict_parser.add_argument("--iou", type=float, default=0.45, help="IOU threshold")
    predict_parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output path for predictions JSON",
    )
    predict_parser.add_argument(
        "--visualize", action="store_true", help="Visualize predictions"
    )

    # ==================== Validate Command ====================
    validate_parser = subparsers.add_parser("validate", help="Validate model")
    validate_parser.add_argument(
        "--model-path",
        "-m",
        type=str,
        required=True,
        help="Path to trained model weights",
    )
    validate_parser.add_argument(
        "--data-yaml",
        type=str,
        default="configs/data.yaml",
        help="Path to data.yaml file",
    )

    # ==================== Export Command ====================
    export_parser = subparsers.add_parser("export", help="Export model")
    export_parser.add_argument(
        "--model-path",
        "-m",
        type=str,
        required=True,
        help="Path to trained model weights",
    )
    export_parser.add_argument(
        "--format",
        type=str,
        default="onnx",
        choices=["onnx", "torchscript", "tflite"],
        help="Export format",
    )

    # ==================== Info Command ====================
    info_parser = subparsers.add_parser("info", help="Show configuration")
    info_parser.add_argument(
        "--config",
        "-c",
        type=str,
        default="configs/default.yaml",
        help="Path to configuration file",
    )

    return parser


def cmd_train(args, logger):
    """训练命令处理函数

    负责加载配置、检查数据集、构造 Trainer 并启动训练流程。
    返回进程退出码: 0 表示成功, 非 0 表示失败。
    """
    # 加载配置文件(若存在), 否则使用默认配置
    if os.path.exists(args.config):
        config = Config.from_yaml(args.config)
        logger.info(f"Configuration loaded from {args.config}")
    else:
        config = Config()
        logger.info("Using default configuration")

    # 使用命令行参数覆盖配置中的对应字段(如果指定了)
    if args.epochs:
        config.train.epochs = args.epochs
    if args.batch_size:
        config.train.batch_size = args.batch_size
    if args.device:
        config.train.device = args.device
    if args.model:
        config.model.model_name = args.model

    print_config(config)

    # 验证数据集结构和样本数
    logger.info("\nVerifying dataset...")
    data_dir = Path(config.data.data_dir)
    if not data_dir.is_absolute():
        data_dir = project_root / data_dir

    dataset_manager = DatasetManager(
        data_dir=str(data_dir),
        num_classes=_read_num_classes(args.data_yaml, logger),
    )
    train_count, val_count, _test_count = dataset_manager.verify_dataset()

    if train_count == 0 or val_count == 0:
        logger.error("Dataset not properly set up!")
        logger.error("Please run: python preprocess.py split-dataset --help")
        return 1

    # 启动训练
    trainer = Trainer(config)
    result = trainer.train(data_yaml=args.data_yaml, output_dir=config.output_dir)

    return 0 if result["success"] else 1


def cmd_predict(args, logger):
    """推理命令处理函数

    负责加载模型并在指定源上运行推理, 结果可选保存并可视化。
    """
    config = Config()

    logger.info(f"Loading model from {args.model_path}...")
    predictor = Predictor(config, model_path=args.model_path)

    logger.info(f"Running inference on {args.source}...")
    result = predictor.predict(
        source=args.source, conf=args.conf, iou=args.iou, visualize=args.visualize
    )

    if not result["success"]:
        logger.error(f"Inference failed: {result.get('error')}")
        return 1

    predictions = result["predictions"]
    logger.info(f"\nFound {sum(len(p['detections']) for p in predictions)} detections")

    # 如果指定了输出路径, 则保存预测结果
    if args.output:
        predictor.save_predictions(predictions, args.output)
        logger.info(f"Predictions saved to {args.output}")

    return 0


def cmd_validate(args, logger):
    """验证命令处理函数

    使用 Trainer 的验证方法对数据集进行评估并返回结果状态。
    """
    config = Config()
    trainer = Trainer(config)

    logger.info(f"Validating model {args.model_path}...")
    result = trainer.validate(data_yaml=args.data_yaml)

    if not result["success"]:
        logger.error(f"Validation failed: {result.get('error')}")
        return 1

    logger.info("Validation completed!")
    return 0


def cmd_export(args, logger):
    """模型导出命令处理函数

    将指定权重加载到 Trainer 的模型并调用导出接口, 输出导出路径。
    """
    config = Config()
    trainer = Trainer(config)
    trainer.model.model.load(args.model_path)  # type: ignore

    logger.info(f"Exporting model to {args.format}...")
    result = trainer.export_model(
        export_format=args.format, output_path=f"outputs/exported_model.{args.format}"
    )

    if not result["success"]:
        logger.error(f"Export failed: {result.get('error')}")
        return 1

    logger.info(f"Model exported to {result['export_path']}")
    return 0


def cmd_info(args, logger):
    """配置展示命令处理函数

    仅打印并展示配置信息, 便于用户确认参数设置。
    """
    config = Config.from_yaml(args.config) if os.path.exists(args.config) else Config()

    print_config(config)
    return 0


def main():
    """主函数: 解析命令并分发到对应的处理函数

    负责创建日志目录并初始化日志组件, 然后根据子命令调用对应的处理器。
    """
    # Setup parser
    parser = setup_parser()
    args = parser.parse_args()

    # Create output directories
    os.makedirs("outputs/logs", exist_ok=True)

    # Setup logger
    logger = setup_logger(name="yolo-tmr", log_file="outputs/logs/yolo-tmr.log")

    logger.info("=" * 60)
    logger.info("YOLO Traffic Sign Recognition - Training & Inference")
    logger.info("=" * 60)

    # Handle commands
    if args.command == "train":
        return cmd_train(args, logger)
    elif args.command == "predict":
        return cmd_predict(args, logger)
    elif args.command == "validate":
        return cmd_validate(args, logger)
    elif args.command == "export":
        return cmd_export(args, logger)
    elif args.command == "info":
        return cmd_info(args, logger)
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
