"""
Example script demonstrating the complete workflow
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.config import Config, ModelConfig, TrainConfig
from src.data import DatasetManager
from src.inference import Predictor
from src.trainer import Trainer
from src.utils import print_config, set_seed, setup_logger


def example_1_custom_config():
    """Example 1: Create custom configuration"""
    print("\n" + "=" * 60)
    print("Example 1: Custom Configuration")
    print("=" * 60)

    # Create custom configuration
    config = Config()
    config.model = ModelConfig(model_name="yolov8s", num_classes=43, pretrained=True)
    config.train = TrainConfig(
        epochs=100, batch_size=16, imgsz=640, device="cuda", lr0=0.01, warmup_epochs=5
    )

    print_config(config)

    # Save configuration
    config.save("custom_config.yaml")
    print("\nConfig saved to custom_config.yaml")


def example_2_load_config():
    """Example 2: Load configuration from file"""
    print("\n" + "=" * 60)
    print("Example 2: Load Configuration from File")
    print("=" * 60)

    # Load from YAML
    config = Config.from_yaml("configs/default.yaml")
    print_config(config)


def example_3_prepare_dataset():
    """Example 3: Prepare dataset"""
    print("\n" + "=" * 60)
    print("Example 3: Prepare Dataset")
    print("=" * 60)

    # Initialize dataset manager
    dataset_manager = DatasetManager(data_dir="data", num_classes=43)

    # Create YOLO structure
    print("\nCreating YOLO dataset structure...")
    data_yaml = dataset_manager.create_yolo_structure()
    print(f"Dataset structure created. data.yaml: {data_yaml}")

    # Note: In real scenario, split dataset from raw images
    # dataset_manager.split_dataset(
    #     images_dir="raw/images",
    #     labels_dir="raw/labels"
    # )

    # Verify dataset
    print("\nVerifying dataset...")
    train_count, val_count, test_count = dataset_manager.verify_dataset()


def example_4_train_model():
    """Example 4: Train model"""
    print("\n" + "=" * 60)
    print("Example 4: Train Model")
    print("=" * 60)

    # Load configuration
    config = Config.from_yaml("configs/quick.yaml")  # Use quick config for demo
    print_config(config)

    # Create trainer
    trainer = Trainer(config)

    # Train model
    print("\nStarting training...")
    result = trainer.train(data_yaml="configs/data.yaml", output_dir="outputs")

    if result["success"]:
        print("✓ Training completed successfully!")
        print(f"Results: {result['results']}")
    else:
        print(f"✗ Training failed: {result['error']}")


def example_5_validate_model():
    """Example 5: Validate model"""
    print("\n" + "=" * 60)
    print("Example 5: Validate Model")
    print("=" * 60)

    config = Config()
    trainer = Trainer(config)

    # Validate (requires trained model)
    result = trainer.validate(data_yaml="configs/data.yaml")

    if result["success"]:
        print("✓ Validation completed!")
    else:
        print(f"✗ Validation failed: {result['error']}")


def example_6_inference():
    """Example 6: Run inference"""
    print("\n" + "=" * 60)
    print("Example 6: Run Inference")
    print("=" * 60)

    config = Config()

    # Note: Replace with actual model path
    model_path = "outputs/weights/best.pt"

    try:
        predictor = Predictor(config, model_path=model_path)

        # Single image prediction
        print("\nRunning inference on single image...")
        result = predictor.predict("test_image.jpg", conf=0.5)

        if result["success"]:
            print("✓ Inference completed!")
            predictions = result["predictions"]
            print(f"Found {sum(len(p['detections']) for p in predictions)} detections")
        else:
            print(f"✗ Inference failed: {result['error']}")

    except Exception as e:
        print(f"Note: Model file not found. Error: {e}")
        print("In real scenario, use trained model weights.")


def example_7_export_model():
    """Example 7: Export model"""
    print("\n" + "=" * 60)
    print("Example 7: Export Model")
    print("=" * 60)

    config = Config()
    trainer = Trainer(config)

    print("\nExporting model to ONNX format...")
    result = trainer.export_model(
        export_format="onnx", output_path="outputs/traffic_sign_model.onnx"
    )

    if result["success"]:
        print(f"✓ Model exported: {result['export_path']}")
    else:
        print(f"✗ Export failed: {result['error']}")


def example_8_batch_inference():
    """Example 8: Batch inference on images"""
    print("\n" + "=" * 60)
    print("Example 8: Batch Inference")
    print("=" * 60)

    config = Config()

    try:
        model_path = "outputs/weights/best.pt"
        predictor = Predictor(config, model_path=model_path)

        # Batch prediction
        print("\nRunning batch inference...")
        result = predictor.predict_batch(image_dir="test_images/", conf=0.5)

        if result["success"]:
            print("✓ Batch inference completed!")
            predictions = result["predictions"]
            print(f"Processed {len(predictions)} images")

            # Save results
            predictor.save_predictions(predictions, "batch_predictions.json")
            print("Results saved to batch_predictions.json")
        else:
            print(f"✗ Batch inference failed: {result['error']}")

    except Exception as e:
        print(f"Note: Model file not found. Error: {e}")


def example_9_programmatic_training():
    """Example 9: Complete training workflow"""
    print("\n" + "=" * 60)
    print("Example 9: Complete Programmatic Workflow")
    print("=" * 60)

    # Setup
    logger = setup_logger(log_file="outputs/logs/example.log")
    set_seed(42)

    # 1. Load configuration
    logger.info("\n1. Loading configuration...")
    config = Config.from_yaml("configs/default.yaml")

    # 2. Prepare dataset
    logger.info("2. Preparing dataset...")
    dataset_manager = DatasetManager(data_dir="data")
    data_yaml = dataset_manager.create_yolo_structure()
    train_count, val_count, test_count = dataset_manager.verify_dataset()

    logger.info(f"   Train: {train_count}, Val: {val_count}, Test: {test_count}")

    if train_count == 0:
        logger.warning("   No training data found. Skipping training.")
        return

    # 3. Train model
    logger.info("3. Training model...")
    trainer = Trainer(config)
    train_result = trainer.train(data_yaml=data_yaml, output_dir="outputs")

    if not train_result["success"]:
        logger.error(f"   Training failed: {train_result['error']}")
        return

    logger.info("   Training completed!")

    # 4. Validate
    logger.info("4. Validating model...")
    val_result = trainer.validate(data_yaml=data_yaml)
    if val_result["success"]:
        logger.info("   Validation completed!")

    # 5. Export
    logger.info("5. Exporting model...")
    export_result = trainer.export_model(export_format="onnx")
    if export_result["success"]:
        logger.info(f"   Model exported: {export_result['export_path']}")

    logger.info("\n✓ Complete workflow finished!")


def main():
    """Run examples"""
    import argparse

    parser = argparse.ArgumentParser(description="Run examples")
    parser.add_argument(
        "--example",
        type=int,
        default=1,
        choices=range(1, 10),
        help="Which example to run (1-9)",
    )
    parser.add_argument("--all", action="store_true", help="Run all examples")

    args = parser.parse_args()

    examples = {
        1: example_1_custom_config,
        2: example_2_load_config,
        3: example_3_prepare_dataset,
        4: example_4_train_model,
        5: example_5_validate_model,
        6: example_6_inference,
        7: example_7_export_model,
        8: example_8_batch_inference,
        9: example_9_programmatic_training,
    }

    if args.all:
        for i in range(1, 10):
            try:
                examples[i]()
            except Exception as e:
                print(f"\n✗ Example {i} failed: {e}")
    else:
        try:
            examples[args.example]()
        except Exception as e:
            print(f"\n✗ Example failed: {e}")
            import traceback

            traceback.print_exc()


if __name__ == "__main__":
    main()
