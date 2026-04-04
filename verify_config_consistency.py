"""
Verification script for config consistency improvements
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.config import Config
from src.data import DatasetManager
from src.utils import get_traffic_sign_classes


def test_get_traffic_sign_classes():
    """Test get_traffic_sign_classes function"""
    print("\n" + "=" * 60)
    print("TEST 1: get_traffic_sign_classes()")
    print("=" * 60)

    # Test 1.1: Auto-load from configs/data.yaml
    print("\n1.1 Auto-load from configs/data.yaml:")
    classes1 = get_traffic_sign_classes()
    print(f"  ✓ Loaded {len(classes1)} classes")
    print(f"  Classes: {list(classes1.values())[:3]}... (showing first 3)")

    # Test 1.2: Specify explicit path
    print("\n1.2 Specify explicit path:")
    data_yaml = project_root / "configs" / "data.yaml"
    classes2 = get_traffic_sign_classes(str(data_yaml))
    print(f"  ✓ Loaded {len(classes2)} classes from {data_yaml}")

    # Test 1.3: Consistency check
    print("\n1.3 Consistency check:")
    if classes1 == classes2:
        print("  ✓ Auto-load and explicit path produce identical results")
    else:
        print("  ✗ WARNING: Results differ!")
        return False

    return True


def test_dataset_manager():
    """Test DatasetManager uses get_traffic_sign_classes"""
    print("\n" + "=" * 60)
    print("TEST 2: DatasetManager class names")
    print("=" * 60)

    manager = DatasetManager("test_data")
    classes_from_manager = manager._get_class_names()
    classes_from_utils = get_traffic_sign_classes()

    print(f"\n2.1 Classes from DatasetManager: {len(classes_from_manager)}")
    print(f"2.2 Classes from get_traffic_sign_classes: {len(classes_from_utils)}")

    print("\n2.3 Consistency check:")
    if classes_from_manager == classes_from_utils:
        print("  ✓ Both methods return identical class definitions")
        return True
    else:
        print("  ✗ WARNING: Methods return different class definitions!")
        # Show differences
        for k in set(classes_from_manager.keys()) | set(classes_from_utils.keys()):
            if classes_from_manager.get(k) != classes_from_utils.get(k):
                print(f"    Difference at class {k}:")
                print(f"      Manager: {classes_from_manager.get(k)}")
                print(f"      Utils:   {classes_from_utils.get(k)}")
        return False


def test_predictor():
    """Test Predictor integrates correctly"""
    print("\n" + "=" * 60)
    print("TEST 3: Predictor class")
    print("=" * 60)

    try:
        from src.inference import Predictor

        config = Config()
        predictor = Predictor(config)

        print("\n3.1 Predictor initialized successfully")
        print(f"3.2 Classes loaded: {len(predictor.classes)}")
        print(f"3.3 First class: {predictor.classes.get(0)}")

        # Verify against get_traffic_sign_classes
        expected_classes = get_traffic_sign_classes()
        if predictor.classes == expected_classes:
            print("  ✓ Predictor classes match get_traffic_sign_classes()")
            return True
        else:
            print("  ✗ WARNING: Predictor classes don't match!")
            return False

    except Exception as e:
        print(f"\n✗ Error initializing Predictor: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_complete_workflow():
    """Test complete workflow"""
    print("\n" + "=" * 60)
    print("TEST 4: Complete workflow consistency")
    print("=" * 60)

    # 1. Get classes from utils
    print("\n4.1 Getting classes from utils...")
    utils_classes = get_traffic_sign_classes()

    # 2. Create dataset manager and get classes
    print("4.2 Creating DatasetManager...")
    manager = DatasetManager("test_data")
    manager_classes = manager._get_class_names()

    # 3. Print summary
    print("\n4.3 Summary:")
    print(f"  Utils classes: {len(utils_classes)}")
    print(f"  Manager classes: {len(manager_classes)}")

    if utils_classes == manager_classes:
        print("  ✓ All classes are consistent across the workflow")
        return True
    else:
        print("  ✗ WARNING: Inconsistency detected!")
        return False


def verify_yaml_format():
    """Verify configs/data.yaml has correct format"""
    print("\n" + "=" * 60)
    print("TEST 5: Verify configs/data.yaml format")
    print("=" * 60)

    import yaml

    data_yaml = project_root / "configs" / "data.yaml"

    if not data_yaml.exists():
        print(f"\n✗ Warning: {data_yaml} not found")
        return False

    try:
        with open(data_yaml, encoding="utf-8") as f:
            config = yaml.safe_load(f)

        print("\n5.1 YAML file loaded successfully")
        print(f"5.2 Keys in config: {list(config.keys())}")

        if "names" not in config:
            print("  ✗ WARNING: 'names' key not found in data.yaml")
            return False

        names = config["names"]
        print(f"5.3 Classes in 'names': {len(names)}")
        print(f"5.4 First 3 classes: {dict(list(names.items())[:3])}")

        # Verify classes are strings
        all_strings = all(isinstance(v, str) for v in names.values())
        if all_strings:
            print("  ✓ All class values are strings")
        else:
            print("  ✗ WARNING: Some class values are not strings!")
            return False

        return True

    except Exception as e:
        print(f"\n✗ Error reading YAML: {e}")
        return False


def main():
    """Run all verification tests"""
    print("\n" + "#" * 60)
    print("# Configuration Consistency Verification")
    print("#" * 60)

    results = {
        "get_traffic_sign_classes": test_get_traffic_sign_classes(),
        "DatasetManager classes": test_dataset_manager(),
        "Predictor integration": test_predictor(),
        "Complete workflow": test_complete_workflow(),
        "YAML format": verify_yaml_format(),
    }

    # Summary
    print("\n" + "=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)

    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")

    total_pass = sum(results.values())
    total_tests = len(results)

    print(f"\nTotal: {total_pass}/{total_tests} tests passed")

    if total_pass == total_tests:
        print("\n✓ All verification tests passed! Configuration consistency improved.")
        return 0
    else:
        print(f"\n✗ {total_tests - total_pass} test(s) failed!")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
