#!/usr/bin/env python3
"""
Installation verification script for Hip Implant AI.
Checks all dependencies and module imports.
"""

import sys
from pathlib import Path

print("=" * 70)
print("HIP IMPLANT AI - Installation Verification")
print("=" * 70)
print()

# Track results
all_passed = True

# 1. Check Python version
print("1. Checking Python version...")
py_version = sys.version_info
if py_version.major == 3 and py_version.minor >= 10:
    print(f"   ✓ Python {py_version.major}.{py_version.minor}.{py_version.micro}")
else:
    print(f"   ✗ Python {py_version.major}.{py_version.minor}.{py_version.micro} (requires 3.10+)")
    all_passed = False
print()

# 2. Check core dependencies
print("2. Checking core dependencies...")
dependencies = {
    'torch': 'PyTorch',
    'torchvision': 'TorchVision',
    'timm': 'timm',
    'transformers': 'Transformers',
    'monai': 'MONAI',
    'cv2': 'OpenCV',
    'albumentations': 'Albumentations',
    'numpy': 'NumPy',
    'pandas': 'Pandas',
    'scipy': 'SciPy',
    'sklearn': 'scikit-learn',
    'yaml': 'PyYAML',
    'tqdm': 'tqdm',
}

for module_name, display_name in dependencies.items():
    try:
        __import__(module_name)
        print(f"   ✓ {display_name}")
    except ImportError as e:
        print(f"   ✗ {display_name} - {str(e)}")
        all_passed = False

print()

# 3. Check CUDA availability
print("3. Checking CUDA availability...")
try:
    import torch
    if torch.cuda.is_available():
        print(f"   ✓ CUDA available - {torch.cuda.get_device_name(0)}")
        print(f"   ✓ CUDA version: {torch.version.cuda}")
    else:
        print("   ⚠ CUDA not available (CPU-only mode)")
        print("   Note: Training will be significantly slower on CPU")
except Exception as e:
    print(f"   ✗ Error checking CUDA: {str(e)}")
    all_passed = False

print()

# 4. Check project structure
print("4. Checking project structure...")
required_dirs = [
    'configs',
    'data/raw',
    'data/processed',
    'data/masks',
    'datasets',
    'models/segmentation',
    'models/classification',
    'training',
    'inference',
    'utils'
]

project_root = Path(__file__).parent
for dir_path in required_dirs:
    full_path = project_root / dir_path
    if full_path.exists():
        print(f"   ✓ {dir_path}/")
    else:
        print(f"   ✗ {dir_path}/ - missing")
        all_passed = False

print()

# 5. Check configuration files
print("5. Checking configuration files...")
config_files = [
    'configs/segmentation.yaml',
    'configs/classification.yaml'
]

for config_file in config_files:
    config_path = project_root / config_file
    if config_path.exists():
        print(f"   ✓ {config_file}")
        # Try to load YAML
        try:
            import yaml
            with open(config_path, 'r') as f:
                yaml.safe_load(f)
            print(f"      (valid YAML)")
        except Exception as e:
            print(f"      ✗ Invalid YAML: {str(e)}")
            all_passed = False
    else:
        print(f"   ✗ {config_file} - missing")
        all_passed = False

print()

# 6. Check module imports
print("6. Checking project module imports...")
modules_to_test = [
    ('utils.preprocessing', 'ImagePreprocessor'),
    ('utils.augmentation', 'SegmentationAugmentation'),
    ('utils.metrics', 'DiceScore'),
    ('utils.uncertainty', 'UncertaintyEstimator'),
    ('datasets.xray_dataset', 'XraySegmentationDataset'),
    ('models.segmentation.segformer', 'SegFormer'),
    ('models.classification.swin', 'SwinTransformer'),
]

# Add project root to path
sys.path.insert(0, str(project_root))

for module_path, class_name in modules_to_test:
    try:
        module = __import__(module_path, fromlist=[class_name])
        getattr(module, class_name)
        print(f"   ✓ {module_path}.{class_name}")
    except Exception as e:
        print(f"   ✗ {module_path}.{class_name} - {str(e)}")
        all_passed = False

print()

# 7. Quick functionality test
print("7. Running quick functionality tests...")
try:
    # Test preprocessing
    from utils.preprocessing import ImagePreprocessor
    import numpy as np

    preprocessor = ImagePreprocessor()
    test_image = np.random.randint(0, 255, (512, 512), dtype=np.uint8)
    processed = preprocessor.process(test_image)
    assert processed.shape == (512, 512), "Preprocessing failed"
    print("   ✓ Preprocessing pipeline")

    # Test augmentation
    from utils.augmentation import SegmentationAugmentation
    augmenter = SegmentationAugmentation(is_training=False)
    print("   ✓ Augmentation pipeline")

    # Test metrics
    from utils.metrics import DiceScore
    dice_calculator = DiceScore()
    print("   ✓ Metrics calculation")

    # Test uncertainty
    from utils.uncertainty import UncertaintyEstimator
    estimator = UncertaintyEstimator()
    test_probs = np.array([0.7, 0.2, 0.1])
    confidence, pred = estimator.compute_confidence(test_probs)
    assert 0 <= confidence <= 1, "Uncertainty estimation failed"
    print("   ✓ Uncertainty estimation")

except Exception as e:
    print(f"   ✗ Functionality test failed: {str(e)}")
    all_passed = False

print()

# Final summary
print("=" * 70)
if all_passed:
    print("✓ ALL CHECKS PASSED - Installation is complete and ready to use!")
    print()
    print("Next steps:")
    print("  1. Prepare your data in data/raw/train and data/raw/val")
    print("  2. Configure training in configs/segmentation.yaml")
    print("  3. Run training: python main.py --mode train_seg")
    print("  4. See QUICKSTART.md for detailed examples")
else:
    print("✗ SOME CHECKS FAILED - Please resolve the issues above")
    print()
    print("Common fixes:")
    print("  - Install missing dependencies: pip install -r requirements.txt")
    print("  - Check Python version: python --version (requires 3.10+)")
    print("  - Ensure all project files are present")
print("=" * 70)

sys.exit(0 if all_passed else 1)
