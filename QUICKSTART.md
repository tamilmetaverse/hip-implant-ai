# Quick Start Guide

Get started with Hip Implant AI in minutes!

## Installation

```bash
# Clone and setup
git clone <repository-url>
cd hip_implant_ai

# Create environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Test (Without Training)

### 1. Test Inference Pipeline

```python
# test_inference.py
import cv2
import numpy as np
from inference.classify import ClassificationInference

# Create dummy checkpoint for testing
# (In production, use real trained model)

# Load an image
image = cv2.imread("path/to/xray.png", cv2.IMREAD_GRAYSCALE)

# Initialize classifier (requires trained checkpoint)
# classifier = ClassificationInference(
#     checkpoint_path="checkpoints/classification/best.pth",
#     class_names=["Type A", "Type B", "Type C"]
# )

# Get prediction
# result = classifier.predict(image, top_k=5)
# print(result)
```

## Training Workflow

### Step 1: Prepare Data

```bash
# Organize your data
hip_implant_ai/
└── data/
    ├── raw/
    │   ├── train/
    │   │   ├── image001.png
    │   │   └── ...
    │   └── val/
    │       └── ...
    └── masks/
        ├── image001_mask.png
        └── ...
```

### Step 2: Configure Training

Edit [configs/segmentation.yaml](configs/segmentation.yaml):

```yaml
data:
  train_dir: "data/raw/train"
  val_dir: "data/raw/val"
  batch_size: 8

training:
  epochs: 100
  learning_rate: 1e-4
```

### Step 3: Train Segmentation Model

```bash
python main.py --mode train_seg
```

### Step 4: Train Classification Model

```bash
# First, organize classification data
python main.py --mode train_cls
```

### Step 5: Run Inference

```bash
# Segmentation
python main.py --mode segment \
    --input test_image.png \
    --checkpoint checkpoints/segmentation/best.pth \
    --output results/mask.png

# Classification
python main.py --mode classify \
    --input test_image.png \
    --checkpoint checkpoints/classification/best.pth \
    --clinical-report
```

## Example: Complete Pipeline

```python
#!/usr/bin/env python3
"""
Complete inference pipeline example
"""

import cv2
from inference.segment import SegmentationInference
from inference.classify import ClassificationInference
from inference.ensemble import MultiModalEnsemble

# Image path
image_path = "data/test_xray.png"
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Step 1: Segment the implant
print("Step 1: Segmentation...")
segmenter = SegmentationInference(
    checkpoint_path="checkpoints/segmentation/best.pth"
)
seg_result = segmenter.predict(image)
mask = seg_result['mask']

# Step 2: Extract ROI
print("Step 2: Extracting ROI...")
masked_image = segmenter.extract_roi(image, mask)

# Step 3: Classify original image
print("Step 3: Classifying original image...")
classifier_original = ClassificationInference(
    checkpoint_path="checkpoints/classification/original_best.pth",
    class_names=["Zimmer", "DePuy", "Stryker", "Smith & Nephew", "Biomet"]
)
result_original = classifier_original.predict(image)

# Step 4: Classify masked image
print("Step 4: Classifying masked image...")
classifier_masked = ClassificationInference(
    checkpoint_path="checkpoints/classification/masked_best.pth",
    class_names=["Zimmer", "DePuy", "Stryker", "Smith & Nephew", "Biomet"]
)
result_masked = classifier_masked.predict(masked_image)

# Step 5: Generate clinical report
print("\n" + "="*60)
print("CLINICAL DECISION SUPPORT REPORT")
print("="*60)

print(f"\nPrimary Prediction: {result_original['primary_prediction']}")
print(f"Confidence: {result_original['confidence']:.1%}")

print("\nTop 5 Predictions:")
for i, pred in enumerate(result_original['predictions'][:5], 1):
    print(f"  {i}. {pred['class_name']}: {pred['probability']:.1%}")

print("\nUncertainty Analysis:")
unc = result_original['uncertainty']
print(f"  Needs Human Review: {unc['needs_human_review']}")
print(f"  Uncertainty Level: {unc['metrics'].get('confidence', 0):.1%}")

if unc['needs_human_review']:
    print("\n⚠️  HUMAN REVIEW REQUIRED")
    print("  The prediction confidence is below threshold.")
    print("  Please verify with clinical expertise.")
else:
    print("\n✓ High confidence prediction - suitable for clinical use")

print("="*60)
```

## Common Tasks

### Export Predictions to CSV

```python
import pandas as pd
from pathlib import Path

# Process multiple images
results = []
for image_path in Path("data/test").glob("*.png"):
    result = classifier.predict_from_file(str(image_path))
    results.append({
        'filename': image_path.name,
        'prediction': result['primary_prediction'],
        'confidence': result['confidence'],
        'needs_review': result['uncertainty']['needs_human_review']
    })

# Save to CSV
df = pd.DataFrame(results)
df.to_csv("predictions.csv", index=False)
```

### Batch Processing

```python
from glob import glob
import cv2

# Load all test images
image_paths = glob("data/test/*.png")
images = [cv2.imread(p, cv2.IMREAD_GRAYSCALE) for p in image_paths]

# Batch predict
results = classifier.batch_predict(images, batch_size=32)

# Process results
for path, result in zip(image_paths, results):
    print(f"{path}: {result['primary_prediction']} ({result['confidence']:.1%})")
```

## Troubleshooting

### CUDA Out of Memory

Reduce batch size in config:

```yaml
data:
  batch_size: 4  # Reduce from 8
```

### Slow Training

Enable mixed precision:

```python
# In training code
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

with autocast():
    outputs = model(images)
    loss = outputs['loss']

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### Poor Predictions

1. Check data quality
2. Increase training epochs
3. Use data augmentation
4. Try ensemble models

## Next Steps

1. Review [README.md](README.md) for complete documentation
2. Customize configurations in [configs/](configs/)
3. Implement custom data loaders if needed
4. Add your own models or modify existing ones
5. Set up experiment tracking (TensorBoard/W&B)

## Support

For issues and questions:
- Check [README.md](README.md)
- Review code documentation
- Open an issue on GitHub

Happy researching! 🚀
