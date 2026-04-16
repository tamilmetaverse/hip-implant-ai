#!/usr/bin/env python3
"""
Dataset Preparation Script for Hip Implant AI
Creates stratified train/validation/test splits from the original dataset
"""

import os
import shutil
from pathlib import Path
from collections import defaultdict
import random
import json
import cv2
import numpy as np
from tqdm import tqdm

# Set random seed for reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Split ratios
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Paths
SOURCE_DIR = Path("data/Hip - 10 Implant")
TARGET_DIR = Path("data/processed")

def analyze_image_properties(image_path):
    """Analyze a single image"""
    try:
        img = cv2.imread(str(image_path))
        if img is None:
            return None
        return {
            'shape': img.shape,
            'dtype': str(img.dtype),
            'size_kb': image_path.stat().st_size / 1024
        }
    except Exception as e:
        print(f"Error reading {image_path}: {e}")
        return None

def collect_images_by_class(source_dir):
    """Collect all images organized by class"""
    images_by_class = defaultdict(list)

    print("Scanning dataset...")
    for class_dir in source_dir.iterdir():
        if not class_dir.is_dir():
            continue

        class_name = class_dir.name

        # Handle subdirectories (AP, LATERAL)
        image_files = []

        # Check if this class has subdirectories
        subdirs = [d for d in class_dir.iterdir() if d.is_dir()]

        if subdirs:
            # Has subdirectories (like AP/LATERAL)
            for subdir in subdirs:
                for img_file in subdir.glob("*"):
                    if img_file.suffix.lower() in ['.png', '.jpg', '.jpeg']:
                        image_files.append(img_file)
        else:
            # No subdirectories, images directly in class folder
            for img_file in class_dir.glob("*"):
                if img_file.suffix.lower() in ['.png', '.jpg', '.jpeg']:
                    image_files.append(img_file)

        images_by_class[class_name] = image_files

    return images_by_class

def stratified_split(images_by_class, train_ratio, val_ratio, test_ratio):
    """Create stratified train/val/test splits"""
    train_images = defaultdict(list)
    val_images = defaultdict(list)
    test_images = defaultdict(list)

    print("\nCreating stratified splits...")
    for class_name, images in images_by_class.items():
        # Shuffle images
        random.shuffle(images)

        n_total = len(images)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)

        # Split
        train_images[class_name] = images[:n_train]
        val_images[class_name] = images[n_train:n_train + n_val]
        test_images[class_name] = images[n_train + n_val:]

        print(f"  {class_name}:")
        print(f"    Train: {len(train_images[class_name])} | "
              f"Val: {len(val_images[class_name])} | "
              f"Test: {len(test_images[class_name])}")

    return train_images, val_images, test_images

def copy_images_to_split(images_dict, target_base_dir, split_name):
    """Copy images to the target directory structure"""
    print(f"\nCopying {split_name} images...")

    for class_name, images in tqdm(images_dict.items(), desc=f"Copying {split_name}"):
        # Create class directory
        class_dir = target_base_dir / split_name / class_name
        class_dir.mkdir(parents=True, exist_ok=True)

        # Copy images
        for img_path in images:
            dest_path = class_dir / img_path.name
            shutil.copy2(img_path, dest_path)

def analyze_dataset_statistics(images_by_class):
    """Analyze dataset statistics"""
    print("\n" + "="*70)
    print("DATASET ANALYSIS")
    print("="*70)

    total_images = sum(len(imgs) for imgs in images_by_class.values())

    print(f"\nTotal Classes: {len(images_by_class)}")
    print(f"Total Images: {total_images}")

    print("\nClass Distribution:")
    print("-" * 70)

    stats = []
    for class_name, images in sorted(images_by_class.items(),
                                     key=lambda x: len(x[1]),
                                     reverse=True):
        count = len(images)
        percentage = (count / total_images) * 100
        stats.append({
            'class': class_name,
            'count': count,
            'percentage': percentage
        })
        print(f"  {class_name:35s} : {count:4d} images ({percentage:5.2f}%)")

    # Check class imbalance
    counts = [len(imgs) for imgs in images_by_class.values()]
    max_count = max(counts)
    min_count = min(counts)
    imbalance_ratio = max_count / min_count

    print("\n" + "-" * 70)
    print(f"Class Imbalance Ratio: {imbalance_ratio:.2f}:1")
    if imbalance_ratio > 3:
        print("WARNING: Significant class imbalance detected!")
        print("Recommendation: Use class weights during training")

    return stats

def analyze_image_properties_dataset(images_by_class):
    """Analyze image properties across dataset"""
    print("\n" + "="*70)
    print("IMAGE PROPERTIES ANALYSIS")
    print("="*70)

    all_shapes = []
    all_sizes = []

    # Sample a few images from each class
    sample_count = 0
    max_samples = 50  # Don't analyze all images, just sample

    for class_name, images in images_by_class.items():
        sample_size = min(5, len(images))
        samples = random.sample(images, sample_size)

        for img_path in samples:
            props = analyze_image_properties(img_path)
            if props:
                all_shapes.append(props['shape'])
                all_sizes.append(props['size_kb'])
                sample_count += 1

                if sample_count >= max_samples:
                    break

        if sample_count >= max_samples:
            break

    if all_shapes:
        print(f"\nAnalyzed {sample_count} sample images:")

        # Get unique shapes
        unique_shapes = list(set([tuple(s) for s in all_shapes]))
        print(f"\nImage Dimensions:")
        for shape in unique_shapes:
            count = sum(1 for s in all_shapes if tuple(s) == shape)
            print(f"  {shape}: {count} images")

        # Size statistics
        avg_size = np.mean(all_sizes)
        print(f"\nAverage File Size: {avg_size:.2f} KB")
        print(f"Size Range: {min(all_sizes):.2f} KB - {max(all_sizes):.2f} KB")

def create_class_names_file(images_by_class, target_dir):
    """Create class_names.txt file"""
    class_names = sorted(images_by_class.keys())

    class_names_file = target_dir / "class_names.txt"
    with open(class_names_file, 'w') as f:
        for name in class_names:
            f.write(f"{name}\n")

    print(f"\n[OK] Created class names file: {class_names_file}")
    return class_names

def create_dataset_info_json(images_by_class, train_images, val_images,
                             test_images, target_dir):
    """Create dataset_info.json with metadata"""
    info = {
        'total_classes': len(images_by_class),
        'total_images': sum(len(imgs) for imgs in images_by_class.values()),
        'split_ratios': {
            'train': TRAIN_RATIO,
            'val': VAL_RATIO,
            'test': TEST_RATIO
        },
        'splits': {
            'train': sum(len(imgs) for imgs in train_images.values()),
            'val': sum(len(imgs) for imgs in val_images.values()),
            'test': sum(len(imgs) for imgs in test_images.values())
        },
        'class_distribution': {
            class_name: {
                'total': len(images),
                'train': len(train_images[class_name]),
                'val': len(val_images[class_name]),
                'test': len(test_images[class_name])
            }
            for class_name, images in images_by_class.items()
        },
        'class_names': sorted(images_by_class.keys())
    }

    info_file = target_dir / "dataset_info.json"
    with open(info_file, 'w') as f:
        json.dump(info, f, indent=2)

    print(f"[OK] Created dataset info: {info_file}")
    return info

def main():
    print("="*70)
    print("HIP IMPLANT AI - DATASET PREPARATION")
    print("="*70)
    print(f"\nSource: {SOURCE_DIR}")
    print(f"Target: {TARGET_DIR}")
    print(f"\nSplit Ratios: Train={TRAIN_RATIO} | Val={VAL_RATIO} | Test={TEST_RATIO}")
    print(f"Random Seed: {RANDOM_SEED}")

    # Check if source exists
    if not SOURCE_DIR.exists():
        print(f"\n❌ Error: Source directory not found: {SOURCE_DIR}")
        return

    # Collect images
    images_by_class = collect_images_by_class(SOURCE_DIR)

    if not images_by_class:
        print("\n❌ Error: No images found in source directory")
        return

    # Analyze dataset
    stats = analyze_dataset_statistics(images_by_class)
    analyze_image_properties_dataset(images_by_class)

    # Create splits
    train_images, val_images, test_images = stratified_split(
        images_by_class, TRAIN_RATIO, VAL_RATIO, TEST_RATIO
    )

    # Ask for confirmation
    print("\n" + "="*70)
    print("READY TO ORGANIZE DATASET")
    print("="*70)
    print(f"\nThis will create the following structure in {TARGET_DIR}:")
    print("""
    processed/
      train/
        Aesculp Bicontact/
        Biomet - Echo Bimetric/
        ...
      val/
        ...
      test/
        ...
    """)

    response = input("\nProceed with dataset preparation? (yes/no): ")
    if response.lower() not in ['yes', 'y']:
        print("Cancelled.")
        return

    # Clean target directory if exists
    if TARGET_DIR.exists():
        print(f"\nCleaning existing directory: {TARGET_DIR}")
        shutil.rmtree(TARGET_DIR)

    # Create target directory
    TARGET_DIR.mkdir(parents=True, exist_ok=True)

    # Copy images to splits
    copy_images_to_split(train_images, TARGET_DIR, "train")
    copy_images_to_split(val_images, TARGET_DIR, "val")
    copy_images_to_split(test_images, TARGET_DIR, "test")

    # Create metadata files
    class_names = create_class_names_file(images_by_class, TARGET_DIR)
    dataset_info = create_dataset_info_json(
        images_by_class, train_images, val_images, test_images, TARGET_DIR
    )

    # Final summary
    print("\n" + "="*70)
    print("DATASET PREPARATION COMPLETE!")
    print("="*70)
    print(f"\n[OK] Train images: {dataset_info['splits']['train']}")
    print(f"[OK] Validation images: {dataset_info['splits']['val']}")
    print(f"[OK] Test images: {dataset_info['splits']['test']}")
    print(f"\n[OK] Output directory: {TARGET_DIR}")
    print(f"[OK] Class names file: {TARGET_DIR / 'class_names.txt'}")
    print(f"[OK] Dataset info: {TARGET_DIR / 'dataset_info.json'}")

    print("\n" + "="*70)
    print("NEXT STEPS:")
    print("="*70)
    print("1. Review the splits in data/processed/")
    print("2. Run training: python training/train_classification.py")
    print("3. Check dataset_info.json for detailed statistics")
    print("\n")

if __name__ == "__main__":
    main()
