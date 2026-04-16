#!/usr/bin/env python3
"""
Evaluation script for Hip Implant Classification Model
Tests the trained model on the held-out test set
"""

import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import numpy as np
from tqdm import tqdm
import argparse
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Import project modules
from datasets.multi_ext_dataset import MultiExtensionDataset
from models.classification.swin import SwinTransformer
from utils.augmentation import ClassificationAugmentation

# Configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_model(checkpoint_path, num_classes, model_name, in_channels):
    """Load trained model from checkpoint"""
    print(f"Loading model from: {checkpoint_path}")

    # Create model
    model = SwinTransformer(
        num_classes=num_classes,
        model_name=model_name,
        pretrained=False,  # We're loading trained weights
        in_channels=in_channels,
        dropout=0.3
    )

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(DEVICE)
    model.eval()

    print(f"Model loaded successfully!")
    print(f"Best validation accuracy from training: {checkpoint['best_acc']:.2f}%")

    return model


def evaluate(model, dataloader, class_names, device):
    """Evaluate model on test set"""
    print("\nEvaluating on test set...")

    all_preds = []
    all_labels = []
    all_probs = []

    correct = 0
    total = 0
    top5_correct = 0

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Testing"):
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)

            # Top-1 predictions
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # Top-5 predictions
            _, top5_pred = outputs.topk(min(5, outputs.size(1)), 1, True, True)
            top5_correct += top5_pred.eq(labels.view(-1, 1).expand_as(top5_pred)).sum().item()

            # Store for detailed analysis
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    # Calculate metrics
    test_acc = 100. * correct / total
    test_top5_acc = 100. * top5_correct / total

    return test_acc, test_top5_acc, all_preds, all_labels, all_probs


def plot_confusion_matrix(cm, class_names, output_dir):
    """Plot confusion matrix"""
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix - Test Set', fontsize=16, pad=20)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    save_path = output_dir / 'confusion_matrix.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nConfusion matrix saved to: {save_path}")
    plt.close()


def plot_per_class_accuracy(report_dict, output_dir):
    """Plot per-class accuracy"""
    # Extract per-class metrics
    classes = []
    accuracies = []

    for class_name, metrics in report_dict.items():
        if class_name not in ['accuracy', 'macro avg', 'weighted avg']:
            classes.append(class_name)
            accuracies.append(metrics['recall'] * 100)  # recall = per-class accuracy

    # Sort by accuracy
    sorted_indices = np.argsort(accuracies)
    classes = [classes[i] for i in sorted_indices]
    accuracies = [accuracies[i] for i in sorted_indices]

    # Plot
    plt.figure(figsize=(12, 8))
    bars = plt.barh(classes, accuracies, color='steelblue')

    # Color bars based on performance
    for i, bar in enumerate(bars):
        if accuracies[i] >= 70:
            bar.set_color('green')
        elif accuracies[i] >= 50:
            bar.set_color('orange')
        else:
            bar.set_color('red')

    plt.xlabel('Accuracy (%)', fontsize=12)
    plt.title('Per-Class Accuracy on Test Set', fontsize=16, pad=20)
    plt.xlim(0, 100)

    # Add value labels
    for i, (cls, acc) in enumerate(zip(classes, accuracies)):
        plt.text(acc + 1, i, f'{acc:.1f}%', va='center', fontsize=10)

    plt.tight_layout()
    save_path = output_dir / 'per_class_accuracy.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Per-class accuracy plot saved to: {save_path}")
    plt.close()


def save_results(test_acc, test_top5_acc, report_dict, cm, class_names, output_dir):
    """Save evaluation results to file"""
    results = {
        'test_accuracy': float(test_acc),
        'test_top5_accuracy': float(test_top5_acc),
        'num_classes': len(class_names),
        'class_names': class_names,
        'per_class_metrics': {},
        'confusion_matrix': cm.tolist()
    }

    # Add per-class metrics
    for class_name in class_names:
        if class_name in report_dict:
            results['per_class_metrics'][class_name] = {
                'precision': float(report_dict[class_name]['precision']),
                'recall': float(report_dict[class_name]['recall']),
                'f1_score': float(report_dict[class_name]['f1-score']),
                'support': int(report_dict[class_name]['support'])
            }

    # Save to JSON
    results_file = output_dir / 'test_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {results_file}")

    # Also save human-readable report
    report_file = output_dir / 'test_report.txt'
    with open(report_file, 'w') as f:
        f.write("="*70 + "\n")
        f.write("HIP IMPLANT AI - TEST SET EVALUATION\n")
        f.write("="*70 + "\n\n")
        f.write(f"Test Accuracy (Top-1): {test_acc:.2f}%\n")
        f.write(f"Test Accuracy (Top-5): {test_top5_acc:.2f}%\n\n")
        f.write("-"*70 + "\n")
        f.write("PER-CLASS RESULTS\n")
        f.write("-"*70 + "\n\n")

        for class_name in class_names:
            if class_name in report_dict:
                metrics = report_dict[class_name]
                f.write(f"{class_name}:\n")
                f.write(f"  Precision: {metrics['precision']*100:.2f}%\n")
                f.write(f"  Recall:    {metrics['recall']*100:.2f}%\n")
                f.write(f"  F1-Score:  {metrics['f1-score']*100:.2f}%\n")
                f.write(f"  Support:   {metrics['support']} samples\n\n")

    print(f"Human-readable report saved to: {report_file}")


def main(args):
    print("="*70)
    print("HIP IMPLANT AI - MODEL EVALUATION")
    print("="*70)
    print(f"\nDevice: {DEVICE}")
    print(f"Checkpoint: {args.checkpoint}")

    # Load dataset info
    info_file = Path(args.data_dir) / 'dataset_info.json'
    with open(info_file, 'r') as f:
        dataset_info = json.load(f)

    num_classes = dataset_info['total_classes']
    class_names = dataset_info['class_names']

    print(f"\nDataset: {args.data_dir}")
    print(f"Number of classes: {num_classes}")
    print(f"Test images: {dataset_info['splits']['test']}")

    # Create test transform
    test_transform = ClassificationAugmentation(
        image_size=(args.image_size, args.image_size),
        is_training=False
    )

    # Load test dataset
    print("\nLoading test dataset...")
    test_dataset = MultiExtensionDataset(
        image_dir=Path(args.data_dir) / 'test',
        transform=test_transform,
        use_folders=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )

    print(f"Test dataset: {len(test_dataset)} images")

    # Load trained model
    model = load_model(args.checkpoint, num_classes, args.model_name, args.in_channels)

    # Evaluate
    test_acc, test_top5_acc, all_preds, all_labels, all_probs = evaluate(
        model, test_loader, class_names, DEVICE
    )

    # Print results
    print("\n" + "="*70)
    print("TEST SET RESULTS")
    print("="*70)
    print(f"\nTop-1 Accuracy: {test_acc:.2f}%")
    print(f"Top-5 Accuracy: {test_top5_acc:.2f}%")

    # Classification report
    print("\n" + "-"*70)
    print("DETAILED CLASSIFICATION REPORT")
    print("-"*70 + "\n")

    report = classification_report(
        all_labels, all_preds,
        target_names=class_names,
        digits=3
    )
    print(report)

    # Get report as dict for saving
    report_dict = classification_report(
        all_labels, all_preds,
        target_names=class_names,
        output_dict=True
    )

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate visualizations
    print("\nGenerating visualizations...")
    plot_confusion_matrix(cm, class_names, output_dir)
    plot_per_class_accuracy(report_dict, output_dir)

    # Save results
    save_results(test_acc, test_top5_acc, report_dict, cm, class_names, output_dir)

    # Final summary
    print("\n" + "="*70)
    print("EVALUATION COMPLETE!")
    print("="*70)
    print(f"\nResults saved to: {output_dir}")
    print(f"  - test_results.json (machine-readable)")
    print(f"  - test_report.txt (human-readable)")
    print(f"  - confusion_matrix.png")
    print(f"  - per_class_accuracy.png")
    print("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Hip Implant Classification Model")

    # Data arguments
    parser.add_argument('--data-dir', type=str, default='data/processed',
                       help='Path to processed dataset directory')
    parser.add_argument('--checkpoint', type=str,
                       default='experiments/run_20260130_203457/checkpoints/best.pth',
                       help='Path to model checkpoint')
    parser.add_argument('--output-dir', type=str, default='evaluation_results',
                       help='Output directory for evaluation results')

    # Model arguments
    parser.add_argument('--model-name', type=str, default='swin_tiny_patch4_window7_224',
                       help='Swin model variant')
    parser.add_argument('--in-channels', type=int, default=1,
                       help='Number of input channels (1 for grayscale, 3 for RGB)')
    parser.add_argument('--image-size', type=int, default=224,
                       help='Input image size')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for evaluation')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of data loading workers')

    args = parser.parse_args()

    main(args)
