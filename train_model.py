#!/usr/bin/env python3
"""
Training script for Hip Implant Classification using Swin Transformer
Handles class imbalance, includes data augmentation, and saves checkpoints
"""

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import numpy as np
from tqdm import tqdm
import argparse
from datetime import datetime

# Import project modules
from datasets.multi_ext_dataset import MultiExtensionDataset
from models.classification.swin import SwinTransformer
from utils.augmentation import ClassificationAugmentation

# Configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
RANDOM_SEED = 42

# Set seeds for reproducibility
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(RANDOM_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def calculate_class_weights(dataset):
    """Calculate class weights for imbalanced dataset"""
    class_counts = {}
    for _, label in dataset.samples:
        class_counts[label] = class_counts.get(label, 0) + 1

    total_samples = len(dataset)
    num_classes = len(class_counts)

    # Calculate weights using inverse frequency
    weights = []
    for class_idx in range(num_classes):
        count = class_counts.get(class_idx, 1)
        weight = total_samples / (num_classes * count)
        weights.append(weight)

    return torch.FloatTensor(weights)


def train_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]")

    for batch_idx, (images, labels) in enumerate(progress_bar):
        images = images.to(device)
        labels = labels.to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Statistics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        # Update progress bar
        progress_bar.set_postfix({
            'loss': running_loss / (batch_idx + 1),
            'acc': 100. * correct / total
        })

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100. * correct / total

    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device, epoch):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    top5_correct = 0

    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch} [Val]")

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(progress_bar):
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # Top-5 accuracy
            _, top5_pred = outputs.topk(min(5, outputs.size(1)), 1, True, True)
            top5_correct += top5_pred.eq(labels.view(-1, 1).expand_as(top5_pred)).sum().item()

            # Store for confusion matrix
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            # Update progress bar
            progress_bar.set_postfix({
                'loss': running_loss / (batch_idx + 1),
                'acc': 100. * correct / total
            })

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100. * correct / total
    epoch_top5_acc = 100. * top5_correct / total

    return epoch_loss, epoch_acc, epoch_top5_acc, all_preds, all_labels


def save_checkpoint(model, optimizer, epoch, best_acc, checkpoint_dir, is_best=False):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_acc': best_acc,
    }

    # Save last checkpoint
    checkpoint_path = checkpoint_dir / 'last.pth'
    torch.save(checkpoint, checkpoint_path)

    # Save best checkpoint
    if is_best:
        best_path = checkpoint_dir / 'best.pth'
        torch.save(checkpoint, best_path)
        print(f"[OK] Best model saved with accuracy: {best_acc:.2f}%")


def load_dataset_info(data_dir):
    """Load dataset information"""
    info_file = Path(data_dir) / 'dataset_info.json'
    with open(info_file, 'r') as f:
        return json.load(f)


def main(args):
    print("="*70)
    print("HIP IMPLANT AI - TRAINING")
    print("="*70)
    print(f"\nDevice: {DEVICE}")
    print(f"Random Seed: {RANDOM_SEED}")

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"run_{timestamp}"
    checkpoint_dir = output_dir / "checkpoints"
    log_dir = output_dir / "logs"

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    # TensorBoard
    writer = SummaryWriter(log_dir)

    # Load dataset info
    dataset_info = load_dataset_info(args.data_dir)
    num_classes = dataset_info['total_classes']
    class_names = dataset_info['class_names']

    print(f"\nDataset: {args.data_dir}")
    print(f"Number of classes: {num_classes}")
    print(f"Total images: {dataset_info['total_images']}")
    print(f"  Train: {dataset_info['splits']['train']}")
    print(f"  Val: {dataset_info['splits']['val']}")
    print(f"  Test: {dataset_info['splits']['test']}")

    # Data augmentation
    print("\nCreating data augmentation pipelines...")
    train_transform = ClassificationAugmentation(
        image_size=(args.image_size, args.image_size),
        is_training=True
    )
    val_transform = ClassificationAugmentation(
        image_size=(args.image_size, args.image_size),
        is_training=False
    )

    # Create datasets
    print("Loading datasets...")
    train_dataset = MultiExtensionDataset(
        image_dir=Path(args.data_dir) / 'train',
        transform=train_transform,
        use_folders=True
    )

    val_dataset = MultiExtensionDataset(
        image_dir=Path(args.data_dir) / 'val',
        transform=val_transform,
        use_folders=True
    )

    print(f"Train dataset: {len(train_dataset)} images")
    print(f"Val dataset: {len(val_dataset)} images")

    # Calculate class weights
    print("\nCalculating class weights for imbalanced dataset...")
    class_weights = calculate_class_weights(train_dataset)
    print("Class weights:")
    for idx, (name, weight) in enumerate(zip(class_names, class_weights)):
        print(f"  {name}: {weight:.3f}")

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )

    # Create model
    print(f"\nCreating Swin Transformer model...")
    print(f"Model: {args.model_name}")
    print(f"Pretrained: {args.pretrained}")
    print(f"Input channels: {args.in_channels}")

    model = SwinTransformer(
        num_classes=num_classes,
        model_name=args.model_name,
        pretrained=args.pretrained,
        in_channels=args.in_channels,
        dropout=args.dropout
    )
    model = model.to(DEVICE)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Loss function with class weights
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(DEVICE))

    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.learning_rate / 100
    )

    # Training loop
    print("\n" + "="*70)
    print("STARTING TRAINING")
    print("="*70)

    best_val_acc = 0.0
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        print("-" * 70)

        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, DEVICE, epoch
        )

        # Validate
        val_loss, val_acc, val_top5_acc, val_preds, val_labels = validate(
            model, val_loader, criterion, DEVICE, epoch
        )

        # Learning rate step
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        # Log metrics
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)
        writer.add_scalar('Accuracy/val_top5', val_top5_acc, epoch)
        writer.add_scalar('Learning_rate', current_lr, epoch)

        # Print epoch summary
        print(f"\nEpoch {epoch} Summary:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | Val Top-5: {val_top5_acc:.2f}%")
        print(f"  Learning Rate: {current_lr:.6f}")

        # Save checkpoint
        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc
            patience_counter = 0
        else:
            patience_counter += 1

        save_checkpoint(model, optimizer, epoch, best_val_acc, checkpoint_dir, is_best)

        # Early stopping
        if patience_counter >= args.patience:
            print(f"\nEarly stopping triggered after {epoch} epochs")
            print(f"No improvement for {args.patience} consecutive epochs")
            break

    # Training complete
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print(f"\nBest validation accuracy: {best_val_acc:.2f}%")
    print(f"Checkpoints saved to: {checkpoint_dir}")
    print(f"TensorBoard logs saved to: {log_dir}")
    print(f"\nTo view training logs, run:")
    print(f"  tensorboard --logdir {log_dir}")

    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Hip Implant Classification Model")

    # Data arguments
    parser.add_argument('--data-dir', type=str, default='data/processed',
                       help='Path to processed dataset directory')
    parser.add_argument('--output-dir', type=str, default='experiments',
                       help='Output directory for checkpoints and logs')

    # Model arguments
    parser.add_argument('--model-name', type=str, default='swin_tiny_patch4_window7_224',
                       help='Swin model variant (tiny, small, base)')
    parser.add_argument('--pretrained', action='store_true', default=True,
                       help='Use ImageNet pretrained weights')
    parser.add_argument('--in-channels', type=int, default=1,
                       help='Number of input channels (1 for grayscale, 3 for RGB)')
    parser.add_argument('--dropout', type=float, default=0.3,
                       help='Dropout rate')

    # Training arguments
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=1e-4,
                       help='Initial learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                       help='Weight decay')
    parser.add_argument('--image-size', type=int, default=224,
                       help='Input image size')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of data loading workers')
    parser.add_argument('--patience', type=int, default=15,
                       help='Early stopping patience')

    args = parser.parse_args()

    main(args)
