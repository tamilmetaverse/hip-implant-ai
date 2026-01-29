"""
Training pipeline for classification models.
Supports mixup, cutmix, and ensemble training.
"""

import os
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import Dict
import random

import sys
sys.path.append(str(Path(__file__).parent.parent))

from models.classification.swin import build_swin_model
from models.classification.convnext import build_convnext_model
from datasets.xray_dataset import XrayClassificationDataset, MaskedXrayDataset
from utils.augmentation import ClassificationAugmentation, mixup_data, cutmix_data
from utils.metrics import ClassificationMetrics


def set_seed(seed: int):
    """
    Set random seeds for reproducibility.

    Args:
        seed: Random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class ClassificationTrainer:
    """
    Trainer for classification models.
    """

    def __init__(self, config: Dict):
        """
        Initialize trainer.

        Args:
            config: Training configuration
        """
        self.config = config

        # Set seed
        set_seed(config['reproducibility']['seed'])

        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # Build model
        self.model = self._build_model()
        self.model = self.model.to(self.device)

        # Build datasets
        self.train_loader, self.val_loader = self._build_dataloaders()

        # Optimizer
        self.optimizer = self._build_optimizer()

        # Scheduler
        self.scheduler = self._build_scheduler()

        # Metrics
        self.metrics_calculator = ClassificationMetrics()

        # Training state
        self.current_epoch = 0
        self.best_metric = 0.0
        self.epochs_without_improvement = 0

        # Create checkpoint directory
        self.checkpoint_dir = Path(config['checkpoint']['save_dir'])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Mixup/Cutmix parameters
        self.use_mixup = config['augmentation'].get('mixup_alpha', 0.0) > 0
        self.use_cutmix = config['augmentation'].get('cutmix_alpha', 0.0) > 0
        self.mixup_alpha = config['augmentation'].get('mixup_alpha', 0.2)
        self.cutmix_alpha = config['augmentation'].get('cutmix_alpha', 1.0)

    def _build_model(self) -> nn.Module:
        """Build classification model."""
        model_name = self.config['model']['name']

        if model_name == 'swin':
            model = build_swin_model(self.config['model'])
        elif model_name == 'convnext':
            model = build_convnext_model(self.config['model'])
        else:
            raise ValueError(f"Unknown model: {model_name}")

        print(f"Built model: {model_name}")
        return model

    def _build_dataloaders(self):
        """Build train and validation dataloaders."""
        # Augmentations
        train_transform = ClassificationAugmentation(
            image_size=tuple(self.config['data']['image_size']),
            rotation_range=self.config['augmentation']['rotation_range'],
            scale_range=tuple(self.config['augmentation']['scale_range']),
            horizontal_flip=self.config['augmentation']['horizontal_flip'],
            vertical_flip=self.config['augmentation']['vertical_flip'],
            brightness=self.config['augmentation']['brightness'],
            contrast=self.config['augmentation']['contrast'],
            random_erasing=self.config['augmentation']['random_erasing'],
            is_training=True
        )

        val_transform = ClassificationAugmentation(
            image_size=tuple(self.config['data']['image_size']),
            is_training=False
        )

        # Choose dataset type
        if self.config['data'].get('use_masked_images', False):
            dataset_class = MaskedXrayDataset
            extra_args = {'mask_dir': self.config['data'].get('mask_dir', 'data/masks')}
        else:
            dataset_class = XrayClassificationDataset
            extra_args = {}

        # Datasets
        train_dataset = dataset_class(
            image_dir=self.config['data']['train_dir'],
            transform=train_transform,
            **extra_args
        )

        val_dataset = dataset_class(
            image_dir=self.config['data']['val_dir'],
            transform=val_transform,
            **extra_args
        )

        # Dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['data']['batch_size'],
            shuffle=True,
            num_workers=self.config['data']['num_workers'],
            pin_memory=True
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['data']['batch_size'],
            shuffle=False,
            num_workers=self.config['data']['num_workers'],
            pin_memory=True
        )

        print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
        return train_loader, val_loader

    def _build_optimizer(self):
        """Build optimizer."""
        return AdamW(
            self.model.parameters(),
            lr=self.config['training']['learning_rate'],
            weight_decay=self.config['training']['weight_decay']
        )

    def _build_scheduler(self):
        """Build learning rate scheduler."""
        total_steps = self.config['training']['epochs']
        return CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps - self.config['training']['warmup_epochs']
        )

    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch.

        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        num_batches = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1}")

        for batch_idx, (images, targets) in enumerate(pbar):
            images = images.to(self.device)
            targets = targets.to(self.device)

            # Apply mixup or cutmix
            apply_mixup = self.use_mixup and random.random() < 0.5
            apply_cutmix = self.use_cutmix and random.random() < 0.5 and not apply_mixup

            if apply_mixup:
                images, targets_a, targets_b, lam = mixup_data(images, targets, self.mixup_alpha)
                outputs = self.model(images, targets_a, lam, targets_a, targets_b)
            elif apply_cutmix:
                images, targets_a, targets_b, lam = cutmix_data(images, targets, self.cutmix_alpha)
                outputs = self.model(images, targets_a, lam, targets_a, targets_b)
            else:
                outputs = self.model(images, targets)

            loss = outputs['loss']

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            if self.config['training'].get('gradient_clip', 0) > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['training']['gradient_clip']
                )

            self.optimizer.step()

            # Collect predictions
            with torch.no_grad():
                logits = outputs['logits']
                predictions = torch.argmax(logits, dim=1)
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

            total_loss += loss.item()
            num_batches += 1

            # Update progress bar
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        # Compute metrics
        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)
        metrics = self.metrics_calculator.compute(all_predictions, all_targets)

        train_metrics = {
            'train_loss': total_loss / num_batches,
            'train_acc': metrics['accuracy'],
            'train_f1': metrics['f1_score']
        }

        return train_metrics

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """
        Validate the model.

        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        all_probabilities = []
        num_batches = 0

        pbar = tqdm(self.val_loader, desc="Validation")

        for images, targets in pbar:
            images = images.to(self.device)
            targets = targets.to(self.device)

            # Forward pass
            outputs = self.model(images, targets)
            loss = outputs['loss']
            logits = outputs['logits']

            # Get predictions
            probabilities = torch.softmax(logits, dim=1)
            predictions = torch.argmax(probabilities, dim=1)

            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())

            total_loss += loss.item()
            num_batches += 1

            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        # Compute metrics
        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)
        all_probabilities = np.array(all_probabilities)

        metrics = self.metrics_calculator.compute(
            all_predictions,
            all_targets,
            all_probabilities,
            num_classes=self.config['model']['num_classes']
        )

        val_metrics = {
            'val_loss': total_loss / num_batches,
            'val_acc': metrics['accuracy'],
            'val_precision': metrics['precision'],
            'val_recall': metrics['recall'],
            'val_f1': metrics['f1_score']
        }

        if 'top5_accuracy' in metrics:
            val_metrics['val_top5_acc'] = metrics['top5_accuracy']

        return val_metrics

    def save_checkpoint(self, metrics: Dict[str, float], is_best: bool = False):
        """
        Save model checkpoint.

        Args:
            metrics: Current metrics
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'config': self.config
        }

        # Save latest checkpoint
        checkpoint_path = self.checkpoint_dir / 'latest.pth'
        torch.save(checkpoint, checkpoint_path)

        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / 'best.pth'
            torch.save(checkpoint, best_path)
            print(f"Saved best model with {self.config['checkpoint']['monitor']}: {metrics[self.config['checkpoint']['monitor']]:.4f}")

    def train(self):
        """
        Main training loop.
        """
        print("Starting training...")

        for epoch in range(self.config['training']['epochs']):
            self.current_epoch = epoch

            # Train
            train_metrics = self.train_epoch()

            # Validate
            val_metrics = self.validate()

            # Combine metrics
            all_metrics = {**train_metrics, **val_metrics}

            # Print metrics
            print(f"\nEpoch {epoch + 1}/{self.config['training']['epochs']}")
            print(f"Train Loss: {train_metrics['train_loss']:.4f}, Train Acc: {train_metrics['train_acc']:.4f}, Train F1: {train_metrics['train_f1']:.4f}")
            print(f"Val Loss: {val_metrics['val_loss']:.4f}, Val Acc: {val_metrics['val_acc']:.4f}, Val F1: {val_metrics['val_f1']:.4f}")

            # Check if best model
            monitor_metric = val_metrics[self.config['checkpoint']['monitor']]
            is_best = monitor_metric > self.best_metric

            if is_best:
                self.best_metric = monitor_metric
                self.epochs_without_improvement = 0
            else:
                self.epochs_without_improvement += 1

            # Save checkpoint
            self.save_checkpoint(all_metrics, is_best)

            # Learning rate scheduling
            if epoch >= self.config['training']['warmup_epochs']:
                self.scheduler.step()

            # Early stopping
            if self.epochs_without_improvement >= self.config['training']['early_stopping_patience']:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                break

        print(f"\nTraining completed! Best {self.config['checkpoint']['monitor']}: {self.best_metric:.4f}")


def main():
    """Main training function."""
    # Load config
    config_path = Path(__file__).parent.parent / 'configs' / 'classification.yaml'

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Create trainer
    trainer = ClassificationTrainer(config)

    # Train
    trainer.train()


if __name__ == '__main__':
    main()
