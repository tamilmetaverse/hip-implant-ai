"""
Evaluation metrics for segmentation and classification tasks.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    roc_auc_score,
    top_k_accuracy_score
)


class DiceScore:
    """
    Dice coefficient for segmentation evaluation.
    """

    def __init__(self, num_classes: int = 2, smooth: float = 1e-6):
        """
        Initialize Dice score calculator.

        Args:
            num_classes: Number of segmentation classes
            smooth: Smoothing factor to avoid division by zero
        """
        self.num_classes = num_classes
        self.smooth = smooth

    def __call__(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        reduce: bool = True
    ) -> torch.Tensor:
        """
        Compute Dice score.

        Args:
            pred: Predicted segmentation [B, C, H, W] or [B, H, W]
            target: Ground truth segmentation [B, H, W]
            reduce: Whether to average across batch

        Returns:
            Dice score
        """
        # Convert predictions to class indices if needed
        if len(pred.shape) == 4:
            pred = torch.argmax(pred, dim=1)

        # One-hot encode
        pred_one_hot = F.one_hot(pred, num_classes=self.num_classes).permute(0, 3, 1, 2).float()
        target_one_hot = F.one_hot(target, num_classes=self.num_classes).permute(0, 3, 1, 2).float()

        # Compute Dice
        intersection = (pred_one_hot * target_one_hot).sum(dim=(2, 3))
        union = pred_one_hot.sum(dim=(2, 3)) + target_one_hot.sum(dim=(2, 3))

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)

        if reduce:
            return dice.mean()
        return dice


class IoUScore:
    """
    Intersection over Union (IoU) for segmentation evaluation.
    """

    def __init__(self, num_classes: int = 2, smooth: float = 1e-6):
        """
        Initialize IoU score calculator.

        Args:
            num_classes: Number of segmentation classes
            smooth: Smoothing factor
        """
        self.num_classes = num_classes
        self.smooth = smooth

    def __call__(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        reduce: bool = True
    ) -> torch.Tensor:
        """
        Compute IoU score.

        Args:
            pred: Predicted segmentation
            target: Ground truth segmentation
            reduce: Whether to average across batch

        Returns:
            IoU score
        """
        # Convert predictions to class indices if needed
        if len(pred.shape) == 4:
            pred = torch.argmax(pred, dim=1)

        # One-hot encode
        pred_one_hot = F.one_hot(pred, num_classes=self.num_classes).permute(0, 3, 1, 2).float()
        target_one_hot = F.one_hot(target, num_classes=self.num_classes).permute(0, 3, 1, 2).float()

        # Compute IoU
        intersection = (pred_one_hot * target_one_hot).sum(dim=(2, 3))
        union = pred_one_hot.sum(dim=(2, 3)) + target_one_hot.sum(dim=(2, 3)) - intersection

        iou = (intersection + self.smooth) / (union + self.smooth)

        if reduce:
            return iou.mean()
        return iou


class ClassificationMetrics:
    """
    Comprehensive classification metrics.
    """

    @staticmethod
    def compute(
        predictions: np.ndarray,
        targets: np.ndarray,
        probabilities: Optional[np.ndarray] = None,
        num_classes: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Compute classification metrics.

        Args:
            predictions: Predicted class labels
            targets: Ground truth labels
            probabilities: Class probabilities (for AUC)
            num_classes: Number of classes

        Returns:
            Dictionary of metrics
        """
        metrics = {}

        # Accuracy
        metrics['accuracy'] = accuracy_score(targets, predictions)

        # Precision, Recall, F1
        precision, recall, f1, _ = precision_recall_fscore_support(
            targets,
            predictions,
            average='weighted',
            zero_division=0
        )
        metrics['precision'] = precision
        metrics['recall'] = recall
        metrics['f1_score'] = f1

        # Per-class metrics
        precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(
            targets,
            predictions,
            average=None,
            zero_division=0
        )
        metrics['precision_per_class'] = precision_per_class
        metrics['recall_per_class'] = recall_per_class
        metrics['f1_per_class'] = f1_per_class

        # Confusion matrix
        metrics['confusion_matrix'] = confusion_matrix(targets, predictions)

        # Top-5 accuracy (if applicable)
        if probabilities is not None and num_classes is not None and num_classes >= 5:
            metrics['top5_accuracy'] = top_k_accuracy_score(
                targets,
                probabilities,
                k=5,
                labels=np.arange(num_classes)
            )

        # AUC (if probabilities provided and binary/multiclass)
        if probabilities is not None and num_classes is not None:
            try:
                if num_classes == 2:
                    metrics['auc'] = roc_auc_score(targets, probabilities[:, 1])
                else:
                    metrics['auc'] = roc_auc_score(
                        targets,
                        probabilities,
                        multi_class='ovr',
                        average='weighted'
                    )
            except ValueError:
                # Not all classes present in targets
                pass

        return metrics


class SegmentationMetrics:
    """
    Comprehensive segmentation metrics.
    """

    def __init__(self, num_classes: int = 2):
        """
        Initialize segmentation metrics calculator.

        Args:
            num_classes: Number of segmentation classes
        """
        self.num_classes = num_classes
        self.dice_score = DiceScore(num_classes)
        self.iou_score = IoUScore(num_classes)

    def compute(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> Dict[str, float]:
        """
        Compute segmentation metrics.

        Args:
            predictions: Predicted segmentation
            targets: Ground truth segmentation

        Returns:
            Dictionary of metrics
        """
        metrics = {}

        # Dice score
        metrics['dice'] = self.dice_score(predictions, targets).item()

        # IoU score
        metrics['iou'] = self.iou_score(predictions, targets).item()

        # Pixel accuracy
        if len(predictions.shape) == 4:
            pred_classes = torch.argmax(predictions, dim=1)
        else:
            pred_classes = predictions

        correct = (pred_classes == targets).sum().item()
        total = targets.numel()
        metrics['pixel_accuracy'] = correct / total

        # Per-class metrics
        dice_per_class = self.dice_score(predictions, targets, reduce=False).mean(dim=0)
        iou_per_class = self.iou_score(predictions, targets, reduce=False).mean(dim=0)

        for i in range(self.num_classes):
            metrics[f'dice_class_{i}'] = dice_per_class[i].item()
            metrics[f'iou_class_{i}'] = iou_per_class[i].item()

        return metrics


def compute_confidence_metrics(
    probabilities: np.ndarray,
    predictions: np.ndarray,
    targets: np.ndarray
) -> Dict[str, float]:
    """
    Compute confidence-related metrics.

    Args:
        probabilities: Predicted probabilities
        predictions: Predicted class labels
        targets: Ground truth labels

    Returns:
        Dictionary of confidence metrics
    """
    metrics = {}

    # Maximum probability (confidence)
    max_probs = np.max(probabilities, axis=1)
    metrics['mean_confidence'] = np.mean(max_probs)
    metrics['median_confidence'] = np.median(max_probs)

    # Confidence on correct predictions
    correct_mask = (predictions == targets)
    if correct_mask.sum() > 0:
        metrics['mean_confidence_correct'] = np.mean(max_probs[correct_mask])

    # Confidence on incorrect predictions
    incorrect_mask = ~correct_mask
    if incorrect_mask.sum() > 0:
        metrics['mean_confidence_incorrect'] = np.mean(max_probs[incorrect_mask])

    # Expected Calibration Error (ECE)
    metrics['ece'] = compute_ece(probabilities, predictions, targets)

    return metrics


def compute_ece(
    probabilities: np.ndarray,
    predictions: np.ndarray,
    targets: np.ndarray,
    n_bins: int = 10
) -> float:
    """
    Compute Expected Calibration Error.

    Args:
        probabilities: Predicted probabilities
        predictions: Predicted class labels
        targets: Ground truth labels
        n_bins: Number of bins for calibration

    Returns:
        ECE score
    """
    confidences = np.max(probabilities, axis=1)
    accuracies = (predictions == targets).astype(float)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = in_bin.mean()

        if prop_in_bin > 0:
            accuracy_in_bin = accuracies[in_bin].mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return ece
