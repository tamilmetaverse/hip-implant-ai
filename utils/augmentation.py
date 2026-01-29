"""
Data augmentation utilities for training.
Supports both segmentation and classification tasks.
"""

import torch
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import Dict, Optional, Tuple, Any


class SegmentationAugmentation:
    """
    Augmentation pipeline for segmentation tasks.
    Applies same transforms to image and mask.
    """

    def __init__(
        self,
        image_size: Tuple[int, int] = (512, 512),
        rotation_range: int = 15,
        scale_range: Tuple[float, float] = (0.9, 1.1),
        horizontal_flip: bool = True,
        vertical_flip: bool = False,
        brightness: float = 0.2,
        contrast: float = 0.2,
        is_training: bool = True
    ):
        """
        Initialize segmentation augmentation pipeline.

        Args:
            image_size: Target image size
            rotation_range: Max rotation degrees
            scale_range: Scale factor range
            horizontal_flip: Enable horizontal flip
            vertical_flip: Enable vertical flip
            brightness: Brightness adjustment range
            contrast: Contrast adjustment range
            is_training: Training mode flag
        """
        self.image_size = image_size
        self.is_training = is_training

        if is_training:
            self.transform = A.Compose([
                A.Resize(height=image_size[0], width=image_size[1]),
                A.Rotate(limit=rotation_range, p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.1,
                    scale_limit=(scale_range[0] - 1.0, scale_range[1] - 1.0),
                    rotate_limit=0,
                    p=0.5
                ),
                A.HorizontalFlip(p=0.5 if horizontal_flip else 0.0),
                A.VerticalFlip(p=0.5 if vertical_flip else 0.0),
                A.RandomBrightnessContrast(
                    brightness_limit=brightness,
                    contrast_limit=contrast,
                    p=0.5
                ),
                A.GaussNoise(p=0.2),
                A.Normalize(mean=0.5, std=0.5),
                ToTensorV2()
            ])
        else:
            self.transform = A.Compose([
                A.Resize(height=image_size[0], width=image_size[1]),
                A.Normalize(mean=0.5, std=0.5),
                ToTensorV2()
            ])

    def __call__(
        self,
        image: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Apply augmentation.

        Args:
            image: Input image
            mask: Segmentation mask (optional)

        Returns:
            Dictionary with augmented image and mask
        """
        if mask is not None:
            augmented = self.transform(image=image, mask=mask)
            return {
                'image': augmented['image'],
                'mask': torch.from_numpy(augmented['mask']).long()
            }
        else:
            augmented = self.transform(image=image)
            return {'image': augmented['image']}


class ClassificationAugmentation:
    """
    Augmentation pipeline for classification tasks.
    Includes advanced augmentations like mixup and cutmix.
    """

    def __init__(
        self,
        image_size: Tuple[int, int] = (224, 224),
        rotation_range: int = 20,
        scale_range: Tuple[float, float] = (0.85, 1.15),
        horizontal_flip: bool = True,
        vertical_flip: bool = False,
        brightness: float = 0.3,
        contrast: float = 0.3,
        random_erasing: bool = True,
        is_training: bool = True
    ):
        """
        Initialize classification augmentation pipeline.

        Args:
            image_size: Target image size
            rotation_range: Max rotation degrees
            scale_range: Scale factor range
            horizontal_flip: Enable horizontal flip
            vertical_flip: Enable vertical flip
            brightness: Brightness adjustment range
            contrast: Contrast adjustment range
            random_erasing: Enable random erasing
            is_training: Training mode flag
        """
        self.image_size = image_size
        self.is_training = is_training

        if is_training:
            augmentations = [
                A.Resize(height=image_size[0], width=image_size[1]),
                A.Rotate(limit=rotation_range, p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.1,
                    scale_limit=(scale_range[0] - 1.0, scale_range[1] - 1.0),
                    rotate_limit=0,
                    p=0.5
                ),
                A.HorizontalFlip(p=0.5 if horizontal_flip else 0.0),
                A.VerticalFlip(p=0.5 if vertical_flip else 0.0),
                A.RandomBrightnessContrast(
                    brightness_limit=brightness,
                    contrast_limit=contrast,
                    p=0.5
                ),
                A.OneOf([
                    A.GaussNoise(p=1.0),
                    A.GaussianBlur(p=1.0),
                    A.MotionBlur(p=1.0),
                ], p=0.3),
                A.GridDistortion(p=0.2),
                A.ElasticTransform(p=0.2),
            ]

            if random_erasing:
                augmentations.append(
                    A.CoarseDropout(
                        max_holes=8,
                        max_height=32,
                        max_width=32,
                        p=0.3
                    )
                )

            augmentations.extend([
                A.Normalize(mean=0.5, std=0.5),
                ToTensorV2()
            ])

            self.transform = A.Compose(augmentations)
        else:
            self.transform = A.Compose([
                A.Resize(height=image_size[0], width=image_size[1]),
                A.Normalize(mean=0.5, std=0.5),
                ToTensorV2()
            ])

    def __call__(self, image: np.ndarray) -> torch.Tensor:
        """
        Apply augmentation.

        Args:
            image: Input image

        Returns:
            Augmented image tensor
        """
        augmented = self.transform(image=image)
        return augmented['image']


def mixup_data(
    x: torch.Tensor,
    y: torch.Tensor,
    alpha: float = 0.2
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """
    Apply mixup augmentation.

    Args:
        x: Input images
        y: Target labels
        alpha: Mixup alpha parameter

    Returns:
        Mixed inputs, targets_a, targets_b, lambda
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, lam


def cutmix_data(
    x: torch.Tensor,
    y: torch.Tensor,
    alpha: float = 1.0
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """
    Apply cutmix augmentation.

    Args:
        x: Input images
        y: Target labels
        alpha: Cutmix alpha parameter

    Returns:
        Mixed inputs, targets_a, targets_b, lambda
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)

    y_a, y_b = y, y[index]

    # Generate random bounding box
    _, _, H, W = x.size()
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    # Apply cutmix
    x[:, :, bby1:bby2, bbx1:bbx2] = x[index, :, bby1:bby2, bbx1:bbx2]

    # Adjust lambda based on actual box area
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))

    return x, y_a, y_b, lam


def mixup_criterion(
    criterion: Any,
    pred: torch.Tensor,
    y_a: torch.Tensor,
    y_b: torch.Tensor,
    lam: float
) -> torch.Tensor:
    """
    Compute loss for mixup/cutmix.

    Args:
        criterion: Loss function
        pred: Predictions
        y_a: First set of targets
        y_b: Second set of targets
        lam: Mixing lambda

    Returns:
        Mixed loss
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
