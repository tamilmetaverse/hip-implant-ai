"""
SegFormer model for medical image segmentation.
Transformer-based encoder-decoder architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple
from transformers import SegformerForSemanticSegmentation, SegformerConfig


class SegFormer(nn.Module):
    """
    SegFormer model for hip implant/bone segmentation.
    """

    def __init__(
        self,
        num_classes: int = 2,
        encoder_name: str = "nvidia/mit-b3",
        pretrained: bool = True,
        in_channels: int = 1
    ):
        """
        Initialize SegFormer model.

        Args:
            num_classes: Number of segmentation classes
            encoder_name: Pretrained encoder name
            pretrained: Whether to use pretrained weights
            in_channels: Number of input channels
        """
        super(SegFormer, self).__init__()

        self.num_classes = num_classes
        self.in_channels = in_channels

        if pretrained:
            # Load pretrained model
            self.model = SegformerForSemanticSegmentation.from_pretrained(
                encoder_name,
                num_labels=num_classes,
                ignore_mismatched_sizes=True
            )
        else:
            # Create model from config
            config = SegformerConfig(
                num_labels=num_classes
            )
            self.model = SegformerForSemanticSegmentation(config)

        # Modify first conv layer if in_channels != 3
        if in_channels != 3:
            self._modify_first_conv(in_channels)

    def _modify_first_conv(self, in_channels: int):
        """
        Modify first convolutional layer for grayscale input.

        Args:
            in_channels: Number of input channels
        """
        # Get the first patch embedding layer
        first_conv = self.model.segformer.encoder.patch_embeddings[0].proj

        # Create new conv layer
        new_conv = nn.Conv2d(
            in_channels,
            first_conv.out_channels,
            kernel_size=first_conv.kernel_size,
            stride=first_conv.stride,
            padding=first_conv.padding
        )

        # Initialize weights
        if in_channels == 1:
            # Average RGB weights for grayscale
            with torch.no_grad():
                new_conv.weight[:, 0:1, :, :] = first_conv.weight.mean(dim=1, keepdim=True)
                new_conv.bias = first_conv.bias

        # Replace
        self.model.segformer.encoder.patch_embeddings[0].proj = new_conv

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor [B, C, H, W]

        Returns:
            Segmentation logits [B, num_classes, H, W]
        """
        outputs = self.model(pixel_values=x)
        logits = outputs.logits

        # Upsample to input size
        logits = F.interpolate(
            logits,
            size=x.shape[2:],
            mode='bilinear',
            align_corners=False
        )

        return logits


class SegFormerWithLoss(nn.Module):
    """
    SegFormer model with integrated loss computation.
    """

    def __init__(
        self,
        num_classes: int = 2,
        encoder_name: str = "nvidia/mit-b3",
        pretrained: bool = True,
        in_channels: int = 1,
        dice_weight: float = 0.5,
        ce_weight: float = 0.5
    ):
        """
        Initialize SegFormer with loss.

        Args:
            num_classes: Number of segmentation classes
            encoder_name: Pretrained encoder name
            pretrained: Whether to use pretrained weights
            in_channels: Number of input channels
            dice_weight: Weight for Dice loss
            ce_weight: Weight for Cross Entropy loss
        """
        super(SegFormerWithLoss, self).__init__()

        self.model = SegFormer(num_classes, encoder_name, pretrained, in_channels)
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight

        self.ce_loss = nn.CrossEntropyLoss()
        self.dice_loss = DiceLoss(num_classes=num_classes)

    def forward(
        self,
        x: torch.Tensor,
        targets: torch.Tensor = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with loss computation.

        Args:
            x: Input tensor [B, C, H, W]
            targets: Target masks [B, H, W] (optional)

        Returns:
            Dictionary with logits and loss (if targets provided)
        """
        logits = self.model(x)

        outputs = {'logits': logits}

        if targets is not None:
            # Compute losses
            ce_loss = self.ce_loss(logits, targets)
            dice_loss = self.dice_loss(logits, targets)

            # Combined loss
            total_loss = self.dice_weight * dice_loss + self.ce_weight * ce_loss

            outputs['loss'] = total_loss
            outputs['ce_loss'] = ce_loss
            outputs['dice_loss'] = dice_loss

        return outputs


class DiceLoss(nn.Module):
    """
    Dice loss for segmentation.
    """

    def __init__(self, num_classes: int = 2, smooth: float = 1e-6):
        """
        Initialize Dice loss.

        Args:
            num_classes: Number of classes
            smooth: Smoothing factor
        """
        super(DiceLoss, self).__init__()
        self.num_classes = num_classes
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute Dice loss.

        Args:
            logits: Model predictions [B, C, H, W]
            targets: Ground truth [B, H, W]

        Returns:
            Dice loss
        """
        # Convert logits to probabilities
        probs = F.softmax(logits, dim=1)

        # One-hot encode targets
        targets_one_hot = F.one_hot(targets, num_classes=self.num_classes)
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()

        # Compute Dice coefficient
        intersection = (probs * targets_one_hot).sum(dim=(2, 3))
        union = probs.sum(dim=(2, 3)) + targets_one_hot.sum(dim=(2, 3))

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)

        # Return Dice loss (1 - Dice)
        return 1.0 - dice.mean()


class FocalLoss(nn.Module):
    """
    Focal loss for handling class imbalance.
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        """
        Initialize Focal loss.

        Args:
            alpha: Weighting factor
            gamma: Focusing parameter
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute Focal loss.

        Args:
            logits: Model predictions [B, C, H, W]
            targets: Ground truth [B, H, W]

        Returns:
            Focal loss
        """
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


def build_segmentation_model(config: dict) -> nn.Module:
    """
    Build segmentation model from configuration.

    Args:
        config: Model configuration dictionary

    Returns:
        Segmentation model
    """
    model_name = config.get('name', 'segformer')
    num_classes = config.get('num_classes', 2)
    encoder = config.get('encoder', 'nvidia/mit-b3')
    pretrained = config.get('pretrained', True)
    in_channels = config.get('in_channels', 1)

    if model_name == 'segformer':
        model = SegFormerWithLoss(
            num_classes=num_classes,
            encoder_name=encoder,
            pretrained=pretrained,
            in_channels=in_channels
        )
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    return model
