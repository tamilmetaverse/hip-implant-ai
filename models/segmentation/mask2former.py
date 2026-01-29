"""
Mask2Former model for medical image segmentation.
Universal segmentation architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerConfig


class Mask2Former(nn.Module):
    """
    Mask2Former model for hip implant/bone segmentation.
    """

    def __init__(
        self,
        num_classes: int = 2,
        backbone: str = "facebook/mask2former-swin-small-ade-semantic",
        pretrained: bool = True,
        in_channels: int = 1
    ):
        """
        Initialize Mask2Former model.

        Args:
            num_classes: Number of segmentation classes
            backbone: Pretrained backbone name
            pretrained: Whether to use pretrained weights
            in_channels: Number of input channels
        """
        super(Mask2Former, self).__init__()

        self.num_classes = num_classes
        self.in_channels = in_channels

        if pretrained:
            # Load pretrained model
            self.model = Mask2FormerForUniversalSegmentation.from_pretrained(
                backbone,
                num_labels=num_classes,
                ignore_mismatched_sizes=True
            )
        else:
            # Create model from config
            config = Mask2FormerConfig(
                num_labels=num_classes
            )
            self.model = Mask2FormerForUniversalSegmentation(config)

        # Modify first conv layer if in_channels != 3
        if in_channels != 3:
            self._modify_first_conv(in_channels)

    def _modify_first_conv(self, in_channels: int):
        """
        Modify first convolutional layer for grayscale input.

        Args:
            in_channels: Number of input channels
        """
        try:
            # Access the pixel-level module's first conv
            first_conv = self.model.model.pixel_level_module.encoder.embeddings.patch_embeddings.projection

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
                with torch.no_grad():
                    new_conv.weight[:, 0:1, :, :] = first_conv.weight.mean(dim=1, keepdim=True)
                    if first_conv.bias is not None:
                        new_conv.bias = first_conv.bias

            # Replace
            self.model.model.pixel_level_module.encoder.embeddings.patch_embeddings.projection = new_conv
        except AttributeError:
            print("Warning: Could not modify first conv layer for Mask2Former")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor [B, C, H, W]

        Returns:
            Segmentation logits [B, num_classes, H, W]
        """
        # Get original size
        orig_h, orig_w = x.shape[2:]

        # Forward pass
        outputs = self.model(pixel_values=x)

        # Get semantic segmentation
        # Mask2Former outputs class queries, we need to convert to dense prediction
        class_queries_logits = outputs.class_queries_logits  # [B, num_queries, num_classes]
        masks_queries_logits = outputs.masks_queries_logits  # [B, num_queries, H, W]

        # Upsample masks to original size
        masks_queries_logits = F.interpolate(
            masks_queries_logits,
            size=(orig_h, orig_w),
            mode='bilinear',
            align_corners=False
        )

        # Combine queries into dense prediction
        # Take the class with maximum probability for each query
        class_probs = F.softmax(class_queries_logits, dim=-1)  # [B, num_queries, num_classes]

        # Create dense segmentation map
        batch_size = x.shape[0]
        logits = torch.zeros(batch_size, self.num_classes, orig_h, orig_w, device=x.device)

        for b in range(batch_size):
            for q in range(class_queries_logits.shape[1]):
                # Get predicted class for this query
                query_class_probs = class_probs[b, q]  # [num_classes]
                query_mask = torch.sigmoid(masks_queries_logits[b, q])  # [H, W]

                # Add weighted contribution to each class
                for c in range(self.num_classes):
                    logits[b, c] += query_class_probs[c] * query_mask

        return logits


class Mask2FormerWithLoss(nn.Module):
    """
    Mask2Former model with integrated loss computation.
    """

    def __init__(
        self,
        num_classes: int = 2,
        backbone: str = "facebook/mask2former-swin-small-ade-semantic",
        pretrained: bool = True,
        in_channels: int = 1,
        dice_weight: float = 0.5,
        ce_weight: float = 0.5
    ):
        """
        Initialize Mask2Former with loss.

        Args:
            num_classes: Number of segmentation classes
            backbone: Pretrained backbone name
            pretrained: Whether to use pretrained weights
            in_channels: Number of input channels
            dice_weight: Weight for Dice loss
            ce_weight: Weight for Cross Entropy loss
        """
        super(Mask2FormerWithLoss, self).__init__()

        self.model = Mask2Former(num_classes, backbone, pretrained, in_channels)
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight

        self.ce_loss = nn.CrossEntropyLoss()

        # Import DiceLoss from segformer module
        from .segformer import DiceLoss
        self.dice_loss = DiceLoss(num_classes=num_classes)

    def forward(
        self,
        x: torch.Tensor,
        targets: Optional[torch.Tensor] = None
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


def build_mask2former_model(config: dict) -> nn.Module:
    """
    Build Mask2Former model from configuration.

    Args:
        config: Model configuration dictionary

    Returns:
        Mask2Former model
    """
    num_classes = config.get('num_classes', 2)
    backbone = config.get('backbone', 'facebook/mask2former-swin-small-ade-semantic')
    pretrained = config.get('pretrained', True)
    in_channels = config.get('in_channels', 1)

    model = Mask2FormerWithLoss(
        num_classes=num_classes,
        backbone=backbone,
        pretrained=pretrained,
        in_channels=in_channels
    )

    return model
