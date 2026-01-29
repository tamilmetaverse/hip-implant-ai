"""
ConvNeXt for hip implant classification.
"""

import torch
import torch.nn as nn
import timm
from typing import Dict, Optional


class ConvNeXt(nn.Module):
    """
    ConvNeXt classifier for implant identification.
    """

    def __init__(
        self,
        num_classes: int = 50,
        model_name: str = "convnext_base",
        pretrained: bool = True,
        in_channels: int = 1,
        dropout: float = 0.3
    ):
        """
        Initialize ConvNeXt.

        Args:
            num_classes: Number of implant classes
            model_name: timm model name
            pretrained: Whether to use ImageNet pretrained weights
            in_channels: Number of input channels
            dropout: Dropout rate
        """
        super(ConvNeXt, self).__init__()

        self.num_classes = num_classes
        self.in_channels = in_channels

        # Create model
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes,
            drop_rate=dropout,
            in_chans=in_channels
        )

        # If in_channels=1 and pretrained, adapt the weights
        if in_channels == 1 and pretrained:
            self._adapt_first_conv()

    def _adapt_first_conv(self):
        """
        Adapt first convolutional layer from RGB to grayscale.
        """
        try:
            # ConvNeXt uses stem conv
            first_conv = self.model.stem[0]  # First conv in stem

            if hasattr(first_conv, 'weight') and first_conv.weight.shape[1] == 3:
                # Average the RGB weights
                with torch.no_grad():
                    new_weight = first_conv.weight.mean(dim=1, keepdim=True)
                    first_conv.weight = nn.Parameter(new_weight)
        except (AttributeError, IndexError):
            print("Warning: Could not adapt first conv layer")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor [B, C, H, W]

        Returns:
            Class logits [B, num_classes]
        """
        return self.model(x)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features before classification head.

        Args:
            x: Input tensor [B, C, H, W]

        Returns:
            Feature tensor
        """
        return self.model.forward_features(x)


class ConvNeXtWithMixup(nn.Module):
    """
    ConvNeXt with mixup/cutmix support.
    """

    def __init__(
        self,
        num_classes: int = 50,
        model_name: str = "convnext_base",
        pretrained: bool = True,
        in_channels: int = 1,
        dropout: float = 0.3,
        label_smoothing: float = 0.1
    ):
        """
        Initialize ConvNeXt with mixup support.

        Args:
            num_classes: Number of implant classes
            model_name: timm model name
            pretrained: Whether to use ImageNet pretrained weights
            in_channels: Number of input channels
            dropout: Dropout rate
            label_smoothing: Label smoothing factor
        """
        super(ConvNeXtWithMixup, self).__init__()

        self.model = ConvNeXt(
            num_classes=num_classes,
            model_name=model_name,
            pretrained=pretrained,
            in_channels=in_channels,
            dropout=dropout
        )

        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def forward(
        self,
        x: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        mixup_lambda: Optional[float] = None,
        targets_a: Optional[torch.Tensor] = None,
        targets_b: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with optional mixup loss.

        Args:
            x: Input tensor
            targets: Original targets
            mixup_lambda: Mixup lambda parameter
            targets_a: First set of mixed targets
            targets_b: Second set of mixed targets

        Returns:
            Dictionary with logits and optional loss
        """
        logits = self.model(x)

        outputs = {'logits': logits}

        if targets is not None:
            if mixup_lambda is not None and targets_a is not None and targets_b is not None:
                # Mixup loss
                loss_a = self.criterion(logits, targets_a)
                loss_b = self.criterion(logits, targets_b)
                loss = mixup_lambda * loss_a + (1 - mixup_lambda) * loss_b
            else:
                # Standard loss
                loss = self.criterion(logits, targets)

            outputs['loss'] = loss

        return outputs


def build_convnext_model(config: dict) -> nn.Module:
    """
    Build ConvNeXt from configuration.

    Args:
        config: Model configuration dictionary

    Returns:
        ConvNeXt model
    """
    num_classes = config.get('num_classes', 50)
    variant = config.get('variant', 'convnext_base')
    pretrained = config.get('pretrained', True)
    in_channels = config.get('in_channels', 1)
    dropout = config.get('dropout', 0.3)
    label_smoothing = config.get('label_smoothing', 0.1)

    model = ConvNeXtWithMixup(
        num_classes=num_classes,
        model_name=variant,
        pretrained=pretrained,
        in_channels=in_channels,
        dropout=dropout,
        label_smoothing=label_smoothing
    )

    return model
