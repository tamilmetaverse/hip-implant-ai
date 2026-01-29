"""
Segmentation inference module.
"""

import torch
import torch.nn as nn
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, Tuple, Optional
import yaml

import sys
sys.path.append(str(Path(__file__).parent.parent))

from models.segmentation.segformer import build_segmentation_model
from models.segmentation.mask2former import build_mask2former_model
from utils.preprocessing import ImagePreprocessor, save_image


class SegmentationInference:
    """
    Inference pipeline for segmentation models.
    """

    def __init__(
        self,
        checkpoint_path: str,
        config_path: Optional[str] = None,
        device: Optional[torch.device] = None
    ):
        """
        Initialize segmentation inference.

        Args:
            checkpoint_path: Path to model checkpoint
            config_path: Path to configuration file
            device: Device to run inference on
        """
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.config = checkpoint.get('config', None)

        # Load config if not in checkpoint
        if self.config is None and config_path is not None:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)

        if self.config is None:
            raise ValueError("Configuration not found in checkpoint and config_path not provided")

        # Build model
        self.model = self._build_model()
        self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        self.model = self.model.to(self.device)
        self.model.eval()

        # Preprocessor
        self.preprocessor = ImagePreprocessor(
            target_size=tuple(self.config['data']['image_size']),
            normalization='minmax',
            apply_clahe=True,
            apply_filtering=True
        )

        print(f"Loaded segmentation model from {checkpoint_path}")

    def _build_model(self) -> nn.Module:
        """Build segmentation model."""
        model_name = self.config['model']['name']

        if model_name == 'segformer':
            model = build_segmentation_model(self.config['model'])
        elif model_name == 'mask2former':
            model = build_mask2former_model(self.config['model'])
        else:
            raise ValueError(f"Unknown model: {model_name}")

        return model

    @torch.no_grad()
    def predict(
        self,
        image: np.ndarray,
        return_probabilities: bool = False
    ) -> Dict[str, np.ndarray]:
        """
        Predict segmentation mask.

        Args:
            image: Input image (grayscale or RGB)
            return_probabilities: Whether to return class probabilities

        Returns:
            Dictionary with mask and optional probabilities
        """
        # Preprocess
        preprocessed = self.preprocessor.process(image)

        # Convert to tensor
        input_tensor = self.preprocessor.to_tensor(preprocessed, add_channel=True)
        input_tensor = input_tensor.unsqueeze(0).to(self.device)  # Add batch dimension

        # Forward pass
        if hasattr(self.model, 'model'):
            # Model with loss wrapper
            logits = self.model.model(input_tensor)
        else:
            logits = self.model(input_tensor)

        # Get probabilities
        probabilities = torch.softmax(logits, dim=1)

        # Get mask
        mask = torch.argmax(probabilities, dim=1).squeeze(0).cpu().numpy()

        # Resize to original image size
        mask = cv2.resize(
            mask.astype(np.uint8),
            (image.shape[1], image.shape[0]),
            interpolation=cv2.INTER_NEAREST
        )

        results = {'mask': mask}

        if return_probabilities:
            probs = probabilities.squeeze(0).cpu().numpy()
            # Resize probabilities
            resized_probs = []
            for c in range(probs.shape[0]):
                resized_prob = cv2.resize(
                    probs[c],
                    (image.shape[1], image.shape[0]),
                    interpolation=cv2.INTER_LINEAR
                )
                resized_probs.append(resized_prob)
            results['probabilities'] = np.stack(resized_probs, axis=0)

        return results

    def predict_from_file(
        self,
        image_path: str,
        save_path: Optional[str] = None
    ) -> Dict[str, np.ndarray]:
        """
        Predict segmentation from image file.

        Args:
            image_path: Path to input image
            save_path: Optional path to save mask

        Returns:
            Dictionary with mask and optional probabilities
        """
        # Load image
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        if image is None:
            raise FileNotFoundError(f"Failed to load image: {image_path}")

        # Predict
        results = self.predict(image)

        # Save mask if requested
        if save_path:
            mask_visual = (results['mask'] * 255).astype(np.uint8)
            save_image(mask_visual, save_path)
            print(f"Saved segmentation mask to {save_path}")

        return results

    def extract_roi(
        self,
        image: np.ndarray,
        mask: np.ndarray
    ) -> np.ndarray:
        """
        Extract region of interest using mask.

        Args:
            image: Original image
            mask: Segmentation mask

        Returns:
            Masked image
        """
        # Apply mask
        masked_image = image * (mask > 0).astype(np.uint8)
        return masked_image

    def batch_predict(
        self,
        images: list,
        batch_size: int = 8
    ) -> list:
        """
        Predict segmentation for multiple images.

        Args:
            images: List of images
            batch_size: Batch size for inference

        Returns:
            List of masks
        """
        masks = []

        for i in range(0, len(images), batch_size):
            batch_images = images[i:i + batch_size]
            batch_masks = []

            for img in batch_images:
                result = self.predict(img)
                batch_masks.append(result['mask'])

            masks.extend(batch_masks)

        return masks
