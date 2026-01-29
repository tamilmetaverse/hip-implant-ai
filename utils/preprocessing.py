"""
Image preprocessing utilities for hip implant identification.
Implements resize, normalization, filtering, and CLAHE enhancement.
"""

import cv2
import numpy as np
import torch
from typing import Tuple, Optional, Union
from scipy.ndimage import median_filter, gaussian_filter


class ImagePreprocessor:
    """
    Comprehensive image preprocessing pipeline for medical imaging.
    """

    def __init__(
        self,
        target_size: Tuple[int, int] = (512, 512),
        normalization: str = "minmax",  # Options: minmax, zscore
        apply_clahe: bool = True,
        apply_filtering: bool = True,
        filter_type: str = "gaussian"  # Options: gaussian, median
    ):
        """
        Initialize preprocessor.

        Args:
            target_size: Target image dimensions (height, width)
            normalization: Normalization method
            apply_clahe: Whether to apply CLAHE enhancement
            apply_filtering: Whether to apply noise filtering
            filter_type: Type of filter to apply
        """
        self.target_size = target_size
        self.normalization = normalization
        self.apply_clahe = apply_clahe
        self.apply_filtering = apply_filtering
        self.filter_type = filter_type

        # CLAHE configuration
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def resize(self, image: np.ndarray) -> np.ndarray:
        """
        Resize image to target size.

        Args:
            image: Input image

        Returns:
            Resized image
        """
        return cv2.resize(image, self.target_size, interpolation=cv2.INTER_LINEAR)

    def normalize(self, image: np.ndarray) -> np.ndarray:
        """
        Normalize image intensity.

        Args:
            image: Input image

        Returns:
            Normalized image
        """
        if self.normalization == "minmax":
            # Min-max normalization to [0, 1]
            image = image.astype(np.float32)
            min_val = np.min(image)
            max_val = np.max(image)
            if max_val - min_val > 0:
                image = (image - min_val) / (max_val - min_val)
            return image

        elif self.normalization == "zscore":
            # Z-score normalization
            image = image.astype(np.float32)
            mean = np.mean(image)
            std = np.std(image)
            if std > 0:
                image = (image - mean) / std
            return image

        else:
            raise ValueError(f"Unknown normalization method: {self.normalization}")

    def apply_filter(self, image: np.ndarray) -> np.ndarray:
        """
        Apply noise filtering.

        Args:
            image: Input image

        Returns:
            Filtered image
        """
        if self.filter_type == "gaussian":
            return gaussian_filter(image, sigma=1.0)
        elif self.filter_type == "median":
            return median_filter(image, size=3)
        else:
            raise ValueError(f"Unknown filter type: {self.filter_type}")

    def apply_clahe_enhancement(self, image: np.ndarray) -> np.ndarray:
        """
        Apply CLAHE contrast enhancement.

        Args:
            image: Input image (should be uint8)

        Returns:
            Enhanced image
        """
        # Convert to uint8 if needed
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)

        # Apply CLAHE
        enhanced = self.clahe.apply(image)

        # Convert back to float
        return enhanced.astype(np.float32) / 255.0

    def process(self, image: np.ndarray) -> np.ndarray:
        """
        Complete preprocessing pipeline.

        Args:
            image: Input image (grayscale or RGB)

        Returns:
            Preprocessed image
        """
        # Handle RGB images (convert to grayscale for medical imaging)
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Resize
        image = self.resize(image)

        # Apply filtering
        if self.apply_filtering:
            image = self.apply_filter(image)

        # Normalize
        image = self.normalize(image)

        # Apply CLAHE
        if self.apply_clahe:
            image = self.apply_clahe_enhancement(image)

        return image

    def process_batch(self, images: np.ndarray) -> np.ndarray:
        """
        Process a batch of images.

        Args:
            images: Batch of images [B, H, W] or [B, H, W, C]

        Returns:
            Preprocessed batch
        """
        processed = []
        for img in images:
            processed.append(self.process(img))
        return np.array(processed)

    def to_tensor(self, image: np.ndarray, add_channel: bool = True) -> torch.Tensor:
        """
        Convert numpy array to PyTorch tensor.

        Args:
            image: Preprocessed image
            add_channel: Whether to add channel dimension

        Returns:
            PyTorch tensor
        """
        tensor = torch.from_numpy(image).float()
        if add_channel and len(tensor.shape) == 2:
            tensor = tensor.unsqueeze(0)  # Add channel dimension
        return tensor


def load_image(image_path: str, color_mode: str = "grayscale") -> np.ndarray:
    """
    Load image from file.

    Args:
        image_path: Path to image file
        color_mode: Color mode (grayscale or rgb)

    Returns:
        Loaded image as numpy array
    """
    if color_mode == "grayscale":
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    elif color_mode == "rgb":
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        raise ValueError(f"Unknown color mode: {color_mode}")

    if image is None:
        raise FileNotFoundError(f"Failed to load image: {image_path}")

    return image


def save_image(image: np.ndarray, save_path: str) -> None:
    """
    Save image to file.

    Args:
        image: Image to save
        save_path: Output path
    """
    # Convert to uint8 if needed
    if image.dtype != np.uint8:
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)

    cv2.imwrite(save_path, image)


def extract_roi_from_mask(
    image: np.ndarray,
    mask: np.ndarray,
    background_value: int = 0
) -> np.ndarray:
    """
    Extract region of interest using mask.

    Args:
        image: Original image
        mask: Binary mask
        background_value: Value for background pixels

    Returns:
        Masked image
    """
    # Ensure mask is binary
    mask = (mask > 0).astype(np.uint8)

    # Apply mask
    if len(image.shape) == 2:
        masked = image * mask + background_value * (1 - mask)
    else:
        masked = image * mask[..., np.newaxis] + background_value * (1 - mask[..., np.newaxis])

    return masked


def compute_bbox_from_mask(mask: np.ndarray) -> Tuple[int, int, int, int]:
    """
    Compute bounding box from binary mask.

    Args:
        mask: Binary mask

    Returns:
        Bounding box coordinates (x_min, y_min, x_max, y_max)
    """
    coords = np.argwhere(mask > 0)
    if len(coords) == 0:
        return (0, 0, mask.shape[1], mask.shape[0])

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    return (int(x_min), int(y_min), int(x_max), int(y_max))
