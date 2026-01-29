"""
X-ray dataset for hip implant segmentation and classification.
"""

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Optional, Callable, Tuple, List
from pathlib import Path


class XraySegmentationDataset(Dataset):
    """
    Dataset for X-ray segmentation tasks (implant or bone segmentation).
    """

    def __init__(
        self,
        image_dir: str,
        mask_dir: str,
        transform: Optional[Callable] = None,
        image_ext: str = '.png'
    ):
        """
        Initialize X-ray segmentation dataset.

        Args:
            image_dir: Directory containing X-ray images
            mask_dir: Directory containing segmentation masks
            transform: Augmentation transform
            image_ext: Image file extension
        """
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.transform = transform
        self.image_ext = image_ext

        # Get list of image files
        self.image_files = sorted([
            f for f in os.listdir(image_dir)
            if f.endswith(image_ext)
        ])

        if len(self.image_files) == 0:
            raise ValueError(f"No images found in {image_dir} with extension {image_ext}")

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get image and mask pair.

        Args:
            idx: Index

        Returns:
            Image tensor and mask tensor
        """
        # Load image
        image_name = self.image_files[idx]
        image_path = self.image_dir / image_name
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

        if image is None:
            raise FileNotFoundError(f"Failed to load image: {image_path}")

        # Load mask
        mask_name = image_name.replace(self.image_ext, '_mask' + self.image_ext)
        mask_path = self.mask_dir / mask_name
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        if mask is None:
            raise FileNotFoundError(f"Failed to load mask: {mask_path}")

        # Convert to binary mask
        mask = (mask > 127).astype(np.uint8)

        # Apply transforms
        if self.transform:
            augmented = self.transform(image, mask)
            image = augmented['image']
            mask = augmented['mask']

        return image, mask

    def get_image_name(self, idx: int) -> str:
        """Get image filename at index."""
        return self.image_files[idx]


class XrayClassificationDataset(Dataset):
    """
    Dataset for X-ray classification tasks (implant identification).
    """

    def __init__(
        self,
        image_dir: str,
        labels_file: Optional[str] = None,
        class_to_idx: Optional[dict] = None,
        transform: Optional[Callable] = None,
        image_ext: str = '.png',
        use_folders: bool = True
    ):
        """
        Initialize X-ray classification dataset.

        Args:
            image_dir: Directory containing X-ray images
            labels_file: Path to CSV file with labels (if not using folder structure)
            class_to_idx: Mapping from class names to indices
            transform: Augmentation transform
            image_ext: Image file extension
            use_folders: Whether images are organized in class folders
        """
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.image_ext = image_ext
        self.use_folders = use_folders

        # Build dataset
        if use_folders:
            self.samples, self.class_to_idx = self._load_from_folders()
        else:
            if labels_file is None:
                raise ValueError("labels_file must be provided when use_folders=False")
            self.samples, self.class_to_idx = self._load_from_csv(labels_file, class_to_idx)

        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        self.num_classes = len(self.class_to_idx)

    def _load_from_folders(self) -> Tuple[List[Tuple[Path, int]], dict]:
        """
        Load dataset from folder structure.

        Returns:
            List of (image_path, label) tuples and class_to_idx mapping
        """
        samples = []
        class_to_idx = {}
        class_idx = 0

        for class_folder in sorted(self.image_dir.iterdir()):
            if not class_folder.is_dir():
                continue

            class_name = class_folder.name
            class_to_idx[class_name] = class_idx

            for image_file in sorted(class_folder.glob(f'*{self.image_ext}')):
                samples.append((image_file, class_idx))

            class_idx += 1

        return samples, class_to_idx

    def _load_from_csv(
        self,
        labels_file: str,
        class_to_idx: Optional[dict]
    ) -> Tuple[List[Tuple[Path, int]], dict]:
        """
        Load dataset from CSV file.

        Args:
            labels_file: Path to CSV file
            class_to_idx: Optional predefined class mapping

        Returns:
            List of (image_path, label) tuples and class_to_idx mapping
        """
        import pandas as pd

        df = pd.read_csv(labels_file)
        samples = []

        # Build class_to_idx if not provided
        if class_to_idx is None:
            unique_classes = sorted(df['label'].unique())
            class_to_idx = {cls: idx for idx, cls in enumerate(unique_classes)}

        for _, row in df.iterrows():
            image_path = self.image_dir / row['filename']
            label = class_to_idx[row['label']]
            samples.append((image_path, label))

        return samples, class_to_idx

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get image and label.

        Args:
            idx: Index

        Returns:
            Image tensor and label
        """
        image_path, label = self.samples[idx]

        # Load image
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

        if image is None:
            raise FileNotFoundError(f"Failed to load image: {image_path}")

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        return image, label

    def get_class_name(self, idx: int) -> str:
        """Get class name from index."""
        return self.idx_to_class[idx]


class MaskedXrayDataset(Dataset):
    """
    Dataset for masked X-ray images (using segmentation masks to isolate ROI).
    """

    def __init__(
        self,
        image_dir: str,
        mask_dir: str,
        labels_file: Optional[str] = None,
        class_to_idx: Optional[dict] = None,
        transform: Optional[Callable] = None,
        image_ext: str = '.png',
        use_folders: bool = True
    ):
        """
        Initialize masked X-ray dataset.

        Args:
            image_dir: Directory containing X-ray images
            mask_dir: Directory containing segmentation masks
            labels_file: Path to CSV file with labels
            class_to_idx: Mapping from class names to indices
            transform: Augmentation transform
            image_ext: Image file extension
            use_folders: Whether images are organized in class folders
        """
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.transform = transform
        self.image_ext = image_ext
        self.use_folders = use_folders

        # Build dataset
        if use_folders:
            self.samples, self.class_to_idx = self._load_from_folders()
        else:
            if labels_file is None:
                raise ValueError("labels_file must be provided when use_folders=False")
            self.samples, self.class_to_idx = self._load_from_csv(labels_file, class_to_idx)

        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        self.num_classes = len(self.class_to_idx)

    def _load_from_folders(self) -> Tuple[List[Tuple[Path, int]], dict]:
        """Load dataset from folder structure."""
        samples = []
        class_to_idx = {}
        class_idx = 0

        for class_folder in sorted(self.image_dir.iterdir()):
            if not class_folder.is_dir():
                continue

            class_name = class_folder.name
            class_to_idx[class_name] = class_idx

            for image_file in sorted(class_folder.glob(f'*{self.image_ext}')):
                samples.append((image_file, class_idx))

            class_idx += 1

        return samples, class_to_idx

    def _load_from_csv(
        self,
        labels_file: str,
        class_to_idx: Optional[dict]
    ) -> Tuple[List[Tuple[Path, int]], dict]:
        """Load dataset from CSV file."""
        import pandas as pd

        df = pd.read_csv(labels_file)
        samples = []

        if class_to_idx is None:
            unique_classes = sorted(df['label'].unique())
            class_to_idx = {cls: idx for idx, cls in enumerate(unique_classes)}

        for _, row in df.iterrows():
            image_path = self.image_dir / row['filename']
            label = class_to_idx[row['label']]
            samples.append((image_path, label))

        return samples, class_to_idx

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get masked image and label.

        Args:
            idx: Index

        Returns:
            Masked image tensor and label
        """
        image_path, label = self.samples[idx]

        # Load image
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

        if image is None:
            raise FileNotFoundError(f"Failed to load image: {image_path}")

        # Load corresponding mask
        mask_name = image_path.name.replace(self.image_ext, '_mask' + self.image_ext)
        mask_path = self.mask_dir / mask_name

        if mask_path.exists():
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            mask = (mask > 127).astype(np.uint8)

            # Apply mask to image
            image = image * mask
        else:
            print(f"Warning: Mask not found for {image_path.name}, using original image")

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        return image, label
