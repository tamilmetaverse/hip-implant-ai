"""
CT dataset for hip implant analysis.
Supports 3D volumetric data and 2D slice extraction.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Optional, Callable, Tuple, List
from pathlib import Path
import SimpleITK as sitk


class CTVolumeDataset(Dataset):
    """
    Dataset for 3D CT volumes.
    """

    def __init__(
        self,
        data_dir: str,
        transform: Optional[Callable] = None,
        slice_axis: int = 2,  # 0: sagittal, 1: coronal, 2: axial
        file_ext: str = '.nii.gz'
    ):
        """
        Initialize CT volume dataset.

        Args:
            data_dir: Directory containing CT volumes
            transform: Augmentation transform
            slice_axis: Axis for slice extraction
            file_ext: File extension for CT volumes
        """
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.slice_axis = slice_axis
        self.file_ext = file_ext

        # Get list of volume files
        self.volume_files = sorted([
            f for f in os.listdir(data_dir)
            if f.endswith(file_ext)
        ])

        if len(self.volume_files) == 0:
            raise ValueError(f"No volumes found in {data_dir} with extension {file_ext}")

    def __len__(self) -> int:
        return len(self.volume_files)

    def load_volume(self, volume_path: Path) -> np.ndarray:
        """
        Load CT volume from file.

        Args:
            volume_path: Path to volume file

        Returns:
            Volume as numpy array
        """
        image = sitk.ReadImage(str(volume_path))
        volume = sitk.GetArrayFromImage(image)
        return volume

    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Get CT volume.

        Args:
            idx: Index

        Returns:
            Volume tensor
        """
        volume_name = self.volume_files[idx]
        volume_path = self.data_dir / volume_name
        volume = self.load_volume(volume_path)

        # Apply transforms if provided
        if self.transform:
            volume = self.transform(volume)

        # Convert to tensor
        if not isinstance(volume, torch.Tensor):
            volume = torch.from_numpy(volume).float()

        return volume


class CTSliceDataset(Dataset):
    """
    Dataset for 2D slices extracted from CT volumes.
    """

    def __init__(
        self,
        data_dir: str,
        labels_file: Optional[str] = None,
        class_to_idx: Optional[dict] = None,
        transform: Optional[Callable] = None,
        slice_axis: int = 2,
        center_slices_only: bool = True,
        num_slices_per_volume: int = 5,
        file_ext: str = '.nii.gz'
    ):
        """
        Initialize CT slice dataset.

        Args:
            data_dir: Directory containing CT volumes
            labels_file: Path to CSV file with labels
            class_to_idx: Mapping from class names to indices
            transform: Augmentation transform
            slice_axis: Axis for slice extraction
            center_slices_only: Whether to use only center slices
            num_slices_per_volume: Number of slices to extract per volume
            file_ext: File extension for CT volumes
        """
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.slice_axis = slice_axis
        self.center_slices_only = center_slices_only
        self.num_slices_per_volume = num_slices_per_volume
        self.file_ext = file_ext

        # Load labels if provided
        if labels_file:
            self.samples, self.class_to_idx = self._load_from_csv(labels_file, class_to_idx)
            self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
            self.num_classes = len(self.class_to_idx)
        else:
            self.samples = None
            self.class_to_idx = None

        # Get list of volume files
        self.volume_files = sorted([
            f for f in os.listdir(data_dir)
            if f.endswith(file_ext)
        ])

        # Build slice index
        self.slice_index = self._build_slice_index()

    def _load_from_csv(
        self,
        labels_file: str,
        class_to_idx: Optional[dict]
    ) -> Tuple[dict, dict]:
        """
        Load labels from CSV file.

        Args:
            labels_file: Path to CSV file
            class_to_idx: Optional predefined class mapping

        Returns:
            Dictionary mapping filenames to labels and class_to_idx mapping
        """
        import pandas as pd

        df = pd.read_csv(labels_file)
        samples = {}

        # Build class_to_idx if not provided
        if class_to_idx is None:
            unique_classes = sorted(df['label'].unique())
            class_to_idx = {cls: idx for idx, cls in enumerate(unique_classes)}

        for _, row in df.iterrows():
            samples[row['filename']] = class_to_idx[row['label']]

        return samples, class_to_idx

    def _build_slice_index(self) -> List[Tuple[int, int]]:
        """
        Build index of (volume_idx, slice_idx) pairs.

        Returns:
            List of tuples
        """
        slice_index = []

        for vol_idx, volume_name in enumerate(self.volume_files):
            volume_path = self.data_dir / volume_name
            volume = self.load_volume(volume_path)

            num_slices = volume.shape[self.slice_axis]

            if self.center_slices_only:
                # Extract slices around center
                center = num_slices // 2
                start = max(0, center - self.num_slices_per_volume // 2)
                end = min(num_slices, center + self.num_slices_per_volume // 2 + 1)
                slice_indices = range(start, end)
            else:
                # Uniformly sample slices
                slice_indices = np.linspace(
                    0,
                    num_slices - 1,
                    self.num_slices_per_volume,
                    dtype=int
                )

            for slice_idx in slice_indices:
                slice_index.append((vol_idx, slice_idx))

        return slice_index

    def load_volume(self, volume_path: Path) -> np.ndarray:
        """Load CT volume from file."""
        image = sitk.ReadImage(str(volume_path))
        volume = sitk.GetArrayFromImage(image)
        return volume

    def __len__(self) -> int:
        return len(self.slice_index)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Optional[int]]:
        """
        Get CT slice.

        Args:
            idx: Index

        Returns:
            Slice tensor and optional label
        """
        vol_idx, slice_idx = self.slice_index[idx]
        volume_name = self.volume_files[vol_idx]
        volume_path = self.data_dir / volume_name

        # Load volume
        volume = self.load_volume(volume_path)

        # Extract slice
        if self.slice_axis == 0:
            slice_2d = volume[slice_idx, :, :]
        elif self.slice_axis == 1:
            slice_2d = volume[:, slice_idx, :]
        else:  # axis == 2
            slice_2d = volume[:, :, slice_idx]

        # Apply transforms
        if self.transform:
            slice_2d = self.transform(slice_2d)

        # Convert to tensor if needed
        if not isinstance(slice_2d, torch.Tensor):
            slice_2d = torch.from_numpy(slice_2d).float()
            if len(slice_2d.shape) == 2:
                slice_2d = slice_2d.unsqueeze(0)  # Add channel dimension

        # Get label if available
        label = None
        if self.samples is not None:
            label = self.samples.get(volume_name, -1)

        if label is not None:
            return slice_2d, label
        else:
            return slice_2d


class MultiModalDataset(Dataset):
    """
    Dataset combining X-ray and CT data for multi-modal learning.
    """

    def __init__(
        self,
        xray_dir: str,
        ct_dir: str,
        labels_file: str,
        xray_transform: Optional[Callable] = None,
        ct_transform: Optional[Callable] = None,
        class_to_idx: Optional[dict] = None
    ):
        """
        Initialize multi-modal dataset.

        Args:
            xray_dir: Directory containing X-ray images
            ct_dir: Directory containing CT volumes/slices
            labels_file: Path to CSV file with labels
            xray_transform: Transform for X-ray images
            ct_transform: Transform for CT images
            class_to_idx: Mapping from class names to indices
        """
        self.xray_dir = Path(xray_dir)
        self.ct_dir = Path(ct_dir)
        self.xray_transform = xray_transform
        self.ct_transform = ct_transform

        # Load labels
        import pandas as pd
        df = pd.read_csv(labels_file)

        # Build class_to_idx if not provided
        if class_to_idx is None:
            unique_classes = sorted(df['label'].unique())
            class_to_idx = {cls: idx for idx, cls in enumerate(unique_classes)}

        self.class_to_idx = class_to_idx
        self.idx_to_class = {v: k for k, v in class_to_idx.items()}
        self.num_classes = len(class_to_idx)

        # Build samples list
        self.samples = []
        for _, row in df.iterrows():
            xray_path = self.xray_dir / row['xray_filename']
            ct_path = self.ct_dir / row['ct_filename']
            label = class_to_idx[row['label']]
            self.samples.append((xray_path, ct_path, label))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Get X-ray, CT, and label.

        Args:
            idx: Index

        Returns:
            X-ray tensor, CT tensor, and label
        """
        xray_path, ct_path, label = self.samples[idx]

        # Load X-ray
        import cv2
        xray = cv2.imread(str(xray_path), cv2.IMREAD_GRAYSCALE)
        if xray is None:
            raise FileNotFoundError(f"Failed to load X-ray: {xray_path}")

        # Load CT
        if ct_path.suffix in ['.nii', '.gz']:
            image = sitk.ReadImage(str(ct_path))
            ct = sitk.GetArrayFromImage(image)
        else:
            ct = cv2.imread(str(ct_path), cv2.IMREAD_GRAYSCALE)

        # Apply transforms
        if self.xray_transform:
            xray = self.xray_transform(xray)
        if self.ct_transform:
            ct = self.ct_transform(ct)

        return xray, ct, label
