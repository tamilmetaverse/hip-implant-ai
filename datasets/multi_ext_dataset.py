"""
Multi-extension dataset wrapper for handling various image formats
"""

from pathlib import Path
from typing import List, Tuple
from datasets.xray_dataset import XrayClassificationDataset


class MultiExtensionDataset(XrayClassificationDataset):
    """Dataset that handles multiple image extensions"""

    def __init__(self, *args, extensions=None, **kwargs):
        """
        Initialize with multiple extensions

        Args:
            extensions: List of extensions to look for (e.g., ['.jpg', '.jpeg', '.png'])
        """
        if extensions is None:
            extensions = ['.jpg', '.jpeg', '.png', '.PNG', '.JPG', '.JPEG']

        self.extensions = extensions

        # Remove image_ext from kwargs if present
        kwargs.pop('image_ext', None)

        # Call parent init with a dummy extension
        kwargs['image_ext'] = '.jpg'  # Dummy, we'll override the loading
        super().__init__(*args, **kwargs)

        # Now reload with multiple extensions
        self.samples, self.class_to_idx = self._load_from_folders_multi_ext()
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        self.num_classes = len(self.class_to_idx)

    def _load_from_folders_multi_ext(self) -> Tuple[List[Tuple[Path, int]], dict]:
        """Load dataset from folder structure with multiple extensions"""
        samples = []
        class_to_idx = {}
        class_idx = 0

        for class_folder in sorted(self.image_dir.iterdir()):
            if not class_folder.is_dir():
                continue

            class_name = class_folder.name
            class_to_idx[class_name] = class_idx

            # Look for files with any of the specified extensions
            for ext in self.extensions:
                for image_file in sorted(class_folder.glob(f'*{ext}')):
                    samples.append((image_file, class_idx))

            class_idx += 1

        return samples, class_to_idx
