"""
Classification inference module.
"""

import torch
import torch.nn as nn
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import yaml

import sys
sys.path.append(str(Path(__file__).parent.parent))

from models.classification.swin import build_swin_model
from models.classification.convnext import build_convnext_model
from utils.preprocessing import ImagePreprocessor
from utils.uncertainty import UncertaintyEstimator, ClinicalDecisionSupport


class ClassificationInference:
    """
    Inference pipeline for classification models.
    """

    def __init__(
        self,
        checkpoint_path: str,
        config_path: Optional[str] = None,
        class_names: Optional[List[str]] = None,
        device: Optional[torch.device] = None
    ):
        """
        Initialize classification inference.

        Args:
            checkpoint_path: Path to model checkpoint
            config_path: Path to configuration file
            class_names: List of class names
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

        # Class names
        self.class_names = class_names if class_names else [f"Class {i}" for i in range(self.config['model']['num_classes'])]

        # Uncertainty estimator
        uncertainty_config = self.config.get('uncertainty', {})
        self.uncertainty_estimator = UncertaintyEstimator(
            confidence_threshold=uncertainty_config.get('confidence_threshold', 0.7),
            variance_threshold=uncertainty_config.get('variance_threshold', 0.1)
        )

        # Clinical decision support
        self.clinical_support = ClinicalDecisionSupport(
            self.uncertainty_estimator,
            self.class_names
        )

        print(f"Loaded classification model from {checkpoint_path}")

    def _build_model(self) -> nn.Module:
        """Build classification model."""
        model_name = self.config['model']['name']

        if model_name == 'swin':
            model = build_swin_model(self.config['model'])
        elif model_name == 'convnext':
            model = build_convnext_model(self.config['model'])
        else:
            raise ValueError(f"Unknown model: {model_name}")

        return model

    @torch.no_grad()
    def predict(
        self,
        image: np.ndarray,
        return_probabilities: bool = True,
        top_k: int = 5
    ) -> Dict:
        """
        Predict implant class.

        Args:
            image: Input image (grayscale or RGB)
            return_probabilities: Whether to return class probabilities
            top_k: Number of top predictions to return

        Returns:
            Dictionary with predictions and probabilities
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
        probabilities = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

        # Get top-k predictions
        top_indices = np.argsort(probabilities)[::-1][:top_k]
        top_probs = probabilities[top_indices]

        predictions = []
        for idx, prob in zip(top_indices, top_probs):
            predictions.append({
                'class_name': self.class_names[idx],
                'class_index': int(idx),
                'probability': float(prob)
            })

        results = {
            'predictions': predictions,
            'primary_prediction': predictions[0]['class_name'],
            'confidence': float(predictions[0]['probability'])
        }

        if return_probabilities:
            results['all_probabilities'] = probabilities

        # Uncertainty estimation
        needs_review, uncertainty_metrics = self.uncertainty_estimator.needs_human_review(probabilities)
        results['uncertainty'] = {
            'needs_human_review': needs_review,
            'metrics': uncertainty_metrics
        }

        return results

    def predict_from_file(
        self,
        image_path: str,
        top_k: int = 5
    ) -> Dict:
        """
        Predict classification from image file.

        Args:
            image_path: Path to input image
            top_k: Number of top predictions to return

        Returns:
            Dictionary with predictions
        """
        # Load image
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        if image is None:
            raise FileNotFoundError(f"Failed to load image: {image_path}")

        # Predict
        results = self.predict(image, top_k=top_k)

        return results

    def generate_clinical_report(
        self,
        image: np.ndarray,
        top_k: int = 5
    ) -> Dict:
        """
        Generate comprehensive clinical decision support report.

        Args:
            image: Input image
            top_k: Number of top predictions to include

        Returns:
            Clinical report dictionary
        """
        # Get predictions
        results = self.predict(image, return_probabilities=True, top_k=top_k)

        # Generate report
        report = self.clinical_support.generate_report(
            probabilities=results['all_probabilities'],
            top_k=top_k
        )

        return report

    def batch_predict(
        self,
        images: List[np.ndarray],
        batch_size: int = 32
    ) -> List[Dict]:
        """
        Predict classification for multiple images.

        Args:
            images: List of images
            batch_size: Batch size for inference

        Returns:
            List of prediction dictionaries
        """
        all_results = []

        for i in range(0, len(images), batch_size):
            batch_images = images[i:i + batch_size]

            # Preprocess batch
            preprocessed = []
            for img in batch_images:
                proc = self.preprocessor.process(img)
                tensor = self.preprocessor.to_tensor(proc, add_channel=True)
                preprocessed.append(tensor)

            # Stack tensors
            batch_tensor = torch.stack(preprocessed).to(self.device)

            # Forward pass
            if hasattr(self.model, 'model'):
                logits = self.model.model(batch_tensor)
            else:
                logits = self.model(batch_tensor)

            # Get probabilities
            probabilities = torch.softmax(logits, dim=1).cpu().numpy()

            # Process each prediction
            for j, probs in enumerate(probabilities):
                top_indices = np.argsort(probs)[::-1][:5]
                top_probs = probs[top_indices]

                predictions = []
                for idx, prob in zip(top_indices, top_probs):
                    predictions.append({
                        'class_name': self.class_names[idx],
                        'class_index': int(idx),
                        'probability': float(prob)
                    })

                needs_review, uncertainty_metrics = self.uncertainty_estimator.needs_human_review(probs)

                result = {
                    'predictions': predictions,
                    'primary_prediction': predictions[0]['class_name'],
                    'confidence': float(predictions[0]['probability']),
                    'uncertainty': {
                        'needs_human_review': needs_review,
                        'metrics': uncertainty_metrics
                    }
                }

                all_results.append(result)

        return all_results
