"""
Ensemble learning for combining multiple models.
Supports both segmentation and classification ensembles.
"""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import yaml

import sys
sys.path.append(str(Path(__file__).parent.parent))

from inference.classify import ClassificationInference
from inference.segment import SegmentationInference
from utils.uncertainty import UncertaintyEstimator, ClinicalDecisionSupport


class ClassificationEnsemble:
    """
    Ensemble of classification models for robust prediction.
    """

    def __init__(
        self,
        model_configs: List[Dict],
        class_names: Optional[List[str]] = None,
        ensemble_strategy: str = 'soft_voting',
        weights: Optional[List[float]] = None,
        device: Optional[torch.device] = None
    ):
        """
        Initialize classification ensemble.

        Args:
            model_configs: List of dictionaries with 'checkpoint_path' and 'config_path'
            class_names: List of class names
            ensemble_strategy: Ensemble strategy ('soft_voting', 'hard_voting', 'weighted')
            weights: Weights for each model (for weighted ensemble)
            device: Device to run inference on
        """
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.ensemble_strategy = ensemble_strategy
        self.class_names = class_names

        # Load models
        self.models = []
        for config in model_configs:
            model = ClassificationInference(
                checkpoint_path=config['checkpoint_path'],
                config_path=config.get('config_path'),
                class_names=class_names,
                device=self.device
            )
            self.models.append(model)

        # Set weights
        if weights is None:
            self.weights = [1.0 / len(self.models)] * len(self.models)
        else:
            if len(weights) != len(self.models):
                raise ValueError("Number of weights must match number of models")
            # Normalize weights
            total_weight = sum(weights)
            self.weights = [w / total_weight for w in weights]

        print(f"Initialized ensemble with {len(self.models)} models")

        # Uncertainty estimator
        self.uncertainty_estimator = UncertaintyEstimator()

        # Clinical decision support
        if class_names:
            self.clinical_support = ClinicalDecisionSupport(
                self.uncertainty_estimator,
                class_names
            )

    def predict(
        self,
        image: np.ndarray,
        top_k: int = 5
    ) -> Dict:
        """
        Predict using ensemble.

        Args:
            image: Input image
            top_k: Number of top predictions to return

        Returns:
            Dictionary with ensemble predictions
        """
        # Collect predictions from all models
        all_probabilities = []

        for model in self.models:
            result = model.predict(image, return_probabilities=True)
            all_probabilities.append(result['all_probabilities'])

        # Combine predictions
        if self.ensemble_strategy == 'soft_voting':
            # Weighted average of probabilities
            ensemble_probs = np.zeros_like(all_probabilities[0])
            for probs, weight in zip(all_probabilities, self.weights):
                ensemble_probs += weight * probs

        elif self.ensemble_strategy == 'hard_voting':
            # Majority voting
            predictions = [np.argmax(probs) for probs in all_probabilities]
            ensemble_pred = np.bincount(predictions).argmax()

            # Convert to probabilities (one-hot)
            ensemble_probs = np.zeros(len(all_probabilities[0]))
            ensemble_probs[ensemble_pred] = 1.0

        elif self.ensemble_strategy == 'weighted':
            # Weighted average (same as soft_voting)
            ensemble_probs = np.zeros_like(all_probabilities[0])
            for probs, weight in zip(all_probabilities, self.weights):
                ensemble_probs += weight * probs

        else:
            raise ValueError(f"Unknown ensemble strategy: {self.ensemble_strategy}")

        # Get top-k predictions
        top_indices = np.argsort(ensemble_probs)[::-1][:top_k]
        top_probs = ensemble_probs[top_indices]

        predictions = []
        for idx, prob in zip(top_indices, top_probs):
            predictions.append({
                'class_name': self.class_names[idx] if self.class_names else f"Class {idx}",
                'class_index': int(idx),
                'probability': float(prob)
            })

        # Uncertainty estimation with ensemble variance
        needs_review, uncertainty_metrics = self.uncertainty_estimator.needs_human_review(
            ensemble_probs,
            ensemble_probabilities=all_probabilities
        )

        results = {
            'predictions': predictions,
            'primary_prediction': predictions[0]['class_name'] if self.class_names else predictions[0]['class_index'],
            'confidence': float(predictions[0]['probability']),
            'ensemble_probabilities': ensemble_probs,
            'individual_probabilities': all_probabilities,
            'uncertainty': {
                'needs_human_review': needs_review,
                'metrics': uncertainty_metrics,
                'ensemble_variance': uncertainty_metrics.get('ensemble_variance', 0.0)
            }
        }

        return results

    def generate_clinical_report(
        self,
        image: np.ndarray,
        top_k: int = 5
    ) -> Dict:
        """
        Generate comprehensive clinical report with ensemble predictions.

        Args:
            image: Input image
            top_k: Number of top predictions to include

        Returns:
            Clinical report dictionary
        """
        # Get ensemble predictions
        results = self.predict(image, top_k=top_k)

        # Generate report
        if hasattr(self, 'clinical_support'):
            report = self.clinical_support.generate_report(
                probabilities=results['ensemble_probabilities'],
                ensemble_probabilities=results['individual_probabilities'],
                top_k=top_k
            )
        else:
            # Basic report without clinical support
            report = {
                'primary_prediction': results['primary_prediction'],
                'confidence': results['confidence'],
                'top_predictions': results['predictions'],
                'uncertainty': results['uncertainty']
            }

        return report


class MultiModalEnsemble:
    """
    Ensemble combining original and masked image predictions.
    """

    def __init__(
        self,
        original_model_config: Dict,
        masked_model_config: Dict,
        segmentation_model_config: Dict,
        class_names: Optional[List[str]] = None,
        fusion_weight: float = 0.5,
        device: Optional[torch.device] = None
    ):
        """
        Initialize multi-modal ensemble.

        Args:
            original_model_config: Config for original image model
            masked_model_config: Config for masked image model
            segmentation_model_config: Config for segmentation model
            class_names: List of class names
            fusion_weight: Weight for original model (1 - weight for masked model)
            device: Device to run inference on
        """
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.fusion_weight = fusion_weight
        self.class_names = class_names

        # Load segmentation model
        self.segmentation_model = SegmentationInference(
            checkpoint_path=segmentation_model_config['checkpoint_path'],
            config_path=segmentation_model_config.get('config_path'),
            device=self.device
        )

        # Load classification models
        self.original_classifier = ClassificationInference(
            checkpoint_path=original_model_config['checkpoint_path'],
            config_path=original_model_config.get('config_path'),
            class_names=class_names,
            device=self.device
        )

        self.masked_classifier = ClassificationInference(
            checkpoint_path=masked_model_config['checkpoint_path'],
            config_path=masked_model_config.get('config_path'),
            class_names=class_names,
            device=self.device
        )

        # Uncertainty estimator
        self.uncertainty_estimator = UncertaintyEstimator()

        # Clinical decision support
        if class_names:
            self.clinical_support = ClinicalDecisionSupport(
                self.uncertainty_estimator,
                class_names
            )

        print("Initialized multi-modal ensemble")

    def predict(
        self,
        image: np.ndarray,
        top_k: int = 5
    ) -> Dict:
        """
        Predict using multi-modal ensemble.

        Args:
            image: Input image
            top_k: Number of top predictions to return

        Returns:
            Dictionary with predictions
        """
        # Step 1: Segment the image
        seg_result = self.segmentation_model.predict(image)
        mask = seg_result['mask']

        # Step 2: Create masked image
        masked_image = self.segmentation_model.extract_roi(image, mask)

        # Step 3: Classify original image
        original_result = self.original_classifier.predict(image, return_probabilities=True)
        original_probs = original_result['all_probabilities']

        # Step 4: Classify masked image
        masked_result = self.masked_classifier.predict(masked_image, return_probabilities=True)
        masked_probs = masked_result['all_probabilities']

        # Step 5: Fuse predictions
        fused_probs = (
            self.fusion_weight * original_probs +
            (1 - self.fusion_weight) * masked_probs
        )

        # Get top-k predictions
        top_indices = np.argsort(fused_probs)[::-1][:top_k]
        top_probs = fused_probs[top_indices]

        predictions = []
        for idx, prob in zip(top_indices, top_probs):
            predictions.append({
                'class_name': self.class_names[idx] if self.class_names else f"Class {idx}",
                'class_index': int(idx),
                'probability': float(prob)
            })

        # Uncertainty estimation
        needs_review, uncertainty_metrics = self.uncertainty_estimator.needs_human_review(
            fused_probs,
            ensemble_probabilities=[original_probs, masked_probs]
        )

        results = {
            'predictions': predictions,
            'primary_prediction': predictions[0]['class_name'] if self.class_names else predictions[0]['class_index'],
            'confidence': float(predictions[0]['probability']),
            'segmentation_mask': mask,
            'original_probabilities': original_probs,
            'masked_probabilities': masked_probs,
            'fused_probabilities': fused_probs,
            'uncertainty': {
                'needs_human_review': needs_review,
                'metrics': uncertainty_metrics
            }
        }

        return results

    def generate_clinical_report(
        self,
        image: np.ndarray,
        top_k: int = 5
    ) -> Dict:
        """
        Generate comprehensive clinical report.

        Args:
            image: Input image
            top_k: Number of top predictions to include

        Returns:
            Clinical report dictionary
        """
        # Get predictions
        results = self.predict(image, top_k=top_k)

        # Generate report
        if hasattr(self, 'clinical_support'):
            report = self.clinical_support.generate_report(
                probabilities=results['fused_probabilities'],
                ensemble_probabilities=[
                    results['original_probabilities'],
                    results['masked_probabilities']
                ],
                top_k=top_k
            )

            # Add segmentation info
            report['segmentation'] = {
                'mask_area': int(np.sum(results['segmentation_mask'] > 0)),
                'total_pixels': int(results['segmentation_mask'].size),
                'coverage_percentage': float(
                    100.0 * np.sum(results['segmentation_mask'] > 0) / results['segmentation_mask'].size
                )
            }
        else:
            report = {
                'primary_prediction': results['primary_prediction'],
                'confidence': results['confidence'],
                'top_predictions': results['predictions'],
                'uncertainty': results['uncertainty']
            }

        return report
