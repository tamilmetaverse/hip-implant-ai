"""
Uncertainty estimation for clinical decision support.
Implements confidence scoring, ensemble variance, and human review flagging.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy.stats import entropy


class UncertaintyEstimator:
    """
    Comprehensive uncertainty estimation for model predictions.
    """

    def __init__(
        self,
        confidence_threshold: float = 0.7,
        variance_threshold: float = 0.1,
        entropy_threshold: float = 1.0
    ):
        """
        Initialize uncertainty estimator.

        Args:
            confidence_threshold: Minimum confidence for automatic decision
            variance_threshold: Maximum variance for ensemble predictions
            entropy_threshold: Maximum entropy for prediction distribution
        """
        self.confidence_threshold = confidence_threshold
        self.variance_threshold = variance_threshold
        self.entropy_threshold = entropy_threshold

    def compute_confidence(
        self,
        probabilities: np.ndarray
    ) -> Tuple[float, int]:
        """
        Compute softmax confidence score.

        Args:
            probabilities: Class probabilities

        Returns:
            Confidence score and predicted class
        """
        max_prob = np.max(probabilities)
        predicted_class = np.argmax(probabilities)
        return float(max_prob), int(predicted_class)

    def compute_entropy(self, probabilities: np.ndarray) -> float:
        """
        Compute prediction entropy.

        Args:
            probabilities: Class probabilities

        Returns:
            Entropy value
        """
        return float(entropy(probabilities))

    def compute_ensemble_variance(
        self,
        ensemble_probabilities: List[np.ndarray]
    ) -> float:
        """
        Compute variance across ensemble predictions.

        Args:
            ensemble_probabilities: List of probability arrays from different models

        Returns:
            Variance score
        """
        # Stack probabilities from all models
        stacked = np.stack(ensemble_probabilities, axis=0)  # [num_models, num_classes]

        # Compute variance across models for each class
        variances = np.var(stacked, axis=0)

        # Return mean variance
        return float(np.mean(variances))

    def compute_margin(self, probabilities: np.ndarray) -> float:
        """
        Compute margin between top two predictions.

        Args:
            probabilities: Class probabilities

        Returns:
            Margin score (higher is more confident)
        """
        sorted_probs = np.sort(probabilities)[::-1]
        if len(sorted_probs) >= 2:
            margin = sorted_probs[0] - sorted_probs[1]
        else:
            margin = sorted_probs[0]
        return float(margin)

    def needs_human_review(
        self,
        probabilities: np.ndarray,
        ensemble_probabilities: Optional[List[np.ndarray]] = None
    ) -> Tuple[bool, Dict[str, float]]:
        """
        Determine if prediction needs human review.

        Args:
            probabilities: Final ensemble probabilities
            ensemble_probabilities: Individual model probabilities

        Returns:
            Boolean flag and uncertainty metrics
        """
        uncertainty_metrics = {}

        # Compute confidence
        confidence, _ = self.compute_confidence(probabilities)
        uncertainty_metrics['confidence'] = confidence

        # Compute entropy
        pred_entropy = self.compute_entropy(probabilities)
        uncertainty_metrics['entropy'] = pred_entropy

        # Compute margin
        margin = self.compute_margin(probabilities)
        uncertainty_metrics['margin'] = margin

        # Compute ensemble variance if available
        if ensemble_probabilities is not None and len(ensemble_probabilities) > 1:
            variance = self.compute_ensemble_variance(ensemble_probabilities)
            uncertainty_metrics['ensemble_variance'] = variance
        else:
            variance = 0.0
            uncertainty_metrics['ensemble_variance'] = 0.0

        # Decision logic
        needs_review = False

        if confidence < self.confidence_threshold:
            needs_review = True
            uncertainty_metrics['review_reason'] = 'low_confidence'

        elif pred_entropy > self.entropy_threshold:
            needs_review = True
            uncertainty_metrics['review_reason'] = 'high_entropy'

        elif variance > self.variance_threshold:
            needs_review = True
            uncertainty_metrics['review_reason'] = 'high_ensemble_variance'

        else:
            uncertainty_metrics['review_reason'] = 'none'

        return needs_review, uncertainty_metrics

    def get_uncertainty_level(
        self,
        probabilities: np.ndarray
    ) -> str:
        """
        Categorize uncertainty level.

        Args:
            probabilities: Class probabilities

        Returns:
            Uncertainty level: 'low', 'medium', or 'high'
        """
        confidence, _ = self.compute_confidence(probabilities)

        if confidence >= 0.9:
            return 'low'
        elif confidence >= 0.7:
            return 'medium'
        else:
            return 'high'


class MonteCarloDropout:
    """
    Monte Carlo Dropout for uncertainty estimation.
    """

    def __init__(self, num_samples: int = 10):
        """
        Initialize MC Dropout estimator.

        Args:
            num_samples: Number of forward passes with dropout
        """
        self.num_samples = num_samples

    def predict_with_uncertainty(
        self,
        model: torch.nn.Module,
        input_tensor: torch.Tensor,
        device: torch.device
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate predictions with uncertainty using MC Dropout.

        Args:
            model: PyTorch model (must have dropout layers)
            input_tensor: Input tensor
            device: Device to run inference on

        Returns:
            Mean probabilities and variance
        """
        model.train()  # Enable dropout
        predictions = []

        with torch.no_grad():
            for _ in range(self.num_samples):
                output = model(input_tensor.to(device))
                probs = torch.softmax(output, dim=1)
                predictions.append(probs.cpu().numpy())

        model.eval()  # Disable dropout

        # Stack predictions
        predictions = np.stack(predictions, axis=0)  # [num_samples, batch_size, num_classes]

        # Compute mean and variance
        mean_probs = np.mean(predictions, axis=0)
        variance = np.var(predictions, axis=0)

        return mean_probs, variance


class DeepEnsembleUncertainty:
    """
    Uncertainty estimation using deep ensembles.
    """

    def __init__(self, models: List[torch.nn.Module]):
        """
        Initialize deep ensemble estimator.

        Args:
            models: List of trained models
        """
        self.models = models

    def predict_with_uncertainty(
        self,
        input_tensor: torch.Tensor,
        device: torch.device
    ) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
        """
        Generate predictions with uncertainty using ensemble.

        Args:
            input_tensor: Input tensor
            device: Device to run inference on

        Returns:
            Mean probabilities, variance, and individual predictions
        """
        predictions = []

        for model in self.models:
            model.eval()
            with torch.no_grad():
                output = model(input_tensor.to(device))
                probs = torch.softmax(output, dim=1)
                predictions.append(probs.cpu().numpy())

        # Stack predictions
        predictions_array = np.stack(predictions, axis=0)  # [num_models, batch_size, num_classes]

        # Compute mean and variance
        mean_probs = np.mean(predictions_array, axis=0)
        variance = np.var(predictions_array, axis=0)

        return mean_probs, variance, predictions


def temperature_scaling(
    logits: torch.Tensor,
    temperature: float = 1.0
) -> torch.Tensor:
    """
    Apply temperature scaling for calibration.

    Args:
        logits: Model logits
        temperature: Temperature parameter

    Returns:
        Calibrated probabilities
    """
    scaled_logits = logits / temperature
    return torch.softmax(scaled_logits, dim=1)


class ClinicalDecisionSupport:
    """
    Clinical decision support system with uncertainty-aware recommendations.
    """

    def __init__(
        self,
        uncertainty_estimator: UncertaintyEstimator,
        class_names: List[str]
    ):
        """
        Initialize clinical decision support.

        Args:
            uncertainty_estimator: Uncertainty estimator instance
            class_names: List of class names
        """
        self.uncertainty_estimator = uncertainty_estimator
        self.class_names = class_names

    def generate_report(
        self,
        probabilities: np.ndarray,
        ensemble_probabilities: Optional[List[np.ndarray]] = None,
        top_k: int = 5
    ) -> Dict:
        """
        Generate clinical decision support report.

        Args:
            probabilities: Final ensemble probabilities
            ensemble_probabilities: Individual model probabilities
            top_k: Number of top predictions to include

        Returns:
            Comprehensive report dictionary
        """
        # Get top predictions
        top_indices = np.argsort(probabilities)[::-1][:top_k]
        top_probs = probabilities[top_indices]

        # Prepare top predictions
        top_predictions = []
        for idx, prob in zip(top_indices, top_probs):
            top_predictions.append({
                'class': self.class_names[idx] if idx < len(self.class_names) else f'Class {idx}',
                'probability': float(prob),
                'rank': len(top_predictions) + 1
            })

        # Get primary prediction
        confidence, predicted_class = self.uncertainty_estimator.compute_confidence(probabilities)

        # Check if needs review
        needs_review, uncertainty_metrics = self.uncertainty_estimator.needs_human_review(
            probabilities,
            ensemble_probabilities
        )

        # Get uncertainty level
        uncertainty_level = self.uncertainty_estimator.get_uncertainty_level(probabilities)

        # Build report
        report = {
            'primary_prediction': {
                'class': self.class_names[predicted_class] if predicted_class < len(self.class_names) else f'Class {predicted_class}',
                'class_index': int(predicted_class),
                'confidence': float(confidence)
            },
            'top_predictions': top_predictions,
            'uncertainty': {
                'level': uncertainty_level,
                'needs_human_review': needs_review,
                'metrics': uncertainty_metrics
            },
            'recommendation': self._generate_recommendation(
                needs_review,
                uncertainty_level,
                confidence
            )
        }

        return report

    def _generate_recommendation(
        self,
        needs_review: bool,
        uncertainty_level: str,
        confidence: float
    ) -> str:
        """
        Generate clinical recommendation text.

        Args:
            needs_review: Whether human review is needed
            uncertainty_level: Uncertainty level
            confidence: Confidence score

        Returns:
            Recommendation text
        """
        if needs_review:
            return (
                f"HUMAN REVIEW REQUIRED: The model prediction has {uncertainty_level} uncertainty "
                f"(confidence: {confidence:.2%}). Please verify this prediction with clinical expertise "
                "before making treatment decisions."
            )
        elif uncertainty_level == 'medium':
            return (
                f"MODERATE CONFIDENCE: The model prediction appears reliable (confidence: {confidence:.2%}), "
                "but consider additional verification if this impacts critical clinical decisions."
            )
        else:
            return (
                f"HIGH CONFIDENCE: The model prediction is highly confident (confidence: {confidence:.2%}). "
                "This prediction can be used to support clinical decision-making."
            )
