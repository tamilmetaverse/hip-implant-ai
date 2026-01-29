"""
Main entry point for Hip Implant AI system.
Provides unified interface for training and inference.
"""

import argparse
import json
from pathlib import Path
import cv2
import numpy as np
import sys

from inference.segment import SegmentationInference
from inference.classify import ClassificationInference
from inference.ensemble import ClassificationEnsemble, MultiModalEnsemble
from training.train_segmentation import main as train_segmentation
from training.train_classification import main as train_classification


def segment_image(args):
    """
    Perform segmentation inference.

    Args:
        args: Command-line arguments
    """
    print("Running segmentation inference...")

    # Initialize segmentation model
    segmenter = SegmentationInference(
        checkpoint_path=args.checkpoint,
        config_path=args.config
    )

    # Predict
    results = segmenter.predict_from_file(
        image_path=args.input,
        save_path=args.output
    )

    print(f"Segmentation completed. Mask saved to {args.output}")

    # If requested, also extract ROI
    if args.extract_roi:
        image = cv2.imread(args.input, cv2.IMREAD_GRAYSCALE)
        roi = segmenter.extract_roi(image, results['mask'])

        roi_path = Path(args.output).parent / f"{Path(args.output).stem}_roi.png"
        cv2.imwrite(str(roi_path), roi)
        print(f"ROI saved to {roi_path}")


def classify_image(args):
    """
    Perform classification inference.

    Args:
        args: Command-line arguments
    """
    print("Running classification inference...")

    # Load class names if provided
    class_names = None
    if args.class_names:
        with open(args.class_names, 'r') as f:
            class_names = [line.strip() for line in f.readlines()]

    # Initialize classification model
    classifier = ClassificationInference(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        class_names=class_names
    )

    # Predict
    if args.clinical_report:
        # Load image
        image = cv2.imread(args.input, cv2.IMREAD_GRAYSCALE)
        report = classifier.generate_clinical_report(image, top_k=args.top_k)
    else:
        report = classifier.predict_from_file(args.input, top_k=args.top_k)

    # Print results
    print("\n" + "=" * 50)
    print("CLASSIFICATION RESULTS")
    print("=" * 50)
    print(f"\nPrimary Prediction: {report.get('primary_prediction', {}).get('class', 'N/A')}")
    print(f"Confidence: {report.get('primary_prediction', {}).get('confidence', 0.0):.2%}")

    print(f"\nTop {args.top_k} Predictions:")
    for i, pred in enumerate(report.get('top_predictions', [])[:args.top_k], 1):
        print(f"  {i}. {pred['class']} ({pred['probability']:.2%})")

    print("\nUncertainty Analysis:")
    uncertainty = report.get('uncertainty', {})
    print(f"  Uncertainty Level: {uncertainty.get('level', 'N/A')}")
    print(f"  Needs Human Review: {uncertainty.get('needs_human_review', False)}")
    print(f"  Confidence Score: {uncertainty.get('metrics', {}).get('confidence', 0.0):.2%}")

    if 'recommendation' in report:
        print(f"\nClinical Recommendation:")
        print(f"  {report['recommendation']}")

    print("=" * 50 + "\n")

    # Save results if output path provided
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"Results saved to {args.output}")


def ensemble_inference(args):
    """
    Perform ensemble inference.

    Args:
        args: Command-line arguments
    """
    print("Running ensemble inference...")

    # Load class names if provided
    class_names = None
    if args.class_names:
        with open(args.class_names, 'r') as f:
            class_names = [line.strip() for line in f.readlines()]

    # Parse model checkpoints
    checkpoints = args.checkpoint.split(',')
    configs = args.config.split(',') if args.config else [None] * len(checkpoints)

    if len(checkpoints) != len(configs):
        raise ValueError("Number of checkpoints must match number of configs")

    # Build model configs
    model_configs = []
    for ckpt, cfg in zip(checkpoints, configs):
        model_configs.append({
            'checkpoint_path': ckpt.strip(),
            'config_path': cfg.strip() if cfg else None
        })

    # Initialize ensemble
    if args.ensemble_type == 'multimodal':
        if len(model_configs) != 3:
            raise ValueError("Multi-modal ensemble requires 3 models: original, masked, segmentation")

        ensemble = MultiModalEnsemble(
            original_model_config=model_configs[0],
            masked_model_config=model_configs[1],
            segmentation_model_config=model_configs[2],
            class_names=class_names,
            fusion_weight=args.fusion_weight
        )
    else:
        # Standard ensemble
        weights = None
        if args.ensemble_weights:
            weights = [float(w) for w in args.ensemble_weights.split(',')]

        ensemble = ClassificationEnsemble(
            model_configs=model_configs,
            class_names=class_names,
            ensemble_strategy=args.ensemble_strategy,
            weights=weights
        )

    # Load and predict
    image = cv2.imread(args.input, cv2.IMREAD_GRAYSCALE)

    if args.clinical_report:
        report = ensemble.generate_clinical_report(image, top_k=args.top_k)
    else:
        report = ensemble.predict(image, top_k=args.top_k)

    # Print results
    print("\n" + "=" * 50)
    print("ENSEMBLE CLASSIFICATION RESULTS")
    print("=" * 50)
    print(f"\nPrimary Prediction: {report.get('primary_prediction', {}).get('class', report.get('primary_prediction', 'N/A'))}")
    print(f"Confidence: {report.get('primary_prediction', {}).get('confidence', report.get('confidence', 0.0)):.2%}")

    if 'uncertainty' in report:
        print("\nUncertainty Analysis:")
        uncertainty = report['uncertainty']
        print(f"  Needs Human Review: {uncertainty.get('needs_human_review', False)}")
        print(f"  Ensemble Variance: {uncertainty.get('metrics', {}).get('ensemble_variance', 0.0):.4f}")

    print("=" * 50 + "\n")

    # Save results
    if args.output:
        # Convert numpy arrays to lists for JSON serialization
        if 'ensemble_probabilities' in report:
            report['ensemble_probabilities'] = report['ensemble_probabilities'].tolist()
        if 'individual_probabilities' in report:
            report['individual_probabilities'] = [p.tolist() for p in report['individual_probabilities']]
        if 'segmentation_mask' in report:
            # Save mask separately
            mask_path = Path(args.output).parent / f"{Path(args.output).stem}_mask.png"
            cv2.imwrite(str(mask_path), report['segmentation_mask'].astype(np.uint8) * 255)
            report['segmentation_mask_path'] = str(mask_path)
            del report['segmentation_mask']

        with open(args.output, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"Results saved to {args.output}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Hip Implant AI - Segmentation and Classification System"
    )

    parser.add_argument(
        '--mode',
        type=str,
        required=True,
        choices=['train_seg', 'train_cls', 'segment', 'classify', 'ensemble'],
        help='Operation mode'
    )

    parser.add_argument(
        '--input',
        type=str,
        help='Input image path (for inference)'
    )

    parser.add_argument(
        '--output',
        type=str,
        help='Output path for results'
    )

    parser.add_argument(
        '--checkpoint',
        type=str,
        help='Model checkpoint path(s) (comma-separated for ensemble)'
    )

    parser.add_argument(
        '--config',
        type=str,
        help='Configuration file path(s) (comma-separated for ensemble)'
    )

    parser.add_argument(
        '--class-names',
        type=str,
        help='Path to file with class names (one per line)'
    )

    parser.add_argument(
        '--top-k',
        type=int,
        default=5,
        help='Number of top predictions to return'
    )

    parser.add_argument(
        '--clinical-report',
        action='store_true',
        help='Generate comprehensive clinical report'
    )

    parser.add_argument(
        '--extract-roi',
        action='store_true',
        help='Extract and save ROI (for segmentation)'
    )

    parser.add_argument(
        '--ensemble-strategy',
        type=str,
        default='soft_voting',
        choices=['soft_voting', 'hard_voting', 'weighted'],
        help='Ensemble strategy'
    )

    parser.add_argument(
        '--ensemble-weights',
        type=str,
        help='Comma-separated weights for ensemble models'
    )

    parser.add_argument(
        '--ensemble-type',
        type=str,
        default='standard',
        choices=['standard', 'multimodal'],
        help='Type of ensemble'
    )

    parser.add_argument(
        '--fusion-weight',
        type=float,
        default=0.5,
        help='Fusion weight for multi-modal ensemble'
    )

    args = parser.parse_args()

    # Route to appropriate function
    if args.mode == 'train_seg':
        print("Starting segmentation training...")
        train_segmentation()

    elif args.mode == 'train_cls':
        print("Starting classification training...")
        train_classification()

    elif args.mode == 'segment':
        if not args.input or not args.checkpoint:
            parser.error("--input and --checkpoint required for segmentation")
        if not args.output:
            args.output = str(Path(args.input).parent / f"{Path(args.input).stem}_mask.png")
        segment_image(args)

    elif args.mode == 'classify':
        if not args.input or not args.checkpoint:
            parser.error("--input and --checkpoint required for classification")
        classify_image(args)

    elif args.mode == 'ensemble':
        if not args.input or not args.checkpoint:
            parser.error("--input and --checkpoint required for ensemble")
        ensemble_inference(args)


if __name__ == '__main__':
    main()
