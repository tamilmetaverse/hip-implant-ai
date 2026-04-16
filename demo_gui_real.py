#!/usr/bin/env python3
"""
Real Hip Implant AI Demo with Trained Model
Uses actual trained Swin Transformer for predictions
"""

import gradio as gr
import numpy as np
import torch
import cv2
from PIL import Image
import json
from pathlib import Path

# Import project modules
from models.classification.swin import SwinTransformer
from utils.augmentation import ClassificationAugmentation

# Configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CHECKPOINT_PATH = 'experiments/run_20260130_203457/checkpoints/best.pth'
DATA_DIR = 'data/processed'

# Global model variable
model = None
class_names = []
transform = None


def load_model():
    """Load trained model on startup"""
    global model, class_names, transform

    print("Loading trained model...")

    # Load dataset info for class names
    info_file = Path(DATA_DIR) / 'dataset_info.json'
    with open(info_file, 'r') as f:
        dataset_info = json.load(f)

    class_names = dataset_info['class_names']
    num_classes = len(class_names)

    # Create model
    model = SwinTransformer(
        num_classes=num_classes,
        model_name='swin_tiny_patch4_window7_224',
        pretrained=False,
        in_channels=1,  # Grayscale
        dropout=0.3
    )

    # Load checkpoint
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(DEVICE)
    model.eval()

    # Create transform
    transform = ClassificationAugmentation(
        image_size=(224, 224),
        is_training=False
    )

    print("[OK] Model loaded successfully!")
    print(f"[OK] Best validation accuracy: {checkpoint['best_acc']:.2f}%")
    print(f"[OK] Device: {DEVICE}")
    print(f"[OK] Classes: {num_classes}")


def preprocess_image(image):
    """Preprocess uploaded image"""
    if image is None:
        return None

    # Convert to grayscale if RGB
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Apply preprocessing
    preprocessed = transform(image)

    # Convert back to uint8 for display
    display_img = (preprocessed.squeeze().cpu().numpy() * 255).astype(np.uint8)

    return Image.fromarray(display_img), preprocessed


def predict_implant(image):
    """Run real model inference"""
    if image is None:
        return None, "Please upload an X-ray image first."

    # Preprocess
    display_img, tensor = preprocess_image(image)

    # Add batch dimension
    tensor = tensor.unsqueeze(0).to(DEVICE)

    # Inference
    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)

    # Get top 5 predictions
    top5_probs, top5_indices = torch.topk(probs, min(5, len(class_names)))
    top5_probs = top5_probs.cpu().numpy()[0]
    top5_indices = top5_indices.cpu().numpy()[0]

    # Format results
    results = {}
    result_text = "## 🏥 Classification Results\n\n"
    result_text += "### Top 5 Predictions:\n\n"

    for i, (idx, prob) in enumerate(zip(top5_indices, top5_probs)):
        implant_name = class_names[idx]
        confidence = prob * 100
        results[implant_name] = float(confidence)

        # Add emoji for top prediction
        emoji = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else f"{i+1}."

        result_text += f"{emoji} **{implant_name}**\n"
        result_text += f"   - Confidence: {confidence:.2f}%\n"
        result_text += f"   - {'[HIGH CONFIDENCE]' if confidence > 70 else '[MODERATE]' if confidence > 40 else '[LOW]'}\n\n"

    # Add clinical interpretation
    result_text += "\n---\n\n"
    result_text += "### 🔬 Clinical Interpretation:\n\n"

    top_confidence = top5_probs[0] * 100

    if top_confidence > 80:
        result_text += "✅ **High Confidence**: The model strongly suggests the top prediction.\n"
    elif top_confidence > 60:
        result_text += "⚠️ **Moderate Confidence**: Consider the top 2-3 predictions.\n"
    else:
        result_text += "❌ **Low Confidence**: Review all top 5 predictions carefully.\n"

    result_text += "\n*Note: This is AI-assisted decision support. Final diagnosis should be made by qualified medical professionals.*"

    return results, result_text, display_img


def process_full_pipeline(image):
    """Complete pipeline: preprocessing + inference"""
    if image is None:
        return None, None, "Please upload an image first."

    # Run prediction
    results, result_text, preprocessed = predict_implant(image)

    return preprocessed, results, result_text


# Custom CSS for medical theme
custom_css = """
    .gradio-container {
        font-family: 'Inter', sans-serif;
    }
    .medical-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 30px;
        border-radius: 10px;
        margin-bottom: 20px;
        text-align: center;
    }
    .result-box {
        border: 2px solid #667eea;
        border-radius: 10px;
        padding: 20px;
        margin-top: 20px;
        background: #f8f9fa;
    }
    .footer {
        text-align: center;
        margin-top: 30px;
        padding: 20px;
        color: #666;
        font-size: 14px;
    }
"""


def create_interface():
    """Create Gradio interface"""

    with gr.Blocks(title="Hip Implant AI - Real Model Demo") as demo:

        # Header
        gr.HTML("""
            <div class="medical-header">
                <h1>🏥 Hip Implant AI Classification System</h1>
                <p style="font-size: 18px; margin-top: 10px;">
                    AI-Powered Hip Implant Identification using Swin Transformer
                </p>
                <p style="font-size: 14px; opacity: 0.9;">
                    62.2% Top-1 Accuracy | 97.3% Top-5 Accuracy | Trained on 910 Clinical X-rays
                </p>
            </div>
        """)

        # Main content
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("## 📤 Upload X-Ray Image")
                input_image = gr.Image(
                    label="Hip X-Ray Image",
                    type="numpy",
                    height=400
                )

                with gr.Row():
                    classify_btn = gr.Button("🔍 Classify Implant", variant="primary", size="lg")
                    clear_btn = gr.ClearButton([input_image], value="🗑️ Clear")

                gr.Markdown("""
                ### 📋 Instructions:
                1. Upload a hip X-ray image (grayscale or RGB)
                2. Click "Classify Implant" button
                3. View top 5 predictions with confidence scores
                4. Review preprocessed image on the right

                **Supported Formats:** JPG, PNG, JPEG
                """)

            with gr.Column(scale=1):
                gr.Markdown("## 🖼️ Preprocessed Image")
                preprocessed_image = gr.Image(
                    label="Preprocessed (224x224 Grayscale)",
                    height=400
                )

                gr.Markdown("## 📊 Confidence Scores")
                confidence_plot = gr.Label(
                    label="Top 5 Predictions",
                    num_top_classes=5
                )

        # Results section
        with gr.Row():
            with gr.Column():
                result_text = gr.Markdown(
                    label="Detailed Results",
                    value="Upload an image and click 'Classify Implant' to see results."
                )

        # Model Information
        with gr.Accordion("ℹ️ Model Information", open=False):
            gr.Markdown(f"""
            ### Model Architecture:
            - **Model:** Swin Transformer Tiny
            - **Parameters:** 27.5M
            - **Input:** 224×224 Grayscale X-ray
            - **Classes:** 11 Hip Implant Types
            - **Device:** {DEVICE}

            ### Performance (Test Set):
            - **Top-1 Accuracy:** 62.16%
            - **Top-5 Accuracy:** 97.30%
            - **Training Data:** 910 images (631 train, 131 val, 148 test)

            ### Supported Implant Types:
            {', '.join(class_names)}

            ### Technical Details:
            - Transfer learning from ImageNet
            - Class-weighted loss for imbalance handling
            - Data augmentation (rotation, flip, brightness)
            - Early stopping with patience=10
            - GPU-accelerated training (RTX 4070)
            """)

        # Usage Examples
        with gr.Accordion("📚 Usage Examples", open=False):
            gr.Markdown("""
            ### Example Use Cases:

            1. **Pre-operative Planning:**
               - Upload patient X-ray before revision surgery
               - Identify existing implant manufacturer and model
               - Plan appropriate surgical approach

            2. **Emergency Situations:**
               - Quick implant identification in trauma cases
               - Assist when patient history unavailable
               - Reduce time spent searching implant databases

            3. **Research and Education:**
               - Study implant characteristics
               - Train medical students on implant recognition
               - Analyze large datasets of hip replacements

            ### Clinical Workflow:
            1. System provides top 5 candidates (97% contain correct implant)
            2. Surgeon reviews candidates with visual references
            3. Final decision made by medical professional
            4. AI serves as decision support, not autonomous diagnosis
            """)

        # Footer
        gr.HTML("""
            <div class="footer">
                <p><strong>Hip Implant AI Classification System</strong></p>
                <p>Developed using PyTorch, Swin Transformer, and Gradio</p>
                <p style="color: #e74c3c; margin-top: 10px;">
                    ⚠️ For research and educational purposes only.
                    Not FDA approved for clinical use.
                </p>
            </div>
        """)

        # Event handlers
        classify_btn.click(
            fn=process_full_pipeline,
            inputs=[input_image],
            outputs=[preprocessed_image, confidence_plot, result_text]
        )

    return demo


# Main execution
if __name__ == "__main__":
    print("="*70)
    print("HIP IMPLANT AI - REAL MODEL DEMO")
    print("="*70)

    # Load model
    load_model()

    # Create interface
    demo = create_interface()

    # Launch
    print("\nLaunching Gradio interface...")
    print("="*70)

    demo.launch(
        share=False,
        server_name="127.0.0.1",
        show_error=True,
        inbrowser=True,
        css=custom_css
    )
