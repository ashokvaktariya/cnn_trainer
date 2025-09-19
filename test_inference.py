#!/usr/bin/env python3
"""
Simple inference script for testing individual images
"""

import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
import argparse
import os
import logging
from pathlib import Path

# Import our model
from models import BinaryEfficientNet

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BinaryClassifierInference:
    """Binary classifier inference class"""
    
    def __init__(self, model_path, device='cuda'):
        """Initialize inference"""
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        logger.info(f"📱 Using device: {self.device}")
        
        # Load model
        logger.info(f"📦 Loading model from: {model_path}")
        self.model = BinaryEfficientNet(num_classes=2)
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        # Setup transforms (same as training)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        logger.info("✅ Model loaded successfully!")
    
    def predict(self, image_path):
        """Predict on a single image"""
        try:
            # Load and preprocess image
            image = Image.open(image_path)
            
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Apply transforms
            input_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # Inference
            with torch.no_grad():
                logits = self.model(input_tensor)
                probabilities = F.softmax(logits, dim=1)
                predicted_class = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0][predicted_class].item()
            
            # Convert to labels
            class_names = ['NEGATIVE', 'POSITIVE']
            predicted_label = class_names[predicted_class]
            
            return {
                'predicted_class': predicted_class,
                'predicted_label': predicted_label,
                'confidence': confidence,
                'probabilities': {
                    'NEGATIVE': probabilities[0][0].item(),
                    'POSITIVE': probabilities[0][1].item()
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Error processing image {image_path}: {e}")
            return None
    
    def test_image(self, image_path, true_label=None):
        """Test a single image and show results"""
        logger.info(f"🔍 Testing image: {os.path.basename(image_path)}")
        
        if true_label:
            logger.info(f"📋 True label: {true_label}")
        
        # Get prediction
        result = self.predict(image_path)
        
        if result is None:
            logger.error("❌ Failed to process image")
            return
        
        # Display results
        logger.info("📊 Prediction Results:")
        logger.info(f"   Predicted: {result['predicted_label']}")
        logger.info(f"   Confidence: {result['confidence']:.4f} ({result['confidence']*100:.2f}%)")
        logger.info(f"   Probabilities:")
        logger.info(f"     NEGATIVE: {result['probabilities']['NEGATIVE']:.4f} ({result['probabilities']['NEGATIVE']*100:.2f}%)")
        logger.info(f"     POSITIVE: {result['probabilities']['POSITIVE']:.4f} ({result['probabilities']['POSITIVE']*100:.2f}%)")
        
        # Check if prediction is correct
        if true_label:
            is_correct = result['predicted_label'] == true_label
            status = "✅ CORRECT" if is_correct else "❌ INCORRECT"
            logger.info(f"   Result: {status}")
        
        return result

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Test binary classifier on individual images')
    parser.add_argument('--image', type=str, required=True, help='Path to image file')
    parser.add_argument('--model', type=str, default='/sharedata01/CNN_data/medical_classification/checkpoints/binary_classifier_best.pth', help='Path to model checkpoint')
    parser.add_argument('--label', type=str, help='True label (POSITIVE/NEGATIVE) for validation')
    parser.add_argument('--batch', action='store_true', help='Process all images in a directory')
    parser.add_argument('--output', type=str, help='Output file for batch results')
    
    args = parser.parse_args()
    
    # Initialize inference
    inference = BinaryClassifierInference(args.model)
    
    if args.batch:
        # Batch processing
        if not os.path.isdir(args.image):
            logger.error("❌ Batch mode requires a directory path")
            return
        
        results = []
        logger.info(f"📁 Processing directory: {args.image}")
        
        # Get all image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(Path(args.image).glob(f'*{ext}'))
            image_files.extend(Path(args.image).glob(f'*{ext.upper()}'))
        
        logger.info(f"🖼️ Found {len(image_files)} images")
        
        for image_path in sorted(image_files):
            # Extract label from filename if available
            filename = image_path.name
            true_label = None
            
            if 'LABEL_POSITIVE' in filename:
                true_label = 'POSITIVE'
            elif 'LABEL_NEGATIVE' in filename:
                true_label = 'NEGATIVE'
            elif 'LABEL_DOUBT' in filename:
                true_label = 'DOUBT'
            
            logger.info(f"\n{'='*60}")
            result = inference.test_image(str(image_path), true_label)
            
            if result:
                results.append({
                    'image': filename,
                    'true_label': true_label,
                    'predicted_label': result['predicted_label'],
                    'confidence': result['confidence'],
                    'probabilities': result['probabilities']
                })
        
        # Summary
        logger.info(f"\n{'='*60}")
        logger.info("📊 BATCH SUMMARY:")
        
        correct = 0
        total = 0
        
        for result in results:
            if result['true_label'] and result['true_label'] != 'DOUBT':
                total += 1
                if result['predicted_label'] == result['true_label']:
                    correct += 1
        
        if total > 0:
            accuracy = correct / total
            logger.info(f"   Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
            logger.info(f"   Correct: {correct}/{total}")
        
        # Save results if output file specified
        if args.output:
            import json
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"💾 Results saved: {args.output}")
    
    else:
        # Single image processing
        if not os.path.isfile(args.image):
            logger.error(f"❌ Image file not found: {args.image}")
            return
        
        logger.info(f"\n{'='*60}")
        inference.test_image(args.image, args.label)

if __name__ == "__main__":
    main()
