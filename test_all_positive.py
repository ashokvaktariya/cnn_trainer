#!/usr/bin/env python3
"""
Script to test model on all positive images from test_images folder
Uses CSV to identify positive images and tests them all
"""

import os
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import pandas as pd
import numpy as np
import random
import logging
import glob

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_model():
    """Create EfficientNet-B7 model architecture matching the checkpoint"""
    try:
        from efficientnet_pytorch import EfficientNet
    except ImportError:
        logger.error("❌ efficientnet_pytorch not installed. Install with: pip install efficientnet-pytorch")
        return None
    
    # Create BinaryEfficientNet model exactly like training
    class BinaryEfficientNet(torch.nn.Module):
        def __init__(self, num_classes=2, dropout_rate=0.3, pretrained=True, **kwargs):
            super().__init__()
            
            # Use EfficientNet-B7 backbone without classifier
            self.backbone = EfficientNet.from_pretrained(
                "efficientnet-b7",
                num_classes=1000,  # Use default ImageNet classes
                dropout_rate=dropout_rate
            )
            
            # Remove the original classifier
            self.backbone._fc = torch.nn.Identity()
            
            # Get feature dimension from EfficientNet-B7
            feature_dim = 2560  # EfficientNet-B7 feature dimension
            
            # Binary classification head
            self.classifier = torch.nn.Sequential(
                torch.nn.AdaptiveAvgPool2d((1, 1)),
                torch.nn.Flatten(),
                torch.nn.Dropout(dropout_rate),
                torch.nn.Linear(feature_dim, 512),
                torch.nn.ReLU(),
                torch.nn.Dropout(dropout_rate),
                torch.nn.Linear(512, 256),
                torch.nn.ReLU(),
                torch.nn.Dropout(dropout_rate),
                torch.nn.Linear(256, num_classes)
            )
        
        def forward(self, x):
            # Extract features from backbone
            features = self.backbone.extract_features(x)
            return self.classifier(features)
    
    return BinaryEfficientNet(num_classes=2, dropout_rate=0.3, pretrained=True)

def load_model_from_hf():
    """Load model from Hugging Face downloaded files in root directory"""
    logger.info("📂 Loading model from Hugging Face downloaded files...")
    
    # Look for the model file in root directory
    model_file = "binary_classifier_best.pth"
    
    if not os.path.exists(model_file):
        logger.error(f"❌ Model file not found: {model_file}")
        logger.info("💡 Run download_models.py first to download the model")
        return None, None
    
    # Create model architecture
    model = create_model()
    
    # Load checkpoint
    logger.info(f"📥 Loading checkpoint: {model_file}")
    checkpoint = torch.load(model_file, map_location='cpu')
    
    # Load state dict
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    
    # Get class names
    class_names = ["NEGATIVE", "POSITIVE"]
    if 'class_names' in checkpoint:
        class_names = checkpoint['class_names']
    
    logger.info(f"✅ Model loaded successfully from HF download")
    if 'best_accuracy' in checkpoint:
        logger.info(f"🏆 Best training accuracy: {checkpoint['best_accuracy']:.2f}%")
    
    return model, class_names

def preprocess_image(image_path, input_size=600):
    """Preprocess image for inference - copy from working inference_api.py"""
    try:
        # Read and process image - simple approach like inference_api.py
        image = Image.open(image_path).convert('RGB')
        
        # Define transforms - copy from inference_api.py
        transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Apply transforms
        image_tensor = transform(image).unsqueeze(0)
        
        return image_tensor, image
        
    except Exception as e:
        logger.error(f"❌ Error preprocessing image {image_path}: {e}")
        raise e

def predict_image(model, image_tensor, class_names):
    """Run inference on image tensor"""
    with torch.no_grad():
        # Forward pass
        outputs = model(image_tensor)
        probabilities = F.softmax(outputs, dim=1)
        
        # Get prediction
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item()
        
        # Get predicted label
        predicted_label = class_names[predicted_class]
        
        # Get probabilities for both classes
        prob_dict = {
            class_names[0].lower(): probabilities[0][0].item(),
            class_names[1].lower(): probabilities[0][1].item()
        }
        
        return {
            "predicted_class": predicted_class,
            "predicted_label": predicted_label,
            "confidence": confidence,
            "probabilities": prob_dict
        }

def get_all_positive_images(csv_file, test_images_folder):
    """Get all positive images from test_images folder using CSV"""
    logger.info(f"📊 Loading dataset from: {csv_file}")
    
    # Load CSV
    df = pd.read_csv(csv_file)
    
    # Filter for positive images in test_images folder
    positive_samples = []
    for idx, row in df.iterrows():
        # Look for image in test_images folder
        image_filename = os.path.basename(row['image_path'])
        test_image_path = os.path.join(test_images_folder, image_filename)
        
        # Only include positive images that exist in test_images folder
        if os.path.exists(test_image_path) and row['binary_label'] == 1:
            positive_samples.append({
                'image_path': test_image_path,
                'label': row['label'],
                'binary_label': row['binary_label']
            })
    
    logger.info(f"📁 Found {len(positive_samples)} positive images in test_images folder")
    
    if len(positive_samples) == 0:
        logger.error("❌ No positive images found in test_images folder")
        logger.info(f"💡 Make sure test_images folder exists and contains positive images from CSV")
        return []
    
    logger.info(f"🎯 Testing all {len(positive_samples)} positive images")
    return positive_samples

def run_positive_inference():
    """Main inference function for all positive images"""
    logger.info("🚀 Starting Positive Images Inference Test")
    
    # Configuration
    csv_file = "final_dataset_cnn.csv"  # Use final dataset
    test_images_folder = "test_images"  # Local test images folder
    
    # Check if paths exist
    if not os.path.exists(csv_file):
        logger.error(f"❌ CSV file not found: {csv_file}")
        return False
    
    if not os.path.exists(test_images_folder):
        logger.error(f"❌ Test images folder not found: {test_images_folder}")
        logger.info("💡 Run copy_test_images.py first to create test_images folder")
        return False
    
    # Load model from HF download
    model, class_names = load_model_from_hf()
    if model is None:
        logger.error("❌ Failed to load model")
        return False
    
    # Get all positive images from test_images folder
    positive_samples = get_all_positive_images(csv_file, test_images_folder)
    if not positive_samples:
        logger.error("❌ No positive samples found")
        return False
    
    # Run inference
    logger.info("🔍 Running inference on all positive images...")
    logger.info("=" * 80)
    
    correct_predictions = 0
    total_predictions = len(positive_samples)
    high_confidence_correct = 0
    low_confidence_correct = 0
    
    for i, sample in enumerate(positive_samples, 1):
        image_path = sample['image_path']
        true_label = sample['label']
        true_binary = sample['binary_label']
        
        try:
            # Preprocess image
            image_tensor, image = preprocess_image(image_path)
            
            # Run prediction
            result = predict_image(model, image_tensor, class_names)
            
            # Check if prediction is correct (should be POSITIVE = 1)
            is_correct = (result['predicted_class'] == true_binary)
            if is_correct:
                correct_predictions += 1
                
                # Track confidence levels
                if result['confidence'] >= 0.8:
                    high_confidence_correct += 1
                else:
                    low_confidence_correct += 1
            
            # Display results
            status = "✅" if is_correct else "❌"
            logger.info(f"{status} Sample {i}/{total_predictions}:")
            logger.info(f"   📁 Image: {os.path.basename(image_path)}")
            logger.info(f"   🏷️  True Label: {true_label}")
            logger.info(f"   🔮 Predicted: {result['predicted_label']}")
            logger.info(f"   📊 Confidence: {result['confidence']:.3f}")
            logger.info(f"   📈 Probabilities: NEGATIVE={result['probabilities']['negative']:.3f}, POSITIVE={result['probabilities']['positive']:.3f}")
            logger.info(f"   {'✅ CORRECT' if is_correct else '❌ INCORRECT'}")
            logger.info("-" * 60)
            
        except Exception as e:
            logger.error(f"❌ Error processing sample {i}: {e}")
            continue
    
    # Final results
    accuracy = (correct_predictions / total_predictions) * 100
    logger.info("=" * 80)
    logger.info("📊 FINAL RESULTS FOR POSITIVE IMAGES:")
    logger.info(f"   🎯 Total Positive Images: {total_predictions}")
    logger.info(f"   ✅ Correctly Predicted as Positive: {correct_predictions}")
    logger.info(f"   ❌ Incorrectly Predicted as Negative: {total_predictions - correct_predictions}")
    logger.info(f"   📈 Accuracy: {accuracy:.2f}%")
    logger.info(f"   🔥 High Confidence Correct (≥80%): {high_confidence_correct}")
    logger.info(f"   🔸 Low Confidence Correct (<80%): {low_confidence_correct}")
    logger.info("=" * 80)
    
    if accuracy >= 90:
        logger.info("🎉 Excellent performance on positive images!")
    elif accuracy >= 80:
        logger.info("✅ Good performance on positive images!")
    elif accuracy >= 70:
        logger.info("⚠️ Moderate performance on positive images")
    else:
        logger.info("🚨 Poor performance on positive images - needs improvement")
    
    return True

if __name__ == "__main__":
    success = run_positive_inference()
    if success:
        logger.info("🎉 Positive images inference test completed successfully!")
    else:
        logger.error("❌ Positive images inference test failed!")
        exit(1)
