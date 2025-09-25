#!/usr/bin/env python3
"""
Simple inference script for server testing
Tests model on random images from the mount directory
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

def get_test_images_from_csv(csv_file, test_images_folder, num_samples=8):
    """Get test images from CSV dataset using test_images folder"""
    logger.info(f"📊 Loading dataset from: {csv_file}")
    
    # Load CSV
    df = pd.read_csv(csv_file)
    
    # Filter for valid image paths in test_images folder
    valid_samples = []
    for idx, row in df.iterrows():
        # Look for image in test_images folder
        image_filename = os.path.basename(row['image_path'])
        test_image_path = os.path.join(test_images_folder, image_filename)
        
        if os.path.exists(test_image_path):
            valid_samples.append({
                'image_path': test_image_path,
                'label': row['label'],
                'binary_label': row['binary_label']
            })
    
    logger.info(f"📁 Found {len(valid_samples)} valid images in test_images folder")
    
    if len(valid_samples) == 0:
        logger.error("❌ No valid images found in test_images folder")
        logger.info(f"💡 Make sure test_images folder exists and contains images from CSV")
        return []
    
    # Sample random images
    if len(valid_samples) >= num_samples:
        selected_samples = random.sample(valid_samples, num_samples)
    else:
        selected_samples = valid_samples
    
    logger.info(f"🎯 Selected {len(selected_samples)} images for testing")
    return selected_samples

def run_simple_inference():
    """Main inference function"""
    logger.info("🚀 Starting RunPod Medical Fracture Detection Inference Test")
    
    # Configuration
    csv_file = "final_dataset_cnn.csv"  # Use final dataset
    test_images_folder = "test_images"  # Local test images folder
    num_samples = 8  # Test 8 samples
    
    # Check if paths exist
    if not os.path.exists(csv_file):
        logger.error(f"❌ CSV file not found: {csv_file}")
        return False
    
    if not os.path.exists(test_images_folder):
        logger.error(f"❌ Test images folder not found: {test_images_folder}")
        logger.info("💡 Create test_images folder and add some images from the dataset")
        return False
    
    # Load model from HF download
    model, class_names = load_model_from_hf()
    if model is None:
        logger.error("❌ Failed to load model")
        return False
    
    # Get test images from CSV dataset
    test_samples = get_test_images_from_csv(csv_file, test_images_folder, num_samples)
    if not test_samples:
        logger.error("❌ No test samples found")
        return False
    
    # Run inference
    logger.info("🔍 Running inference on test_images folder samples...")
    logger.info("=" * 80)
    
    correct_predictions = 0
    total_predictions = len(test_samples)
    
    for i, sample in enumerate(test_samples, 1):
        image_path = sample['image_path']
        true_label = sample['label']
        true_binary = sample['binary_label']
        
        try:
            # Preprocess image
            image_tensor, image = preprocess_image(image_path)
            
            # Run prediction
            result = predict_image(model, image_tensor, class_names)
            
            # Check if prediction is correct
            is_correct = (result['predicted_class'] == true_binary)
            if is_correct:
                correct_predictions += 1
            
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
    logger.info("📊 FINAL RESULTS:")
    logger.info(f"   🎯 Total Samples: {total_predictions}")
    logger.info(f"   ✅ Correct Predictions: {correct_predictions}")
    logger.info(f"   ❌ Incorrect Predictions: {total_predictions - correct_predictions}")
    logger.info(f"   📈 Accuracy: {accuracy:.2f}%")
    logger.info("=" * 80)
    
    if accuracy >= 80:
        logger.info("🎉 Model performance looks good!")
    elif accuracy >= 60:
        logger.info("⚠️ Model performance is moderate")
    else:
        logger.info("🚨 Model performance needs improvement")
    
    return True

if __name__ == "__main__":
    success = run_simple_inference()
    if success:
        logger.info("🎉 RunPod inference test completed successfully!")
    else:
        logger.error("❌ RunPod inference test failed!")
        exit(1)
