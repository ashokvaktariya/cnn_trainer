#!/usr/bin/env python3
"""
Simple inference script to test the trained model on sample images
Runs on server with checkpoint path and test images from mount
"""

import os
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import pandas as pd
import random
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_model():
    """Create EfficientNet-B7 model architecture matching training script"""
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
            features = self.backbone(x)
            return self.classifier(features)
    
    return BinaryEfficientNet(num_classes=2, dropout_rate=0.3, pretrained=True)

def load_model_from_checkpoint(checkpoint_path):
    """Load model from checkpoint directory"""
    logger.info(f"📂 Loading model from: {checkpoint_path}")
    
    # Look for the best model file
    possible_files = [
        'binary_classifier_best.pth',
        'best_model.pth',
        'model_best.pth', 
        'checkpoint_best.pth',
        'best_checkpoint.pth',
        'final_model.pth',
        'model.pth'
    ]
    
    model_file = None
    for filename in possible_files:
        full_path = os.path.join(checkpoint_path, filename)
        if os.path.exists(full_path):
            model_file = full_path
            logger.info(f"📄 Found model file: {filename}")
            break
    
    # If no standard name found, look for any .pth file
    if model_file is None:
        pth_files = [f for f in os.listdir(checkpoint_path) if f.endswith('.pth')]
        if pth_files:
            model_file = os.path.join(checkpoint_path, pth_files[0])
            logger.info(f"📄 Found model file: {pth_files[0]}")
        else:
            logger.error(f"❌ No .pth files found in directory: {checkpoint_path}")
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
    
    # Get class names from checkpoint if available
    class_names = ["NEGATIVE", "POSITIVE"]
    if 'class_names' in checkpoint:
        class_names = checkpoint['class_names']
    
    logger.info(f"✅ Model loaded successfully")
    if 'best_accuracy' in checkpoint:
        logger.info(f"🏆 Best training accuracy: {checkpoint['best_accuracy']:.2f}%")
    
    return model, class_names

def preprocess_image(image_path, input_size=(600, 600)):
    """Preprocess image for inference"""
    try:
        # Load image and ensure it's RGB
        image = Image.open(image_path)
        
        # Convert to RGB if needed (handles grayscale, RGBA, etc.)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Define transforms
        transform = transforms.Compose([
            transforms.Resize(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Apply transforms
        image_tensor = transform(image).unsqueeze(0)
        
        # Verify tensor shape
        if image_tensor.shape[1] != 3:
            logger.warning(f"⚠️ Image has {image_tensor.shape[1]} channels, converting to RGB")
            # Convert grayscale to RGB by repeating the channel
            if image_tensor.shape[1] == 1:
                image_tensor = image_tensor.repeat(1, 3, 1, 1)
            else:
                # Take first 3 channels if more than 3
                image_tensor = image_tensor[:, :3, :, :]
        
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

def get_test_samples(csv_file, image_root, num_samples=10):
    """Get random test samples from the dataset"""
    logger.info(f"📊 Loading dataset from: {csv_file}")
    
    # Load CSV
    df = pd.read_csv(csv_file)
    
    # Filter for valid image paths
    valid_samples = []
    for idx, row in df.iterrows():
        image_path = os.path.join(image_root, row['image_path'])
        if os.path.exists(image_path):
            valid_samples.append({
                'image_path': image_path,
                'label': row['label'],
                'binary_label': row['binary_label']
            })
    
    logger.info(f"📁 Found {len(valid_samples)} valid images")
    
    # Sample random images
    if len(valid_samples) >= num_samples:
        test_samples = random.sample(valid_samples, num_samples)
    else:
        test_samples = valid_samples
    
    logger.info(f"🎯 Selected {len(test_samples)} samples for testing")
    return test_samples

def run_inference():
    """Main inference function"""
    logger.info("🚀 Starting Medical Fracture Detection Inference Test")
    
    # Configuration
    checkpoint_path = "/home/avaktariya/binary_classifier_20250924_131737"
    csv_file = "balanced_dataset_cnn.csv"  # Use balanced dataset
    image_root = "/mount/civiescaks01storage01/aksfileshare01/CNN/gleamer-images/"
    num_samples = 8  # Test 8 samples
    
    # Check if files exist
    if not os.path.exists(checkpoint_path):
        logger.error(f"❌ Checkpoint path not found: {checkpoint_path}")
        return False
    
    if not os.path.exists(csv_file):
        logger.error(f"❌ CSV file not found: {csv_file}")
        return False
    
    if not os.path.exists(image_root):
        logger.error(f"❌ Image root not found: {image_root}")
        return False
    
    # Load model
    model, class_names = load_model_from_checkpoint(checkpoint_path)
    if model is None:
        logger.error("❌ Failed to load model")
        return False
    
    # Get test samples
    test_samples = get_test_samples(csv_file, image_root, num_samples)
    if not test_samples:
        logger.error("❌ No test samples found")
        return False
    
    # Run inference
    logger.info("🔍 Running inference on test samples...")
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
    success = run_inference()
    if success:
        logger.info("✅ Inference test completed successfully!")
    else:
        logger.error("❌ Inference test failed!")
        exit(1)
