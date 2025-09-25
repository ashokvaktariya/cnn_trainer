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
import random
import logging
import glob

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
    model_file = os.path.join(checkpoint_path, 'binary_classifier_best.pth')
    
    if not os.path.exists(model_file):
        logger.error(f"❌ Model file not found: {model_file}")
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
    
    logger.info(f"✅ Model loaded successfully")
    if 'best_accuracy' in checkpoint:
        logger.info(f"🏆 Best training accuracy: {checkpoint['best_accuracy']:.2f}%")
    
    return model, class_names

def preprocess_image(image_path, input_size=(600, 600)):
    """Preprocess image for inference"""
    # Load image
    image = Image.open(image_path).convert('RGB')
    
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Apply transforms
    image_tensor = transform(image).unsqueeze(0)
    return image_tensor, image

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

def get_random_images(image_root, num_samples=8):
    """Get random images from the mount directory"""
    logger.info(f"📁 Searching for images in: {image_root}")
    
    # Find all image files
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
    image_files = []
    
    for ext in image_extensions:
        pattern = os.path.join(image_root, '**', ext)
        image_files.extend(glob.glob(pattern, recursive=True))
    
    logger.info(f"📊 Found {len(image_files)} images")
    
    if len(image_files) == 0:
        logger.error("❌ No images found in the directory")
        return []
    
    # Sample random images
    if len(image_files) >= num_samples:
        selected_images = random.sample(image_files, num_samples)
    else:
        selected_images = image_files
    
    logger.info(f"🎯 Selected {len(selected_images)} images for testing")
    return selected_images

def run_simple_inference():
    """Main inference function"""
    logger.info("🚀 Starting Simple Medical Fracture Detection Inference Test")
    
    # Configuration
    checkpoint_path = "/home/avaktariya/binary_classifier_20250924_131737"
    image_root = "/mount/civiescaks01storage01/aksfileshare01/CNN/gleamer-images/"
    num_samples = 8  # Test 8 samples
    
    # Check if paths exist
    if not os.path.exists(checkpoint_path):
        logger.error(f"❌ Checkpoint path not found: {checkpoint_path}")
        return False
    
    if not os.path.exists(image_root):
        logger.error(f"❌ Image root not found: {image_root}")
        return False
    
    # Load model
    model, class_names = load_model_from_checkpoint(checkpoint_path)
    if model is None:
        logger.error("❌ Failed to load model")
        return False
    
    # Get random images
    test_images = get_random_images(image_root, num_samples)
    if not test_images:
        logger.error("❌ No test images found")
        return False
    
    # Run inference
    logger.info("🔍 Running inference on random images...")
    logger.info("=" * 80)
    
    for i, image_path in enumerate(test_images, 1):
        try:
            # Preprocess image
            image_tensor, image = preprocess_image(image_path)
            
            # Run prediction
            result = predict_image(model, image_tensor, class_names)
            
            # Display results
            logger.info(f"📸 Sample {i}/{len(test_images)}:")
            logger.info(f"   📁 Image: {os.path.basename(image_path)}")
            logger.info(f"   🔮 Predicted: {result['predicted_label']}")
            logger.info(f"   📊 Confidence: {result['confidence']:.3f}")
            logger.info(f"   📈 Probabilities: NEGATIVE={result['probabilities']['negative']:.3f}, POSITIVE={result['probabilities']['positive']:.3f}")
            logger.info("-" * 60)
            
        except Exception as e:
            logger.error(f"❌ Error processing image {i}: {e}")
            continue
    
    logger.info("=" * 80)
    logger.info("✅ Simple inference test completed!")
    logger.info(f"🎯 Tested {len(test_images)} images")
    logger.info("=" * 80)
    
    return True

if __name__ == "__main__":
    success = run_simple_inference()
    if success:
        logger.info("🎉 Simple inference test completed successfully!")
    else:
        logger.error("❌ Simple inference test failed!")
        exit(1)
