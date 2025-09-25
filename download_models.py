#!/usr/bin/env python3
"""
Download models from Hugging Face and setup for inference
Downloads the trained model and EfficientNet backbone
"""

import os
import torch
from huggingface_hub import hf_hub_download, snapshot_download
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_trained_model():
    """Download the trained model from Hugging Face"""
    logger.info("📥 Downloading trained model from Hugging Face...")
    
    try:
        # Download the best model file
        model_path = hf_hub_download(
            repo_id="5techlab-research/cnn_med_33k",
            filename="binary_classifier_best.pth",
            local_dir=".",
            local_dir_use_symlinks=False
        )
        
        logger.info(f"✅ Trained model downloaded to: {model_path}")
        return model_path
        
    except Exception as e:
        logger.error(f"❌ Failed to download trained model: {e}")
        return None

def download_efficientnet_backbone():
    """Download EfficientNet-B7 backbone"""
    logger.info("📥 Downloading EfficientNet-B7 backbone...")
    
    try:
        # Download from torchvision hub
        model = torch.hub.load('pytorch/vision', 'efficientnet_b7', pretrained=True)
        
        # Save the backbone
        backbone_path = "./models/efficientnet_b7_backbone.pth"
        os.makedirs("./models", exist_ok=True)
        torch.save(model.state_dict(), backbone_path)
        
        logger.info(f"✅ EfficientNet-B7 backbone saved to: {backbone_path}")
        return backbone_path
        
    except Exception as e:
        logger.error(f"❌ Failed to download EfficientNet backbone: {e}")
        return None

def create_model_config():
    """Create model configuration file"""
    config = {
        "model_architecture": "efficientnet_b7",
        "num_classes": 2,
        "class_names": ["NEGATIVE", "POSITIVE"],
        "input_size": [600, 600],
        "pretrained": True,
        "dropout": 0.3,
        "model_path": "binary_classifier_best.pth",
        "backbone_path": "efficientnet_b7_backbone.pth"
    }
    
    import json
    config_path = "model_config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    logger.info(f"✅ Model configuration saved to: {config_path}")
    return config_path

def setup_models():
    """Setup all models for inference"""
    logger.info("🚀 Setting up models for inference...")
    
    # Files will be saved to root directory
    
    # Download trained model
    trained_model = download_trained_model()
    if not trained_model:
        logger.error("❌ Failed to download trained model")
        return False
    
    # Download EfficientNet backbone
    backbone = download_efficientnet_backbone()
    if not backbone:
        logger.error("❌ Failed to download EfficientNet backbone")
        return False
    
    # Create model configuration
    config = create_model_config()
    
    logger.info("🎉 All models setup complete!")
    logger.info("📁 Root directory structure:")
    logger.info("   ├── binary_classifier_best.pth")
    logger.info("   ├── efficientnet_b7_backbone.pth")
    logger.info("   └── model_config.json")
    
    return True

if __name__ == "__main__":
    success = setup_models()
    if success:
        logger.info("✅ Model setup completed successfully!")
        logger.info("🚀 Ready to run inference server!")
    else:
        logger.error("❌ Model setup failed!")
        exit(1)
