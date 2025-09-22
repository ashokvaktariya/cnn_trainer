#!/usr/bin/env python3
"""
Medical Fracture Detection - FastAPI Inference Server
Optimized for H100 GPU with single file solution
"""

import os
import io
import base64
import logging
from typing import List, Dict, Optional
from pathlib import Path
import json

import torch
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import yaml

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Medical Fracture Detection API",
    description="Binary classification API for detecting fractures in medical images",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
model = None
device = None
transform = None
config = None

class BinaryClassifier(torch.nn.Module):
    """Binary classifier model architecture"""
    def __init__(self, num_classes=2, dropout=0.2):
        super(BinaryClassifier, self).__init__()
        
        # Use EfficientNet-B0 as backbone
        self.backbone = models.efficientnet_b0(pretrained=False)
        
        # Modify the classifier for binary classification
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = torch.nn.Sequential(
            torch.nn.Dropout(dropout),
            torch.nn.Linear(in_features, num_classes)
        )
        
    def forward(self, x):
        return self.backbone(x)

def load_config():
    """Load configuration"""
    global config
    config_path = "config_training.yaml"
    
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info("✅ Configuration loaded")
    else:
        # Default configuration
        config = {
            'model': {'num_classes': 2, 'dropout': 0.2},
            'data': {'image_size': [224, 224]},
            'hardware': {'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
        }
        logger.info("⚠️ Using default configuration")

def load_model(model_path: str = None):
    """Load the trained binary classifier model"""
    global model, device, transform
    
    try:
        # Set device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")
        
        # Define image preprocessing first
        image_size = config['data']['image_size']
        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Initialize model
        model = BinaryClassifier(
            num_classes=config['model']['num_classes'],
            dropout=config['model']['dropout']
        )
        
        # Try to load model weights
        if model_path and os.path.exists(model_path):
            logger.info(f"Loading model from: {model_path}")
            checkpoint = torch.load(model_path, map_location=device)
            
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            
            logger.info("✅ Model weights loaded successfully!")
        else:
            logger.warning("⚠️ No model weights found, using random initialization")
        
        model.to(device)
        model.eval()
        
        logger.info("✅ Model loaded successfully!")
        logger.info(f"✅ Transform pipeline initialized")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        return False

def preprocess_image(image_data: bytes) -> torch.Tensor:
    """Preprocess image for inference"""
    try:
        # Load image
        image = Image.open(io.BytesIO(image_data))
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Apply transforms
        if transform is None:
            raise HTTPException(status_code=503, detail="Model not properly loaded")
        
        tensor = transform(image)
        tensor = tensor.unsqueeze(0)  # Add batch dimension
        
        return tensor
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Image preprocessing failed: {e}")
        raise HTTPException(status_code=400, detail=f"Image preprocessing failed: {str(e)}")

def predict_image(image_tensor: torch.Tensor) -> Dict:
    """Run inference on a single image"""
    try:
        with torch.no_grad():
            image_tensor = image_tensor.to(device)
            output = model(image_tensor)
            probabilities = F.softmax(output, dim=1)
            
            # Get predictions
            confidence, predicted = torch.max(probabilities, 1)
            predicted_class = predicted.item()
            confidence_score = confidence.item()
            
            # Map class indices to labels
            class_labels = {0: "NEGATIVE", 1: "POSITIVE"}
            predicted_label = class_labels[predicted_class]
            
            # Get probability for each class
            prob_negative = probabilities[0][0].item()
            prob_positive = probabilities[0][1].item()
            
            return {
                "predicted_class": predicted_class,
                "predicted_label": predicted_label,
                "confidence": confidence_score,
                "probabilities": {
                    "negative": prob_negative,
                    "positive": prob_positive
                }
            }
            
    except Exception as e:
        logger.error(f"❌ Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.on_event("startup")
async def startup_event():
    """Initialize the model on startup"""
    logger.info("🚀 Starting Medical Fracture Detection API...")
    
    # Load configuration
    load_config()
    
    # Try to find model file
    model_paths = [
        "/mount/civiescaks01storage01/aksfileshare01/CNN/gleamer-images/best_model.pth",
        "best_model.pth",
        "binary_classifier_best.pth"
    ]
    
    model_loaded = False
    for model_path in model_paths:
        if os.path.exists(model_path):
            if load_model(model_path):
                model_loaded = True
                break
    
    if not model_loaded:
        logger.warning("⚠️ No model file found, using random initialization")
        load_model()
    
    logger.info("✅ API ready for inference!")

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Medical Fracture Detection API v2.0",
        "version": "2.0.0",
        "status": "running",
        "model_loaded": model is not None,
        "device": str(device) if device else "unknown",
        "config_loaded": config is not None
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": str(device) if device else "unknown",
        "cuda_available": torch.cuda.is_available(),
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0
    }

@app.post("/predict")
async def predict_single_image(file: UploadFile = File(...)):
    """Predict fracture in a single image"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read image data
        image_data = await file.read()
        
        # Preprocess image
        image_tensor = preprocess_image(image_data)
        
        # Run prediction
        result = predict_image(image_tensor)
        
        # Add metadata
        result.update({
            "filename": file.filename,
            "file_size": len(image_data),
            "content_type": file.content_type
        })
        
        logger.info(f"✅ Prediction completed for {file.filename}: {result['predicted_label']} ({result['confidence']:.3f})")
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/predict_batch")
async def predict_batch_images(files: List[UploadFile] = File(...)):
    """Predict fractures in multiple images"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if len(files) > 10:  # Limit batch size
        raise HTTPException(status_code=400, detail="Maximum 10 images per batch")
    
    results = []
    
    for file in files:
        try:
            # Validate file type
            if not file.content_type.startswith('image/'):
                results.append({
                    "filename": file.filename,
                    "error": "File must be an image"
                })
                continue
            
            # Read image data
            image_data = await file.read()
            
            # Preprocess image
            image_tensor = preprocess_image(image_data)
            
            # Run prediction
            result = predict_image(image_tensor)
            
            # Add metadata
            result.update({
                "filename": file.filename,
                "file_size": len(image_data),
                "content_type": file.content_type
            })
            
            results.append(result)
            
        except Exception as e:
            logger.error(f"❌ Error processing {file.filename}: {e}")
            results.append({
                "filename": file.filename,
                "error": str(e)
            })
    
    logger.info(f"✅ Batch prediction completed for {len(files)} images")
    
    return {
        "total_images": len(files),
        "successful_predictions": len([r for r in results if "error" not in r]),
        "results": results
    }

@app.post("/predict_base64")
async def predict_base64_image(image_data: str = Form(...)):
    """Predict fracture from base64 encoded image"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Decode base64 image
        image_bytes = base64.b64decode(image_data)
        
        # Preprocess image
        image_tensor = preprocess_image(image_bytes)
        
        # Run prediction
        result = predict_image(image_tensor)
        
        logger.info(f"✅ Base64 prediction completed: {result['predicted_label']} ({result['confidence']:.3f})")
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Base64 prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Base64 prediction failed: {str(e)}")

@app.get("/model_info")
async def get_model_info():
    """Get model information"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        "model_architecture": "EfficientNet-B0",
        "num_classes": config['model']['num_classes'],
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    }

@app.get("/config")
async def get_config():
    """Get current configuration"""
    return config

if __name__ == "__main__":
    # For local development
    uvicorn.run(
        "inference_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )