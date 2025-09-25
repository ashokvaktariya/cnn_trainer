#!/usr/bin/env python3
"""
FastAPI inference server for medical fracture detection
"""

import os
import json
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
import logging
from typing import Dict, Any
import io

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load model configuration
def load_model_config():
    """Load model configuration"""
    config_path = "model_config.json"
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Model config not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    return config

# EfficientNet model class
class BinaryEfficientNet(nn.Module):
    """Binary Classification EfficientNet for Medical Images"""
    
    def __init__(self, num_classes=2, dropout_rate=0.3, pretrained=True):
        super(BinaryEfficientNet, self).__init__()
        
        # Load EfficientNet-B7 backbone
        import torchvision.models as models
        self.backbone = models.efficientnet_b7(pretrained=pretrained)
        
        # Replace classifier for binary classification
        num_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(num_features, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)

# Global model variable
model = None
config = None
transform = None

def load_model():
    """Load the trained model"""
    global model, config, transform
    
    logger.info("📥 Loading model configuration...")
    config = load_model_config()
    
    logger.info("🏗️ Initializing EfficientNet-B7 model...")
    model = BinaryEfficientNet(
        num_classes=config['num_classes'],
        dropout_rate=config['dropout'],
        pretrained=False  # We'll load our trained weights
    )
    
    logger.info("📂 Loading trained weights...")
    model_path = config['model_path']
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model weights not found: {model_path}")
    
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    logger.info("🔄 Setting up image transforms...")
    transform = transforms.Compose([
        transforms.Resize(config['input_size']),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    logger.info("✅ Model loaded successfully!")

def preprocess_image(image_bytes: bytes) -> torch.Tensor:
    """Preprocess image for inference"""
    try:
        # Load image from bytes
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Apply transforms
        image_tensor = transform(image).unsqueeze(0)
        
        return image_tensor
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Image preprocessing failed: {str(e)}")

def predict_fracture(image_tensor: torch.Tensor) -> Dict[str, Any]:
    """Predict fracture from image tensor"""
    global model, config
    
    with torch.no_grad():
        # Get model predictions
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        
        # Get prediction
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item()
        
        # Get class name
        class_name = config['class_names'][predicted_class]
        
        # Get probabilities for both classes
        neg_prob = probabilities[0][0].item()
        pos_prob = probabilities[0][1].item()
        
        return {
            "prediction": class_name,
            "confidence": confidence,
            "probabilities": {
                "NEGATIVE": neg_prob,
                "POSITIVE": pos_prob
            },
            "binary_prediction": predicted_class,
            "fracture_detected": predicted_class == 1
        }

# FastAPI app
app = FastAPI(
    title="Medical Fracture Detection API",
    description="EfficientNet-B7 model for detecting fractures in X-ray images",
    version="1.0.0"
)

@app.on_event("startup")
async def startup_event():
    """Initialize model on startup"""
    logger.info("🚀 Starting Medical Fracture Detection API...")
    load_model()
    logger.info("✅ API ready for inference!")

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Medical Fracture Detection API",
        "model": "EfficientNet-B7",
        "classes": ["NEGATIVE", "POSITIVE"],
        "status": "ready"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Predict fracture from uploaded image"""
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read image bytes
        image_bytes = await file.read()
        
        # Preprocess image
        image_tensor = preprocess_image(image_bytes)
        
        # Make prediction
        result = predict_fracture(image_tensor)
        
        # Add metadata
        result["filename"] = file.filename
        result["file_size"] = len(image_bytes)
        
        return JSONResponse(content=result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict_batch")
async def predict_batch(files: list[UploadFile] = File(...)):
    """Predict fractures from multiple uploaded images"""
    try:
        results = []
        
        for file in files:
            # Validate file type
            if not file.content_type.startswith('image/'):
                results.append({
                    "filename": file.filename,
                    "error": "File must be an image"
                })
                continue
            
            try:
                # Read image bytes
                image_bytes = await file.read()
                
                # Preprocess image
                image_tensor = preprocess_image(image_bytes)
                
                # Make prediction
                result = predict_fracture(image_tensor)
                result["filename"] = file.filename
                result["file_size"] = len(image_bytes)
                
                results.append(result)
                
            except Exception as e:
                results.append({
                    "filename": file.filename,
                    "error": str(e)
                })
        
        return JSONResponse(content={"results": results})
        
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

@app.get("/model_info")
async def model_info():
    """Get model information"""
    return {
        "architecture": config['model_architecture'],
        "num_classes": config['num_classes'],
        "class_names": config['class_names'],
        "input_size": config['input_size'],
        "model_path": config['model_path']
    }

def main():
    """Start the FastAPI server"""
    logger.info("🌐 Starting FastAPI server...")
    
    uvicorn.run(
        "inference_server:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        workers=1
    )

if __name__ == "__main__":
    main()
