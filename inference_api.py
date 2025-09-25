#!/usr/bin/env python3
"""
Medical Fracture Detection FastAPI Application
Provides binary classification for detecting fractures in medical images
"""

import os
import json
import logging
from typing import Dict, List, Optional, Union
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Medical Fracture Detection API",
    description="Binary classification for detecting fractures in medical images",
    version="1.0.0"
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
model_config = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_source = "unknown"

# Hugging Face settings
HF_USERNAME = "5techlab-research"
HF_REPO_NAME = "cnn_med_33k"
HF_REPO_ID = "5techlab-research/cnn_med_33k"
HF_TOKEN = os.getenv('HF_TOKEN', '')

def load_model_config():
    """Load model configuration"""
    config_path = "model_config.json"
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Model config not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    return config

def create_model():
    """Create EfficientNet-B7 model architecture"""
    import torchvision.models as models
    
    # Load EfficientNet-B7 backbone
    backbone = models.efficientnet_b7(pretrained=False)
    
    # Modify classifier for binary classification
    num_features = backbone.classifier[1].in_features
    backbone.classifier = torch.nn.Sequential(
        torch.nn.Dropout(0.3),
        torch.nn.Linear(num_features, 2)
    )
    
    return backbone

def load_model_from_local():
    """Load model from local directory"""
    global model, model_config, model_source
    
    try:
        model_config = load_model_config()
        model_path = model_config.get("model_path", "./models/binary_classifier_best.pth")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Create model architecture
        model = create_model()
        
        # Load state dict
        checkpoint = torch.load(model_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.to(device)
        model.eval()
        
        model_source = "local"
        logger.info(f"✅ Model loaded successfully from local directory: {model_path}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to load model from local: {e}")
        return False

def load_model_from_huggingface():
    """Load model from Hugging Face Hub"""
    global model, model_config, model_source
    
    try:
        from huggingface_hub import hf_hub_download
        
        logger.info("📥 Downloading model from Hugging Face Hub...")
        
        # Download model file
        model_path = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename="binary_classifier_best.pth",
            local_dir="./models",
            local_dir_use_symlinks=False,
            token=HF_TOKEN
        )
        
        # Create model architecture
        model = create_model()
        
        # Load state dict
        checkpoint = torch.load(model_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.to(device)
        model.eval()
        
        # Create config
        model_config = {
            "model_architecture": "efficientnet_b7",
            "num_classes": 2,
            "class_names": ["NEGATIVE", "POSITIVE"],
            "input_size": [600, 600],
            "pretrained": True,
            "dropout": 0.3,
            "model_path": model_path,
            "backbone_path": "./models/efficientnet_b7_backbone.bin"
        }
        
        model_source = "huggingface"
        logger.info(f"✅ Model loaded successfully from Hugging Face Hub: {model_path}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to load model from Hugging Face: {e}")
        return False

def auto_load_model():
    """Auto-load model (try HF first, then local)"""
    logger.info("🔄 Auto-loading model...")
    
    # Try Hugging Face first
    if load_model_from_huggingface():
        return True
    
    # Fallback to local
    logger.info("💡 Falling back to local model...")
    return load_model_from_local()

def preprocess_image(image: Image.Image) -> torch.Tensor:
    """Preprocess image for inference"""
    # Get input size from config
    input_size = model_config.get("input_size", [600, 600])
    
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Apply transforms
    image_tensor = transform(image).unsqueeze(0)
    return image_tensor.to(device)

def predict_image(image_tensor: torch.Tensor) -> Dict:
    """Run inference on image tensor"""
    global model, model_config
    
    with torch.no_grad():
        # Forward pass
        outputs = model(image_tensor)
        probabilities = F.softmax(outputs, dim=1)
        
        # Get prediction
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item()
        
        # Get class names
        class_names = model_config.get("class_names", ["NEGATIVE", "POSITIVE"])
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

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize model on startup"""
    logger.info("🚀 Starting Medical Fracture Detection API...")
    
    # Auto-load model
    if auto_load_model():
        logger.info("✅ API ready for inference!")
    else:
        logger.error("❌ Failed to load model. API may not work properly.")

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API status"""
    return {
        "message": "Medical Fracture Detection API",
        "version": "1.0.0",
        "status": "running",
        "model_loaded": model is not None,
        "model_source": model_source,
        "device": str(device),
        "endpoints": {
            "health": "/health",
            "model_info": "/model_info",
            "predict": "/predict",
            "predict_local": "/predict_local",
            "predict_global": "/predict_global",
            "upload_interface": "/upload_interface",
            "api_docs": "/api_docs"
        }
    }

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_source": model_source,
        "device": str(device),
        "timestamp": torch.cuda.Event(enable_timing=True).elapsed_time(0) if torch.cuda.is_available() else 0
    }

# Model info endpoint
@app.get("/model_info")
async def model_info():
    """Get detailed model information"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_architecture": model_config.get("model_architecture", "unknown"),
        "num_classes": model_config.get("num_classes", 2),
        "class_names": model_config.get("class_names", ["NEGATIVE", "POSITIVE"]),
        "input_size": model_config.get("input_size", [600, 600]),
        "model_source": model_source,
        "device": str(device),
        "hf_repo": HF_REPO_ID,
        "model_path": model_config.get("model_path", "unknown")
    }

# Load model endpoint
@app.post("/load_model")
async def load_model(source: str = Query("auto", description="Model source: auto, local, or huggingface")):
    """Load model from specified source"""
    global model_source
    
    success = False
    if source == "huggingface":
        success = load_model_from_huggingface()
    elif source == "local":
        success = load_model_from_local()
    elif source == "auto":
        success = auto_load_model()
    else:
        raise HTTPException(status_code=400, detail="Invalid source. Use: auto, local, or huggingface")
    
    if success:
        return {
            "message": f"Model loaded successfully from {model_source}",
            "model_source": model_source,
            "model_info": await model_info()
        }
    else:
        raise HTTPException(status_code=500, detail="Failed to load model")

# Prediction endpoints
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Original single image prediction"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Read and process image
        image = Image.open(file.file).convert('RGB')
        image_tensor = preprocess_image(image)
        
        # Run prediction
        result = predict_image(image_tensor)
        
        # Add file info
        result.update({
            "filename": file.filename,
            "file_size": file.size,
            "content_type": file.content_type,
            "inference_type": "standard"
        })
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict_local")
async def predict_local(file: UploadFile = File(...)):
    """Local inference on uploaded image"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Read and process image
        image = Image.open(file.file).convert('RGB')
        image_tensor = preprocess_image(image)
        
        # Run prediction
        result = predict_image(image_tensor)
        
        # Add file info
        result.update({
            "filename": file.filename,
            "file_size": file.size,
            "content_type": file.content_type,
            "inference_type": "local",
            "model_source": model_source
        })
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Local prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Local prediction failed: {str(e)}")

@app.post("/predict_global")
async def predict_global(file: UploadFile = File(...)):
    """Global inference via HF Hub"""
    # Force reload from HF Hub
    if not load_model_from_huggingface():
        raise HTTPException(status_code=503, detail="Failed to load model from Hugging Face")
    
    try:
        # Read and process image
        image = Image.open(file.file).convert('RGB')
        image_tensor = preprocess_image(image)
        
        # Run prediction
        result = predict_image(image_tensor)
        
        # Add file info
        result.update({
            "filename": file.filename,
            "file_size": file.size,
            "content_type": file.content_type,
            "inference_type": "global",
            "model_source": "huggingface"
        })
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Global prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Global prediction failed: {str(e)}")

# Upload interface
@app.get("/upload_interface", response_class=HTMLResponse)
async def upload_interface():
    """HTML web interface for image upload"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Medical Fracture Detection</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }
            .container { max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
            h1 { color: #2c3e50; text-align: center; }
            .upload-area { border: 2px dashed #3498db; padding: 40px; text-align: center; margin: 20px 0; border-radius: 10px; }
            .upload-area:hover { background-color: #f8f9fa; }
            input[type="file"] { margin: 10px 0; }
            button { background-color: #3498db; color: white; padding: 12px 24px; border: none; border-radius: 5px; cursor: pointer; margin: 10px; font-size: 16px; }
            button:hover { background-color: #2980b9; }
            .result { margin-top: 20px; padding: 20px; background-color: #ecf0f1; border-radius: 5px; }
            .positive { background-color: #e74c3c; color: white; }
            .negative { background-color: #27ae60; color: white; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🏥 Medical Fracture Detection</h1>
            <div class="upload-area">
                <h3>Upload Medical Image</h3>
                <input type="file" id="imageFile" accept="image/*">
                <br>
                <button onclick="predictLocal()">🔍 Local Inference</button>
                <button onclick="predictGlobal()">🌐 Global Inference</button>
            </div>
            <div id="result"></div>
        </div>
        
        <script>
            async function predictLocal() {
                const file = document.getElementById('imageFile').files[0];
                if (!file) {
                    alert('Please select an image file');
                    return;
                }
                
                const formData = new FormData();
                formData.append('file', file);
                
                try {
                    const response = await fetch('/predict_local', {
                        method: 'POST',
                        body: formData
                    });
                    const result = await response.json();
                    displayResult(result);
                } catch (error) {
                    document.getElementById('result').innerHTML = '<div class="result">Error: ' + error.message + '</div>';
                }
            }
            
            async function predictGlobal() {
                const file = document.getElementById('imageFile').files[0];
                if (!file) {
                    alert('Please select an image file');
                    return;
                }
                
                const formData = new FormData();
                formData.append('file', file);
                
                try {
                    const response = await fetch('/predict_global', {
                        method: 'POST',
                        body: formData
                    });
                    const result = await response.json();
                    displayResult(result);
                } catch (error) {
                    document.getElementById('result').innerHTML = '<div class="result">Error: ' + error.message + '</div>';
                }
            }
            
            function displayResult(result) {
                const resultDiv = document.getElementById('result');
                const className = result.predicted_label === 'POSITIVE' ? 'positive' : 'negative';
                resultDiv.innerHTML = `
                    <div class="result ${className}">
                        <h3>Prediction Result</h3>
                        <p><strong>Prediction:</strong> ${result.predicted_label}</p>
                        <p><strong>Confidence:</strong> ${(result.confidence * 100).toFixed(2)}%</p>
                        <p><strong>Negative Probability:</strong> ${(result.probabilities.negative * 100).toFixed(2)}%</p>
                        <p><strong>Positive Probability:</strong> ${(result.probabilities.positive * 100).toFixed(2)}%</p>
                        <p><strong>File:</strong> ${result.filename}</p>
                        <p><strong>Inference Type:</strong> ${result.inference_type}</p>
                        <p><strong>Model Source:</strong> ${result.model_source}</p>
                    </div>
                `;
            }
        </script>
    </body>
    </html>
    """
    return html_content

# API documentation endpoint
@app.get("/api_docs")
async def api_docs():
    """Complete API documentation"""
    return {
        "title": "Medical Fracture Detection API",
        "version": "1.0.0",
        "description": "Binary classification for detecting fractures in medical images",
        "endpoints": {
            "GET /": "Root endpoint with API status",
            "GET /health": "Health check endpoint",
            "GET /model_info": "Get detailed model information",
            "POST /load_model": "Load model from specified source",
            "POST /predict": "Original single image prediction",
            "POST /predict_local": "Local inference on uploaded image",
            "POST /predict_global": "Global inference via HF Hub",
            "GET /upload_interface": "HTML web interface for image upload",
            "GET /api_docs": "Complete API documentation"
        },
        "model_info": await model_info() if model else "Model not loaded",
        "supported_formats": ["JPEG", "PNG", "BMP", "TIFF"],
        "hf_repo": HF_REPO_ID
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
