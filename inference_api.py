#!/usr/bin/env python3
"""
FastAPI application for binary fracture classification inference
Supports both local and Hugging Face Hub model loading
"""

import os
import io
import base64
import logging
import tempfile
from typing import List, Dict, Optional, Union
from pathlib import Path
import json

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Query
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn

# Hugging Face Hub integration
from huggingface_hub import hf_hub_download, HfApi
import requests

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

# Hugging Face Configuration
HF_USERNAME = "5techlab-research"
HF_REPO_NAME = "medical-fracture-detection-v1"
HF_REPO_ID = f"{HF_USERNAME}/{HF_REPO_NAME}"
HF_TOKEN = os.getenv('HF_TOKEN', '')  # Get token from environment variable

# Local model directory
MODEL_DIR = "./models"
MODEL_FILENAME = "binary_classifier_best.pth"
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILENAME)

# Global variables
model = None
device = None
transform = None
model_source = None  # Track if model is loaded from local or HF

class BinaryEfficientNet(torch.nn.Module):
    """Binary Classification Model using EfficientNet-B7 for fracture detection"""
    
    def __init__(self, num_classes=2, dropout_rate=0.3, pretrained=True, **kwargs):
        super().__init__()
        
        # Use EfficientNet-B7 backbone without classifier
        try:
            from efficientnet_pytorch import EfficientNet
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
            
            # Initialize classifier weights
            self._initialize_weights()
            
        except ImportError:
            # Fallback to torchvision EfficientNet-B0
            from torchvision.models import efficientnet_b0
            self.backbone = efficientnet_b0(pretrained=False)
            
            # Modify the classifier for binary classification
            self.backbone.classifier = torch.nn.Sequential(
                torch.nn.Dropout(0.2),
                torch.nn.Linear(self.backbone.classifier[1].in_features, num_classes)
            )
            self.classifier = None
    
    def _initialize_weights(self):
        """Initialize classifier weights"""
        for m in self.classifier:
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                torch.nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
        # x shape: (batch_size, 3, 224, 224) - single image
        if x.dim() == 5:
            # If batch contains multiple images, take the first one
            x = x[:, 0]  # (batch_size, 3, 224, 224)
        
        if self.classifier is not None:
            # Extract features using EfficientNet backbone (without classifier)
            features = self.backbone.extract_features(x)  # Get raw features
            
            # Apply classifier
            logits = self.classifier(features)
        else:
            # Fallback EfficientNet-B0
            logits = self.backbone(x)
        
        return logits

def download_model_from_hf():
    """Download model from Hugging Face Hub to local directory"""
    try:
        # Create models directory if it doesn't exist
        os.makedirs(MODEL_DIR, exist_ok=True)
        
        # Check if model already exists locally
        if os.path.exists(MODEL_PATH):
            logger.info(f"✅ Model already exists locally: {MODEL_PATH}")
            return True
        
        # Download model from Hugging Face Hub
        logger.info("📥 Downloading model from Hugging Face Hub to local directory...")
        downloaded_path = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename="checkpoints/binary_classifier_best.pth",
            token=HF_TOKEN,
            local_dir=MODEL_DIR,
            local_dir_use_symlinks=False  # Download actual file, not symlink
        )
        
        # Move to our desired filename if needed
        if downloaded_path != MODEL_PATH:
            import shutil
            shutil.move(downloaded_path, MODEL_PATH)
        
        logger.info(f"✅ Model downloaded to: {MODEL_PATH}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to download model from Hugging Face Hub: {e}")
        return False

def load_model_from_hf():
    """Load model from Hugging Face Hub (downloaded to local directory)"""
    global model, device, transform, model_source
    
    try:
        # Set device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")
        
        # Define image preprocessing (matches your existing code)
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Download model if not exists locally
        if not os.path.exists(MODEL_PATH):
            logger.info("📥 Model not found locally, downloading from Hugging Face Hub...")
            if not download_model_from_hf():
                return False
        
        logger.info(f"✅ Loading model from: {MODEL_PATH}")
        
        # Initialize model
        model = BinaryEfficientNet(num_classes=2, dropout_rate=0.3, pretrained=False)
        
        # Load checkpoint
        checkpoint = torch.load(MODEL_PATH, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Set to evaluation mode
        model.eval()
        model.to(device)
        
        model_source = "huggingface"
        logger.info("✅ Model loaded successfully from Hugging Face Hub!")
        logger.info(f"✅ Model path: {MODEL_PATH}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to load model from Hugging Face Hub: {e}")
        return False

def load_model_local(model_path: str = None):
    """Load the trained binary classifier model from local path"""
    global model, device, transform, model_source
    
    try:
        # Use default model path if none provided
        if model_path is None:
            model_path = MODEL_PATH
        
        # Set device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")
        
        # Define image preprocessing (matches your existing code)
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Check if model file exists
        if not os.path.exists(model_path):
            logger.error(f"❌ Model file not found: {model_path}")
            return False
        
        logger.info(f"✅ Loading model from: {model_path}")
        
        # Initialize model
        model = BinaryEfficientNet(num_classes=2, dropout_rate=0.3, pretrained=False)
        
        # Load model weights
        checkpoint = torch.load(model_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.to(device)
        model.eval()
        model_source = "local"
        
        logger.info("✅ Model loaded successfully from local path!")
        logger.info(f"✅ Model path: {model_path}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to load model from local path: {e}")
        return False

def load_model(model_source: str = "auto"):
    """Load model from specified source (auto, local, huggingface)"""
    if model_source == "auto":
        # Check if model exists locally first
        if os.path.exists(MODEL_PATH):
            logger.info("🎯 Found local model, loading from local directory")
            return load_model_local()
        else:
            # Try Hugging Face first, then fallback to local
            logger.info("📥 No local model found, downloading from Hugging Face Hub...")
            if load_model_from_hf():
                return True
            else:
                logger.error("❌ Failed to load model from any source")
                return False
    elif model_source == "huggingface":
        return load_model_from_hf()
    elif model_source == "local":
        return load_model_local()
    else:
        logger.error(f"❌ Unknown model source: {model_source}")
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
            outputs = model(image_tensor)
            probabilities = F.softmax(outputs, dim=1)
            
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
    
    if not load_model("auto"):
        logger.error("❌ Failed to load model. API will not function properly.")
    else:
        logger.info("✅ API ready for inference!")

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Medical Fracture Detection API",
        "version": "1.0.0",
        "status": "running",
        "model_loaded": model is not None
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": str(device) if device else "unknown"
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

@app.get("/test_images")
async def get_test_images():
    """Get list of available test images"""
    test_dir = Path("test_images/test_images/test_images")
    
    if not test_dir.exists():
        return {"error": "Test images directory not found"}
    
    images = {}
    for category in ["positive", "negative", "doubt"]:
        category_dir = test_dir / category
        if category_dir.exists():
            images[category] = [f.name for f in category_dir.glob("*.jpg")]
    
    return {
        "total_categories": len(images),
        "images": images
    }

@app.post("/test_inference")
async def test_inference_on_downloaded_images():
    """Run inference on all downloaded test images"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    test_dir = Path("test_images/test_images/test_images")
    
    if not test_dir.exists():
        raise HTTPException(status_code=404, detail="Test images directory not found")
    
    results = {}
    total_images = 0
    correct_predictions = 0
    
    for category in ["positive", "negative", "doubt"]:
        category_dir = test_dir / category
        if not category_dir.exists():
            continue
        
        category_results = []
        for image_file in category_dir.glob("*.jpg"):
            try:
                # Read image
                with open(image_file, 'rb') as f:
                    image_data = f.read()
                
                # Preprocess image
                image_tensor = preprocess_image(image_data)
                
                # Run prediction
                result = predict_image(image_tensor)
                
                # Add ground truth
                result["ground_truth"] = category.upper()
                result["filename"] = image_file.name
                
                # Check if prediction is correct (for positive/negative only)
                if category in ["positive", "negative"]:
                    is_correct = (result["predicted_label"] == category.upper())
                    result["correct"] = is_correct
                    if is_correct:
                        correct_predictions += 1
                
                category_results.append(result)
                total_images += 1
                
            except Exception as e:
                logger.error(f"❌ Error processing {image_file}: {e}")
                category_results.append({
                    "filename": image_file.name,
                    "ground_truth": category.upper(),
                    "error": str(e)
                })
        
        results[category] = category_results
    
    # Calculate accuracy for positive/negative only
    positive_negative_total = sum(len(results.get(cat, [])) for cat in ["positive", "negative"])
    accuracy = correct_predictions / positive_negative_total if positive_negative_total > 0 else 0
    
    logger.info(f"✅ Test inference completed: {total_images} images, Accuracy: {accuracy:.3f}")
    
    return {
        "total_images": total_images,
        "accuracy": accuracy,
        "correct_predictions": correct_predictions,
        "positive_negative_total": positive_negative_total,
        "results": results
    }

@app.get("/model_info")
async def get_model_info():
    """Get information about the loaded model"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_source": model_source,
        "device": str(device) if device else "unknown",
        "huggingface_repo": HF_REPO_ID,
        "model_loaded": model is not None,
        "transform_available": transform is not None
    }

@app.post("/load_model")
async def load_model_endpoint(source: str = Query("auto", description="Model source: auto, local, huggingface")):
    """Load model from specified source"""
    global model_source
    
    try:
        success = load_model(source)
        if success:
            return {
                "message": f"Model loaded successfully from {model_source}",
                "model_source": model_source,
                "device": str(device),
                "status": "success"
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to load model")
    except Exception as e:
        logger.error(f"❌ Error loading model: {e}")
        raise HTTPException(status_code=500, detail=f"Error loading model: {str(e)}")

@app.post("/predict_local")
async def predict_local_image(file: UploadFile = File(...)):
    """Predict fracture in a single image (local inference)"""
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
            "content_type": file.content_type,
            "inference_type": "local",
            "model_source": model_source
        })
        
        logger.info(f"✅ Local prediction completed for {file.filename}: {result['predicted_label']} ({result['confidence']:.3f})")
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Local prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/predict_global")
async def predict_global_image(file: UploadFile = File(...)):
    """Predict fracture in a single image (global inference via HF Hub)"""
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
            "content_type": file.content_type,
            "inference_type": "global",
            "model_source": model_source,
            "huggingface_repo": HF_REPO_ID
        })
        
        logger.info(f"✅ Global prediction completed for {file.filename}: {result['predicted_label']} ({result['confidence']:.3f})")
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Global prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/upload_interface", response_class=HTMLResponse)
async def upload_interface():
    """Simple HTML interface for image upload and inference"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Medical Fracture Detection - Upload Interface</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
            .container { background: #f5f5f5; padding: 20px; border-radius: 10px; }
            .upload-area { border: 2px dashed #ccc; padding: 20px; text-align: center; margin: 20px 0; }
            .result { background: #e8f5e8; padding: 15px; border-radius: 5px; margin: 10px 0; }
            .error { background: #ffe8e8; padding: 15px; border-radius: 5px; margin: 10px 0; }
            button { background: #007bff; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; }
            button:hover { background: #0056b3; }
            .model-info { background: #e8f4fd; padding: 15px; border-radius: 5px; margin: 10px 0; }
        </style>
    </head>
    <body>
        <h1>🏥 Medical Fracture Detection API</h1>
        
        <div class="model-info">
            <h3>Model Information</h3>
            <div id="modelInfo">Loading...</div>
        </div>
        
        <div class="container">
            <h2>📤 Upload Medical Image for Analysis</h2>
            
            <div class="upload-area">
                <input type="file" id="imageFile" accept="image/*" style="margin: 10px;">
                <br>
                <button onclick="predictLocal()">🔍 Local Inference</button>
                <button onclick="predictGlobal()">🌐 Global Inference</button>
            </div>
            
            <div id="result"></div>
        </div>
        
        <script>
            // Load model info on page load
            async function loadModelInfo() {
                try {
                    const response = await fetch('/model_info');
                    const data = await response.json();
                    document.getElementById('modelInfo').innerHTML = `
                        <strong>Source:</strong> ${data.model_source}<br>
                        <strong>Device:</strong> ${data.device}<br>
                        <strong>HF Repo:</strong> ${data.huggingface_repo}<br>
                        <strong>Status:</strong> ${data.model_loaded ? '✅ Loaded' : '❌ Not Loaded'}
                    `;
                } catch (error) {
                    document.getElementById('modelInfo').innerHTML = '❌ Error loading model info';
                }
            }
            
            async function predictLocal() {
                const fileInput = document.getElementById('imageFile');
                const file = fileInput.files[0];
                
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
                    displayResult(result, 'Local Inference');
                } catch (error) {
                    displayError('Error: ' + error.message);
                }
            }
            
            async function predictGlobal() {
                const fileInput = document.getElementById('imageFile');
                const file = fileInput.files[0];
                
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
                    displayResult(result, 'Global Inference');
                } catch (error) {
                    displayError('Error: ' + error.message);
                }
            }
            
            function displayResult(result, type) {
                const resultDiv = document.getElementById('result');
                resultDiv.innerHTML = `
                    <div class="result">
                        <h3>🎯 ${type} Result</h3>
                        <p><strong>Prediction:</strong> ${result.predicted_label}</p>
                        <p><strong>Confidence:</strong> ${(result.confidence * 100).toFixed(2)}%</p>
                        <p><strong>Probabilities:</strong></p>
                        <ul>
                            <li>Negative: ${(result.probabilities.negative * 100).toFixed(2)}%</li>
                            <li>Positive: ${(result.probabilities.positive * 100).toFixed(2)}%</li>
                        </ul>
                        <p><strong>File:</strong> ${result.filename}</p>
                        <p><strong>Model Source:</strong> ${result.model_source}</p>
                    </div>
                `;
            }
            
            function displayError(message) {
                const resultDiv = document.getElementById('result');
                resultDiv.innerHTML = `<div class="error">❌ ${message}</div>`;
            }
            
            // Load model info when page loads
            loadModelInfo();
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.get("/api_docs")
async def api_documentation():
    """API documentation endpoint"""
    return {
        "title": "Medical Fracture Detection API",
        "version": "1.0.0",
        "description": "Binary classification API for detecting fractures in medical images",
        "endpoints": {
            "/": "Root endpoint with API status",
            "/health": "Health check endpoint",
            "/model_info": "Get model information",
            "/load_model": "Load model from specified source (auto, local, huggingface)",
            "/predict": "Original single image prediction endpoint",
            "/predict_local": "Local inference endpoint",
            "/predict_global": "Global inference endpoint (via HF Hub)",
            "/predict_batch": "Batch prediction endpoint (max 10 images)",
            "/predict_base64": "Base64 encoded image prediction",
            "/upload_interface": "HTML interface for image upload",
            "/test_images": "Get list of available test images",
            "/test_inference": "Run inference on downloaded test images",
            "/api_docs": "This documentation endpoint"
        },
        "huggingface_repo": HF_REPO_ID,
        "model_sources": ["auto", "local", "huggingface"],
        "supported_formats": ["jpg", "jpeg", "png", "bmp", "tiff"],
        "max_batch_size": 10
    }

if __name__ == "__main__":
    uvicorn.run(
        "inference_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
