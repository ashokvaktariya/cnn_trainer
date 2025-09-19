#!/usr/bin/env python3
"""
Simple script to run the Medical Fracture Detection API
"""

import uvicorn
import os
import sys

def check_model_exists():
    """Check if the model file exists"""
    model_path = "./models/binary_classifier_best.pth"
    
    if os.path.exists(model_path):
        file_size = os.path.getsize(model_path) / (1024*1024)
        print(f"✅ Model found locally: {file_size:.2f} MB")
        return True
    else:
        print("⚠️ Model not found locally, will download from HF Hub to ./models/")
        return False

def main():
    """Run the FastAPI application"""
    print("🚀 Starting Medical Fracture Detection API...")
    print("📱 Hugging Face Repository: 5techlab-research/medical-fracture-detection-v1")
    print("🌐 API will be available at: http://localhost:8000")
    print("📤 Upload interface: http://localhost:8000/upload_interface")
    print("📚 API Documentation: http://localhost:8000/api_docs")
    print("=" * 60)
    
    # Check if model exists
    print("🔍 Checking model availability...")
    check_model_exists()
    
    print("\n🚀 Starting API server...")
    print("💡 If model loading fails, check the logs above")
    print("💡 You can also test model loading with: python test_model_loading.py")
    print("=" * 60)
    
    # Run the API
    uvicorn.run(
        "inference_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

if __name__ == "__main__":
    main()
