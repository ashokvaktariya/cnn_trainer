#!/usr/bin/env python3
"""
Download the model from Hugging Face Hub to local directory for team sharing
"""

import os
import sys
from huggingface_hub import hf_hub_download

# Configuration
HF_USERNAME = "5techlab-research"
HF_REPO_NAME = "medical-fracture-detection-v1"
HF_REPO_ID = f"{HF_USERNAME}/{HF_REPO_NAME}"
HF_TOKEN = os.getenv('HF_TOKEN', '')  # Get token from environment variable

MODEL_DIR = "./models"
MODEL_FILENAME = "binary_classifier_best.pth"
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILENAME)

def download_model():
    """Download model from Hugging Face Hub to local directory"""
    try:
        print("🚀 Medical Fracture Detection Model Downloader")
        print("=" * 50)
        print(f"📱 Repository: {HF_REPO_ID}")
        print(f"📁 Local Directory: {MODEL_DIR}")
        print(f"📄 Model File: {MODEL_FILENAME}")
        print("=" * 50)
        
        # Create models directory if it doesn't exist
        os.makedirs(MODEL_DIR, exist_ok=True)
        print(f"✅ Created directory: {MODEL_DIR}")
        
        # Check if model already exists locally
        if os.path.exists(MODEL_PATH):
            file_size = os.path.getsize(MODEL_PATH) / (1024*1024)
            print(f"✅ Model already exists: {MODEL_PATH}")
            print(f"📊 File size: {file_size:.2f} MB")
            return True
        
        # Download model from Hugging Face Hub
        print("📥 Downloading model from Hugging Face Hub...")
        print("⏳ This may take a few minutes depending on your internet connection...")
        
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
        
        # Check final file size
        file_size = os.path.getsize(MODEL_PATH) / (1024*1024)
        
        print("=" * 50)
        print(f"✅ Model downloaded successfully!")
        print(f"📁 Location: {MODEL_PATH}")
        print(f"📊 File size: {file_size:.2f} MB")
        print("=" * 50)
        print("🎯 Your team can now use this model file directly!")
        print("💡 Run 'python run_api.py' to start the API")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to download model: {e}")
        return False

if __name__ == "__main__":
    success = download_model()
    if not success:
        sys.exit(1)
