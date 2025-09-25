#!/usr/bin/env python3
"""
Simple script to push model to Hugging Face
"""

import os
from huggingface_hub import HfApi, create_repo

def push_model():
    """Push model to HF"""
    token = os.getenv('HF_TOKEN')
    if not token:
        print("Set HF_TOKEN environment variable")
        return False
    
    api = HfApi(token=token)
    repo_id = "5techlab-research/cnn_med_33k"
    
    # Create repository first
    print("📝 Creating repository...")
    create_repo(
        repo_id=repo_id,
        repo_type="model",
        private=False,
        exist_ok=True,
        token=token
    )
    print("✅ Repository created!")
    
    # Upload model files
    print("⬆️ Uploading model files...")
    api.upload_folder(
        folder_path="/home/avaktariya/binary_classifier_20250924_131737",
        repo_id=repo_id,
        repo_type="model",
        commit_message="Upload medical fracture detection model"
    )
    
    print("✅ Model uploaded successfully!")
    print(f"🔗 Repository: https://huggingface.co/{repo_id}")
    return True

if __name__ == "__main__":
    push_model()
