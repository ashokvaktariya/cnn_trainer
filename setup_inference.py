#!/usr/bin/env python3
"""
Setup script for inference server
Downloads models and installs dependencies
"""

import subprocess
import sys
import os

def install_requirements():
    """Install required packages"""
    print("📦 Installing requirements...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements_inference.txt"])
        print("✅ Requirements installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install requirements: {e}")
        return False

def setup_models():
    """Setup models for inference"""
    print("📥 Setting up models...")
    try:
        from download_models import setup_models
        success = setup_models()
        if success:
            print("✅ Models setup successfully!")
            return True
        else:
            print("❌ Model setup failed!")
            return False
    except Exception as e:
        print(f"❌ Model setup error: {e}")
        return False

def main():
    """Main setup function"""
    print("🚀 Setting up Medical Fracture Detection Inference Server...")
    
    # Install requirements
    if not install_requirements():
        return False
    
    # Setup models
    if not setup_models():
        return False
    
    print("🎉 Setup completed successfully!")
    print("📋 Next steps:")
    print("   1. Run: python inference_server.py")
    print("   2. Visit: http://localhost:8000")
    print("   3. API docs: http://localhost:8000/docs")
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        print("❌ Setup failed!")
        sys.exit(1)
