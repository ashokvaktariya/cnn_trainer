#!/usr/bin/env python3
"""
Test script to verify model loading from Hugging Face Hub
"""

import os
import sys
import torch
from pathlib import Path

# Add current directory to path
sys.path.append('.')

def test_model_path():
    """Test if the model file exists at the expected path"""
    cache_path = "./hf_cache/models--5techlab-research--medical-fracture-detection-v1/snapshots/43612ae31cbd84e3120fe731c41472fa0c71b1d4/checkpoints/binary_classifier_best.pth"
    
    print(f"🔍 Checking model path: {cache_path}")
    
    if os.path.exists(cache_path):
        print("✅ Model file found!")
        
        # Check file size
        file_size = os.path.getsize(cache_path)
        print(f"📊 File size: {file_size / (1024*1024):.2f} MB")
        
        # Try to load the model
        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"📱 Using device: {device}")
            
            checkpoint = torch.load(cache_path, map_location=device)
            print("✅ Model checkpoint loaded successfully!")
            
            if 'model_state_dict' in checkpoint:
                print("📋 Checkpoint contains 'model_state_dict'")
                print(f"📊 Model state dict keys: {len(checkpoint['model_state_dict'])}")
            else:
                print("📋 Checkpoint contains direct model weights")
                print(f"📊 Direct model keys: {len(checkpoint)}")
            
            if 'best_accuracy' in checkpoint:
                print(f"🏆 Best training accuracy: {checkpoint['best_accuracy']:.2f}%")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    else:
        print("❌ Model file not found!")
        return False

def test_api_import():
    """Test if we can import the API functions"""
    try:
        from inference_api import load_model_from_hf, load_model_local, load_model
        print("✅ Successfully imported API functions")
        return True
    except Exception as e:
        print(f"❌ Error importing API functions: {e}")
        return False

def main():
    """Main test function"""
    print("🧪 Testing Model Loading from Hugging Face Hub")
    print("=" * 50)
    
    # Test 1: Check if model file exists
    print("\n1️⃣ Testing model file existence...")
    model_exists = test_model_path()
    
    # Test 2: Test API import
    print("\n2️⃣ Testing API import...")
    api_import = test_api_import()
    
    # Test 3: Test model loading function
    if api_import:
        print("\n3️⃣ Testing model loading function...")
        try:
            from inference_api import load_model_from_hf
            success = load_model_from_hf()
            if success:
                print("✅ Model loading function works!")
            else:
                print("❌ Model loading function failed!")
        except Exception as e:
            print(f"❌ Error testing model loading: {e}")
    
    print("\n" + "=" * 50)
    if model_exists and api_import:
        print("🎉 All tests passed! Model is ready for API.")
    else:
        print("⚠️ Some tests failed. Check the errors above.")

if __name__ == "__main__":
    main()
