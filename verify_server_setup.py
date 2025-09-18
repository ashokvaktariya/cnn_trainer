#!/usr/bin/env python3
"""
Server Setup Verification Script
Verify that everything is properly configured on the H200 GPU server
"""

import os
import sys
import subprocess
import importlib.util

def check_python_version():
    """Check Python version"""
    print("🐍 Checking Python version...")
    version = sys.version_info
    print(f"   Python {version.major}.{version.minor}.{version.micro}")
    
    if version.major == 3 and version.minor >= 8:
        print("   ✅ Python version is compatible")
        return True
    else:
        print("   ❌ Python 3.8+ required")
        return False

def check_gpu():
    """Check GPU availability"""
    print("\n🎮 Checking GPU availability...")
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            print(f"   ✅ CUDA available: {gpu_count} GPU(s)")
            print(f"   🎯 GPU 0: {gpu_name}")
            return True
        else:
            print("   ❌ CUDA not available")
            return False
    except ImportError:
        print("   ❌ PyTorch not installed")
        return False

def check_dependencies():
    """Check required dependencies"""
    print("\n📦 Checking dependencies...")
    
    required_packages = [
        'torch', 'torchvision', 'monai', 'pandas', 'numpy', 
        'PIL', 'requests', 'transformers', 'scikit-learn',
        'matplotlib', 'seaborn', 'tqdm'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'PIL':
                import PIL
            else:
                importlib.import_module(package)
            print(f"   ✅ {package}")
        except ImportError:
            print(f"   ❌ {package}")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n   📋 Missing packages: {', '.join(missing_packages)}")
        print(f"   💡 Install with: pip install {' '.join(missing_packages)}")
        return False
    else:
        print("   ✅ All dependencies installed")
        return True

def check_data_paths():
    """Check data paths and files"""
    print("\n📁 Checking data paths...")
    
    # Check if we're on server
    gleamer_path = "/sharedata01/CNN_data/gleamer/gleamer"
    csv_path = os.path.join(gleamer_path, "dicom_image_url_file.csv")
    
    if os.path.exists(gleamer_path):
        print(f"   ✅ Gleamer directory found: {gleamer_path}")
        
        # Check CSV file
        if os.path.exists(csv_path):
            print(f"   ✅ CSV file found: {csv_path}")
            
            # Check CSV size
            try:
                import pandas as pd
                df = pd.read_csv(csv_path)
                print(f"   📊 CSV contains {len(df)} records")
            except Exception as e:
                print(f"   ❌ Error reading CSV: {e}")
                return False
        else:
            print(f"   ❌ CSV file not found: {csv_path}")
            return False
        
        # Check for image files
        try:
            files = os.listdir(gleamer_path)
            image_files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            print(f"   🖼️ Found {len(image_files)} image files")
            
            if len(image_files) > 0:
                print(f"   📁 Sample files: {image_files[:3]}")
            else:
                print("   ⚠️ No image files found in gleamer directory")
                
        except Exception as e:
            print(f"   ❌ Error listing directory: {e}")
            return False
            
    else:
        print(f"   ❌ Gleamer directory not found: {gleamer_path}")
        print("   💡 Make sure you're running on the H200 server")
        return False
    
    return True

def check_config():
    """Check configuration file"""
    print("\n⚙️ Checking configuration...")
    
    try:
        from config import (
            CSV_FILE, EXISTING_IMAGES_DIR, DATA_ROOT,
            PREPROCESSING_CONFIG, TRAINING_CONFIG
        )
        
        print(f"   ✅ Config loaded successfully")
        print(f"   📁 CSV File: {CSV_FILE}")
        print(f"   🖼️ Images Dir: {EXISTING_IMAGES_DIR}")
        print(f"   💾 Data Root: {DATA_ROOT}")
        print(f"   🔧 Use Existing Images: {PREPROCESSING_CONFIG.get('use_existing_images', False)}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error loading config: {e}")
        return False

def run_quick_test():
    """Run a quick preprocessing test"""
    print("\n🧪 Running quick preprocessing test...")
    
    try:
        # Test image finding
        from step1_preprocessing import DatasetPreprocessor
        
        preprocessor = DatasetPreprocessor()
        
        # Test with a small sample
        print("   📊 Testing with 5 records...")
        
        # This will test the image finding logic
        df = preprocessor.load_dataset()
        if len(df) > 0:
            sample_df = df.head(5)
            processed_data, stats = preprocessor.preprocess_batch(sample_df)
            
            print(f"   ✅ Successfully processed {len(processed_data)} records")
            print(f"   📈 Stats: {stats}")
            return True
        else:
            print("   ❌ No data loaded")
            return False
            
    except Exception as e:
        print(f"   ❌ Preprocessing test failed: {e}")
        return False

def main():
    """Main verification function"""
    print("🚀 H200 GPU Server Setup Verification")
    print("=" * 50)
    
    checks = [
        ("Python Version", check_python_version),
        ("GPU Availability", check_gpu),
        ("Dependencies", check_dependencies),
        ("Data Paths", check_data_paths),
        ("Configuration", check_config),
        ("Quick Test", run_quick_test)
    ]
    
    results = []
    
    for check_name, check_func in checks:
        try:
            result = check_func()
            results.append((check_name, result))
        except Exception as e:
            print(f"   ❌ {check_name} check failed: {e}")
            results.append((check_name, False))
    
    # Summary
    print("\n📋 Verification Summary:")
    print("=" * 30)
    
    all_passed = True
    for check_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {check_name}: {status}")
        if not result:
            all_passed = False
    
    print(f"\n🎯 Overall Status: {'✅ READY' if all_passed else '❌ ISSUES FOUND'}")
    
    if all_passed:
        print("\n🚀 Ready to run preprocessing!")
        print("   Next steps:")
        print("   1. python step1_preprocessing.py --sample-size 100")
        print("   2. python run_all_steps.py")
    else:
        print("\n🔧 Please fix the issues above before proceeding")
    
    return all_passed

if __name__ == "__main__":
    main()
