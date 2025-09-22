#!/usr/bin/env python3
"""
Setup script for medical fracture detection training on H100 server
"""

import os
import subprocess
import sys
from pathlib import Path

def check_gpu():
    """Check GPU availability and specs"""
    print("🔍 Checking GPU configuration...")
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ NVIDIA GPU detected")
            print(result.stdout)
            return True
        else:
            print("❌ No NVIDIA GPU detected")
            return False
    except FileNotFoundError:
        print("❌ nvidia-smi not found. Please install NVIDIA drivers.")
        return False

def check_python_packages():
    """Check if required packages are installed"""
    print("\n📦 Checking Python packages...")
    
    required_packages = [
        'torch', 'torchvision', 'pandas', 'numpy', 'scikit-learn',
        'matplotlib', 'seaborn', 'PIL', 'yaml', 'tqdm'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - MISSING")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️ Missing packages: {missing_packages}")
        print("Run: pip install -r requirements_training.txt")
        return False
    
    return True

def check_data_paths():
    """Check if data paths exist"""
    print("\n📁 Checking data paths...")
    
    config_path = "config_training.yaml"
    if not os.path.exists(config_path):
        print(f"❌ Config file not found: {config_path}")
        return False
    
    # Read config
    import yaml
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    csv_path = config['data']['csv_path']
    image_root = config['data']['image_root']
    output_dir = config['data']['output_dir']
    
    # Check CSV file
    if os.path.exists(csv_path):
        print(f"✅ CSV file found: {csv_path}")
    else:
        print(f"❌ CSV file not found: {csv_path}")
        return False
    
    # Check image root
    if os.path.exists(image_root):
        print(f"✅ Image root found: {image_root}")
    else:
        print(f"❌ Image root not found: {image_root}")
        return False
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    print(f"✅ Output directory created: {output_dir}")
    
    return True

def install_requirements():
    """Install required packages"""
    print("\n📦 Installing requirements...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements_training.txt"])
        print("✅ Requirements installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install requirements: {e}")
        return False

def run_data_analysis():
    """Run data analysis"""
    print("\n📊 Running data analysis...")
    try:
        subprocess.check_call([sys.executable, "data_overview.py"])
        print("✅ Data analysis completed!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Data analysis failed: {e}")
        return False

def create_training_script():
    """Create a simple training script"""
    script_content = '''#!/bin/bash
# Medical Fracture Detection Training Script

echo "🏥 Starting Medical Fracture Detection Training"
echo "=============================================="

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Run data analysis first
echo "📊 Running data analysis..."
python data_overview.py

# Start training
echo "🚀 Starting model training..."
python train_model.py

echo "✅ Training completed!"
'''
    
    with open('run_training.sh', 'w') as f:
        f.write(script_content)
    
    # Make executable
    os.chmod('run_training.sh', 0o755)
    print("✅ Training script created: run_training.sh")

def main():
    """Main setup function"""
    print("🏥 Medical Fracture Detection - Training Setup")
    print("=" * 50)
    
    # Check GPU
    if not check_gpu():
        print("⚠️ GPU check failed. Training may not work optimally.")
    
    # Check packages
    if not check_python_packages():
        print("📦 Installing missing packages...")
        if not install_requirements():
            print("❌ Setup failed at package installation")
            return False
    
    # Check data paths
    if not check_data_paths():
        print("❌ Setup failed at data path validation")
        return False
    
    # Run data analysis
    print("\n📊 Running initial data analysis...")
    if not run_data_analysis():
        print("⚠️ Data analysis failed, but continuing with setup")
    
    # Create training script
    create_training_script()
    
    print("\n" + "=" * 50)
    print("✅ SETUP COMPLETED SUCCESSFULLY!")
    print("=" * 50)
    print("📋 Next steps:")
    print("   1. Review the data analysis report")
    print("   2. Check visualizations in the output directory")
    print("   3. Run training: ./run_training.sh")
    print("   4. Or run manually: python train_model.py")
    print("\n🔧 Configuration files:")
    print("   • config_training.yaml - Training configuration")
    print("   • requirements_training.txt - Python dependencies")
    print("   • data_overview.py - Data analysis script")
    print("   • train_model.py - Main training script")
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)
