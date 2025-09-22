#!/usr/bin/env python3
"""
Simple script to run X-ray classification on your dataset
"""

import os
import sys
from pathlib import Path

def main():
    print("🏥 X-ray Image Classifier")
    print("=" * 50)
    
    # Default paths (update these for your setup)
    csv_path = "/mount/civiescaks01storage01/aksfileshare01/CNN/gleamer-images/processed_dicom_image_url_file.csv"
    image_root = "/mount/civiescaks01storage01/aksfileshare01/CNN/gleamer-images/gleamer-images"
    output_dir = "./filtered_output"
    
    # Check if paths exist
    if not os.path.exists(csv_path):
        print(f"❌ CSV file not found: {csv_path}")
        print("Please update the csv_path in this script")
        return
    
    if not os.path.exists(image_root):
        print(f"❌ Image root directory not found: {image_root}")
        print("Please update the image_root in this script")
        return
    
    print(f"📁 CSV file: {csv_path}")
    print(f"📁 Image root: {image_root}")
    print(f"📁 Output directory: {output_dir}")
    print()
    
    # Ask user for options
    print("Options:")
    print("1. Test on 1000 samples (recommended)")
    print("2. Process all samples")
    print("3. Custom number of samples")
    
    choice = input("Enter your choice (1-3): ").strip()
    
    if choice == "1":
        limit = 1000
        print("🚀 Starting classification on 1000 samples...")
    elif choice == "2":
        limit = 0  # 0 means all
        print("🚀 Starting classification on ALL samples...")
    elif choice == "3":
        try:
            limit = int(input("Enter number of samples: "))
            print(f"🚀 Starting classification on {limit} samples...")
        except ValueError:
            print("❌ Invalid number. Using 1000 samples.")
            limit = 1000
    else:
        print("❌ Invalid choice. Using 1000 samples.")
        limit = 1000
    
    # Import and run classifier
    try:
        from xray_classifier_filter import XrayClassifierFilter
        
        classifier = XrayClassifierFilter(
            csv_path=csv_path,
            image_root=image_root,
            output_dir=output_dir
        )
        
        results = classifier.process_dataset(limit=limit)
        
        print("\n✅ Classification complete!")
        print(f"📊 Results saved in: {output_dir}")
        print(f"📁 Filtered X-ray images: {output_dir}/filtered_gleamer_data")
        print(f"📁 Invalid images: {output_dir}/invalid")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Please install requirements: pip install -r requirements_classifier.txt")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
