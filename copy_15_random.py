#!/usr/bin/env python3
import os
import shutil
import random

# Paths
IMAGE_ROOT = "/mount/civiescaks01storage01/aksfileshare01/CNN/gleamer-dicom/"
OUTPUT_DIR = "./random_dicom_samples"

def main():
    print("Copying 15 random DICOM files...")
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Get all DICOM files
    dicom_files = []
    for root, dirs, files in os.walk(IMAGE_ROOT):
        for file in files:
            if file.lower().endswith('.dcm') or file.lower().endswith('.dicom'):
                dicom_files.append(os.path.join(root, file))
    
    print(f"Found {len(dicom_files)} DICOM files")
    
    # Select 15 random files
    random_files = random.sample(dicom_files, min(15, len(dicom_files)))
    print(f"Selected {len(random_files)} random files")
    
    copied_count = 0
    
    for i, file_path in enumerate(random_files):
        filename = os.path.basename(file_path)
        dest_path = os.path.join(OUTPUT_DIR, filename)
        
        try:
            shutil.copy2(file_path, dest_path)
            print(f"File {i+1}: Copied {filename}")
            copied_count += 1
        except Exception as e:
            print(f"File {i+1}: ERROR copying {filename} - {e}")
    
    print(f"Done! Copied {copied_count} DICOM files to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
