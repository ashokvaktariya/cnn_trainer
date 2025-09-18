#!/usr/bin/env python3
"""
Create sample dataset by copying existing images from server directory
Each sample will be in a folder named by its index, containing all images and data
"""

import pandas as pd
import os
import json
import random
import shutil
from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import tqdm

# Server configuration
SERVER_IMAGES_DIR = "/sharedata01/CNN_data/gleamer/gleamer"
LOCAL_IMAGES_DIR = "sampledb"  # Local fallback
OUTPUT_DIR = "sampledb"  # Output directory name

def create_sample_dataset(csv_file="dicom_image_url_file.csv", num_samples=20, output_dir=OUTPUT_DIR):
    """
    Create sample dataset by copying existing images from server/local directory
    
    Args:
        csv_file: Path to the CSV file
        num_samples: Number of random samples to process
        output_dir: Directory to save the sample dataset
    """
    
    print(f"🚀 Creating sample dataset with {num_samples} random samples...")
    print(f"📁 Output directory: {output_dir}")
    print(f"🔍 Looking for images in: {SERVER_IMAGES_DIR}")
    print(f"🔍 Fallback directory: {LOCAL_IMAGES_DIR}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Check which image directory to use
    images_source_dir = find_image_directory()
    print(f"✅ Using image directory: {images_source_dir}")
    
    # Read CSV file
    print(f"📊 Reading CSV file: {csv_file}")
    try:
        df = pd.read_csv(csv_file)
        print(f"✅ Loaded CSV with {len(df)} rows and {len(df.columns)} columns")
        print(f"📋 Columns: {list(df.columns)}")
    except Exception as e:
        print(f"❌ Error reading CSV: {e}")
        return
    
    # Select random samples
    if len(df) < num_samples:
        num_samples = len(df)
        print(f"⚠️  CSV has only {len(df)} rows, using all of them")
    
    random_indices = random.sample(range(len(df)), num_samples)
    print(f"🎲 Selected random indices: {random_indices}")
    
    # Process each sample
    successful_downloads = 0
    failed_downloads = 0
    
    for i, idx in enumerate(tqdm(random_indices, desc="Processing samples")):
        try:
            # Create folder for this sample
            sample_folder = os.path.join(output_dir, f"sample_{idx}")
            os.makedirs(sample_folder, exist_ok=True)
            
            # Get row data
            row = df.iloc[idx]
            
            # Save all column data to a text file
            save_sample_data(row, sample_folder, idx)
            
            # Copy images for this sample
            images_copied = copy_sample_images(row, sample_folder, idx, images_source_dir)
            
            if images_copied > 0:
                successful_downloads += 1
                print(f"✅ Sample {idx}: {images_copied} images copied")
            else:
                failed_downloads += 1
                print(f"❌ Sample {idx}: No images found/copied")
                
        except Exception as e:
            failed_downloads += 1
            print(f"❌ Error processing sample {idx}: {e}")
    
    # Summary
    print(f"\n{'='*60}")
    print("📊 DOWNLOAD SUMMARY")
    print(f"{'='*60}")
    print(f"✅ Successful samples: {successful_downloads}")
    print(f"❌ Failed samples: {failed_downloads}")
    print(f"📁 Output directory: {output_dir}")
    print(f"🎯 Total samples processed: {len(random_indices)}")
    
    return output_dir

def find_image_directory():
    """Find the best image directory to use (server or local)"""
    
    # Check server directory first
    if os.path.exists(SERVER_IMAGES_DIR):
        print(f"✅ Found server images directory: {SERVER_IMAGES_DIR}")
        return SERVER_IMAGES_DIR
    
    # Check local directory
    if os.path.exists(LOCAL_IMAGES_DIR):
        print(f"✅ Found local images directory: {LOCAL_IMAGES_DIR}")
        return LOCAL_IMAGES_DIR
    
    # Check if we're in the same directory as CSV (for server)
    current_dir = os.getcwd()
    if os.path.exists(os.path.join(current_dir, "dicom_image_url_file.csv")):
        print(f"✅ Using current directory as image source: {current_dir}")
        return current_dir
    
    print(f"⚠️  No image directory found, using current directory: {current_dir}")
    return current_dir

def save_sample_data(row, sample_folder, idx):
    """Save all column data to a text file"""
    
    data_file = os.path.join(sample_folder, "sample_data.txt")
    
    with open(data_file, 'w', encoding='utf-8') as f:
        f.write(f"SAMPLE {idx} - COMPLETE DATA\n")
        f.write("="*50 + "\n\n")
        
        for col in row.index:
            f.write(f"{col}:\n")
            f.write(f"{str(row[col])}\n")
            f.write("-" * 30 + "\n\n")

def copy_sample_images(row, sample_folder, idx, source_dir):
    """Copy all images for a sample from existing directory"""
    
    images_copied = 0
    
    try:
        # Parse SOP_INSTANCE_UID_ARRAY for image naming
        sop_uids = row['SOP_INSTANCE_UID_ARRAY']
        if isinstance(sop_uids, str):
            try:
                uid_list = eval(sop_uids)
            except:
                try:
                    uid_list = json.loads(sop_uids)
                except:
                    uid_list = [sop_uids]
        else:
            uid_list = sop_uids
        
        # Copy each image
        for i, uid in enumerate(uid_list):
            try:
                # Try different possible locations and filenames
                possible_paths = find_image_file(uid, source_dir)
                
                if possible_paths:
                    source_path = possible_paths[0]  # Use first found
                    
                    # Copy image
                    img_filename = f"{uid}.jpg"
                    dest_path = os.path.join(sample_folder, img_filename)
                    
                    shutil.copy2(source_path, dest_path)
                    
                    # Verify image is valid (not blank)
                    if is_valid_image(dest_path):
                        images_copied += 1
                    else:
                        print(f"⚠️  Sample {idx}: Image {img_filename} appears to be blank")
                        os.remove(dest_path)  # Remove blank image
                        
                else:
                    print(f"⚠️  Sample {idx}: Image {uid}.jpg not found in source directory")
                    
            except Exception as e:
                print(f"❌ Sample {idx}: Failed to copy image {uid}: {e}")
                continue
    
    except Exception as e:
        print(f"❌ Sample {idx}: Error processing UIDs: {e}")
    
    return images_copied

def find_image_file(uid, source_dir):
    """Find image file by UID in various possible locations"""
    
    possible_paths = []
    
    # Possible filenames
    possible_names = [
        f"{uid}.jpg",
        f"{uid}.jpeg", 
        f"{uid}.png",
        f"{uid}.dcm"
    ]
    
    # Possible directories to search
    search_dirs = [
        source_dir,
        os.path.join(source_dir, "images"),
        os.path.join(source_dir, "data"),
        os.path.join(source_dir, "positive"),
        os.path.join(source_dir, "negative"),
        os.path.join(source_dir, "Negetive"),
        os.path.join(source_dir, "Negative 2")
    ]
    
    # Search recursively in all directories
    for search_dir in search_dirs:
        if os.path.exists(search_dir):
            for root, dirs, files in os.walk(search_dir):
                for filename in files:
                    if any(filename == name for name in possible_names):
                        possible_paths.append(os.path.join(root, filename))
    
    return possible_paths

def is_valid_image(image_path):
    """Check if image is valid (not blank)"""
    try:
        img = Image.open(image_path)
        img_array = np.array(img)
        
        # Check if image is blank (all same pixel values or very low variance)
        unique_values = len(np.unique(img_array))
        std_dev = np.std(img_array)
        
        return unique_values > 2 and std_dev > 5
        
    except Exception:
        return False

def analyze_sample_dataset(sample_dir):
    """Analyze the created sample dataset"""
    
    print(f"\n{'='*60}")
    print("🔍 SAMPLE DATASET ANALYSIS")
    print(f"{'='*60}")
    
    if not os.path.exists(sample_dir):
        print(f"❌ Sample directory not found: {sample_dir}")
        return
    
    sample_folders = [f for f in os.listdir(sample_dir) if f.startswith('sample_')]
    
    print(f"📁 Found {len(sample_folders)} sample folders")
    
    total_images = 0
    valid_images = 0
    blank_images = 0
    
    for folder in sample_folders:
        folder_path = os.path.join(sample_dir, folder)
        images = [f for f in os.listdir(folder_path) if f.endswith('.jpg')]
        
        print(f"\n📂 {folder}:")
        print(f"   Images: {len(images)}")
        
        total_images += len(images)
        
        for img in images:
            img_path = os.path.join(folder_path, img)
            if is_valid_image(img_path):
                valid_images += 1
            else:
                blank_images += 1
    
    print(f"\n📊 SUMMARY:")
    print(f"   Total images: {total_images}")
    print(f"   Valid images: {valid_images}")
    print(f"   Blank images: {blank_images}")
    print(f"   Success rate: {valid_images/total_images*100:.1f}%" if total_images > 0 else "   Success rate: 0%")

def main():
    """Main function"""
    
    # Set random seed for reproducibility
    random.seed(42)
    
    # Create sample dataset
    sample_dir = create_sample_dataset(
        csv_file="dicom_image_url_file.csv",
        num_samples=20,
        output_dir="sample_dataset"
    )
    
    # Analyze the dataset
    if sample_dir:
        analyze_sample_dataset(sample_dir)
    
    print(f"\n🎉 Sample dataset creation complete!")
    print(f"📁 Check the '{OUTPUT_DIR}' folder to explore your data")
    print(f"🔍 Each folder contains:")
    print(f"   - sample_data.txt (all CSV columns for that row)")
    print(f"   - All images for that sample (copied from existing directory)")
    print(f"   - Folder named by CSV index (sample_0, sample_1, etc.)")

if __name__ == "__main__":
    main()
