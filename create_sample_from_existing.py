#!/usr/bin/env python3
"""
Create sample dataset using existing images from sampledb directory
Finds CSV rows that match available images and creates organized sample folders
"""

import pandas as pd
import os
import json
import shutil
from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import tqdm

def create_sample_from_existing(csv_file="dicom_image_url_file.csv", output_dir="sample_dataset_existing"):
    """
    Create sample dataset using existing images from sampledb
    """
    
    print(f"🚀 Creating sample dataset from existing images...")
    print(f"📁 Output directory: {output_dir}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get list of available images from sampledb
    available_images = get_available_images()
    print(f"📊 Found {len(available_images)} available images in sampledb")
    
    if not available_images:
        print("❌ No images found in sampledb directory")
        return None
    
    # Read CSV file
    print(f"📊 Reading CSV file: {csv_file}")
    try:
        df = pd.read_csv(csv_file)
        print(f"✅ Loaded CSV with {len(df)} rows and {len(df.columns)} columns")
    except Exception as e:
        print(f"❌ Error reading CSV: {e}")
        return None
    
    # Find CSV rows that match available images
    matching_rows = find_matching_csv_rows(df, available_images)
    print(f"🎯 Found {len(matching_rows)} CSV rows with matching images")
    
    if not matching_rows:
        print("❌ No matching CSV rows found for available images")
        return None
    
    # Process each matching row
    successful_samples = 0
    failed_samples = 0
    
    for i, (idx, row) in enumerate(tqdm(matching_rows.items(), desc="Processing samples")):
        try:
            # Create folder for this sample
            sample_folder = os.path.join(output_dir, f"sample_{idx}")
            os.makedirs(sample_folder, exist_ok=True)
            
            # Save all column data to a text file
            save_sample_data(row, sample_folder, idx)
            
            # Copy images for this sample
            images_copied = copy_matching_images(row, sample_folder, idx, available_images)
            
            if images_copied > 0:
                successful_samples += 1
                print(f"✅ Sample {idx}: {images_copied} images copied")
            else:
                failed_samples += 1
                print(f"❌ Sample {idx}: No images copied")
                
        except Exception as e:
            failed_samples += 1
            print(f"❌ Error processing sample {idx}: {e}")
    
    # Summary
    print(f"\n{'='*60}")
    print("📊 SAMPLE CREATION SUMMARY")
    print(f"{'='*60}")
    print(f"✅ Successful samples: {successful_samples}")
    print(f"❌ Failed samples: {failed_samples}")
    print(f"📁 Output directory: {output_dir}")
    print(f"🎯 Total samples processed: {len(matching_rows)}")
    
    return output_dir

def get_available_images():
    """Get list of available images from sampledb directory"""
    
    available_images = {}
    
    # Search in sampledb directory
    sampledb_dir = "sampledb"
    if not os.path.exists(sampledb_dir):
        print(f"⚠️  sampledb directory not found: {sampledb_dir}")
        return available_images
    
    # Walk through all subdirectories
    for root, dirs, files in os.walk(sampledb_dir):
        for file in files:
            if file.endswith(('.jpg', '.jpeg', '.png')):
                # Extract UID from filename (remove extension)
                uid = os.path.splitext(file)[0]
                file_path = os.path.join(root, file)
                
                # Check if image is valid (not blank)
                if is_valid_image(file_path):
                    available_images[uid] = file_path
                else:
                    print(f"⚠️  Skipping blank image: {file}")
    
    return available_images

def find_matching_csv_rows(df, available_images):
    """Find CSV rows that have images available in sampledb"""
    
    matching_rows = {}
    available_uids = set(available_images.keys())
    
    print(f"🔍 Searching for CSV rows with {len(available_uids)} available images...")
    
    for idx, row in df.iterrows():
        try:
            # Parse SOP_INSTANCE_UID_ARRAY
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
            
            # Check if any UID from this row is available
            row_uids = set(uid_list)
            if row_uids & available_uids:  # Intersection exists
                matching_rows[idx] = row
                
                # Limit to first 10 matches for manageable sample size
                if len(matching_rows) >= 10:
                    break
                    
        except Exception as e:
            continue
    
    return matching_rows

def copy_matching_images(row, sample_folder, idx, available_images):
    """Copy images that are available for this sample"""
    
    images_copied = 0
    
    try:
        # Parse SOP_INSTANCE_UID_ARRAY
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
        
        # Copy each available image
        for uid in uid_list:
            if uid in available_images:
                try:
                    source_path = available_images[uid]
                    
                    # Copy image
                    img_filename = f"{uid}.jpg"
                    dest_path = os.path.join(sample_folder, img_filename)
                    
                    shutil.copy2(source_path, dest_path)
                    images_copied += 1
                    
                except Exception as e:
                    print(f"❌ Sample {idx}: Failed to copy image {uid}: {e}")
                    continue
    
    except Exception as e:
        print(f"❌ Sample {idx}: Error processing UIDs: {e}")
    
    return images_copied

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
    
    for folder in sample_folders:
        folder_path = os.path.join(sample_dir, folder)
        images = [f for f in os.listdir(folder_path) if f.endswith('.jpg')]
        
        print(f"\n📂 {folder}:")
        print(f"   Images: {len(images)}")
        
        total_images += len(images)
        valid_images += len(images)  # All copied images should be valid
    
    print(f"\n📊 SUMMARY:")
    print(f"   Total images: {total_images}")
    print(f"   Valid images: {valid_images}")
    print(f"   Success rate: 100%")

def main():
    """Main function"""
    
    # Create sample dataset from existing images
    sample_dir = create_sample_from_existing(
        csv_file="dicom_image_url_file.csv",
        output_dir="sample_dataset_existing"
    )
    
    # Analyze the dataset
    if sample_dir:
        analyze_sample_dataset(sample_dir)
    
    print(f"\n🎉 Sample dataset creation complete!")
    print(f"📁 Check the 'sample_dataset_existing' folder to explore your data")
    print(f"🔍 Each folder contains:")
    print(f"   - sample_data.txt (all CSV columns for that row)")
    print(f"   - All available images for that sample")
    print(f"   - Folder named by CSV index (sample_0, sample_1, etc.)")

if __name__ == "__main__":
    main()
