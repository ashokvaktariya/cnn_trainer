#!/usr/bin/env python3
"""
Script to copy sample images from dataset to test_images folder
Uses final_dataset_cnn.csv to get image paths and copies them locally
"""

import os
import pandas as pd
import shutil
import random
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def copy_test_images(csv_file, source_root, target_folder, num_positive=20, num_negative=25):
    """Copy specific number of positive and negative images from validation dataset to test_images folder"""
    
    logger.info(f"📊 Loading validation dataset from: {csv_file}")
    
    # Load CSV
    df = pd.read_csv(csv_file)
    
    # Filter for valid image paths and separate by class
    positive_samples = []
    negative_samples = []
    
    for idx, row in df.iterrows():
        image_path = os.path.join(source_root, row['image_path'])
        if os.path.exists(image_path):
            sample = {
                'source_path': image_path,
                'filename': os.path.basename(row['image_path']),
                'label': row['label'],
                'binary_label': row['binary_label']
            }
            
            if row['binary_label'] == 1:  # POSITIVE
                positive_samples.append(sample)
            else:  # NEGATIVE
                negative_samples.append(sample)
    
    logger.info(f"📁 Found {len(positive_samples)} positive images in validation set")
    logger.info(f"📁 Found {len(negative_samples)} negative images in validation set")
    
    if len(positive_samples) == 0 and len(negative_samples) == 0:
        logger.error("❌ No valid images found in validation dataset")
        return False
    
    # Create target folders
    os.makedirs(target_folder, exist_ok=True)
    positive_folder = os.path.join(target_folder, "positive")
    negative_folder = os.path.join(target_folder, "negative")
    
    os.makedirs(positive_folder, exist_ok=True)
    os.makedirs(negative_folder, exist_ok=True)
    
    logger.info(f"📁 Created target folders:")
    logger.info(f"   📁 Main folder: {target_folder}")
    logger.info(f"   📈 Positive folder: {positive_folder}")
    logger.info(f"   📉 Negative folder: {negative_folder}")
    
    # Sample exact number of positive images
    if len(positive_samples) >= num_positive:
        selected_positive = random.sample(positive_samples, num_positive)
    else:
        selected_positive = positive_samples
        logger.warning(f"⚠️ Only {len(positive_samples)} positive images available, using all")
    
    # Sample exact number of negative images
    if len(negative_samples) >= num_negative:
        selected_negative = random.sample(negative_samples, num_negative)
    else:
        selected_negative = negative_samples
        logger.warning(f"⚠️ Only {len(negative_samples)} negative images available, using all")
    
    logger.info(f"🎯 Selected images to copy:")
    logger.info(f"   📈 Positive: {len(selected_positive)}")
    logger.info(f"   📉 Negative: {len(selected_negative)}")
    
    # Copy positive images
    copied_positive = 0
    failed_positive = 0
    
    logger.info("📈 Copying positive images...")
    for i, sample in enumerate(selected_positive, 1):
        try:
            source_path = sample['source_path']
            target_path = os.path.join(positive_folder, sample['filename'])
            
            # Copy file
            shutil.copy2(source_path, target_path)
            copied_positive += 1
            
            if i % 5 == 0:  # Log progress every 5 files
                logger.info(f"   📋 Copied {i}/{len(selected_positive)} positive images...")
                
        except Exception as e:
            logger.error(f"❌ Failed to copy positive {sample['filename']}: {e}")
            failed_positive += 1
            continue
    
    # Copy negative images
    copied_negative = 0
    failed_negative = 0
    
    logger.info("📉 Copying negative images...")
    for i, sample in enumerate(selected_negative, 1):
        try:
            source_path = sample['source_path']
            target_path = os.path.join(negative_folder, sample['filename'])
            
            # Copy file
            shutil.copy2(source_path, target_path)
            copied_negative += 1
            
            if i % 5 == 0:  # Log progress every 5 files
                logger.info(f"   📋 Copied {i}/{len(selected_negative)} negative images...")
                
        except Exception as e:
            logger.error(f"❌ Failed to copy negative {sample['filename']}: {e}")
            failed_negative += 1
            continue
    
    # Calculate totals
    copied_count = copied_positive + copied_negative
    failed_count = failed_positive + failed_negative
    
    # Summary
    logger.info("=" * 60)
    logger.info("📊 COPY SUMMARY:")
    logger.info(f"   📈 Positive images: {copied_positive}/{len(selected_positive)} copied")
    logger.info(f"   📉 Negative images: {copied_negative}/{len(selected_negative)} copied")
    logger.info(f"   ✅ Total successfully copied: {copied_count}")
    logger.info(f"   ❌ Total failed to copy: {failed_count}")
    
    logger.info("📁 Folder structure:")
    logger.info(f"   📁 {target_folder}/")
    logger.info(f"   ├── 📈 positive/ ({copied_positive} images)")
    logger.info(f"   └── 📉 negative/ ({copied_negative} images)")
    
    logger.info("=" * 60)
    
    if copied_count > 0:
        logger.info("🎉 Test images copied successfully!")
        logger.info(f"💡 You can now run: python runpod_testing.py")
        logger.info(f"📊 Folder structure: test_images/positive/ and test_images/negative/")
        return True
    else:
        logger.error("❌ No images were copied successfully")
        return False

def main():
    """Main function"""
    logger.info("🚀 Starting Test Images Copy Script from Validation Dataset")
    
    # Configuration
    csv_file = "preprocessed_balanced/balanced_dataset_cnn_val.csv"
    source_root = "/mount/civiescaks01storage01/aksfileshare01/CNN/gleamer-images/"
    target_folder = "test_images"
    num_positive = 20
    num_negative = 25
    
    # Check if CSV exists
    if not os.path.exists(csv_file):
        logger.error(f"❌ Validation CSV file not found: {csv_file}")
        return False
    
    # Check if source root exists
    if not os.path.exists(source_root):
        logger.error(f"❌ Source root not found: {source_root}")
        logger.info("💡 Make sure you're running this on the server with mount access")
        return False
    
    # Copy images
    success = copy_test_images(csv_file, source_root, target_folder, num_positive, num_negative)
    
    return success

if __name__ == "__main__":
    success = main()
    if success:
        logger.info("✅ Test images copy completed successfully!")
    else:
        logger.error("❌ Test images copy failed!")
        exit(1)
