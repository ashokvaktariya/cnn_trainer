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

def copy_test_images(csv_file, source_root, target_folder, num_samples=100):
    """Copy sample images from dataset to test_images folder with balanced positive/negative"""
    
    logger.info(f"📊 Loading dataset from: {csv_file}")
    
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
    
    logger.info(f"📁 Found {len(positive_samples)} positive images")
    logger.info(f"📁 Found {len(negative_samples)} negative images")
    
    if len(positive_samples) == 0 and len(negative_samples) == 0:
        logger.error("❌ No valid images found in dataset")
        return False
    
    # Create target folder
    os.makedirs(target_folder, exist_ok=True)
    logger.info(f"📁 Created target folder: {target_folder}")
    
    # Calculate balanced sampling
    half_samples = num_samples // 2
    remaining_samples = num_samples - (half_samples * 2)
    
    # Sample positive images
    if len(positive_samples) >= half_samples:
        selected_positive = random.sample(positive_samples, half_samples)
    else:
        selected_positive = positive_samples
        logger.warning(f"⚠️ Only {len(positive_samples)} positive images available")
    
    # Sample negative images
    if len(negative_samples) >= half_samples:
        selected_negative = random.sample(negative_samples, half_samples)
    else:
        selected_negative = negative_samples
        logger.warning(f"⚠️ Only {len(negative_samples)} negative images available")
    
    # Add remaining samples from the larger class
    selected_samples = selected_positive + selected_negative
    
    if remaining_samples > 0:
        # Add remaining samples from the class with more available images
        if len(positive_samples) > len(negative_samples):
            remaining_positive = [s for s in positive_samples if s not in selected_positive]
            if len(remaining_positive) >= remaining_samples:
                selected_samples.extend(random.sample(remaining_positive, remaining_samples))
        else:
            remaining_negative = [s for s in negative_samples if s not in selected_negative]
            if len(remaining_negative) >= remaining_samples:
                selected_samples.extend(random.sample(remaining_negative, remaining_samples))
    
    logger.info(f"🎯 Selected {len(selected_samples)} images to copy:")
    logger.info(f"   📈 Positive: {len([s for s in selected_samples if s['binary_label'] == 1])}")
    logger.info(f"   📉 Negative: {len([s for s in selected_samples if s['binary_label'] == 0])}")
    
    # Copy images
    copied_count = 0
    failed_count = 0
    
    for i, sample in enumerate(selected_samples, 1):
        try:
            source_path = sample['source_path']
            target_path = os.path.join(target_folder, sample['filename'])
            
            # Copy file
            shutil.copy2(source_path, target_path)
            copied_count += 1
            
            if i % 10 == 0:  # Log progress every 10 files
                logger.info(f"📋 Copied {i}/{len(selected_samples)} images...")
                
        except Exception as e:
            logger.error(f"❌ Failed to copy {sample['filename']}: {e}")
            failed_count += 1
            continue
    
    # Summary
    logger.info("=" * 60)
    logger.info("📊 COPY SUMMARY:")
    logger.info(f"   🎯 Total selected: {len(selected_samples)}")
    logger.info(f"   ✅ Successfully copied: {copied_count}")
    logger.info(f"   ❌ Failed to copy: {failed_count}")
    logger.info(f"   📁 Target folder: {target_folder}")
    
    # Show label distribution
    label_counts = {}
    for sample in selected_samples:
        label = sample['label']
        label_counts[label] = label_counts.get(label, 0) + 1
    
    logger.info("📈 Label distribution:")
    for label, count in label_counts.items():
        percentage = (count / len(selected_samples)) * 100
        logger.info(f"   {label}: {count} ({percentage:.1f}%)")
    
    logger.info("=" * 60)
    
    if copied_count > 0:
        logger.info("🎉 Test images copied successfully!")
        logger.info(f"💡 You can now run: python runpod_testing.py")
        return True
    else:
        logger.error("❌ No images were copied successfully")
        return False

def main():
    """Main function"""
    logger.info("🚀 Starting Test Images Copy Script")
    
    # Configuration
    csv_file = "final_dataset_cnn.csv"
    source_root = "/mount/civiescaks01storage01/aksfileshare01/CNN/gleamer-images/"
    target_folder = "test_images"
    num_samples = 100
    
    # Check if CSV exists
    if not os.path.exists(csv_file):
        logger.error(f"❌ CSV file not found: {csv_file}")
        return False
    
    # Check if source root exists
    if not os.path.exists(source_root):
        logger.error(f"❌ Source root not found: {source_root}")
        logger.info("💡 Make sure you're running this on the server with mount access")
        return False
    
    # Copy images
    success = copy_test_images(csv_file, source_root, target_folder, num_samples)
    
    return success

if __name__ == "__main__":
    success = main()
    if success:
        logger.info("✅ Test images copy completed successfully!")
    else:
        logger.error("❌ Test images copy failed!")
        exit(1)
