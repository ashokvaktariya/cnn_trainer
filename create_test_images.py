#!/usr/bin/env python3
"""
Create test images with labels for real-time inference testing
Copies valid images from the dataset and adds labels to filenames
"""

import pandas as pd
import numpy as np
import os
import shutil
from pathlib import Path
import logging
from PIL import Image
import argparse

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
CSV_FILE = "/sharedata01/CNN_data/gleamer/gleamer/dicom_image_url_file.csv"
IMAGES_DIR = "/sharedata01/CNN_data/gleamer/gleamer"
OUTPUT_DIR = "/sharedata01/CNN_data/medical_classification/test_images"
SAMPLES_PER_LABEL = 10  # Number of images to copy per label

def is_valid_image(image_path):
    """Check if image is valid (not blank/corrupted)"""
    try:
        img = Image.open(image_path)
        img_array = np.array(img)
        
        # Check if image is not blank (has variation in pixel values)
        if len(img_array.shape) == 2:  # Grayscale
            if img_array.std() < 10:  # Very low variation
                return False
        else:  # RGB
            if img_array.std() < 10:
                return False
        
        # Check if image is not too small
        if img.size[0] < 50 or img.size[1] < 50:
            return False
            
        return True
    except Exception as e:
        logger.warning(f"Invalid image {image_path}: {e}")
        return False

def create_test_images():
    """Create test images with labels in filenames"""
    
    logger.info("🚀 Creating test images for real-time inference")
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    logger.info(f"📁 Output directory: {OUTPUT_DIR}")
    
    # Load CSV data
    logger.info("📊 Loading CSV data...")
    df = pd.read_csv(CSV_FILE)
    logger.info(f"✅ Loaded {len(df)} records")
    
    # Analyze label distribution
    label_counts = df['GLEAMER_FINDING'].value_counts()
    logger.info(f"📊 Label distribution:")
    for label, count in label_counts.items():
        logger.info(f"   {label}: {count}")
    
    # Create subdirectories for each label
    labels = ['POSITIVE', 'NEGATIVE', 'DOUBT']
    for label in labels:
        label_dir = os.path.join(OUTPUT_DIR, label.lower())
        os.makedirs(label_dir, exist_ok=True)
    
    # Copy images with labels in filenames
    copied_count = {label: 0 for label in labels}
    
    logger.info("🖼️ Copying test images...")
    
    for idx, row in df.iterrows():
        if idx % 1000 == 0:
            logger.info(f"   Processed {idx}/{len(df)} records...")
        
        label = row['GLEAMER_FINDING']
        if label not in labels:
            continue
            
        # Check if we have enough samples for this label
        if copied_count[label] >= SAMPLES_PER_LABEL:
            continue
        
        # Parse UIDs
        try:
            uids_str = row['SOP_INSTANCE_UID_ARRAY']
            if pd.isna(uids_str) or uids_str == '':
                continue
                
            # Parse UIDs (handle both string and list formats)
            if isinstance(uids_str, str):
                if uids_str.startswith('[') and uids_str.endswith(']'):
                    # Remove brackets and split
                    uids_str = uids_str[1:-1]
                uids = [uid.strip().strip("'\"") for uid in uids_str.split(',')]
            else:
                uids = [str(uids_str)]
            
            # Try to find valid images for this record
            for uid in uids:
                if copied_count[label] >= SAMPLES_PER_LABEL:
                    break
                    
                uid = uid.strip()
                if not uid:
                    continue
                
                # Look for image file
                possible_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
                image_path = None
                
                for ext in possible_extensions:
                    potential_path = os.path.join(IMAGES_DIR, f"{uid}{ext}")
                    if os.path.exists(potential_path):
                        image_path = potential_path
                        break
                
                if image_path and is_valid_image(image_path):
                    # Create filename with label
                    filename = f"LABEL_{label}_UID_{uid}_IDX_{idx}"
                    if image_path.lower().endswith('.jpg') or image_path.lower().endswith('.jpeg'):
                        filename += '.jpg'
                    elif image_path.lower().endswith('.png'):
                        filename += '.png'
                    else:
                        filename += '.jpg'  # Default to jpg
                    
                    # Copy to appropriate directory
                    output_path = os.path.join(OUTPUT_DIR, label.lower(), filename)
                    
                    try:
                        shutil.copy2(image_path, output_path)
                        copied_count[label] += 1
                        logger.info(f"✅ Copied {label}: {filename}")
                        
                        if copied_count[label] >= SAMPLES_PER_LABEL:
                            break
                            
                    except Exception as e:
                        logger.warning(f"❌ Failed to copy {image_path}: {e}")
                        
        except Exception as e:
            logger.warning(f"⚠️ Error processing row {idx}: {e}")
            continue
    
    # Summary
    logger.info("🎉 Test images creation complete!")
    logger.info("📊 Summary:")
    total_copied = 0
    for label in labels:
        count = copied_count[label]
        total_copied += count
        logger.info(f"   {label}: {count} images")
    
    logger.info(f"📁 Total images copied: {total_copied}")
    logger.info(f"📂 Test images location: {OUTPUT_DIR}")
    
    # Create a summary file
    summary_file = os.path.join(OUTPUT_DIR, "test_images_summary.txt")
    with open(summary_file, 'w') as f:
        f.write("TEST IMAGES SUMMARY\n")
        f.write("==================\n\n")
        f.write(f"Total images: {total_copied}\n")
        f.write(f"Created: {pd.Timestamp.now()}\n\n")
        
        for label in labels:
            count = copied_count[label]
            f.write(f"{label}: {count} images\n")
            f.write(f"  Directory: {os.path.join(OUTPUT_DIR, label.lower())}\n")
            f.write(f"  Filename format: LABEL_{label}_UID_<uid>_IDX_<index>.jpg\n\n")
        
        f.write("\nFILENAME FORMAT:\n")
        f.write("LABEL_<POSITIVE|NEGATIVE|DOUBT>_UID_<unique_id>_IDX_<csv_index>.jpg\n")
        f.write("\nExample: LABEL_POSITIVE_UID_2.25.123456789_IDX_1234.jpg\n")
    
    logger.info(f"📄 Summary saved: {summary_file}")
    
    return OUTPUT_DIR

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Create test images for inference testing')
    parser.add_argument('--samples', type=int, default=10, help='Number of samples per label (default: 10)')
    parser.add_argument('--output', type=str, default='/sharedata01/CNN_data/medical_classification/test_images', help='Output directory (default: /sharedata01/CNN_data/medical_classification/test_images)')
    
    args = parser.parse_args()
    
    # Update global variables
    global SAMPLES_PER_LABEL, OUTPUT_DIR
    SAMPLES_PER_LABEL = args.samples
    OUTPUT_DIR = args.output
    
    try:
        output_dir = create_test_images()
        logger.info(f"✅ Test images created successfully in: {output_dir}")
        logger.info("🚀 Ready for real-time inference testing!")
        
    except Exception as e:
        logger.error(f"❌ Error creating test images: {e}")
        raise

if __name__ == "__main__":
    main()
