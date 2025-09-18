#!/usr/bin/env python3
"""
Dataset Distribution Analysis Script

This script analyzes the dataset to show:
- Distribution of positive, negative, and doubt cases
- Blank image counts for each category
- Overall dataset statistics

Usage Examples:
    # Analyze full dataset with images and detailed logs (slow)
    python3 analyze_dataset_distribution.py
    
    # Analyze only CSV labels with logs (fast)
    python3 analyze_dataset_distribution.py --csv-only
    
    # Analyze sample of 1000 records with images and verbose logging
    python3 analyze_dataset_distribution.py --sample-size 1000 --verbose
    
    # Analyze specific CSV file with full logging
    python3 analyze_dataset_distribution.py --csv /path/to/file.csv --verbose
    
    # Quick CSV analysis without image processing
    python3 analyze_dataset_distribution.py --csv-only --sample-size 100
"""

import os
import pandas as pd
import numpy as np
from PIL import Image
from collections import defaultdict
import argparse

# Configuration
SERVER_BASE_DIR = "/sharedata01/CNN_data/gleamer/gleamer"
SERVER_CSV_FILE = os.path.join(SERVER_BASE_DIR, "dicom_image_url_file.csv")
LOCAL_CSV_FILE = "dicom_image_url_file.csv"  # Local fallback

def is_blank_image(image_path):
    """
    Check if an image is blank (all pixels are same color or very low variance)
    """
    try:
        with Image.open(image_path) as img:
            # Convert to RGB if needed
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Convert to numpy array
            img_array = np.array(img)
            
            # Check if all pixels are the same color (blank)
            if len(img_array.shape) == 3:
                # Check if all RGB values are the same
                if np.all(img_array == img_array[0, 0]):
                    return True
                
                # Check variance - very low variance indicates blank image
                variance = np.var(img_array)
                if variance < 10:  # Very low variance threshold
                    return True
            else:
                # Grayscale image
                if np.all(img_array == img_array[0, 0]):
                    return True
                variance = np.var(img_array)
                if variance < 10:
                    return True
            
            return False
    except Exception as e:
        print(f"Error checking image {image_path}: {e}")
        return True  # Assume blank if we can't read it

def find_image_files(uid, base_dir, verbose=False):
    """Find image files for a given UID in the server directory structure"""
    possible_extensions = ['.jpg', '.jpeg', '.png', '.dcm']
    possible_paths = []
    
    for ext in possible_extensions:
        # Check in main directory (server stores images directly in gleamer folder)
        main_path = os.path.join(base_dir, f"{uid}{ext}")
        if os.path.exists(main_path):
            possible_paths.append(main_path)
            if verbose:
                print(f"      📁 Found in main dir: {main_path}")
        
        # Check in common subdirectories
        for subdir in ['images', 'data', 'positive', 'negative', 'Negetive', 'Negative 2', 'gleamer']:
            subdir_path = os.path.join(base_dir, subdir, f"{uid}{ext}")
            if os.path.exists(subdir_path):
                possible_paths.append(subdir_path)
                if verbose:
                    print(f"      📁 Found in subdir: {subdir_path}")
        
        # Also check if there are nested directories (common in medical imaging)
        try:
            for root, dirs, files in os.walk(base_dir):
                for file in files:
                    if file == f"{uid}{ext}":
                        possible_paths.append(os.path.join(root, file))
                        if verbose:
                            print(f"      📁 Found in nested dir: {os.path.join(root, file)}")
                        break
                if possible_paths:  # Found at least one, break outer loop
                    break
        except Exception as e:
            if verbose:
                print(f"      ⚠️ Error walking directory: {e}")
            pass  # Skip if directory traversal fails
    
    return possible_paths

def analyze_dataset_distribution(csv_file=None, sample_size=None, check_images=True):
    """
    Analyze dataset distribution by label and blank image status
    
    Args:
        csv_file: Path to CSV file (None for auto-detection)
        sample_size: Number of samples to analyze (None for all)
        check_images: Whether to check if images are blank
    """
    # Determine CSV file path
    if csv_file is None:
        if os.path.exists(SERVER_CSV_FILE):
            csv_file = SERVER_CSV_FILE
            base_dir = SERVER_BASE_DIR
            print(f"📁 Using server CSV: {csv_file}")
        elif os.path.exists(LOCAL_CSV_FILE):
            csv_file = LOCAL_CSV_FILE
            base_dir = "."
            print(f"📁 Using local CSV: {csv_file}")
        else:
            print("❌ No CSV file found!")
            return
    
    # Read CSV
    try:
        df = pd.read_csv(csv_file)
        print(f"📊 CSV loaded: {len(df)} records")
        print(f"📋 CSV columns: {list(df.columns)}")
        
        # Check for required columns
        if 'GLEAMER_FINDING' in df.columns:
            print(f"✅ GLEAMER_FINDING column found")
            unique_labels = df['GLEAMER_FINDING'].value_counts()
            print(f"📊 Unique labels in CSV: {dict(unique_labels)}")
        else:
            print(f"⚠️ GLEAMER_FINDING column not found!")
            
        if 'SOP_INSTANCE_UID_ARRAY' in df.columns:
            print(f"✅ SOP_INSTANCE_UID_ARRAY column found")
        else:
            print(f"⚠️ SOP_INSTANCE_UID_ARRAY column not found!")
            
    except Exception as e:
        print(f"❌ Error reading CSV: {e}")
        return
    
    # Limit sample size if specified
    if sample_size and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=42)
        print(f"📊 Analyzing sample of {sample_size} records")
    
    # Initialize counters
    label_counts = defaultdict(int)
    blank_counts = defaultdict(int)
    total_images = defaultdict(int)
    
    # Process each row
    print("\n🔍 Analyzing dataset distribution...")
    print(f"📊 Processing {len(df)} records...")
    
    for idx, row in df.iterrows():
        if idx % 100 == 0:
            print(f"   📈 Progress: {idx}/{len(df)} records ({idx/len(df)*100:.1f}%)")
        
        # Get label from GLEAMER_FINDING column
        if 'GLEAMER_FINDING' in row and pd.notna(row['GLEAMER_FINDING']):
            label = str(row['GLEAMER_FINDING']).strip().upper()
        else:
            label = 'UNKNOWN'
        
        # Count label
        label_counts[label] += 1
        
        # Check images if requested
        if check_images:
            # Get UIDs from SOP_INSTANCE_UID_ARRAY
            if 'SOP_INSTANCE_UID_ARRAY' in row and pd.notna(row['SOP_INSTANCE_UID_ARRAY']):
                try:
                    uid_string = str(row['SOP_INSTANCE_UID_ARRAY'])
                    # Parse UIDs (assuming comma-separated or JSON-like format)
                    if ',' in uid_string:
                        uids = [uid.strip().strip('"\'[]') for uid in uid_string.split(',')]
                    else:
                        uids = [uid_string.strip().strip('"\'[]')]
                    
                    # Log UID processing for first few records
                    if idx < 5:
                        print(f"   🔍 Record {idx}: Found {len(uids)} UIDs for label '{label}'")
                    
                    # Check each image
                    for uid_idx, uid in enumerate(uids):
                        if uid and uid != 'nan':
                            uid = uid.strip()
                            total_images[label] += 1
                            
                            # Log first few UIDs
                            if idx < 3 and uid_idx < 2:
                                print(f"      🖼️ Checking UID: {uid[:50]}...")
                            
                            # Find image file
                            verbose_logging = idx < 3 and uid_idx < 2  # Verbose for first few
                            image_paths = find_image_files(uid, base_dir, verbose=verbose_logging)
                            if image_paths:
                                if is_blank_image(image_paths[0]):
                                    blank_counts[label] += 1
                                    if idx < 5:
                                        print(f"      ⚫ Blank image found: {os.path.basename(image_paths[0])}")
                                else:
                                    if idx < 5:
                                        print(f"      ✅ Valid image found: {os.path.basename(image_paths[0])}")
                            else:
                                # Image not found - count as missing
                                blank_counts[label] += 1
                                if idx < 5:
                                    print(f"      ❌ Image not found: {uid[:50]}...")
                
                except Exception as e:
                    print(f"   ⚠️ Error processing UIDs for row {idx}: {e}")
                    if idx < 10:  # Show more details for first few errors
                        print(f"      📄 Row data: {dict(row)}")
    
    # Print results
    print("\n" + "="*60)
    print("📊 DATASET DISTRIBUTION ANALYSIS")
    print("="*60)
    
    total_records = len(df)
    print(f"📈 Total Records: {total_records:,}")
    
    if check_images:
        total_img_count = sum(total_images.values())
        total_blank_count = sum(blank_counts.values())
        print(f"🖼️ Total Images: {total_img_count:,}")
        print(f"⚫ Total Blank Images: {total_blank_count:,}")
        print(f"✅ Total Valid Images: {total_img_count - total_blank_count:,}")
        print(f"📊 Blank Rate: {(total_blank_count/total_img_count*100):.1f}%")
    
    print("\n📋 BY LABEL:")
    print("-" * 60)
    
    for label in sorted(label_counts.keys()):
        count = label_counts[label]
        percentage = (count / total_records) * 100
        
        if check_images:
            img_count = total_images.get(label, 0)
            blank_count = blank_counts.get(label, 0)
            valid_count = img_count - blank_count
            blank_rate = (blank_count / img_count * 100) if img_count > 0 else 0
            
            print(f"🏷️ {label:15} | Records: {count:6,} ({percentage:5.1f}%) | "
                  f"Images: {img_count:6,} | Valid: {valid_count:6,} | "
                  f"Blank: {blank_count:6,} ({blank_rate:5.1f}%)")
        else:
            print(f"🏷️ {label:15} | Records: {count:6,} ({percentage:5.1f}%)")
    
    # Summary statistics
    print("\n📊 SUMMARY:")
    print("-" * 60)
    
    if 'POSITIVE' in label_counts and 'NEGATIVE' in label_counts:
        pos_count = label_counts['POSITIVE']
        neg_count = label_counts['NEGATIVE']
        total_classified = pos_count + neg_count
        
        if total_classified > 0:
            pos_ratio = (pos_count / total_classified) * 100
            neg_ratio = (neg_count / total_classified) * 100
            print(f"✅ Positive: {pos_count:,} ({pos_ratio:.1f}%)")
            print(f"❌ Negative: {neg_count:,} ({neg_ratio:.1f}%)")
            
            if check_images:
                pos_blanks = blank_counts.get('POSITIVE', 0)
                neg_blanks = blank_counts.get('NEGATIVE', 0)
                pos_total = total_images.get('POSITIVE', 0)
                neg_total = total_images.get('NEGATIVE', 0)
                
                pos_blank_rate = (pos_blanks / pos_total * 100) if pos_total > 0 else 0
                neg_blank_rate = (neg_blanks / neg_total * 100) if neg_total > 0 else 0
                
                print(f"🖼️ Positive Images - Valid: {pos_total - pos_blanks:,}, Blank: {pos_blanks:,} ({pos_blank_rate:.1f}%)")
                print(f"🖼️ Negative Images - Valid: {neg_total - neg_blanks:,}, Blank: {neg_blanks:,} ({neg_blank_rate:.1f}%)")
    
    # Check for doubt/uncertain cases
    doubt_labels = [label for label in label_counts.keys() if 'DOUBT' in label or 'UNCERTAIN' in label]
    if doubt_labels:
        print(f"❓ Doubt/Uncertain: {sum(label_counts[label] for label in doubt_labels):,} cases")
    
    # Final detailed summary
    print("\n" + "="*60)
    print("📋 DETAILED FILE STRUCTURE ANALYSIS")
    print("="*60)
    
    if check_images:
        print(f"📁 Base directory: {base_dir}")
        print(f"📊 Total images processed: {sum(total_images.values()):,}")
        print(f"⚫ Total blank images: {sum(blank_counts.values()):,}")
        print(f"✅ Total valid images: {sum(total_images.values()) - sum(blank_counts.values()):,}")
        
        # Show sample of file paths found
        print(f"\n🔍 Sample file paths found:")
        sample_count = 0
        for label in label_counts.keys():
            if sample_count >= 3:
                break
            if total_images.get(label, 0) > 0:
                print(f"   📂 {label} label: {total_images[label]} images")
                sample_count += 1
    
    print(f"\n📊 Processing completed:")
    print(f"   ⏱️ Total records analyzed: {total_records:,}")
    print(f"   🏷️ Unique labels found: {len(label_counts)}")
    print(f"   📈 Label distribution: {dict(label_counts)}")
    
    print("\n🎉 Analysis complete!")
    return {
        'label_counts': dict(label_counts),
        'blank_counts': dict(blank_counts),
        'total_images': dict(total_images),
        'total_records': total_records
    }

def main():
    parser = argparse.ArgumentParser(description='Analyze dataset distribution with detailed logging')
    parser.add_argument('--csv', type=str, help='Path to CSV file')
    parser.add_argument('--sample-size', type=int, help='Number of samples to analyze (for testing)')
    parser.add_argument('--no-images', action='store_true', help='Skip image analysis (faster)')
    parser.add_argument('--csv-only', action='store_true', help='Only analyze CSV labels (fastest)')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging for all records')
    
    args = parser.parse_args()
    
    print("🚀 Starting dataset distribution analysis...")
    print(f"📋 Arguments: CSV={args.csv}, Sample={args.sample_size}, Images={not args.no_images and not args.csv_only}")
    
    # Run analysis
    analyze_dataset_distribution(
        csv_file=args.csv,
        sample_size=args.sample_size,
        check_images=not args.no_images and not args.csv_only
    )

if __name__ == "__main__":
    main()
