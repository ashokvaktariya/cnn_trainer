#!/usr/bin/env python3
"""
Sample Dataset Creator for Medical Fracture Detection
Creates a sample dataset with 10 samples that have valid images.
"""

import pandas as pd
import numpy as np
import os
import json
import yaml
import shutil
from pathlib import Path
import logging
from PIL import Image

# Load configuration
with open('config_training.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SampleDatasetCreator:
    """Create sample dataset with valid images"""
    
    def __init__(self):
        self.csv_path = config['data']['csv_path']
        self.image_root = config['data']['image_root']
        self.sample_dir = "./sampledata"
        self.num_samples = 10
        
        # Create sample directory
        os.makedirs(self.sample_dir, exist_ok=True)
        
        logger.info("🔍 Sample Dataset Creator initialized")
        logger.info(f"📁 CSV Path: {self.csv_path}")
        logger.info(f"🖼️ Image Root: {self.image_root}")
        logger.info(f"📊 Sample Directory: {self.sample_dir}")
        logger.info(f"🎯 Target Samples: {self.num_samples}")
    
    def load_csv(self):
        """Load the CSV file"""
        logger.info("📋 Loading CSV file...")
        
        try:
            df = pd.read_csv(self.csv_path)
            logger.info(f"✅ CSV loaded successfully! Shape: {df.shape}")
            return df
        except Exception as e:
            logger.error(f"❌ Error loading CSV: {e}")
            raise
    
    def find_valid_samples(self, df):
        """Find samples with valid images (stratified sampling for balanced dataset)"""
        logger.info("🔍 Finding samples with valid images (stratified sampling)...")
        
        # Filter out DOUBT cases
        df_filtered = df[df['GLEAMER_FINDING'] != 'DOUBT']
        logger.info(f"🚫 Filtered DOUBT cases: {len(df)} → {len(df_filtered)}")
        
        # Get POSITIVE and NEGATIVE samples
        pos_samples = df_filtered[df_filtered['GLEAMER_FINDING'] == 'POSITIVE']
        neg_samples = df_filtered[df_filtered['GLEAMER_FINDING'] == 'NEGATIVE']
        
        logger.info(f"📊 Available samples:")
        logger.info(f"   POSITIVE: {len(pos_samples)}")
        logger.info(f"   NEGATIVE: {len(neg_samples)}")
        
        # Calculate how many samples to take from each class
        samples_per_class = self.num_samples // 2
        logger.info(f"🎯 Target: {samples_per_class} samples per class")
        
        # Sample from each class
        pos_sample = pos_samples.sample(n=min(samples_per_class, len(pos_samples)), random_state=42)
        neg_sample = neg_samples.sample(n=min(samples_per_class, len(neg_samples)), random_state=42)
        
        # Combine samples
        balanced_df = pd.concat([pos_sample, neg_sample]).reset_index(drop=True)
        logger.info(f"🎲 Created balanced sample: {len(balanced_df)} samples")
        
        valid_samples = []
        processed = 0
        
        for idx, row in balanced_df.iterrows():
            processed += 1
            
            logger.info(f"   Processing sample {processed}/{len(balanced_df)}: {row['GLEAMER_FINDING']}")
            
            # Parse download URLs to get image filenames
            download_urls_string = str(row['download_urls'])
            try:
                import json
                # Try to parse as JSON array first
                try:
                    download_urls = json.loads(download_urls_string)
                    if not isinstance(download_urls, list):
                        download_urls = [download_urls]
                except json.JSONDecodeError:
                    # Fallback to comma-separated parsing
                    if ',' in download_urls_string:
                        download_urls = [url.strip().strip('"\'[]') for url in download_urls_string.split(',')]
                    else:
                        download_urls = [download_urls_string.strip().strip('"\'[]')]
                
                # Extract filenames from URLs
                image_filenames = []
                for url in download_urls:
                    if url and url != 'nan':
                        # Extract filename from URL
                        filename = os.path.basename(url.strip())
                        if filename:
                            image_filenames.append(filename)
                
                # Find valid images for this sample
                sample_images = []
                for filename in image_filenames:
                    if filename:
                        image_path = self._find_image_file_by_filename(filename)
                        if image_path and self._is_valid_image(image_path):
                            sample_images.append({
                                'filename': filename,
                                'image_path': image_path,
                                'uid': filename.split('.')[0]  # Extract UID from filename
                            })
                
                # If we found valid images, add this sample
                if sample_images:
                    sample_data = {
                        'row_index': row.name,  # Original row index from original dataframe
                        'gleamer_finding': row['GLEAMER_FINDING'],
                        'accession_number': row.get('ACCESSION_NUMBER', ''),
                        'study_description': row.get('STUDY_DESCRIPTION', ''),
                        'body_part': row.get('BODY_PART_ARRAY', ''),
                        'clinical_indication': row.get('clinical_indication', ''),
                        'exam_technique': row.get('exam_technique', ''),
                        'findings': row.get('findings', ''),
                        'images': sample_images
                    }
                    valid_samples.append(sample_data)
                    logger.info(f"   ✅ Added {row['GLEAMER_FINDING']} sample with {len(sample_images)} images")
                else:
                    logger.warning(f"   ⚠️ No valid images found for {row['GLEAMER_FINDING']} sample")
                        
            except Exception as e:
                logger.warning(f"⚠️ Error processing row {idx}: {e}")
                continue
        
        logger.info(f"✅ Found {len(valid_samples)} valid samples (stratified sampling)")
        return valid_samples
    
    def _find_image_file(self, uid):
        """Find image file for given UID"""
        possible_extensions = ['.jpg', '.jpeg', '.png', '.dcm', '.dicom']
        
        for ext in possible_extensions:
            # Check main directory
            image_path = os.path.join(self.image_root, f"{uid}{ext}")
            if os.path.exists(image_path):
                return image_path
            
            # Check subdirectories
            for subdir in ['images', 'data', 'positive', 'negative', 'Negetive', 'Negative 2', 'fracture', 'normal']:
                image_path = os.path.join(self.image_root, subdir, f"{uid}{ext}")
                if os.path.exists(image_path):
                    return image_path
        
        return None
    
    def _find_image_file_by_filename(self, filename):
        """Find image file by exact filename"""
        # Check main directory
        image_path = os.path.join(self.image_root, filename)
        if os.path.exists(image_path):
            return image_path
        
        # Check subdirectories
        for subdir in ['images', 'data', 'positive', 'negative', 'Negetive', 'Negative 2', 'fracture', 'normal']:
            image_path = os.path.join(self.image_root, subdir, filename)
            if os.path.exists(image_path):
                return image_path
        
        return None
    
    def _is_valid_image(self, image_path):
        """Check if image is valid (not blank)"""
        try:
            with Image.open(image_path) as img:
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                img_array = np.array(img)
                
                # Check if image is blank
                if len(img_array.shape) == 3:
                    if np.all(img_array == img_array[0, 0]):
                        return False
                    
                    # Check variance - very low variance indicates blank image
                    variance = np.var(img_array)
                    if variance < 10:
                        return False
                else:
                    if np.all(img_array == img_array[0, 0]):
                        return False
                    variance = np.var(img_array)
                    if variance < 10:
                        return False
                
                return True
        except Exception as e:
            return False
    
    def create_sample_folders(self, valid_samples):
        """Create folders for each sample"""
        logger.info("📁 Creating sample folders...")
        
        for i, sample in enumerate(valid_samples):
            # Create sample folder (using row index for unique naming)
            sample_folder = os.path.join(self.sample_dir, str(sample['row_index']))
            os.makedirs(sample_folder, exist_ok=True)
            
            # Create data.txt file with sample information
            data_file = os.path.join(sample_folder, "data.txt")
            with open(data_file, 'w') as f:
                f.write("=== SAMPLE DATA ===\n")
                f.write(f"Row Index: {sample['row_index']}\n")
                f.write(f"GLEAMER Finding: {sample['gleamer_finding']}\n")
                f.write(f"Accession Number: {sample['accession_number']}\n")
                f.write(f"Study Description: {sample['study_description']}\n")
                f.write(f"Body Part: {sample['body_part']}\n")
                f.write(f"Clinical Indication: {sample['clinical_indication']}\n")
                f.write(f"Exam Technique: {sample['exam_technique']}\n")
                f.write(f"Findings: {sample['findings']}\n")
                f.write(f"Number of Images: {len(sample['images'])}\n")
                f.write("\n=== IMAGES ===\n")
                
                for j, img_info in enumerate(sample['images']):
                    f.write(f"Image {j+1}: {img_info['filename']} (UID: {img_info['uid']})\n")
            
            # Copy images to sample folder
            for j, img_info in enumerate(sample['images']):
                src_path = img_info['image_path']
                dst_filename = f"image{j+1}_{img_info['filename']}"
                dst_path = os.path.join(sample_folder, dst_filename)
                
                try:
                    shutil.copy2(src_path, dst_path)
                    logger.info(f"   Copied: {dst_filename}")
                except Exception as e:
                    logger.warning(f"   Failed to copy {src_path}: {e}")
            
            logger.info(f"✅ Created sample folder: {sample_folder}")
    
    def create_summary_report(self, valid_samples):
        """Create summary report"""
        logger.info("📊 Creating summary report...")
        
        summary_file = os.path.join(self.sample_dir, "summary.txt")
        with open(summary_file, 'w') as f:
            f.write("=== SAMPLE DATASET SUMMARY ===\n")
            f.write(f"Total Samples: {len(valid_samples)}\n")
            f.write(f"Source CSV: {self.csv_path}\n")
            f.write(f"Image Root: {self.image_root}\n")
            f.write(f"Created: {pd.Timestamp.now()}\n\n")
            
            # Count by label
            label_counts = {}
            for sample in valid_samples:
                label = sample['gleamer_finding']
                label_counts[label] = label_counts.get(label, 0) + 1
            
            f.write("=== LABEL DISTRIBUTION ===\n")
            for label, count in label_counts.items():
                f.write(f"{label}: {count} samples\n")
            
            f.write("\n=== SAMPLE DETAILS ===\n")
            for i, sample in enumerate(valid_samples):
                f.write(f"\nSample {i+1} (Row {sample['row_index']}):\n")
                f.write(f"  Label: {sample['gleamer_finding']}\n")
                f.write(f"  Images: {len(sample['images'])}\n")
                f.write(f"  Folder: {sample['row_index']}/\n")
        
        logger.info(f"✅ Summary report created: {summary_file}")
    
    def run(self):
        """Run the complete sample dataset creation"""
        logger.info("🚀 Starting sample dataset creation...")
        
        try:
            # Load CSV
            df = self.load_csv()
            
            # Find valid samples
            valid_samples = self.find_valid_samples(df)
            
            if len(valid_samples) == 0:
                logger.error("❌ No valid samples found!")
                return False
            
            # Create sample folders
            self.create_sample_folders(valid_samples)
            
            # Create summary report
            self.create_summary_report(valid_samples)
            
            logger.info("✅ Sample dataset creation completed!")
            logger.info(f"📁 Sample data saved to: {self.sample_dir}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Sample dataset creation failed: {e}")
            raise

def main():
    """Main function"""
    creator = SampleDatasetCreator()
    creator.run()

if __name__ == "__main__":
    main()
