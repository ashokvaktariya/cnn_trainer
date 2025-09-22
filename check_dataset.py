#!/usr/bin/env python3
"""
Dataset Checker Script for Medical Fracture Detection
Checks the preprocessed dataset and shows training statistics.
"""

import pandas as pd
import numpy as np
import os
import json
import yaml
from pathlib import Path
import logging

# Load configuration
with open('config_training.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DatasetChecker:
    """Check dataset statistics and training readiness"""
    
    def __init__(self):
        self.csv_path = config['data']['csv_path']
        self.image_root = config['data']['image_root']
        self.output_dir = config['data']['output_dir']
        
        logger.info("🔍 Dataset Checker initialized")
        logger.info(f"📁 CSV Path: {self.csv_path}")
        logger.info(f"🖼️ Image Root: {self.image_root}")
    
    def check_csv_exists(self):
        """Check if CSV file exists"""
        logger.info("📋 Checking CSV file...")
        
        if os.path.exists(self.csv_path):
            logger.info(f"✅ CSV file exists: {self.csv_path}")
            return True
        else:
            logger.error(f"❌ CSV file not found: {self.csv_path}")
            return False
    
    def load_and_analyze_csv(self):
        """Load CSV and analyze basic statistics"""
        logger.info("📊 Loading and analyzing CSV...")
        
        try:
            df = pd.read_csv(self.csv_path)
            logger.info(f"✅ CSV loaded successfully!")
            logger.info(f"📊 Dataset shape: {df.shape}")
            
            # Check required columns
            required_columns = ['SOP_INSTANCE_UID_ARRAY', 'GLEAMER_FINDING']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                logger.error(f"❌ Missing required columns: {missing_columns}")
                return None
            
            # Label distribution
            label_counts = df['GLEAMER_FINDING'].value_counts()
            logger.info(f"🏷️ Label distribution:")
            for label, count in label_counts.items():
                percentage = (count / len(df)) * 100
                logger.info(f"   {label}: {count:,} ({percentage:.1f}%)")
            
            return df
            
        except Exception as e:
            logger.error(f"❌ Error loading CSV: {e}")
            return None
    
    def check_image_directory(self):
        """Check image directory structure"""
        logger.info("🖼️ Checking image directory...")
        
        if not os.path.exists(self.image_root):
            logger.error(f"❌ Image directory not found: {self.image_root}")
            return False
        
        # Count image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.dcm', '.dicom']
        total_images = 0
        image_counts = {}
        
        for ext in image_extensions:
            count = len(list(Path(self.image_root).rglob(f"*{ext}")))
            image_counts[ext] = count
            total_images += count
        
        logger.info(f"📊 Image files found:")
        for ext, count in image_counts.items():
            if count > 0:
                logger.info(f"   {ext}: {count:,} files")
        
        logger.info(f"📊 Total images: {total_images:,}")
        
        # Check subdirectories
        subdirs = [d for d in os.listdir(self.image_root) if os.path.isdir(os.path.join(self.image_root, d))]
        if subdirs:
            logger.info(f"📁 Subdirectories found: {len(subdirs)}")
            for subdir in subdirs[:5]:  # Show first 5
                subdir_path = os.path.join(self.image_root, subdir)
                subdir_images = sum(len(list(Path(subdir_path).rglob(f"*{ext}"))) for ext in image_extensions)
                logger.info(f"   {subdir}: {subdir_images:,} images")
        
        return total_images > 0
    
    def sample_image_check(self, df, sample_size=100):
        """Check a sample of images to estimate success rate"""
        logger.info(f"🔍 Checking sample of {sample_size} images...")
        
        sample_df = df.sample(n=min(sample_size, len(df)), random_state=42)
        found_images = 0
        valid_images = 0
        
        for idx, row in sample_df.iterrows():
            # Parse UIDs
            uid_string = str(row['SOP_INSTANCE_UID_ARRAY'])
            try:
                import json
                # Try to parse as JSON array first
                try:
                    uids = json.loads(uid_string)
                    if not isinstance(uids, list):
                        uids = [uids]
                except json.JSONDecodeError:
                    # Fallback to comma-separated parsing
                    if ',' in uid_string:
                        uids = [uid.strip().strip('"\'[]') for uid in uid_string.split(',')]
                    else:
                        uids = [uid_string.strip().strip('"\'[]')]
                
                # Check for image files
                for uid in uids:
                    if uid and uid != 'nan':
                        uid = uid.strip()
                        image_path = self._find_image_file(uid)
                        if image_path:
                            found_images += 1
                            if self._is_valid_image(image_path):
                                valid_images += 1
                            break
                            
            except Exception as e:
                logger.warning(f"⚠️ Error processing UID: {e}")
                continue
        
        success_rate = (valid_images / sample_size) * 100 if sample_size > 0 else 0
        
        logger.info(f"📊 Sample check results:")
        logger.info(f"   Images found: {found_images}/{sample_size}")
        logger.info(f"   Valid images: {valid_images}/{sample_size}")
        logger.info(f"   Success rate: {success_rate:.1f}%")
        
        return success_rate
    
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
    
    def _is_valid_image(self, image_path):
        """Check if image is valid (not blank)"""
        try:
            from PIL import Image
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
    
    def estimate_training_samples(self, df, success_rate):
        """Estimate training samples based on success rate"""
        logger.info("📊 Estimating training samples...")
        
        # Filter out DOUBT cases
        binary_df = df[df['GLEAMER_FINDING'].isin(['POSITIVE', 'NEGATIVE'])]
        
        # Estimate based on success rate
        estimated_valid = int(len(binary_df) * (success_rate / 100))
        
        # Apply train/val/test split
        train_split = config['data']['train_split']
        val_split = config['data']['val_split']
        test_split = config['data']['test_split']
        
        train_samples = int(estimated_valid * train_split)
        val_samples = int(estimated_valid * val_split)
        test_samples = int(estimated_valid * test_split)
        
        logger.info(f"📊 Estimated training samples:")
        logger.info(f"   Total valid samples: {estimated_valid:,}")
        logger.info(f"   Training samples: {train_samples:,} ({train_split*100:.0f}%)")
        logger.info(f"   Validation samples: {val_samples:,} ({val_split*100:.0f}%)")
        logger.info(f"   Test samples: {test_samples:,} ({test_split*100:.0f}%)")
        
        # Class distribution
        pos_count = len(binary_df[binary_df['GLEAMER_FINDING'] == 'POSITIVE'])
        neg_count = len(binary_df[binary_df['GLEAMER_FINDING'] == 'NEGATIVE'])
        
        estimated_pos = int(pos_count * (success_rate / 100))
        estimated_neg = int(neg_count * (success_rate / 100))
        
        logger.info(f"📊 Estimated class distribution:")
        logger.info(f"   POSITIVE: {estimated_pos:,} ({estimated_pos/(estimated_pos+estimated_neg)*100:.1f}%)")
        logger.info(f"   NEGATIVE: {estimated_neg:,} ({estimated_neg/(estimated_pos+estimated_neg)*100:.1f}%)")
        
        return {
            'total_valid': estimated_valid,
            'train_samples': train_samples,
            'val_samples': val_samples,
            'test_samples': test_samples,
            'positive_samples': estimated_pos,
            'negative_samples': estimated_neg
        }
    
    def check_training_readiness(self):
        """Check if dataset is ready for training"""
        logger.info("🚀 Checking training readiness...")
        
        # Check CSV
        if not self.check_csv_exists():
            return False
        
        # Load and analyze CSV
        df = self.load_and_analyze_csv()
        if df is None:
            return False
        
        # Check image directory
        if not self.check_image_directory():
            return False
        
        # Sample check
        success_rate = self.sample_image_check(df)
        
        # Estimate training samples
        estimates = self.estimate_training_samples(df, success_rate)
        
        # Training readiness assessment
        logger.info("🎯 Training Readiness Assessment:")
        
        if success_rate > 50:
            logger.info("✅ Dataset is ready for training!")
        elif success_rate > 20:
            logger.warning("⚠️ Dataset has low success rate but may be usable")
        else:
            logger.error("❌ Dataset has very low success rate - check image paths")
        
        if estimates['train_samples'] > 1000:
            logger.info("✅ Sufficient training samples available")
        else:
            logger.warning("⚠️ Low number of training samples")
        
        return True
    
    def run_check(self):
        """Run complete dataset check"""
        logger.info("🔍 Starting dataset check...")
        
        try:
            self.check_training_readiness()
            logger.info("✅ Dataset check completed!")
        except Exception as e:
            logger.error(f"❌ Dataset check failed: {e}")
            raise

def main():
    """Main function"""
    checker = DatasetChecker()
    checker.run_check()

if __name__ == "__main__":
    main()
