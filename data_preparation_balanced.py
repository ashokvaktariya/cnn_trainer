#!/usr/bin/env python3
"""
Balanced Data Preparation Script
Creates a balanced dataset with 6k negative images and all positive images
Processes corrected CSV data and creates 90/10 train/validation split for CNN training
"""

import pandas as pd
import numpy as np
import os
import logging
import random

# Try to import sklearn, fallback to manual split if not available
try:
    from sklearn.model_selection import train_test_split
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("WARNING: sklearn not available. Using manual stratified split.")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BalancedDataPreparator:
    """Balanced data preparation for train/validation split for CNN training"""
    
    def __init__(self, csv_file=None):
        self.csv_file = csv_file or "final_dataset_cnn.csv"
        self.output_dir = "preprocessed_balanced"
        self.data = None
        self.train_data = None
        self.val_data = None
        
        logger.info(f"🔧 Initializing Balanced Data Preparator with {self.csv_file}")
    
    def load_data(self):
        """Load corrected CSV data"""
        logger.info("📊 Loading corrected CSV data...")
        
        self.data = pd.read_csv(self.csv_file)
        logger.info(f"✅ Loaded {len(self.data)} records from CSV")
        
        # Validate required columns
        required_columns = ['image_path', 'label']
        missing_columns = [col for col in required_columns if col not in self.data.columns]
        if missing_columns:
            logger.error(f"❌ Missing required columns: {missing_columns}")
            logger.info(f"Available columns: {self.data.columns.tolist()}")
            return None
        
        # Show distribution
        if 'label' in self.data.columns:
            label_counts = self.data['label'].value_counts()
            logger.info(f"📊 Original label distribution:")
            for label, count in label_counts.items():
                percentage = (count / len(self.data)) * 100
                logger.info(f"   {label}: {count:,} ({percentage:.1f}%)")
        
        return self.data
    
    def create_balanced_dataset(self, max_negative=6000):
        """Create balanced dataset with 6k negative and all positive images"""
        logger.info(f"⚖️ Creating balanced dataset (max {max_negative:,} negative, all positive)...")
        
        # Filter for binary classification (only POSITIVE/NEGATIVE)
        binary_data = self.data[
            self.data['label'].isin(['POSITIVE', 'NEGATIVE'])
        ].copy()
        
        # Separate positive and negative data
        positive_data = binary_data[binary_data['label'] == 'POSITIVE'].copy()
        negative_data = binary_data[binary_data['label'] == 'NEGATIVE'].copy()
        
        logger.info(f"📊 Original distribution:")
        logger.info(f"   POSITIVE: {len(positive_data):,}")
        logger.info(f"   NEGATIVE: {len(negative_data):,}")
        
        # Sample negative data if we have more than max_negative
        if len(negative_data) > max_negative:
            logger.info(f"📉 Sampling {max_negative:,} negative images from {len(negative_data):,}")
            # Set random seed for reproducibility
            random.seed(42)
            negative_indices = negative_data.index.tolist()
            random.shuffle(negative_indices)
            sampled_negative_indices = negative_indices[:max_negative]
            negative_data = negative_data.loc[sampled_negative_indices].copy()
        else:
            logger.info(f"📊 Using all {len(negative_data):,} negative images")
        
        # Combine positive and sampled negative data
        balanced_data = pd.concat([positive_data, negative_data], ignore_index=True)
        
        # Shuffle the combined dataset
        balanced_data = balanced_data.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Add binary labels
        balanced_data['binary_label'] = balanced_data['label'].map({
            'NEGATIVE': 0,
            'POSITIVE': 1
        })
        
        # Add file path column for CNN training
        balanced_data['file_path'] = balanced_data['image_path']
        
        logger.info(f"✅ Balanced dataset created:")
        logger.info(f"   Total records: {len(balanced_data):,}")
        
        # Show final distribution
        class_counts = balanced_data['binary_label'].value_counts().sort_index()
        for label, count in class_counts.items():
            label_name = 'NEGATIVE' if label == 0 else 'POSITIVE'
            percentage = (count / len(balanced_data)) * 100
            logger.info(f"   {label_name}: {count:,} ({percentage:.1f}%)")
        
        # Calculate positive rate
        positive_rate = (class_counts[1] / len(balanced_data)) * 100
        logger.info(f"📊 Positive rate: {positive_rate:.1f}%")
        
        self.data = balanced_data
        return self.data
    
    def _manual_stratified_split(self, train_ratio):
        """Manual stratified split without sklearn"""
        logger.info("📊 Using manual stratified split...")
        
        # Set random seed for reproducibility
        random.seed(42)
        
        # Group by binary_label
        positive_data = self.data[self.data['binary_label'] == 1]
        negative_data = self.data[self.data['binary_label'] == 0]
        
        # Calculate split sizes
        pos_train_size = int(len(positive_data) * train_ratio)
        neg_train_size = int(len(negative_data) * train_ratio)
        
        # Shuffle and split positive data
        pos_indices = positive_data.index.tolist()
        random.shuffle(pos_indices)
        pos_train_indices = pos_indices[:pos_train_size]
        pos_val_indices = pos_indices[pos_train_size:]
        
        # Shuffle and split negative data
        neg_indices = negative_data.index.tolist()
        random.shuffle(neg_indices)
        neg_train_indices = neg_indices[:neg_train_size]
        neg_val_indices = neg_indices[neg_train_size:]
        
        # Combine train and validation indices
        train_indices = pos_train_indices + neg_train_indices
        val_indices = pos_val_indices + neg_val_indices
        
        # Create train and validation datasets
        train_data = self.data.loc[train_indices].copy()
        val_data = self.data.loc[val_indices].copy()
        
        return train_data, val_data
    
    def split_train_validation(self, train_ratio=0.9):
        """Split data into training and validation sets (90:10)"""
        logger.info("📊 Splitting data into train/validation sets (90:10)...")
        
        if self.data is None:
            logger.error("❌ No data available.")
            return None, None
        
        # Stratified split to maintain class distribution
        if SKLEARN_AVAILABLE:
            train_data, val_data = train_test_split(
                self.data,
                test_size=1-train_ratio,
                random_state=42,
                stratify=self.data['binary_label']
            )
        else:
            # Manual stratified split
            train_data, val_data = self._manual_stratified_split(train_ratio)
        
        # Reset indices
        train_data = train_data.reset_index(drop=True)
        val_data = val_data.reset_index(drop=True)
        
        logger.info(f"✅ Train/Validation split complete:")
        logger.info(f"   Training records: {len(train_data):,} ({len(train_data)/len(self.data)*100:.1f}%)")
        logger.info(f"   Validation records: {len(val_data):,} ({len(val_data)/len(self.data)*100:.1f}%)")
        
        # Show class distribution in each set
        train_counts = train_data['binary_label'].value_counts().sort_index()
        val_counts = val_data['binary_label'].value_counts().sort_index()
        
        logger.info("📊 Training set distribution:")
        for label, count in train_counts.items():
            label_name = 'NEGATIVE' if label == 0 else 'POSITIVE'
            percentage = (count / len(train_data)) * 100
            logger.info(f"   {label_name}: {count:,} ({percentage:.1f}%)")
        
        logger.info("📊 Validation set distribution:")
        for label, count in val_counts.items():
            label_name = 'NEGATIVE' if label == 0 else 'POSITIVE'
            percentage = (count / len(val_data)) * 100
            logger.info(f"   {label_name}: {count:,} ({percentage:.1f}%)")
        
        self.train_data = train_data
        self.val_data = val_data
        
        return train_data, val_data
    
    def save_datasets(self):
        """Save training and validation datasets"""
        if self.train_data is None or self.val_data is None:
            logger.error("❌ No train/validation data to save. Run split_train_validation() first.")
            return None, None
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Save training dataset
        train_file = os.path.join(self.output_dir, "balanced_dataset_cnn_train.csv")
        self.train_data.to_csv(train_file, index=False)
        logger.info(f"💾 Training dataset saved to: {train_file}")
        
        # Save validation dataset
        val_file = os.path.join(self.output_dir, "balanced_dataset_cnn_val.csv")
        self.val_data.to_csv(val_file, index=False)
        logger.info(f"💾 Validation dataset saved to: {val_file}")
        
        # Save combined dataset for reference
        combined_file = os.path.join(self.output_dir, "balanced_dataset_cnn.csv")
        self.data.to_csv(combined_file, index=False)
        logger.info(f"💾 Combined dataset saved to: {combined_file}")
        
        # Also save files in root directory for easier access
        root_train_file = "balanced_dataset_cnn_train.csv"
        self.train_data.to_csv(root_train_file, index=False)
        logger.info(f"💾 Training dataset also saved to root: {root_train_file}")
        
        root_val_file = "balanced_dataset_cnn_val.csv"
        self.val_data.to_csv(root_val_file, index=False)
        logger.info(f"💾 Validation dataset also saved to root: {root_val_file}")
        
        root_combined_file = "balanced_dataset_cnn.csv"
        self.data.to_csv(root_combined_file, index=False)
        logger.info(f"💾 Combined dataset also saved to root: {root_combined_file}")
        
        # Save summary
        summary_file = os.path.join(self.output_dir, "balanced_dataset_summary.txt")
        with open(summary_file, 'w') as f:
            f.write("BALANCED CNN TRAINING DATASET SUMMARY\n")
            f.write("=" * 50 + "\n")
            f.write(f"Total records: {len(self.data):,}\n")
            f.write(f"Training records: {len(self.train_data):,}\n")
            f.write(f"Validation records: {len(self.val_data):,}\n")
            f.write(f"Train ratio: {len(self.train_data)/len(self.data)*100:.1f}%\n")
            f.write(f"Val ratio: {len(self.val_data)/len(self.data)*100:.1f}%\n\n")
            
            f.write("TRAINING SET DISTRIBUTION:\n")
            train_counts = self.train_data['binary_label'].value_counts().sort_index()
            for label, count in train_counts.items():
                label_name = 'NEGATIVE' if label == 0 else 'POSITIVE'
                percentage = (count / len(self.train_data)) * 100
                f.write(f"  {label_name}: {count:,} ({percentage:.1f}%)\n")
            
            f.write("\nVALIDATION SET DISTRIBUTION:\n")
            val_counts = self.val_data['binary_label'].value_counts().sort_index()
            for label, count in val_counts.items():
                label_name = 'NEGATIVE' if label == 0 else 'POSITIVE'
                percentage = (count / len(self.val_data)) * 100
                f.write(f"  {label_name}: {count:,} ({percentage:.1f}%)\n")
        
        logger.info(f"📊 Summary saved to: {summary_file}")
        
        return train_file, val_file

def main():
    """Main balanced data preparation pipeline"""
    logger.info("🚀 Starting Balanced Data Preparation")
    
    # Initialize preparator
    preparator = BalancedDataPreparator()
    
    # Load data
    preparator.load_data()
    
    # Create balanced dataset (6k negative, all positive)
    preparator.create_balanced_dataset(max_negative=6000)
    
    # Split into train/validation (90:10)
    preparator.split_train_validation(train_ratio=0.9)
    
    # Save datasets
    preparator.save_datasets()
    
    logger.info("🎉 Balanced data preparation complete!")
    logger.info("✅ Training and validation datasets ready!")
    
    return preparator

if __name__ == "__main__":
    preparator = main()
