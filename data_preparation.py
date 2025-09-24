#!/usr/bin/env python3
"""
Data Preparation Script
Processes filtered CSV data from team and creates 90/10 train/validation split
Uses GLEAMER image data format
"""

import pandas as pd
import numpy as np
import os
import logging
import yaml
import random

# Try to import sklearn, fallback to manual split if not available
try:
    from sklearn.model_selection import train_test_split
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("WARNING: sklearn not available. Using manual stratified split.")

# Load YAML configuration
with open('config_training.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataPreparator:
    """Data preparation for train/validation split with GLEAMER format"""
    
    def __init__(self, csv_file=None):
        self.csv_file = csv_file or "filtered_training_data.csv"
        self.output_dir = config['data']['output_dir']
        self.data = None
        self.train_data = None
        self.val_data = None
        
        logger.info(f"🔧 Initializing Data Preparator with {self.csv_file}")
    
    def load_data(self):
        """Load CSV data from team"""
        logger.info("📊 Loading filtered CSV data from team...")
        
        self.data = pd.read_csv(self.csv_file)
        logger.info(f"✅ Loaded {len(self.data)} records from CSV")
        
        # Validate required columns
        required_columns = ['file_path', 'label']
        missing_columns = [col for col in required_columns if col not in self.data.columns]
        if missing_columns:
            logger.error(f"❌ Missing required columns: {missing_columns}")
            logger.info(f"Available columns: {self.data.columns.tolist()}")
            return None
        
        # Show distribution
        if 'label' in self.data.columns:
            label_counts = self.data['label'].value_counts()
            logger.info(f"📊 Label distribution:")
            for label, count in label_counts.items():
                percentage = (count / len(self.data)) * 100
                logger.info(f"   {label}: {count:,} ({percentage:.1f}%)")
        
        return self.data
    
    def add_binary_labels(self):
        """Add binary labels for classification"""
        logger.info("🏷️ Adding binary labels...")
        
        # Filter for binary classification (only POSITIVE/NEGATIVE)
        binary_data = self.data[
            self.data['label'].isin(['POSITIVE', 'NEGATIVE'])
        ].copy()
        
        # Add binary labels
        binary_data['binary_label'] = binary_data['label'].map({
            'NEGATIVE': 0,
            'POSITIVE': 1
        })
        
        logger.info(f"✅ Binary labels added:")
        logger.info(f"   Binary records: {len(binary_data):,}")
        
        # Class distribution
        class_counts = binary_data['binary_label'].value_counts().sort_index()
        for label, count in class_counts.items():
            label_name = 'NEGATIVE' if label == 0 else 'POSITIVE'
            percentage = (count / len(binary_data)) * 100
            logger.info(f"   {label_name}: {count:,} ({percentage:.1f}%)")
        
        self.data = binary_data
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
        train_file = os.path.join(self.output_dir, "train_dataset.csv")
        self.train_data.to_csv(train_file, index=False)
        logger.info(f"💾 Training dataset saved to: {train_file}")
        
        # Save validation dataset
        val_file = os.path.join(self.output_dir, "val_dataset.csv")
        self.val_data.to_csv(val_file, index=False)
        logger.info(f"💾 Validation dataset saved to: {val_file}")
        
        # Save summary
        summary_file = os.path.join(self.output_dir, "dataset_summary.txt")
        with open(summary_file, 'w') as f:
            f.write("DATASET SUMMARY\n")
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
    """Main data preparation pipeline"""
    logger.info("🚀 Starting Simple Data Preparation")
    
    # Initialize preparator
    preparator = DataPreparator()
    
    # Load data
    preparator.load_data()
    
    # Add binary labels
    preparator.add_binary_labels()
    
    # Split into train/validation (90:10)
    preparator.split_train_validation(train_ratio=0.9)
    
    # Save datasets
    preparator.save_datasets()
    
    logger.info("🎉 Data preparation complete!")
    logger.info("✅ Training and validation datasets ready!")
    
    return preparator

if __name__ == "__main__":
    preparator = main()