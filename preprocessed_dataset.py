#!/usr/bin/env python3
"""
Preprocessed Dataset Class
Loads data from preprocessed pickle files with local image paths
H200 GPU Server Optimized
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from PIL import Image
import os
import pickle
import json
import logging

from config import (
    DATA_ROOT, PREPROCESSED_DIR, TRAINING_CONFIG, PREPROCESSING_CONFIG,
    LOGGING_CONFIG
)

logger = logging.getLogger(__name__)

class PreprocessedMedicalDataset(Dataset):
    """Dataset for preprocessed medical data with local images"""
    
    def __init__(self, mode='train', transform=None, max_length=None):
        self.mode = mode
        self.transform = transform
        self.max_length = max_length or TRAINING_CONFIG['text_max_length']
        
        # Load preprocessed data
        self.processed_data = self._load_preprocessed_data()
        
        if not self.processed_data:
            raise RuntimeError("No preprocessed data found. Please run step1_preprocessing.py first.")
        
        # Filter data based on mode (if split info is available)
        self._filter_by_mode()
        
        logger.info(f"✅ Loaded {len(self.processed_data)} {mode} samples")
    
    def _load_preprocessed_data(self):
        """Load preprocessed data from pickle file"""
        cache_file = os.path.join(PREPROCESSED_DIR, "preprocessed_data.pkl")
        
        if not os.path.exists(cache_file):
            logger.warning("No preprocessed data found. Please run preprocessing first.")
            return []
        
        try:
            with open(cache_file, 'rb') as f:
                processed_data = pickle.load(f)
            
            logger.info(f"📂 Loaded {len(processed_data)} preprocessed records")
            return processed_data
            
        except Exception as e:
            logger.error(f"Error loading preprocessed data: {e}")
            return []
    
    def _filter_by_mode(self):
        """Filter data by train/val/test mode"""
        # If we have a simple list, split it based on indices
        total_samples = len(self.processed_data)
        train_end = int((1 - TRAINING_CONFIG['val_split'] - TRAINING_CONFIG['test_split']) * total_samples)
        val_end = int((1 - TRAINING_CONFIG['test_split']) * total_samples)
        
        if self.mode == 'train':
            self.processed_data = self.processed_data[:train_end]
        elif self.mode == 'val':
            self.processed_data = self.processed_data[train_end:val_end]
        else:  # test
            self.processed_data = self.processed_data[val_end:]
    
    def __len__(self):
        return len(self.processed_data)
    
    def load_image_from_path(self, image_path):
        """Load image from local path"""
        try:
            if image_path and os.path.exists(image_path):
                image = Image.open(image_path).convert('RGB')
                
                # Validate image size
                width, height = image.size
                min_size = PREPROCESSING_CONFIG['min_image_size']
                max_size = PREPROCESSING_CONFIG['max_image_size']
                
                if width < min_size[0] or height < min_size[1] or width > max_size[0] or height > max_size[1]:
                    if PREPROCESSING_CONFIG['skip_corrupt_images']:
                        logger.warning(f"Image size {width}x{height} outside valid range")
                        return None
                
                return np.array(image)
            else:
                # Return black image as fallback if path doesn't exist
                return np.zeros(PREPROCESSING_CONFIG['image_size'] + (3,), dtype=np.uint8)
                
        except Exception as e:
            logger.error(f"Error loading image from {image_path}: {e}")
            # Return black image as fallback
            return np.zeros(PREPROCESSING_CONFIG['image_size'] + (3,), dtype=np.uint8)
    
    def __getitem__(self, idx):
        record = self.processed_data[idx]
        
        # Load images from local paths
        images = []
        for i in range(TRAINING_CONFIG['max_images_per_study']):
            if i < len(record['image_paths']) and record['image_paths'][i]:
                image = self.load_image_from_path(record['image_paths'][i])
            else:
                # Use cached image data if available
                if i < len(record['images']):
                    image = record['images'][i]
                else:
                    # Create black image
                    image = np.zeros(PREPROCESSING_CONFIG['image_size'] + (3,), dtype=np.uint8)
            
            if self.transform:
                # Apply transforms if available
                image = self.transform(image)
            else:
                # Convert to tensor
                image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            
            images.append(image)
        
        # Stack images
        images = torch.stack(images)
        
        # Enhanced text data processing
        # Combine all available text fields for richer context
        text_parts = []
        
        if record['clinical_indication']:
            text_parts.append(f"Clinical Indication: {record['clinical_indication']}")
        
        if record['exam_technique']:
            text_parts.append(f"Exam Technique: {record['exam_technique']}")
        
        if record['study_description']:
            text_parts.append(f"Study Description: {record['study_description']}")
        
        if record['findings']:
            text_parts.append(f"Findings: {record['findings']}")
        
        # Join all text parts
        enhanced_clinical_text = " ".join(text_parts)
        
        # Fallback to original if enhanced is empty
        if not enhanced_clinical_text.strip():
            enhanced_clinical_text = record['clinical_text']
        
        # Tokenize enhanced text
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        
        text_encoding = tokenizer(
            enhanced_clinical_text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'images': images,
            'text_input_ids': text_encoding['input_ids'].squeeze(0),
            'text_attention_mask': text_encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(record['label'], dtype=torch.long),
            
            # All available text data
            'clinical_text': enhanced_clinical_text,  # Enhanced combined text
            'original_clinical_text': record['clinical_text'],  # Original combined text
            'clinical_indication': record['clinical_indication'],
            'exam_technique': record['exam_technique'],
            'findings': record['findings'],
            
            # Metadata
            'accession_number': record['accession_number'],
            'study_instance_uid': record['study_instance_uid'],
            'study_description': record['study_description'],
            'body_part_array': record['body_part_array'],
            'sop_instance_uid_array': record['sop_instance_uid_array'],
            'gleamer_finding': record['gleamer_finding'],
            
            # Image metadata
            'image_count': record['image_count'],
            'num_images_available': record['num_images_available'],
            'num_images_used': record['num_images_used'],
            'image_paths': record['image_paths']
        }

def create_preprocessed_data_loaders(batch_size=None, num_workers=None):
    """Create data loaders for preprocessed data"""
    # Use config defaults if not provided
    batch_size = batch_size or TRAINING_CONFIG['batch_size']
    num_workers = num_workers or TRAINING_CONFIG['num_workers']
    
    # Create datasets
    train_dataset = PreprocessedMedicalDataset(mode='train')
    val_dataset = PreprocessedMedicalDataset(mode='val')
    test_dataset = PreprocessedMedicalDataset(mode='test')
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    logger.info(f"📊 Created data loaders:")
    logger.info(f"   Train: {len(train_dataset)} samples")
    logger.info(f"   Val: {len(val_dataset)} samples")
    logger.info(f"   Test: {len(test_dataset)} samples")
    
    return train_loader, val_loader, test_loader

def get_dataset_stats():
    """Get statistics about the preprocessed dataset"""
    stats_file = os.path.join(PREPROCESSED_DIR, "dataset_stats.json")
    
    if not os.path.exists(stats_file):
        logger.warning("No dataset stats found")
        return None
    
    try:
        with open(stats_file, 'r') as f:
            stats = json.load(f)
        
        return stats
    except Exception as e:
        logger.error(f"Error loading dataset stats: {e}")
        return None

if __name__ == "__main__":
    # Test the preprocessed dataset
    try:
        train_loader, val_loader, test_loader = create_preprocessed_data_loaders()
        
        # Test loading a batch
        for batch in train_loader:
            print("✅ Batch loaded successfully!")
            print(f"   Images shape: {batch['images'].shape}")
            print(f"   Text input shape: {batch['text_input_ids'].shape}")
            print(f"   Labels shape: {batch['label'].shape}")
            break
        
        # Print stats
        stats = get_dataset_stats()
        if stats:
            print(f"\n📊 Dataset Statistics:")
            print(f"   Total records: {stats['total_records']}")
            print(f"   Total images: {stats['total_images']}")
            print(f"   Label distribution: {stats['label_distribution']}")
            
    except Exception as e:
        print(f"❌ Error testing dataset: {e}")
