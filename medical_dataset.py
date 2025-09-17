import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import requests
from PIL import Image
import io
import numpy as np
from transformers import AutoTokenizer, AutoModel
from lib.monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, 
    ScaleIntensityd, Resized, ToTensord,
    RandRotated, RandZoomed, RandFlipd, 
    RandGaussianNoised, RandAdjustContrastd
)
import json
import ast
import logging
from pathlib import Path
import os
from config import (
    CSV_FILE, DATA_ROOT, PREPROCESSED_DIR, 
    TRAINING_CONFIG, PREPROCESSING_CONFIG, 
    LOGGING_CONFIG
)

class MedicalDataset(Dataset):
    """Dataset for medical image classification with text"""
    
    def __init__(self, csv_file=None, transform=None, text_transform=None, mode='train', max_length=None):
        # Use config defaults if not provided
        csv_file = csv_file or CSV_FILE
        max_length = max_length or TRAINING_CONFIG['text_max_length']
        
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        self.text_transform = text_transform
        self.mode = mode
        self.max_length = max_length
        
        # Initialize text tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        
        # Split data for train/val/test based on config
        total_len = len(self.data)
        train_end = int((1 - TRAINING_CONFIG['val_split'] - TRAINING_CONFIG['test_split']) * total_len)
        val_end = int((1 - TRAINING_CONFIG['test_split']) * total_len)
        
        if mode == 'train':
            self.data = self.data.iloc[:train_end]
        elif mode == 'val':
            self.data = self.data.iloc[train_end:val_end]
        else:  # test
            self.data = self.data.iloc[val_end:]
    
    def __len__(self):
        return len(self.data)
    
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
                        print(f"Warning: Image size {width}x{height} outside valid range, skipping")
                        return None
                
                return np.array(image)
            else:
                # Return black image as fallback if path doesn't exist
                return np.zeros(PREPROCESSING_CONFIG['image_size'] + (3,), dtype=np.uint8)
                
        except Exception as e:
            print(f"Error loading image from {image_path}: {e}")
            # Return black image as fallback
            return np.zeros(PREPROCESSING_CONFIG['image_size'] + (3,), dtype=np.uint8)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # Load and process images
        image_urls = ast.literal_eval(row['download_urls'])
        images = []
        
        # Use configured number of images or pad with zeros
        max_images = TRAINING_CONFIG['max_images_per_study']
        for i in range(max_images):
            if i < len(image_urls):
                image = self.load_image_from_url(image_urls[i])
                if image is None:  # Skip corrupted images
                    image = np.zeros(PREPROCESSING_CONFIG['image_size'] + (3,), dtype=np.uint8)
            else:
                image = np.zeros(PREPROCESSING_CONFIG['image_size'] + (3,), dtype=np.uint8)
            
            if self.transform:
                image = self.transform(image)
            images.append(image)
        
        # Stack images: (3, H, W) for RGB
        images = torch.stack(images)
        
        # Process text data
        clinical_text = f"{row['clinical_indication']} {row['exam_technique']} {row['findings']}"
        clinical_text = clinical_text.replace('\n', ' ').strip()
        
        text_encoding = self.tokenizer(
            clinical_text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Labels
        label = 1 if row['GLEAMER_FINDING'] == 'POSITIVE' else 0
        
        return {
            'images': images,
            'text_input_ids': text_encoding['input_ids'].squeeze(0),
            'text_attention_mask': text_encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.long),
            'clinical_text': clinical_text,
            'body_part': row['BODY_PART_ARRAY'],
            'study_description': row['STUDY_DESCRIPTION']
        }

# Data transforms
def get_transforms(mode='train'):
    """Get data transforms for training or validation"""
    if mode == 'train' and PREPROCESSING_CONFIG['use_augmentation']:
        prob = PREPROCESSING_CONFIG['augmentation_prob']
        return Compose([
            RandRotated(keys=["image"], prob=prob*0.3, range_x=0.1),
            RandZoomed(keys=["image"], prob=prob*0.3, min_zoom=0.9, max_zoom=1.1),
            RandFlipd(keys=["image"], prob=prob*0.3),
            RandGaussianNoised(keys=["image"], prob=prob*0.2, std=0.01),
            RandAdjustContrastd(keys=["image"], prob=prob*0.2, gamma=(0.8, 1.2))
        ])
    else:
        return None

def create_data_loaders(csv_file=None, batch_size=None, num_workers=None):
    """Create data loaders for training and validation"""
    
    # Use config defaults if not provided
    csv_file = csv_file or CSV_FILE
    batch_size = batch_size or TRAINING_CONFIG['batch_size']
    num_workers = num_workers or TRAINING_CONFIG['num_workers']
    
    # Training transforms
    train_transforms = get_transforms('train')
    
    # Create datasets
    train_dataset = MedicalDataset(
        csv_file, 
        transform=train_transforms, 
        mode='train'
    )
    
    val_dataset = MedicalDataset(
        csv_file, 
        transform=None, 
        mode='val'
    )
    
    test_dataset = MedicalDataset(
        csv_file, 
        transform=None, 
        mode='test'
    )
    
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
    
    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    # Test dataset
    train_loader, val_loader = create_data_loaders("dicom_image_url_file.csv")
    
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    
    # Test batch
    for batch in train_loader:
        print(f"Image shape: {batch['images'].shape}")
        print(f"Text input shape: {batch['text_input_ids'].shape}")
        print(f"Labels: {batch['label']}")
        break
