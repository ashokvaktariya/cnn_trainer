import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import numpy as np
from PIL import Image
import io
import os
import logging
from pathlib import Path
# Use PyTorch/torchvision transforms instead of MONAI
import torchvision.transforms as transforms
import yaml

# Load YAML configuration
with open('config_training.yaml', 'r') as f:
    config = yaml.safe_load(f)

class BinaryMedicalDataset(Dataset):
    """Binary Classification Dataset for Medical Image Fracture Detection"""
    
    def __init__(self, csv_file=None, transform=None, mode='train', use_valid_images_only=True):
        # Use config defaults if not provided
        csv_file = csv_file or config['data']['csv_path']
        
        # Setup logging first
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        self.mode = mode
        self.use_valid_images_only = use_valid_images_only
        
        # Filter data for binary classification
        self._filter_data()
        
        # Split data for train/val/test
        self._split_data()
        
        self.logger.info(f"📊 {mode} dataset initialized with {len(self.data)} samples")
    
    def _filter_data(self):
        """Filter data for binary classification"""
        original_count = len(self.data)
        
        # Exclude DOUBT cases if configured
        self.data = self.data[self.data['GLEAMER_FINDING'] != 'DOUBT']
        self.logger.info(f"🚫 Excluded DOUBT cases: {original_count} → {len(self.data)}")
        
        # Filter valid images only if configured
        if self.use_valid_images_only:
            # This will be handled in __getitem__ method
            pass
        
        # Map labels to binary classification
        self.data = self.data[self.data['GLEAMER_FINDING'].isin(['POSITIVE', 'NEGATIVE'])]
        label_mapping = {"NEGATIVE": 0, "POSITIVE": 1}
        self.data['label'] = self.data['GLEAMER_FINDING'].map(label_mapping)
        
        self.logger.info(f"🏷️ Binary labels: POSITIVE={sum(self.data['label']==1)}, NEGATIVE={sum(self.data['label']==0)}")
    
    def _split_data(self):
        """Split data into train/val/test sets"""
        total_len = len(self.data)
        train_end = int((1 - config['data']['val_split'] - config['data']['test_split']) * total_len)
        val_end = int((1 - config['data']['test_split']) * total_len)
        
        if self.mode == 'train':
            self.data = self.data.iloc[:train_end]
        elif self.mode == 'val':
            self.data = self.data.iloc[train_end:val_end]
        elif self.mode == 'test':
            self.data = self.data.iloc[val_end:]
        
        self.logger.info(f"📊 {self.mode} split: {len(self.data)} samples")
    
    def _find_image_file(self, uid):
        """Find image file for given UID"""
        base_dir = config['data']['image_root']  # Use config path
        possible_extensions = ['.jpg', '.jpeg', '.png', '.dcm', '.dicom']
        
        for ext in possible_extensions:
            # Check main directory
            image_path = os.path.join(base_dir, f"{uid}{ext}")
            if os.path.exists(image_path):
                return image_path
            
            # Check subdirectories
            for subdir in ['images', 'data', 'positive', 'negative', 'Negetive', 'Negative 2', 'fracture', 'normal']:
                image_path = os.path.join(base_dir, subdir, f"{uid}{ext}")
                if os.path.exists(image_path):
                    return image_path
        
        return None
    
    def _find_image_file_by_filename(self, filename):
        """Find image file by exact filename"""
        base_dir = config['data']['image_root']  # Use config path
        
        # Check main directory
        image_path = os.path.join(base_dir, filename)
        if os.path.exists(image_path):
            return image_path
        
        # Check subdirectories
        for subdir in ['images', 'data', 'positive', 'negative', 'Negetive', 'Negative 2', 'fracture', 'normal']:
            image_path = os.path.join(base_dir, subdir, filename)
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
                
                # Check if all pixels are the same color (blank)
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
            self.logger.warning(f"⚠️ Error checking image {image_path}: {e}")
            return False
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """Get a single sample from the dataset"""
        row = self.data.iloc[idx]
        
        # Get label
        label = int(row['label'])  # 0 for NEGATIVE, 1 for POSITIVE
        
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
            
            # Find first valid image
            image_path = None
            for filename in image_filenames:
                if filename:
                    found_path = self._find_image_file_by_filename(filename)
                    if found_path and self._is_valid_image(found_path):
                        image_path = found_path
                        break
            
            if image_path is None:
                # No valid image found, skip this sample
                return self.__getitem__((idx + 1) % len(self.data))
            
            # Load image
            try:
                with Image.open(image_path) as img:
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    
                    # Convert to numpy array
                    image = np.array(img)
                    
            except Exception as e:
                self.logger.warning(f"⚠️ Error loading image {image_path}: {e}")
                return self.__getitem__((idx + 1) % len(self.data))
            
        except Exception as e:
            self.logger.warning(f"⚠️ Error processing UIDs for row {idx}: {e}")
            return self.__getitem__((idx + 1) % len(self.data))
        
        # Apply transforms
        if self.transform:
            # Convert numpy array to PIL Image for torchvision transforms
            if len(image.shape) == 3 and image.shape[2] == 3:  # RGB image
                image = Image.fromarray(image.astype(np.uint8), 'RGB')
            else:  # Grayscale image
                image = Image.fromarray(image.astype(np.uint8), 'L')
            
            # Apply torchvision transforms (returns tensor with correct dimensions)
            image = self.transform(image)
        
        return {
            'image': image,
            'label': torch.tensor(label, dtype=torch.long),
            'uid': uids[0] if uids else '',
            'gleamer_finding': row['GLEAMER_FINDING']
        }

def get_transforms(mode='train'):
    """Get data transforms for training/validation using PyTorch transforms"""
    
    if mode == 'train':
        # Training transforms with augmentation
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
        ])
    else:
        # Validation/test transforms (no augmentation)
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
        ])
    
    return transform

def custom_collate_fn(batch):
    """Custom collate function - torchvision transforms ensure consistent dimensions"""
    images = []
    labels = []
    uids = []
    gleamer_findings = []
    
    for item in batch:
        images.append(item['image'])
        labels.append(item['label'])
        uids.append(item['uid'])
        gleamer_findings.append(item['gleamer_finding'])
    
    return {
        'image': torch.stack(images),
        'label': torch.stack(labels),
        'uid': uids,
        'gleamer_finding': gleamer_findings
    }

def create_data_loaders(batch_size=None, num_workers=None, balance_classes=True):
    """Create data loaders for training and validation"""
    
    batch_size = batch_size or config['training']['batch_size']
    num_workers = num_workers or config['hardware']['num_workers']
    
    # Create datasets
    train_dataset = BinaryMedicalDataset(
        transform=get_transforms('train'),
        mode='train'
    )
    
    val_dataset = BinaryMedicalDataset(
        transform=get_transforms('val'),
        mode='val'
    )
    
    test_dataset = BinaryMedicalDataset(
        transform=get_transforms('val'),
        mode='test'
    )
    
    # Create samplers for class balancing
    train_sampler = None
    if balance_classes:
        # Calculate class weights
        class_counts = train_dataset.data['label'].value_counts().sort_index()
        class_weights = 1.0 / class_counts.values
        sample_weights = [class_weights[label] for label in train_dataset.data['label']]
        
        train_sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
    
    # Create data loaders with custom collate function
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=custom_collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=custom_collate_fn
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=custom_collate_fn
    )
    
    return train_loader, val_loader, test_loader

def get_dataset_stats():
    """Get dataset statistics"""
    dataset = BinaryMedicalDataset(mode='train')
    
    stats = {
        'total_samples': len(dataset),
        'positive_samples': sum(dataset.data['label'] == 1),
        'negative_samples': sum(dataset.data['label'] == 0),
        'positive_ratio': sum(dataset.data['label'] == 1) / len(dataset),
        'negative_ratio': sum(dataset.data['label'] == 0) / len(dataset)
    }
    
    return stats

if __name__ == "__main__":
    # Test dataset creation
    print("🧪 Testing Binary Medical Dataset...")
    
    # Create datasets
    train_loader, val_loader, test_loader = create_data_loaders(batch_size=4)
    
    # Test loading
    for batch in train_loader:
        print(f"✅ Batch loaded successfully:")
        print(f"   Images: {batch['image'].shape}")
        print(f"   Labels: {batch['label'].shape}")
        print(f"   Label distribution: {batch['label'].unique(return_counts=True)}")
        break
    
    # Print dataset stats
    stats = get_dataset_stats()
    print(f"📊 Dataset Statistics:")
    print(f"   Total samples: {stats['total_samples']}")
    print(f"   Positive: {stats['positive_samples']} ({stats['positive_ratio']:.1%})")
    print(f"   Negative: {stats['negative_samples']} ({stats['negative_ratio']:.1%})")