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
try:
    from monai.transforms import (
        Compose, LoadImaged, EnsureChannelFirstd, 
        ScaleIntensityd, Resized, ToTensord,
        RandRotated, RandZoomd, RandFlipd, 
        RandGaussianNoised, RandAdjustContrastd,
        RandAffined, RandGaussianSmoothd
    )
except ImportError:
    # If MONAI is not installed, create minimal fallback transforms
    print("Warning: MONAI not installed. Using minimal transforms.")
    # Create dummy classes for basic functionality
    class Compose:
        def __init__(self, transforms):
            self.transforms = transforms
        def __call__(self, data):
            for transform in self.transforms:
                data = transform(data)
            return data
    
    class LoadImaged:
        def __call__(self, data):
            return data
    
    class EnsureChannelFirstd:
        def __call__(self, data):
            return data
    
    class ScaleIntensityd:
        def __call__(self, data):
            return data
    
    class Resized:
        def __call__(self, data):
            return data
    
    class ToTensord:
        def __call__(self, data):
            return data
    
    class RandRotated:
        def __call__(self, data):
            return data
    
    class RandZoomd:
        def __call__(self, data):
            return data
    
    class RandFlipd:
        def __call__(self, data):
            return data
    
    class RandGaussianNoised:
        def __call__(self, data):
            return data
    
    class RandAdjustContrastd:
        def __call__(self, data):
            return data
    
    class RandAffined:
        def __call__(self, data):
            return data
    
    class RandGaussianSmoothd:
        def __call__(self, data):
            return data
from config import (
    CSV_FILE, DATA_ROOT, PREPROCESSED_DIR, 
    TRAINING_CONFIG, PREPROCESSING_CONFIG, 
    LOGGING_CONFIG, BINARY_CONFIG
)

class BinaryMedicalDataset(Dataset):
    """Binary Classification Dataset for Medical Image Fracture Detection"""
    
    def __init__(self, csv_file=None, transform=None, mode='train', use_valid_images_only=True):
        # Use config defaults if not provided
        csv_file = csv_file or CSV_FILE
        
        # Setup logging first
        logging.basicConfig(level=getattr(logging, LOGGING_CONFIG['log_level']))
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
        if TRAINING_CONFIG.get('exclude_doubt_cases', True):
            self.data = self.data[self.data['GLEAMER_FINDING'] != 'DOUBT']
            self.logger.info(f"🚫 Excluded DOUBT cases: {original_count} → {len(self.data)}")
        
        # Filter valid images only if configured
        if self.use_valid_images_only:
            # This will be handled in __getitem__ method
            pass
        
        # Map labels to binary classification
        self.data = self.data[self.data['GLEAMER_FINDING'].isin(['POSITIVE', 'NEGATIVE'])]
        self.data['label'] = self.data['GLEAMER_FINDING'].map(BINARY_CONFIG['label_mapping'])
        
        self.logger.info(f"🏷️ Binary labels: POSITIVE={sum(self.data['label']==1)}, NEGATIVE={sum(self.data['label']==0)}")
    
    def _split_data(self):
        """Split data into train/val/test sets"""
        total_len = len(self.data)
        train_end = int((1 - TRAINING_CONFIG['val_split'] - TRAINING_CONFIG['test_split']) * total_len)
        val_end = int((1 - TRAINING_CONFIG['test_split']) * total_len)
        
        if self.mode == 'train':
            self.data = self.data.iloc[:train_end]
        elif self.mode == 'val':
            self.data = self.data.iloc[train_end:val_end]
        elif self.mode == 'test':
            self.data = self.data.iloc[val_end:]
        
        self.logger.info(f"📊 {self.mode} split: {len(self.data)} samples")
    
    def _find_image_file(self, uid):
        """Find image file for given UID"""
        base_dir = "/sharedata01/CNN_data/gleamer/gleamer"
        possible_extensions = ['.jpg', '.jpeg', '.png', '.dcm']
        
        for ext in possible_extensions:
            # Check main directory
            image_path = os.path.join(base_dir, f"{uid}{ext}")
            if os.path.exists(image_path):
                return image_path
            
            # Check subdirectories
            for subdir in ['images', 'data', 'positive', 'negative', 'Negetive', 'Negative 2']:
                image_path = os.path.join(base_dir, subdir, f"{uid}{ext}")
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
        
        # Parse UIDs from SOP_INSTANCE_UID_ARRAY
        uid_string = str(row['SOP_INSTANCE_UID_ARRAY'])
        try:
            # Parse UIDs (assuming comma-separated or JSON-like format)
            if ',' in uid_string:
                uids = [uid.strip().strip('"\'[]') for uid in uid_string.split(',')]
            else:
                uids = [uid_string.strip().strip('"\'[]')]
            
            # Find first valid image
            image_path = None
            for uid in uids:
                if uid and uid != 'nan':
                    uid = uid.strip()
                    found_path = self._find_image_file(uid)
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
            # Prepare data dict for MONAI transforms (MONAI expects numpy arrays)
            data_dict = {'image': image}  # image is already numpy array
            data_dict = self.transform(data_dict)
            image = data_dict['image']
        
        # Ensure image is a proper tensor with correct dimensions
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image).float()
        
        # Debug: Print tensor shape to understand the issue
        print(f"Image tensor shape after transforms: {image.shape}")
        
        # Ensure image has correct dimensions [C, H, W]
        if image.dim() == 2:  # [H, W]
            image = image.unsqueeze(0)  # Add channel dimension -> [1, H, W]
        elif image.dim() == 3:
            # Check if it's [H, W, C] format
            if image.shape[-1] == 3 or image.shape[-1] == 1:  # Last dimension is channel
                image = image.permute(2, 0, 1)  # Convert [H, W, C] to [C, H, W]
            elif image.shape[0] in [1, 3]:  # First dimension is channel
                pass  # Already correct format [C, H, W]
            else:
                # If we can't determine, assume it's grayscale and add channel dimension
                image = image.unsqueeze(0)  # Add channel dimension
        
        return {
            'image': image,
            'label': torch.tensor(label, dtype=torch.long),
            'uid': uids[0] if uids else '',
            'gleamer_finding': row['GLEAMER_FINDING']
        }

def get_transforms(mode='train'):
    """Get data transforms for training/validation using simple transforms"""
    
    if mode == 'train':
        # Simple training transforms
        transform = Compose([
            Resized(keys=["image"], spatial_size=(224, 224)),
            ScaleIntensityd(keys=["image"], minv=0.0, maxv=1.0),
            ToTensord(keys=["image"])
        ])
    else:
        # Validation/test transforms (no augmentation)
        transform = Compose([
            Resized(keys=["image"], spatial_size=(224, 224)),
            ScaleIntensityd(keys=["image"], minv=0.0, maxv=1.0),
            ToTensord(keys=["image"])
        ])
    
    return transform

def custom_collate_fn(batch):
    """Custom collate function to handle tensor dimension issues"""
    images = []
    labels = []
    uids = []
    gleamer_findings = []
    
    for item in batch:
        # Ensure consistent image dimensions
        image = item['image']
        if isinstance(image, torch.Tensor):
            # Ensure [C, H, W] format
            if image.dim() == 2:  # [H, W]
                image = image.unsqueeze(0)  # [1, H, W]
            elif image.dim() == 3:
                if image.shape[0] not in [1, 3]:  # [H, W, C]
                    image = image.permute(2, 0, 1)  # [C, H, W]
            
            # Ensure all images have the same spatial dimensions (224, 224)
            if image.shape[-2:] != (224, 224):
                # Resize to (224, 224) if needed
                image = torch.nn.functional.interpolate(
                    image.unsqueeze(0), 
                    size=(224, 224), 
                    mode='bilinear', 
                    align_corners=False
                ).squeeze(0)
        
        images.append(image)
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
    
    batch_size = batch_size or TRAINING_CONFIG['batch_size']
    num_workers = num_workers or TRAINING_CONFIG['num_workers']
    
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
    if balance_classes and TRAINING_CONFIG.get('balance_classes', True):
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