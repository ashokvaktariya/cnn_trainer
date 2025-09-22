#!/usr/bin/env python3
"""
Medical Fracture Detection - Model Training Script
Optimized for H100 GPU with mixed precision training
"""

import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from pathlib import Path
import logging
from tqdm import tqdm
import wandb
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MedicalImageDataset(Dataset):
    """Custom dataset for medical images"""
    
    def __init__(self, df, image_root, transform=None, is_training=True):
        self.df = df.reset_index(drop=True)
        self.image_root = image_root
        self.transform = transform
        self.is_training = is_training
        
        # Find image path column
        self.image_col = None
        for col in df.columns:
            if 'image' in col.lower() or 'path' in col.lower() or 'url' in col.lower():
                self.image_col = col
                break
        
        if self.image_col is None:
            raise ValueError("No image path column found in dataset")
        
        # Find label column
        self.label_col = None
        for col in df.columns:
            if 'label' in col.lower() or 'class' in col.lower() or 'target' in col.lower():
                self.label_col = col
                break
        
        if self.label_col is None:
            raise ValueError("No label column found in dataset")
        
        logger.info(f"Using image column: {self.image_col}")
        logger.info(f"Using label column: {self.label_col}")
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        # Get image path
        image_path = self.df.iloc[idx][self.image_col]
        full_path = os.path.join(self.image_root, str(image_path))
        
        # Load image
        try:
            image = Image.open(full_path).convert('RGB')
        except Exception as e:
            logger.warning(f"Failed to load image {full_path}: {e}")
            # Return a blank image as fallback
            image = Image.new('RGB', (224, 224), color='black')
        
        # Get label
        label = self.df.iloc[idx][self.label_col]
        
        # Convert label to binary (assuming 0/1 or NEGATIVE/POSITIVE)
        if isinstance(label, str):
            if 'POSITIVE' in label.upper() or 'FRACTURE' in label.upper():
                label = 1
            else:
                label = 0
        else:
            label = int(label)
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, label

class BinaryClassifier(nn.Module):
    """Binary classifier using EfficientNet"""
    
    def __init__(self, num_classes=2, dropout=0.2):
        super(BinaryClassifier, self).__init__()
        
        # Use EfficientNet-B0 as backbone
        self.backbone = models.efficientnet_b0(pretrained=True)
        
        # Modify classifier for binary classification
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)

class ModelTrainer:
    """Model training class with H100 optimizations"""
    
    def __init__(self, config_path="config_training.yaml"):
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.device = torch.device(self.config['hardware']['device'])
        self.output_dir = self.config['data']['output_dir']
        
        # Create output directories
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'checkpoints'), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'logs'), exist_ok=True)
        
        # Initialize model
        self.model = BinaryClassifier(
            num_classes=self.config['model']['num_classes'],
            dropout=self.config['model']['dropout']
        )
        
        # Move to device
        self.model = self.model.to(self.device)
        
        # Initialize optimizer and scheduler
        self.optimizer = self._get_optimizer()
        self.scheduler = self._get_scheduler()
        
        # Initialize scaler for mixed precision
        self.scaler = torch.cuda.amp.GradScaler() if self.config['hardware']['mixed_precision'] else None
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        
        logger.info(f"Model initialized on {self.device}")
        logger.info(f"Mixed precision: {self.config['hardware']['mixed_precision']}")
    
    def _get_optimizer(self):
        """Initialize optimizer"""
        optimizer_name = self.config['training']['optimizer'].lower()
        lr = self.config['training']['learning_rate']
        weight_decay = self.config['training']['weight_decay']
        
        if optimizer_name == 'adamw':
            return optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name == 'adam':
            return optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name == 'sgd':
            return optim.SGD(self.model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")
    
    def _get_scheduler(self):
        """Initialize learning rate scheduler"""
        scheduler_name = self.config['training']['scheduler'].lower()
        
        if scheduler_name == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, 
                T_max=self.config['training']['num_epochs']
            )
        elif scheduler_name == 'step':
            return optim.lr_scheduler.StepLR(
                self.optimizer, 
                step_size=self.config['training']['num_epochs'] // 3,
                gamma=0.1
            )
        elif scheduler_name == 'plateau':
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, 
                mode='min', 
                patience=5, 
                factor=0.5
            )
        else:
            return None
    
    def _get_transforms(self, is_training=True):
        """Get data transforms"""
        image_size = self.config['data']['image_size']
        
        if is_training and self.config['data']['augmentation']:
            return transforms.Compose([
                transforms.Resize((image_size[0] + 32, image_size[1] + 32)),
                transforms.RandomCrop(image_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            return transforms.Compose([
                transforms.Resize(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    
    def load_data(self):
        """Load and prepare data"""
        logger.info("Loading dataset...")
        
        # Load CSV
        csv_path = self.config['data']['csv_path']
        df = pd.read_csv(csv_path)
        logger.info(f"Loaded {len(df)} samples")
        
        # Split data
        train_split = self.config['data']['train_split']
        val_split = self.config['data']['val_split']
        test_split = self.config['data']['test_split']
        
        # First split: train vs (val + test)
        train_df, temp_df = train_test_split(
            df, 
            test_size=(val_split + test_split), 
            random_state=42,
            stratify=df[self._get_label_column(df)] if self._get_label_column(df) else None
        )
        
        # Second split: val vs test
        val_size = val_split / (val_split + test_split)
        val_df, test_df = train_test_split(
            temp_df, 
            test_size=(1 - val_size), 
            random_state=42,
            stratify=temp_df[self._get_label_column(temp_df)] if self._get_label_column(temp_df) else None
        )
        
        logger.info(f"Data splits: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
        
        # Create datasets
        train_dataset = MedicalImageDataset(
            train_df, 
            self.config['data']['image_root'],
            transform=self._get_transforms(is_training=True),
            is_training=True
        )
        
        val_dataset = MedicalImageDataset(
            val_df, 
            self.config['data']['image_root'],
            transform=self._get_transforms(is_training=False),
            is_training=False
        )
        
        test_dataset = MedicalImageDataset(
            test_df, 
            self.config['data']['image_root'],
            transform=self._get_transforms(is_training=False),
            is_training=False
        )
        
        # Create data loaders
        batch_size = self.config['training']['batch_size']
        num_workers = self.config['hardware']['num_workers']
        
        self.train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=num_workers,
            pin_memory=self.config['hardware']['pin_memory']
        )
        
        self.val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=num_workers,
            pin_memory=self.config['hardware']['pin_memory']
        )
        
        self.test_loader = DataLoader(
            test_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=num_workers,
            pin_memory=self.config['hardware']['pin_memory']
        )
        
        logger.info("Data loaders created successfully")
    
    def _get_label_column(self, df):
        """Find label column in dataframe"""
        for col in df.columns:
            if 'label' in col.lower() or 'class' in col.lower() or 'target' in col.lower():
                return col
        return None
    
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}')
        
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            
            if self.scaler:
                # Mixed precision training
                with torch.cuda.amp.autocast():
                    output = self.model(data)
                    loss = nn.CrossEntropyLoss()(output, target)
                
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Standard training
                output = self.model(data)
                loss = nn.CrossEntropyLoss()(output, target)
                loss.backward()
                self.optimizer.step()
            
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.*correct/total:.2f}%'
            })
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100. * correct / total
        
        self.train_losses.append(avg_loss)
        self.train_accuracies.append(accuracy)
        
        return avg_loss, accuracy
    
    def validate_epoch(self, epoch):
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in tqdm(self.val_loader, desc='Validation'):
                data, target = data.to(self.device), target.to(self.device)
                
                if self.scaler:
                    with torch.cuda.amp.autocast():
                        output = self.model(data)
                        loss = nn.CrossEntropyLoss()(output, target)
                else:
                    output = self.model(data)
                    loss = nn.CrossEntropyLoss()(output, target)
                
                total_loss += loss.item()
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)
                
                all_preds.extend(pred.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = 100. * correct / total
        
        # Calculate additional metrics
        precision, recall, f1, _ = precision_recall_fscore_support(all_targets, all_preds, average='binary')
        
        self.val_losses.append(avg_loss)
        self.val_accuracies.append(accuracy)
        
        return avg_loss, accuracy, precision, recall, f1
    
    def save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies,
            'config': self.config
        }
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        # Save regular checkpoint
        checkpoint_path = os.path.join(self.output_dir, 'checkpoints', f'checkpoint_epoch_{epoch+1}.pth')
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = os.path.join(self.output_dir, 'best_model.pth')
            torch.save(checkpoint, best_path)
            logger.info(f"Best model saved to {best_path}")
    
    def train(self):
        """Main training loop"""
        logger.info("Starting training...")
        
        # Initialize wandb if enabled
        if self.config['logging']['use_wandb']:
            wandb.init(
                project=self.config['logging']['wandb_project'],
                entity=self.config['logging']['wandb_entity'],
                config=self.config,
                name=f"fracture_detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
        
        best_val_acc = 0
        patience_counter = 0
        patience = self.config['training']['patience']
        
        for epoch in range(self.config['training']['num_epochs']):
            # Train
            train_loss, train_acc = self.train_epoch(epoch)
            
            # Validate
            val_loss, val_acc, precision, recall, f1 = self.validate_epoch(epoch)
            
            # Update scheduler
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            # Log metrics
            logger.info(f'Epoch {epoch+1}/{self.config["training"]["num_epochs"]}:')
            logger.info(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            logger.info(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            logger.info(f'  Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}')
            
            # Log to wandb
            if self.config['logging']['use_wandb']:
                wandb.log({
                    'epoch': epoch + 1,
                    'train_loss': train_loss,
                    'train_accuracy': train_acc,
                    'val_loss': val_loss,
                    'val_accuracy': val_acc,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'learning_rate': self.optimizer.param_groups[0]['lr']
                })
            
            # Save checkpoint
            is_best = val_acc > best_val_acc
            if is_best:
                best_val_acc = val_acc
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Save checkpoint
            if (epoch + 1) % self.config['training']['save_frequency'] == 0 or is_best:
                self.save_checkpoint(epoch, is_best)
            
            # Early stopping
            if patience_counter >= patience:
                logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break
        
        logger.info(f"Training completed! Best validation accuracy: {best_val_acc:.2f}%")
        
        if self.config['logging']['use_wandb']:
            wandb.finish()
    
    def evaluate(self):
        """Evaluate on test set"""
        logger.info("Evaluating on test set...")
        
        self.model.eval()
        all_preds = []
        all_targets = []
        all_probs = []
        
        with torch.no_grad():
            for data, target in tqdm(self.test_loader, desc='Testing'):
                data, target = data.to(self.device), target.to(self.device)
                
                if self.scaler:
                    with torch.cuda.amp.autocast():
                        output = self.model(data)
                else:
                    output = self.model(data)
                
                probs = torch.softmax(output, dim=1)
                pred = output.argmax(dim=1)
                
                all_preds.extend(pred.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        # Calculate metrics
        accuracy = accuracy_score(all_targets, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(all_targets, all_preds, average='binary')
        
        # Calculate AUC
        try:
            auc = roc_auc_score(all_targets, [prob[1] for prob in all_probs])
        except:
            auc = 0.0
        
        logger.info("Test Results:")
        logger.info(f"  Accuracy: {accuracy:.4f}")
        logger.info(f"  Precision: {precision:.4f}")
        logger.info(f"  Recall: {recall:.4f}")
        logger.info(f"  F1 Score: {f1:.4f}")
        logger.info(f"  AUC: {auc:.4f}")
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc,
            'predictions': all_preds,
            'targets': all_targets,
            'probabilities': all_probs
        }

def main():
    """Main training function"""
    print("🏥 Medical Fracture Detection - Model Training")
    print("=" * 50)
    
    # Initialize trainer
    trainer = ModelTrainer()
    
    # Load data
    trainer.load_data()
    
    # Train model
    trainer.train()
    
    # Evaluate
    results = trainer.evaluate()
    
    print("\n🎉 Training completed successfully!")
    print(f"📊 Final Test Accuracy: {results['accuracy']:.4f}")
    print(f"📁 Model saved to: {trainer.output_dir}")

if __name__ == "__main__":
    main()
