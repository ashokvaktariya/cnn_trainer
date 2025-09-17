#!/usr/bin/env python3
"""
Step 2: Train Image-Only Models
Trains DenseNet121 and EfficientNet models using only image data
H200 GPU Server Optimized
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import json
import logging
import time
from pathlib import Path
from tqdm import tqdm
import argparse

from config import (
    CHECKPOINTS_DIR, RESULTS_DIR, LOGS_DIR, GPU_CONFIG, TRAINING_CONFIG,
    MODEL_CONFIGS, LOGGING_CONFIG, get_model_path, get_results_path
)
from preprocessed_dataset import create_preprocessed_data_loaders
from models import ImageOnlyDenseNet, ImageOnlyEfficientNet, create_model

# Setup logging
logging.basicConfig(
    level=getattr(logging, LOGGING_CONFIG['log_level']),
    format=LOGGING_CONFIG['log_format'],
    handlers=[
        logging.FileHandler(os.path.join(LOGS_DIR, "image_models_training.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ImageModelTrainer:
    """Trainer for image-only models"""
    
    def __init__(self, model_name, device=None):
        self.model_name = model_name
        self.device = device or torch.device(GPU_CONFIG['device'])
        self.config = MODEL_CONFIGS[model_name]
        
        # Setup paths
        self.checkpoint_dir = os.path.join(CHECKPOINTS_DIR, model_name)
        self.results_dir = os.path.join(RESULTS_DIR, model_name)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
        self.val_aucs = []
        self.best_auc = 0.0
        self.best_epoch = 0
        
        logger.info(f"🔧 Initialized {model_name} trainer")
        logger.info(f"🖥️  Device: {self.device}")
    
    def create_model(self):
        """Create the model"""
        if self.model_name == "image_densenet":
            model = ImageOnlyDenseNet(
                num_classes=2,
                pretrained=self.config['pretrained'],
                freeze_backbone=self.config['freeze_backbone'],
                dropout_rate=self.config['dropout_rate']
            )
        elif self.model_name == "image_efficientnet":
            model = ImageOnlyEfficientNet(
                num_classes=2,
                model_name=self.config['model_name'],
                pretrained=self.config['pretrained'],
                freeze_backbone=self.config['freeze_backbone'],
                dropout_rate=self.config['dropout_rate']
            )
        else:
            raise ValueError(f"Unknown model: {self.model_name}")
        
        # Compile model for H200 optimization
        if GPU_CONFIG['compile_model'] and hasattr(torch, 'compile'):
            model = torch.compile(model)
        
        model = model.to(self.device)
        logger.info(f"✅ Created {self.model_name} model")
        
        return model
    
    def create_optimizer(self, model):
        """Create optimizer and scheduler"""
        optimizer = optim.AdamW(
            model.parameters(),
            lr=TRAINING_CONFIG['learning_rate'],
            weight_decay=TRAINING_CONFIG['weight_decay']
        )
        
        # Learning rate scheduler
        if TRAINING_CONFIG['scheduler'] == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, 
                T_max=TRAINING_CONFIG['num_epochs']
            )
        elif TRAINING_CONFIG['scheduler'] == 'step':
            scheduler = optim.lr_scheduler.StepLR(
                optimizer, 
                step_size=10, 
                gamma=0.1
            )
        else:
            scheduler = None
        
        return optimizer, scheduler
    
    def train_epoch(self, model, train_loader, optimizer, criterion, scaler, epoch):
        """Train for one epoch"""
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{TRAINING_CONFIG['num_epochs']}")
        
        for batch_idx, batch in enumerate(progress_bar):
            images = batch['images'].to(self.device)
            labels = batch['label'].to(self.device)
            
            optimizer.zero_grad()
            
            if TRAINING_CONFIG['use_amp']:
                with torch.cuda.amp.autocast():
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                
                scaler.scale(loss).backward()
                
                # Gradient clipping
                if TRAINING_CONFIG['gradient_clipping']:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), TRAINING_CONFIG['gradient_clipping'])
                
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                
                # Gradient clipping
                if TRAINING_CONFIG['gradient_clipping']:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), TRAINING_CONFIG['gradient_clipping'])
                
                optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.*correct/total:.2f}%'
            })
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def validate_epoch(self, model, val_loader, criterion):
        """Validate for one epoch"""
        model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        all_predictions = []
        all_labels = []
        all_probabilities = []
        
        with torch.no_grad():
            for batch in val_loader:
                images = batch['images'].to(self.device)
                labels = batch['label'].to(self.device)
                
                if TRAINING_CONFIG['use_amp']:
                    with torch.cuda.amp.autocast():
                        outputs = model(images)
                        loss = criterion(outputs, labels)
                else:
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # Store for AUC calculation
                probabilities = torch.softmax(outputs, dim=1)
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(probabilities[:, 1].cpu().numpy())
        
        avg_loss = total_loss / len(val_loader)
        accuracy = 100. * correct / total
        
        # Calculate AUC
        from sklearn.metrics import roc_auc_score
        try:
            auc = roc_auc_score(all_labels, all_probabilities)
        except:
            auc = 0.5  # Default AUC for edge cases
        
        return avg_loss, accuracy, auc
    
    def save_checkpoint(self, model, optimizer, epoch, auc, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'auc': auc,
            'best_auc': self.best_auc,
            'config': self.config,
            'training_config': TRAINING_CONFIG
        }
        
        # Save latest checkpoint
        latest_path = os.path.join(self.checkpoint_dir, 'latest.pth')
        torch.save(checkpoint, latest_path)
        
        # Save best checkpoint
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, 'best.pth')
            torch.save(checkpoint, best_path)
            logger.info(f"💾 Saved best checkpoint (AUC: {auc:.4f})")
    
    def save_metrics(self):
        """Save training metrics"""
        metrics = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_accuracies': self.val_accuracies,
            'val_aucs': self.val_aucs,
            'best_auc': self.best_auc,
            'best_epoch': self.best_epoch,
            'config': self.config,
            'training_config': TRAINING_CONFIG
        }
        
        metrics_path = os.path.join(self.results_dir, 'training_metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
    
    def train(self, train_loader, val_loader):
        """Main training loop"""
        logger.info(f"🚀 Starting training for {self.model_name}")
        
        # Create model
        model = self.create_model()
        
        # Create optimizer and scheduler
        optimizer, scheduler = self.create_optimizer(model)
        
        # Loss function
        criterion = nn.CrossEntropyLoss()
        
        # Mixed precision scaler
        scaler = torch.cuda.amp.GradScaler() if TRAINING_CONFIG['use_amp'] else None
        
        # Training loop
        start_time = time.time()
        patience_counter = 0
        
        for epoch in range(TRAINING_CONFIG['num_epochs']):
            epoch_start = time.time()
            
            # Train
            train_loss, train_acc = self.train_epoch(model, train_loader, optimizer, criterion, scaler, epoch)
            
            # Validate
            val_loss, val_acc, val_auc = self.validate_epoch(model, val_loader, criterion)
            
            # Update scheduler
            if scheduler:
                scheduler.step()
            
            # Store metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)
            self.val_aucs.append(val_auc)
            
            # Check for best model
            is_best = val_auc > self.best_auc
            if is_best:
                self.best_auc = val_auc
                self.best_epoch = epoch
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Save checkpoint
            if (epoch + 1) % TRAINING_CONFIG['save_every_n_epochs'] == 0 or is_best:
                self.save_checkpoint(model, optimizer, epoch, val_auc, is_best)
            
            epoch_time = time.time() - epoch_start
            
            # Log progress
            logger.info(
                f"Epoch {epoch+1}/{TRAINING_CONFIG['num_epochs']} | "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | "
                f"Val AUC: {val_auc:.4f} | Time: {epoch_time:.1f}s"
            )
            
            # Early stopping
            if patience_counter >= TRAINING_CONFIG['patience']:
                logger.info(f"🛑 Early stopping at epoch {epoch+1}")
                break
        
        total_time = time.time() - start_time
        logger.info(f"✅ Training completed in {total_time/60:.1f} minutes")
        logger.info(f"🏆 Best AUC: {self.best_auc:.4f} at epoch {self.best_epoch+1}")
        
        # Save final metrics
        self.save_metrics()
        
        return model

def train_image_models():
    """Train both image-only models"""
    print("🏥 Medical Image Classification - Step 2: Train Image-Only Models")
    print("=" * 70)
    
    # Setup device
    device = torch.device(GPU_CONFIG['device'] if torch.cuda.is_available() else 'cpu')
    logger.info(f"🖥️  Using device: {device}")
    
    # Create data loaders
    logger.info("📊 Creating data loaders...")
    train_loader, val_loader, test_loader = create_preprocessed_data_loaders()
    
    logger.info(f"📈 Train samples: {len(train_loader.dataset)}")
    logger.info(f"📈 Val samples: {len(val_loader.dataset)}")
    logger.info(f"📈 Test samples: {len(test_loader.dataset)}")
    
    # Train models
    trained_models = {}
    
    for model_name in ['image_densenet', 'image_efficientnet']:
        if MODEL_CONFIGS[model_name]:
            print(f"\n🚀 Training {model_name}...")
            
            trainer = ImageModelTrainer(model_name, device)
            model = trainer.train(train_loader, val_loader)
            
            trained_models[model_name] = {
                'model': model,
                'trainer': trainer,
                'best_auc': trainer.best_auc
            }
            
            print(f"✅ {model_name} training completed!")
            print(f"🏆 Best AUC: {trainer.best_auc:.4f}")
    
    # Summary
    print(f"\n📊 Training Summary:")
    for model_name, results in trained_models.items():
        print(f"   {model_name}: AUC = {results['best_auc']:.4f}")
    
    return trained_models

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Train image-only models')
    parser.add_argument('--model', type=str, choices=['image_densenet', 'image_efficientnet', 'both'], 
                       default='both', help='Model to train')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    
    args = parser.parse_args()
    
    if args.model == 'both':
        train_image_models()
    else:
        # Train specific model
        device = torch.device(GPU_CONFIG['device'] if torch.cuda.is_available() else 'cpu')
        train_loader, val_loader, test_loader = create_preprocessed_data_loaders()
        
        trainer = ImageModelTrainer(args.model, device)
        model = trainer.train(train_loader, val_loader)
        
        print(f"✅ {args.model} training completed!")
        print(f"🏆 Best AUC: {trainer.best_auc:.4f}")

if __name__ == "__main__":
    main()
