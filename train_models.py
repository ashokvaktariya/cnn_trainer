import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from lib.monai.engines import SupervisedTrainer
from lib.monai.losses import DiceLoss, FocalLoss
from lib.monai.metrics import ROCAUCMetric, ConfusionMatrixMetric
from lib.monai.handlers import (
    CheckpointSaver, StatsHandler, 
    ValidationHandler, LrScheduleHandler,
    EarlyStopHandler
)
from lib.monai.networks.utils import one_hot
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import os
import time
from tqdm import tqdm
import json

from medical_dataset import create_data_loaders
from models import (
    ImageOnlyDenseNet, ImageOnlyEfficientNet,
    MultimodalDenseNet, MultimodalEfficientNet,
    MedicalEnsemble
)

class MedicalTrainer:
    """Custom trainer for medical image classification"""
    
    def __init__(self, model, device, model_name, save_dir="./checkpoints"):
        self.model = model.to(device)
        self.device = device
        self.model_name = model_name
        self.save_dir = save_dir
        self.best_auc = 0.0
        self.train_losses = []
        self.val_losses = []
        self.val_aucs = []
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
    def train_epoch(self, train_loader, optimizer, criterion, epoch):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = len(train_loader)
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        
        for batch_idx, batch in enumerate(progress_bar):
            optimizer.zero_grad()
            
            # Move data to device
            images = batch['images'].to(self.device)
            labels = batch['label'].to(self.device)
            
            # Forward pass
            if self.model_name.startswith('multimodal'):
                # Multimodal models need text input
                text_input_ids = batch['text_input_ids'].to(self.device)
                text_attention_mask = batch['text_attention_mask'].to(self.device)
                outputs = self.model(images, text_input_ids, text_attention_mask)
            else:
                # Image-only models
                outputs = self.model(images)
            
            # Calculate loss
            loss = criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Avg Loss': f'{total_loss/(batch_idx+1):.4f}'
            })
        
        avg_loss = total_loss / num_batches
        self.train_losses.append(avg_loss)
        
        return avg_loss
    
    def validate(self, val_loader, criterion):
        """Validate the model"""
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                # Move data to device
                images = batch['images'].to(self.device)
                labels = batch['label'].to(self.device)
                
                # Forward pass
                if self.model_name.startswith('multimodal'):
                    text_input_ids = batch['text_input_ids'].to(self.device)
                    text_attention_mask = batch['text_attention_mask'].to(self.device)
                    outputs = self.model(images, text_input_ids, text_attention_mask)
                else:
                    outputs = self.model(images)
                
                # Calculate loss
                loss = criterion(outputs, labels)
                total_loss += loss.item()
                
                # Get predictions
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(probs, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs[:, 1].cpu().numpy())  # Probability of positive class
        
        # Calculate metrics
        avg_loss = total_loss / len(val_loader)
        accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
        
        # Calculate AUC
        try:
            from sklearn.metrics import roc_auc_score
            auc = roc_auc_score(all_labels, all_probs)
        except:
            auc = 0.0
        
        self.val_losses.append(avg_loss)
        self.val_aucs.append(auc)
        
        return avg_loss, accuracy, auc
    
    def train(self, train_loader, val_loader, num_epochs=50, lr=1e-4, weight_decay=1e-5):
        """Complete training pipeline"""
        print(f"\n🚀 Training {self.model_name}")
        print(f"📊 Training samples: {len(train_loader.dataset)}")
        print(f"📊 Validation samples: {len(val_loader.dataset)}")
        
        # Setup
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=5, verbose=True
        )
        
        # Training loop
        for epoch in range(num_epochs):
            start_time = time.time()
            
            # Train
            train_loss = self.train_epoch(train_loader, optimizer, criterion, epoch)
            
            # Validate
            val_loss, val_accuracy, val_auc = self.validate(val_loader, criterion)
            
            # Update scheduler
            scheduler.step(val_auc)
            
            # Save best model
            if val_auc > self.best_auc:
                self.best_auc = val_auc
                self.save_checkpoint(epoch, val_auc, is_best=True)
            
            # Print progress
            epoch_time = time.time() - start_time
            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            print(f"Val Accuracy: {val_accuracy:.4f}, Val AUC: {val_auc:.4f}")
            print(f"Best AUC: {self.best_auc:.4f}, Time: {epoch_time:.2f}s")
            print("-" * 50)
            
            # Early stopping
            if len(self.val_aucs) > 10:
                if max(self.val_aucs[-10:]) <= max(self.val_aucs[:-10]):
                    print("Early stopping triggered!")
                    break
        
        print(f"✅ Training completed! Best AUC: {self.best_auc:.4f}")
        
        # Save final model and metrics
        self.save_checkpoint(epoch, val_auc, is_best=False)
        self.save_metrics()
    
    def save_checkpoint(self, epoch, auc, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'best_auc': self.best_auc,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_aucs': self.val_aucs
        }
        
        if is_best:
            torch.save(checkpoint, f"{self.save_dir}/{self.model_name}_best.pth")
        else:
            torch.save(checkpoint, f"{self.save_dir}/{self.model_name}_final.pth")
    
    def save_metrics(self):
        """Save training metrics"""
        metrics = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_aucs': self.val_aucs,
            'best_auc': self.best_auc
        }
        
        with open(f"{self.save_dir}/{self.model_name}_metrics.json", 'w') as f:
            json.dump(metrics, f, indent=2)

def train_all_models(csv_file, device, save_dir="./checkpoints"):
    """Train all 5 models"""
    
    print("🏥 Medical Image Classification Training Pipeline")
    print("=" * 60)
    
    # Create data loaders
    print("📊 Loading data...")
    train_loader, val_loader = create_data_loaders(csv_file, batch_size=16, num_workers=4)
    
    models_config = {
        'image_densenet': {
            'model': ImageOnlyDenseNet(),
            'epochs': 50,
            'lr': 1e-4
        },
        'image_efficientnet': {
            'model': ImageOnlyEfficientNet(),
            'epochs': 50,
            'lr': 1e-4
        },
        'multimodal_densenet': {
            'model': MultimodalDenseNet(),
            'epochs': 40,
            'lr': 5e-5  # Lower LR for multimodal
        },
        'multimodal_efficientnet': {
            'model': MultimodalEfficientNet(),
            'epochs': 40,
            'lr': 5e-5
        }
    }
    
    trained_models = {}
    
    # Train individual models
    for model_name, config in models_config.items():
        print(f"\n🎯 Training {model_name}")
        
        trainer = MedicalTrainer(
            model=config['model'],
            device=device,
            model_name=model_name,
            save_dir=save_dir
        )
        
        trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=config['epochs'],
            lr=config['lr']
        )
        
        trained_models[model_name] = trainer
    
    # Train ensemble (optional - can be done separately)
    print(f"\n🎯 Training Ensemble Model")
    ensemble_model = MedicalEnsemble(
        model_paths={
            'image_densenet': f"{save_dir}/image_densenet_best.pth",
            'image_efficientnet': f"{save_dir}/image_efficientnet_best.pth",
            'multimodal_densenet': f"{save_dir}/multimodal_densenet_best.pth",
            'multimodal_efficientnet': f"{save_dir}/multimodal_efficientnet_best.pth"
        },
        device=device
    )
    
    # Fine-tune ensemble weights
    ensemble_trainer = MedicalTrainer(
        model=ensemble_model,
        device=device,
        model_name='ensemble',
        save_dir=save_dir
    )
    
    # Create ensemble data loader (smaller batch size)
    train_loader_ensemble, val_loader_ensemble = create_data_loaders(
        csv_file, batch_size=8, num_workers=2
    )
    
    ensemble_trainer.train(
        train_loader=train_loader_ensemble,
        val_loader=val_loader_ensemble,
        num_epochs=20,
        lr=1e-5
    )
    
    print("\n🎉 All models trained successfully!")
    
    return trained_models, ensemble_trainer

def evaluate_models(test_csv_file, model_paths, device):
    """Evaluate all trained models"""
    
    print("📊 Evaluating Models")
    print("=" * 40)
    
    # Load test data
    _, test_loader = create_data_loaders(test_csv_file, batch_size=16, num_workers=4)
    
    # Load ensemble model
    ensemble = MedicalEnsemble(model_paths, device)
    ensemble.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            images = batch['images'].to(device)
            labels = batch['label'].to(device)
            text_input_ids = batch['text_input_ids'].to(device)
            text_attention_mask = batch['text_attention_mask'].to(device)
            
            # Get ensemble prediction
            outputs, probs, weights = ensemble(images, text_input_ids, text_attention_mask)
            
            preds = torch.argmax(outputs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(outputs[:, 1].cpu().numpy())
    
    # Calculate metrics
    from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
    
    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    auc = roc_auc_score(all_labels, all_probs)
    
    print(f"📈 Final Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"AUC: {auc:.4f}")
    print(f"Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=['NEGATIVE', 'POSITIVE']))
    
    return {
        'accuracy': accuracy,
        'auc': auc,
        'predictions': all_preds,
        'labels': all_labels,
        'probabilities': all_probs
    }

if __name__ == "__main__":
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Using device: {device}")
    
    # Train all models
    trained_models, ensemble_trainer = train_all_models(
        csv_file="dicom_image_url_file.csv",
        device=device,
        save_dir="./checkpoints"
    )
    
    # Evaluate ensemble
    model_paths = {
        'image_densenet': "./checkpoints/image_densenet_best.pth",
        'image_efficientnet': "./checkpoints/image_efficientnet_best.pth",
        'multimodal_densenet': "./checkpoints/multimodal_densenet_best.pth",
        'multimodal_efficientnet': "./checkpoints/multimodal_efficientnet_best.pth"
    }
    
    results = evaluate_models(
        test_csv_file="dicom_image_url_file.csv",
        model_paths=model_paths,
        device=device
    )
