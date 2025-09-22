import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
try:
    from torch.amp import GradScaler, autocast
except ImportError:
    from torch.cuda.amp import GradScaler, autocast
import numpy as np
import os
import time
import yaml
from tqdm import tqdm
import json
import logging
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

from medical_dataset import create_data_loaders, get_dataset_stats
from models import create_model, create_loss_function, get_model_summary

# Load YAML configuration
with open('config_training.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BinaryMedicalTrainer:
    """Binary Classification Trainer for Medical Image Fracture Detection"""
    
    def __init__(self, model, device, model_name, save_dir=None):
        self.model = model.to(device)
        self.device = device
        self.model_name = model_name
        self.save_dir = save_dir or os.path.join(config['data']['output_dir'], 'checkpoints')
        self.best_accuracy = 0.0
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
        self.val_aucs = []
        
        # Mixed precision training
        self.scaler = GradScaler() if config['hardware']['mixed_precision'] else None
        
        # Create save directory
        os.makedirs(self.save_dir, exist_ok=True)
        
        logger.info(f"🚀 Binary Medical Trainer initialized for {model_name}")
        logger.info(f"📱 Device: {device}")
        logger.info(f"💾 Save directory: {self.save_dir}")
    
    def train_epoch(self, train_loader, optimizer, criterion, epoch):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        
        for batch_idx, batch in enumerate(progress_bar):
            images = batch['image'].to(self.device)
            labels = batch['label'].to(self.device)
            
            optimizer.zero_grad()
            
            if self.scaler:
                # Mixed precision training
                with autocast(device_type='cuda'):
                    outputs = self.model(images)
                    loss = criterion(outputs, labels)
                
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                if config['training'].get('gradient_clipping'):
                    self.scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), config['training']['gradient_clipping'])
                
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                # Standard training
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                
                # Gradient clipping
                if config['training'].get('gradient_clipping'):
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), config['training']['gradient_clipping'])
                
                optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Update progress bar
            accuracy = 100.0 * correct / total
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{accuracy:.2f}%'
            })
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100.0 * correct / total
        
        self.train_losses.append(avg_loss)
        
        logger.info(f"📊 Epoch {epoch+1} - Train Loss: {avg_loss:.4f}, Train Acc: {accuracy:.2f}%")
        
        return avg_loss, accuracy
    
    def validate_epoch(self, val_loader, criterion, epoch):
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        all_predictions = []
        all_labels = []
        all_probabilities = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Validation {epoch+1}"):
                images = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)
                
                if self.scaler:
                    with autocast(device_type='cuda'):
                        outputs = self.model(images)
                        loss = criterion(outputs, labels)
                else:
                    outputs = self.model(images)
                    loss = criterion(outputs, labels)
                
                # Statistics
                total_loss += loss.item()
                probabilities = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs.data, 1)
                
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # Store for metrics
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(probabilities[:, 1].cpu().numpy())  # POSITIVE class probability
        
        # Calculate metrics
        avg_loss = total_loss / len(val_loader)
        accuracy = 100.0 * correct / total
        
        # Calculate AUC
        auc = roc_auc_score(all_labels, all_probabilities)
        
        # Calculate precision, recall, F1
        precision = 0.0
        recall = 0.0
        f1 = 0.0
        
        if len(set(all_labels)) > 1:  # Check if we have both classes
            report = classification_report(all_labels, all_predictions, output_dict=True)
            precision = report['1']['precision']  # POSITIVE class
            recall = report['1']['recall']        # POSITIVE class
            f1 = report['1']['f1-score']         # POSITIVE class
        
        self.val_losses.append(avg_loss)
        self.val_accuracies.append(accuracy)
        self.val_aucs.append(auc)
        
        logger.info(f"📊 Epoch {epoch+1} - Val Loss: {avg_loss:.4f}, Val Acc: {accuracy:.2f}%, Val AUC: {auc:.4f}")
        logger.info(f"📊 Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
        
        return avg_loss, accuracy, auc, precision, recall, f1
    
    def save_checkpoint(self, epoch, optimizer, scheduler, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'best_accuracy': self.best_accuracy,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_accuracies': self.val_accuracies,
            'val_aucs': self.val_aucs,
            'config': {
                'model_name': self.model_name,
                'training_config': TRAINING_CONFIG,
                'binary_config': BINARY_CONFIG
            }
        }
        
        if is_best:
            checkpoint_path = os.path.join(self.save_dir, f"{self.model_name}_best.pth")
            torch.save(checkpoint, checkpoint_path)
            logger.info(f"💾 Best model saved: {checkpoint_path}")
        else:
            checkpoint_path = os.path.join(self.save_dir, f"{self.model_name}_epoch_{epoch+1}.pth")
            torch.save(checkpoint, checkpoint_path)
        
        # Always save latest
        latest_path = os.path.join(self.save_dir, f"{self.model_name}_latest.pth")
        torch.save(checkpoint, latest_path)
    
    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.best_accuracy = checkpoint['best_accuracy']
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        self.val_accuracies = checkpoint['val_accuracies']
        self.val_aucs = checkpoint['val_aucs']
        
        logger.info(f"📂 Checkpoint loaded: {checkpoint_path}")
        logger.info(f"📊 Best accuracy: {self.best_accuracy:.2f}%")
        
        return checkpoint
    
    def plot_training_curves(self, save_path=None):
        """Plot training curves"""
        if save_path is None:
            save_path = os.path.join(RESULTS_DIR, f"{self.model_name}_training_curves.png")
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss curves
        axes[0, 0].plot(self.train_losses, label='Train Loss')
        axes[0, 0].plot(self.val_losses, label='Val Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Accuracy curves
        axes[0, 1].plot(self.val_accuracies, label='Val Accuracy', color='green')
        axes[0, 1].set_title('Validation Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy (%)')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # AUC curves
        axes[1, 0].plot(self.val_aucs, label='Val AUC', color='orange')
        axes[1, 0].set_title('Validation AUC')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('AUC')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Combined metrics
        axes[1, 1].plot(self.val_accuracies, label='Accuracy', color='green')
        axes[1, 1].plot([x * 100 for x in self.val_aucs], label='AUC × 100', color='orange')
        axes[1, 1].set_title('Validation Metrics Comparison')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Score')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"📊 Training curves saved: {save_path}")
    
    def train(self, train_loader, val_loader, num_epochs=None, learning_rate=None):
        """Complete training pipeline"""
        num_epochs = num_epochs or TRAINING_CONFIG['num_epochs']
        learning_rate = learning_rate or TRAINING_CONFIG['learning_rate']
        
        logger.info(f"🚀 Starting training for {num_epochs} epochs")
        logger.info(f"📊 Training samples: {len(train_loader.dataset)}")
        logger.info(f"📊 Validation samples: {len(val_loader.dataset)}")
        
        # Setup optimizer
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=TRAINING_CONFIG['weight_decay']
        )
        
        # Setup scheduler
        if TRAINING_CONFIG['scheduler'] == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=num_epochs
            )
        elif TRAINING_CONFIG['scheduler'] == 'step':
            scheduler = optim.lr_scheduler.StepLR(
                optimizer, step_size=num_epochs//3, gamma=0.1
            )
        else:
            scheduler = None
        
        # Setup loss function
        if MODEL_CONFIGS['binary_classifier']['use_focal_loss']:
            criterion = create_loss_function(
                use_focal_loss=True,
                alpha=MODEL_CONFIGS['binary_classifier']['focal_alpha'],
                gamma=MODEL_CONFIGS['binary_classifier']['focal_gamma']
            )
        else:
            criterion = nn.CrossEntropyLoss()
        
        # Early stopping
        patience = TRAINING_CONFIG['patience']
        min_delta = TRAINING_CONFIG['min_delta']
        best_epoch = 0
        patience_counter = 0
        
        # Training loop
        start_time = time.time()
        
        for epoch in range(num_epochs):
            epoch_start = time.time()
            
            # Train
            train_loss, train_acc = self.train_epoch(train_loader, optimizer, criterion, epoch)
            
            # Validate
            val_loss, val_acc, val_auc, precision, recall, f1 = self.validate_epoch(val_loader, criterion, epoch)
            
            # Update scheduler
            if scheduler:
                scheduler.step()
            
            # Check for best model
            is_best = val_acc > self.best_accuracy + min_delta
            if is_best:
                self.best_accuracy = val_acc
                best_epoch = epoch
                patience_counter = 0
                logger.info(f"🎯 New best accuracy: {val_acc:.2f}% at epoch {epoch+1}")
            else:
                patience_counter += 1
            
            # Save checkpoint
            if is_best or (epoch + 1) % TRAINING_CONFIG['save_every_n_epochs'] == 0:
                self.save_checkpoint(epoch, optimizer, scheduler, is_best=is_best)
            
            # Log epoch time
            epoch_time = time.time() - epoch_start
            logger.info(f"⏱️ Epoch {epoch+1} completed in {epoch_time:.1f}s")
            
            # Early stopping
            if patience_counter >= patience:
                logger.info(f"🛑 Early stopping at epoch {epoch+1} (patience: {patience})")
                break
        
        # Final training summary
        total_time = time.time() - start_time
        logger.info(f"🎉 Training completed in {total_time/3600:.1f} hours")
        logger.info(f"🏆 Best accuracy: {self.best_accuracy:.2f}% at epoch {best_epoch+1}")
        
        # Plot training curves
        self.plot_training_curves()
        
        # Save final results
        results = {
            'best_accuracy': self.best_accuracy,
            'best_epoch': best_epoch,
            'total_epochs': epoch + 1,
            'training_time_hours': total_time / 3600,
            'final_train_loss': train_loss,
            'final_val_loss': val_loss,
            'final_val_accuracy': val_acc,
            'final_val_auc': val_auc,
            'final_precision': precision,
            'final_recall': recall,
            'final_f1': f1
        }
        
        results_path = os.path.join(RESULTS_DIR, f"{self.model_name}_training_results.json")
        os.makedirs(os.path.dirname(results_path), exist_ok=True)
        
        # Convert numpy types to native Python types for JSON serialization
        def convert_numpy_types(obj):
            if isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return obj
        
        # Convert results to JSON-serializable format
        json_results = convert_numpy_types(results)
        
        with open(results_path, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        logger.info(f"📊 Training results saved: {results_path}")
        
        return results

def train_binary_classifier():
    """Main training function for binary classification"""
    logger.info("🚀 Starting Binary Medical Image Classification Training")
    
    # Setup device
    device = torch.device(config['hardware']['device'])
    logger.info(f"📱 Using device: {device}")
    
    # Create model
    model_config = config['model']
    model = create_model(
        model_type="BinaryEfficientNet",  # Use EfficientNet
        num_classes=model_config['num_classes'],
        dropout_rate=model_config['dropout'],
        pretrained=model_config['pretrained']
    )
    
    # Print model summary
    get_model_summary(model)
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        batch_size=config['training']['batch_size'],
        num_workers=config['hardware']['num_workers'],
        balance_classes=True  # Always balance for medical data
    )
    
    # Print dataset statistics
    stats = get_dataset_stats()
    logger.info(f"📊 Dataset Statistics:")
    logger.info(f"   Total samples: {stats['total_samples']}")
    logger.info(f"   Positive: {stats['positive_samples']} ({stats['positive_ratio']:.1%})")
    logger.info(f"   Negative: {stats['negative_samples']} ({stats['negative_ratio']:.1%})")
    
    # Create trainer
    trainer = BinaryMedicalTrainer(model, device, "binary_classifier")
    
    # Train model
    results = trainer.train(train_loader, val_loader)
    
    # Check if target accuracy reached
    target_accuracy = BINARY_CONFIG['target_accuracy'] * 100
    if results['best_accuracy'] >= target_accuracy:
        logger.info(f"🎯 Target accuracy {target_accuracy:.1f}% reached! ✅")
    else:
        logger.info(f"⚠️ Target accuracy {target_accuracy:.1f}% not reached. Best: {results['best_accuracy']:.2f}%")
    
    return trainer, results

if __name__ == "__main__":
    trainer, results = train_binary_classifier()