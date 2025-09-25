#!/usr/bin/env python3
"""
Quick evaluation script for specific checkpoint
/home/avaktariya/binary_classifier_20250924_131737
"""

import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score, accuracy_score,
    precision_score, recall_score, f1_score
)
import pandas as pd
import os
import json
import argparse
from tqdm import tqdm
import logging
import yaml
from pathlib import Path

from medical_dataset import create_data_loaders, get_dataset_stats
from models import create_model

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_config():
    """Load configuration from YAML file"""
    config_path = Path('config_training.yaml')
    if config_path.exists():
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    else:
        # Fallback configuration
        return {
            'data': {
                'train_csv': 'balanced_dataset_cnn_train.csv',
                'val_csv': 'balanced_dataset_cnn_val.csv',
                'test_csv': 'balanced_dataset_cnn.csv',
                'image_root': '/mount/civiescaks01storage01/aksfileshare01/CNN/gleamer-images/'
            },
            'training': {
                'batch_size': 16,
                'num_workers': 4
            },
            'model': {
                'num_classes': 2,
                'class_names': ['NEGATIVE', 'POSITIVE']
            }
        }

def evaluate_checkpoint(checkpoint_path):
    """Evaluate specific checkpoint"""
    logger.info(f"🔍 Evaluating checkpoint: {checkpoint_path}")
    
    # Load configuration
    config = load_config()
    
    # Setup device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    logger.info(f"🖥️ Using device: {device}")
    
    # Create model
    model = create_model(
        model_type='BinaryEfficientNet',
        num_classes=config['model']['num_classes'],
        dropout_rate=0.3,
        pretrained=False
    )
    
    # Load checkpoint
    if os.path.isdir(checkpoint_path):
        # If it's a directory, look for the best model
        best_model_path = os.path.join(checkpoint_path, 'best_model.pth')
        if os.path.exists(best_model_path):
            checkpoint_path = best_model_path
        else:
            # Look for any .pth file in the directory
            pth_files = [f for f in os.listdir(checkpoint_path) if f.endswith('.pth')]
            if pth_files:
                checkpoint_path = os.path.join(checkpoint_path, pth_files[0])
            else:
                logger.error(f"❌ No .pth files found in {checkpoint_path}")
                return None
    
    if not os.path.exists(checkpoint_path):
        logger.error(f"❌ Checkpoint not found: {checkpoint_path}")
        return None
    
    logger.info(f"📂 Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Create test data loader
    _, _, test_loader = create_data_loaders(
        config['data']['train_csv'],
        config['data']['val_csv'],
        config['data']['test_csv'],
        batch_size=config['training']['batch_size'],
        num_workers=config['training']['num_workers'],
        balance_classes=False
    )
    
    # Evaluate on test set
    logger.info("🧪 Evaluating on test set...")
    
    all_predictions = []
    all_labels = []
    all_probabilities = []
    all_losses = []
    
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Get probabilities and predictions
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            # Store results
            all_losses.append(loss.item())
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(probabilities[:, 1].cpu().numpy())  # POSITIVE class probability
    
    # Calculate metrics
    test_loss = np.mean(all_losses)
    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, average='binary')
    recall = recall_score(all_labels, all_predictions, average='binary')
    f1 = f1_score(all_labels, all_predictions, average='binary')
    auc = roc_auc_score(all_labels, all_probabilities)
    avg_precision = average_precision_score(all_labels, all_probabilities)
    
    # Print results
    logger.info(f"🏆 EVALUATION RESULTS:")
    logger.info(f"   Test Loss: {test_loss:.4f}")
    logger.info(f"   Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    logger.info(f"   Precision: {precision:.4f}")
    logger.info(f"   Recall: {recall:.4f}")
    logger.info(f"   F1-Score: {f1:.4f}")
    logger.info(f"   AUC: {auc:.4f}")
    logger.info(f"   Average Precision: {avg_precision:.4f}")
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    tn, fp, fn, tp = cm.ravel()
    logger.info(f"📊 Confusion Matrix:")
    logger.info(f"   True Negatives: {tn}")
    logger.info(f"   False Positives: {fp}")
    logger.info(f"   False Negatives: {fn}")
    logger.info(f"   True Positives: {tp}")
    
    # Save results
    results = {
        'test_loss': test_loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'auc': auc,
        'average_precision': avg_precision,
        'confusion_matrix': {
            'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)
        }
    }
    
    results_file = 'evaluation_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"💾 Results saved to: {results_file}")
    
    return results

def main():
    """Main evaluation function"""
    parser = argparse.ArgumentParser(description='Evaluate specific checkpoint')
    parser.add_argument('--checkpoint', type=str, 
                       default='/home/avaktariya/binary_classifier_20250924_131737',
                       help='Path to model checkpoint')
    
    args = parser.parse_args()
    
    # Run evaluation
    results = evaluate_checkpoint(args.checkpoint)
    
    if results is not None:
        logger.info("🎉 Evaluation complete!")
    else:
        logger.error("❌ Evaluation failed!")
    
    return results

if __name__ == "__main__":
    results = main()
