#!/usr/bin/env python3
"""
Simple Model Evaluation Script for Current Dataset
Works with final_dataset_cnn.csv structure
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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
                'train_csv': 'final_dataset_cnn_train.csv',
                'val_csv': 'final_dataset_cnn_val.csv',
                'test_csv': 'final_dataset_cnn.csv',
                'image_root': '/mount/civiescaks01storage01/aksfileshare01/CNN/gleamer-images/'
            },
            'training': {
                'batch_size': 8,
                'num_workers': 4
            },
            'model': {
                'num_classes': 2,
                'class_names': ['NEGATIVE', 'POSITIVE']
            }
        }

class SimpleModelEvaluator:
    """Simple evaluator for current dataset"""
    
    def __init__(self, model, device, model_name="binary_classifier"):
        self.model = model.to(device)
        self.device = device
        self.model_name = model_name
        self.results = {}
        
    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint"""
        if not os.path.exists(checkpoint_path):
            logger.error(f"❌ Checkpoint not found: {checkpoint_path}")
            return None
            
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        logger.info(f"✅ Loaded checkpoint: {checkpoint_path}")
        logger.info(f"   Best accuracy: {checkpoint.get('best_accuracy', 'N/A'):.2f}%")
        
        return checkpoint
    
    def evaluate_test_set(self, test_loader):
        """Evaluate model on test set"""
        logger.info("🧪 Evaluating on test set...")
        
        self.model.eval()
        all_predictions = []
        all_labels = []
        all_probabilities = []
        all_losses = []
        
        criterion = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Testing"):
                images = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)
                
                # Forward pass
                outputs = self.model(images)
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
        
        self.results['test_metrics'] = {
            'test_loss': test_loss,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc': auc,
            'average_precision': avg_precision
        }
        
        self.results['predictions'] = all_predictions
        self.results['labels'] = all_labels
        self.results['probabilities'] = all_probabilities
        
        logger.info(f"📊 Test Results:")
        logger.info(f"   Loss: {test_loss:.4f}")
        logger.info(f"   Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        logger.info(f"   Precision: {precision:.4f}")
        logger.info(f"   Recall: {recall:.4f}")
        logger.info(f"   F1-Score: {f1:.4f}")
        logger.info(f"   AUC: {auc:.4f}")
        logger.info(f"   Avg Precision: {avg_precision:.4f}")
        
        return self.results
    
    def create_confusion_matrix(self, predictions, labels, save_path="confusion_matrix.png"):
        """Create and save confusion matrix"""
        cm = confusion_matrix(labels, predictions)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['NEGATIVE', 'POSITIVE'],
                   yticklabels=['NEGATIVE', 'POSITIVE'])
        plt.title(f'Confusion Matrix - {self.model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"📊 Confusion matrix saved to: {save_path}")
        
        # Print confusion matrix details
        tn, fp, fn, tp = cm.ravel()
        logger.info(f"   True Negatives: {tn}")
        logger.info(f"   False Positives: {fp}")
        logger.info(f"   False Negatives: {fn}")
        logger.info(f"   True Positives: {tp}")
    
    def create_roc_curve(self, labels, probabilities, save_path="roc_curve.png"):
        """Create and save ROC curve"""
        fpr, tpr, _ = roc_curve(labels, probabilities)
        auc = roc_auc_score(labels, probabilities)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {self.model_name}')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"📈 ROC curve saved to: {save_path}")
    
    def create_precision_recall_curve(self, labels, probabilities, save_path="precision_recall_curve.png"):
        """Create and save Precision-Recall curve"""
        precision, recall, _ = precision_recall_curve(labels, probabilities)
        avg_precision = average_precision_score(labels, probabilities)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='darkorange', lw=2, 
                label=f'PR curve (AP = {avg_precision:.3f})')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve - {self.model_name}')
        plt.legend(loc="lower left")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"📈 Precision-Recall curve saved to: {save_path}")
    
    def create_classification_report(self, predictions, labels, save_path="classification_report.txt"):
        """Create and save classification report"""
        report = classification_report(labels, predictions, 
                                     target_names=['NEGATIVE', 'POSITIVE'],
                                     output_dict=True)
        
        # Save to file
        with open(save_path, 'w') as f:
            f.write(f"Classification Report - {self.model_name}\n")
            f.write("=" * 50 + "\n\n")
            f.write(classification_report(labels, predictions, 
                                        target_names=['NEGATIVE', 'POSITIVE']))
            f.write("\n\nDetailed Metrics:\n")
            f.write(f"Accuracy: {report['accuracy']:.4f}\n")
            f.write(f"Macro Avg F1: {report['macro avg']['f1-score']:.4f}\n")
            f.write(f"Weighted Avg F1: {report['weighted avg']['f1-score']:.4f}\n")
        
        logger.info(f"📋 Classification report saved to: {save_path}")
    
    def evaluate(self, checkpoint_path, test_loader=None):
        """Complete evaluation pipeline"""
        logger.info(f"🚀 Starting evaluation of {self.model_name}")
        
        # Load checkpoint
        checkpoint = self.load_checkpoint(checkpoint_path)
        if checkpoint is None:
            return None
        
        # Create test loader if not provided
        if test_loader is None:
            config = load_config()
            _, _, test_loader = create_data_loaders(
                config['data']['train_csv'],
                config['data']['val_csv'],
                config['data']['test_csv'],
                batch_size=config['training']['batch_size'],
                num_workers=config['training']['num_workers'],
                balance_classes=False  # Don't balance for evaluation
            )
        
        # Evaluate on test set
        eval_results = self.evaluate_test_set(test_loader)
        
        # Create visualizations
        self.create_confusion_matrix(eval_results['predictions'], eval_results['labels'])
        self.create_roc_curve(eval_results['labels'], eval_results['probabilities'])
        self.create_precision_recall_curve(eval_results['labels'], eval_results['probabilities'])
        
        # Create classification report
        self.create_classification_report(eval_results['predictions'], eval_results['labels'])
        
        logger.info("🎉 Evaluation complete!")
        
        return eval_results

def main():
    """Main evaluation function"""
    parser = argparse.ArgumentParser(description='Evaluate binary medical image classifier')
    parser.add_argument('--checkpoint', type=str, 
                       default='/home/avaktariya/binary_classifier_20250924_131737',
                       help='Path to model checkpoint')
    parser.add_argument('--model-name', type=str, default='binary_classifier',
                       help='Model name for saving results')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config()
    
    # Setup device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    logger.info(f"🖥️  Using device: {device}")
    
    # Create model
    model = create_model(
        model_type='BinaryEfficientNet',
        num_classes=config['model']['num_classes'],
        dropout_rate=0.3,
        pretrained=False  # Don't use pretrained for evaluation
    )
    
    # Create evaluator
    evaluator = SimpleModelEvaluator(model, device, args.model_name)
    
    # Run evaluation
    results = evaluator.evaluate(args.checkpoint)
    
    if results is not None:
        # Print summary
        test_metrics = results['test_metrics']
        logger.info(f"🏆 FINAL EVALUATION SUMMARY:")
        logger.info(f"   Accuracy: {test_metrics['accuracy']:.4f} ({test_metrics['accuracy']*100:.2f}%)")
        logger.info(f"   Precision: {test_metrics['precision']:.4f}")
        logger.info(f"   Recall: {test_metrics['recall']:.4f}")
        logger.info(f"   F1-Score: {test_metrics['f1_score']:.4f}")
        logger.info(f"   AUC: {test_metrics['auc']:.4f}")
        
        # Save results to JSON
        results_path = f"{args.model_name}_evaluation_results.json"
        with open(results_path, 'w') as f:
            json.dump(test_metrics, f, indent=2)
        logger.info(f"💾 Results saved to: {results_path}")
    
    return results

if __name__ == "__main__":
    results = main()
