#!/usr/bin/env python3
"""
Model Evaluation Script for Binary Medical Image Classification

This script:
1. Loads trained binary classifier
2. Evaluates on test dataset
3. Generates comprehensive metrics and visualizations
4. Creates confusion matrix and ROC curves
5. Provides detailed performance analysis
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

from medical_dataset import create_data_loaders, get_dataset_stats
from models import create_model
from train_models import BinaryMedicalTrainer
from config import (
    TRAINING_CONFIG, GPU_CONFIG, CHECKPOINTS_DIR, RESULTS_DIR,
    MODEL_CONFIGS, BINARY_CONFIG
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BinaryModelEvaluator:
    """Comprehensive evaluation for binary medical image classification"""
    
    def __init__(self, model, device, model_name):
        self.model = model.to(device)
        self.device = device
        self.model_name = model_name
        self.results = {}
        
        # Create results directory
        os.makedirs(RESULTS_DIR, exist_ok=True)
        
        logger.info(f"📊 Binary Model Evaluator initialized for {model_name}")
    
    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        logger.info(f"📂 Model checkpoint loaded: {checkpoint_path}")
        
        if 'best_accuracy' in checkpoint:
            logger.info(f"🏆 Best training accuracy: {checkpoint['best_accuracy']:.2f}%")
        
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
            'average_precision': avg_precision,
            'num_samples': len(all_labels)
        }
        
        logger.info(f"📊 Test Results:")
        logger.info(f"   Loss: {test_loss:.4f}")
        logger.info(f"   Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        logger.info(f"   Precision: {precision:.4f}")
        logger.info(f"   Recall: {recall:.4f}")
        logger.info(f"   F1-Score: {f1:.4f}")
        logger.info(f"   AUC: {auc:.4f}")
        logger.info(f"   Average Precision: {avg_precision:.4f}")
        
        return {
            'predictions': all_predictions,
            'labels': all_labels,
            'probabilities': all_probabilities,
            'test_loss': test_loss
        }
    
    def create_confusion_matrix(self, predictions, labels, save_path=None):
        """Create and save confusion matrix"""
        if save_path is None:
            save_path = os.path.join(RESULTS_DIR, f"{self.model_name}_confusion_matrix.png")
        
        cm = confusion_matrix(labels, predictions)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['NEGATIVE', 'POSITIVE'],
                    yticklabels=['NEGATIVE', 'POSITIVE'])
        plt.title(f'Confusion Matrix - {self.model_name}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        # Add accuracy text
        accuracy = accuracy_score(labels, predictions)
        plt.text(0.5, -0.15, f'Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)',
                transform=plt.gca().transAxes, ha='center')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"📊 Confusion matrix saved: {save_path}")
        
        return cm
    
    def create_roc_curve(self, labels, probabilities, save_path=None):
        """Create and save ROC curve"""
        if save_path is None:
            save_path = os.path.join(RESULTS_DIR, f"{self.model_name}_roc_curve.png")
        
        fpr, tpr, thresholds = roc_curve(labels, probabilities)
        auc = roc_auc_score(labels, probabilities)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
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
        
        logger.info(f"📊 ROC curve saved: {save_path}")
        
        return fpr, tpr, thresholds, auc
    
    def create_precision_recall_curve(self, labels, probabilities, save_path=None):
        """Create and save Precision-Recall curve"""
        if save_path is None:
            save_path = os.path.join(RESULTS_DIR, f"{self.model_name}_precision_recall_curve.png")
        
        precision, recall, thresholds = precision_recall_curve(labels, probabilities)
        avg_precision = average_precision_score(labels, probabilities)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='darkorange', lw=2, 
                label=f'PR curve (AP = {avg_precision:.4f})')
        
        # Add baseline (random classifier)
        baseline = np.sum(labels) / len(labels)
        plt.axhline(y=baseline, color='navy', linestyle='--', 
                   label=f'Random (AP = {baseline:.4f})')
        
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
        
        logger.info(f"📊 Precision-Recall curve saved: {save_path}")
        
        return precision, recall, thresholds, avg_precision
    
    def create_classification_report(self, predictions, labels, save_path=None):
        """Create detailed classification report"""
        if save_path is None:
            save_path = os.path.join(RESULTS_DIR, f"{self.model_name}_classification_report.txt")
        
        report = classification_report(labels, predictions, 
                                     target_names=['NEGATIVE', 'POSITIVE'],
                                     output_dict=True)
        
        with open(save_path, 'w') as f:
            f.write(f"BINARY MEDICAL IMAGE CLASSIFICATION REPORT\n")
            f.write(f"Model: {self.model_name}\n")
            f.write("=" * 50 + "\n\n")
            
            # Overall metrics
            f.write("OVERALL METRICS:\n")
            f.write(f"Accuracy: {report['accuracy']:.4f} ({report['accuracy']*100:.2f}%)\n")
            f.write(f"Macro Average Precision: {report['macro avg']['precision']:.4f}\n")
            f.write(f"Macro Average Recall: {report['macro avg']['recall']:.4f}\n")
            f.write(f"Macro Average F1-Score: {report['macro avg']['f1-score']:.4f}\n")
            f.write(f"Weighted Average Precision: {report['weighted avg']['precision']:.4f}\n")
            f.write(f"Weighted Average Recall: {report['weighted avg']['recall']:.4f}\n")
            f.write(f"Weighted Average F1-Score: {report['weighted avg']['f1-score']:.4f}\n\n")
            
            # Per-class metrics
            f.write("PER-CLASS METRICS:\n")
            for class_name, metrics in report.items():
                if class_name not in ['accuracy', 'macro avg', 'weighted avg']:
                    f.write(f"\n{class_name.upper()}:\n")
                    f.write(f"  Precision: {metrics['precision']:.4f}\n")
                    f.write(f"  Recall: {metrics['recall']:.4f}\n")
                    f.write(f"  F1-Score: {metrics['f1-score']:.4f}\n")
                    f.write(f"  Support: {metrics['support']}\n")
            
            # Confusion matrix
            cm = confusion_matrix(labels, predictions)
            f.write(f"\nCONFUSION MATRIX:\n")
            f.write(f"                Predicted\n")
            f.write(f"                NEGATIVE  POSITIVE\n")
            f.write(f"Actual NEGATIVE    {cm[0,0]:4d}     {cm[0,1]:4d}\n")
            f.write(f"       POSITIVE    {cm[1,0]:4d}     {cm[1,1]:4d}\n")
        
        logger.info(f"📊 Classification report saved: {save_path}")
        
        return report
    
    def analyze_predictions(self, predictions, labels, probabilities):
        """Analyze prediction patterns"""
        logger.info("🔍 Analyzing prediction patterns...")
        
        # Convert to numpy arrays
        predictions = np.array(predictions)
        labels = np.array(labels)
        probabilities = np.array(probabilities)
        
        # Confidence analysis
        confidence_analysis = {}
        
        for threshold in [0.5, 0.6, 0.7, 0.8, 0.9]:
            confident_mask = (probabilities >= threshold) | (probabilities <= (1 - threshold))
            confident_predictions = predictions[confident_mask]
            confident_labels = labels[confident_mask]
            
            if len(confident_predictions) > 0:
                acc = accuracy_score(confident_labels, confident_predictions)
                confidence_analysis[threshold] = {
                    'accuracy': acc,
                    'num_samples': len(confident_predictions),
                    'percentage': len(confident_predictions) / len(predictions) * 100
                }
        
        # Error analysis
        errors = predictions != labels
        error_indices = np.where(errors)[0]
        
        error_analysis = {
            'total_errors': len(error_indices),
            'error_rate': len(error_indices) / len(predictions),
            'false_positives': np.sum((predictions == 1) & (labels == 0)),
            'false_negatives': np.sum((predictions == 0) & (labels == 1))
        }
        
        self.results['prediction_analysis'] = {
            'confidence_analysis': confidence_analysis,
            'error_analysis': error_analysis
        }
        
        logger.info(f"📊 Prediction Analysis:")
        logger.info(f"   Total errors: {error_analysis['total_errors']}")
        logger.info(f"   Error rate: {error_analysis['error_rate']:.4f}")
        logger.info(f"   False positives: {error_analysis['false_positives']}")
        logger.info(f"   False negatives: {error_analysis['false_negatives']}")
        
        return confidence_analysis, error_analysis
    
    def generate_evaluation_report(self, save_path=None):
        """Generate comprehensive evaluation report"""
        if save_path is None:
            save_path = os.path.join(RESULTS_DIR, f"{self.model_name}_evaluation_report.json")
        
        # Add dataset statistics
        dataset_stats = get_dataset_stats()
        self.results['dataset_stats'] = dataset_stats
        
        # Add model configuration
        self.results['model_config'] = MODEL_CONFIGS['binary_classifier']
        
        # Save results
        with open(save_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        logger.info(f"📊 Evaluation report saved: {save_path}")
        
        return self.results
    
    def evaluate(self, checkpoint_path, test_loader=None):
        """Complete evaluation pipeline"""
        logger.info(f"🚀 Starting evaluation of {self.model_name}")
        
        # Load checkpoint
        checkpoint = self.load_checkpoint(checkpoint_path)
        
        # Create test loader if not provided
        if test_loader is None:
            _, _, test_loader = create_data_loaders(
                batch_size=TRAINING_CONFIG['batch_size'],
                num_workers=TRAINING_CONFIG['num_workers'],
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
        
        # Analyze predictions
        self.analyze_predictions(eval_results['predictions'], eval_results['labels'], eval_results['probabilities'])
        
        # Generate final report
        final_results = self.generate_evaluation_report()
        
        logger.info("🎉 Evaluation complete!")
        
        return final_results

def main():
    """Main evaluation function"""
    parser = argparse.ArgumentParser(description='Evaluate binary medical image classifier')
    parser.add_argument('--checkpoint', type=str, 
                       default=os.path.join(CHECKPOINTS_DIR, 'binary_classifier_best.pth'),
                       help='Path to model checkpoint')
    parser.add_argument('--model-name', type=str, default='binary_classifier',
                       help='Model name for saving results')
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device(GPU_CONFIG['device'])
    
    # Create model
    model_config = MODEL_CONFIGS['binary_classifier']
    model = create_model(
        model_type=model_config['model_type'],
        num_classes=model_config['num_classes'],
        dropout_rate=model_config['dropout_rate'],
        pretrained=False  # Don't use pretrained for evaluation
    )
    
    # Create evaluator
    evaluator = BinaryModelEvaluator(model, device, args.model_name)
    
    # Run evaluation
    results = evaluator.evaluate(args.checkpoint)
    
    # Print summary
    test_metrics = results['test_metrics']
    logger.info(f"🏆 FINAL EVALUATION SUMMARY:")
    logger.info(f"   Accuracy: {test_metrics['accuracy']:.4f} ({test_metrics['accuracy']*100:.2f}%)")
    logger.info(f"   Precision: {test_metrics['precision']:.4f}")
    logger.info(f"   Recall: {test_metrics['recall']:.4f}")
    logger.info(f"   F1-Score: {test_metrics['f1_score']:.4f}")
    logger.info(f"   AUC: {test_metrics['auc']:.4f}")
    
    return results

if __name__ == "__main__":
    results = main()
