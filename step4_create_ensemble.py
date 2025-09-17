#!/usr/bin/env python3
"""
Step 4: Create and Test Ensemble Model
Creates ensemble model combining all trained models and evaluates performance
H200 GPU Server Optimized
"""

import torch
import torch.nn as nn
import numpy as np
import os
import json
import logging
import time
from pathlib import Path
from tqdm import tqdm
import argparse
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from config import (
    CHECKPOINTS_DIR, RESULTS_DIR, LOGS_DIR, GPU_CONFIG, TRAINING_CONFIG,
    MODEL_CONFIGS, ENSEMBLE_CONFIG, LOGGING_CONFIG, get_model_path, get_results_path
)
from preprocessed_dataset import create_preprocessed_data_loaders
from models import (
    ImageOnlyDenseNet, ImageOnlyEfficientNet,
    MultimodalDenseNet, MultimodalEfficientNet,
    MedicalEnsemble
)

# Setup logging
logging.basicConfig(
    level=getattr(logging, LOGGING_CONFIG['log_level']),
    format=LOGGING_CONFIG['log_format'],
    handlers=[
        logging.FileHandler(os.path.join(LOGS_DIR, "ensemble_creation.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class EnsembleCreator:
    """Creates and evaluates ensemble models"""
    
    def __init__(self, device=None):
        self.device = device or torch.device(GPU_CONFIG['device'])
        self.model_paths = {}
        self.models = {}
        
        # Setup paths
        self.ensemble_dir = os.path.join(RESULTS_DIR, "ensemble")
        os.makedirs(self.ensemble_dir, exist_ok=True)
        
        logger.info(f"🔧 Initialized EnsembleCreator")
        logger.info(f"🖥️  Device: {self.device}")
    
    def find_trained_models(self):
        """Find all trained model checkpoints"""
        logger.info("🔍 Searching for trained models...")
        
        model_names = ['image_densenet', 'image_efficientnet', 'multimodal_densenet', 'multimodal_efficientnet']
        
        for model_name in model_names:
            best_path = os.path.join(CHECKPOINTS_DIR, model_name, 'best.pth')
            if os.path.exists(best_path):
                self.model_paths[model_name] = best_path
                logger.info(f"✅ Found {model_name}: {best_path}")
            else:
                logger.warning(f"❌ Model not found: {model_name}")
        
        if len(self.model_paths) == 0:
            raise FileNotFoundError("No trained models found. Please train models first.")
        
        logger.info(f"📊 Found {len(self.model_paths)} trained models")
        return self.model_paths
    
    def load_individual_models(self):
        """Load all individual trained models"""
        logger.info("📂 Loading individual models...")
        
        for model_name, model_path in self.model_paths.items():
            try:
                # Load checkpoint
                checkpoint = torch.load(model_path, map_location=self.device)
                
                # Create model
                if 'image_densenet' in model_name:
                    model = ImageOnlyDenseNet(num_classes=2)
                elif 'image_efficientnet' in model_name:
                    model = ImageOnlyEfficientNet(num_classes=2)
                elif 'multimodal_densenet' in model_name:
                    model = MultimodalDenseNet(num_classes=2)
                elif 'multimodal_efficientnet' in model_name:
                    model = MultimodalEfficientNet(num_classes=2)
                else:
                    raise ValueError(f"Unknown model type: {model_name}")
                
                # Load weights
                model.load_state_dict(checkpoint['model_state_dict'])
                model = model.to(self.device)
                model.eval()
                
                self.models[model_name] = {
                    'model': model,
                    'auc': checkpoint.get('auc', 0.0),
                    'config': checkpoint.get('config', {})
                }
                
                logger.info(f"✅ Loaded {model_name} (AUC: {checkpoint.get('auc', 0.0):.4f})")
                
            except Exception as e:
                logger.error(f"❌ Failed to load {model_name}: {e}")
                continue
        
        logger.info(f"📊 Loaded {len(self.models)} models successfully")
        return self.models
    
    def create_ensemble(self, method='weighted'):
        """Create ensemble model"""
        logger.info(f"🔧 Creating {method} ensemble...")
        
        if method == 'weighted':
            ensemble = MedicalEnsemble(self.model_paths, device=self.device)
        else:
            # For other methods, we'll implement them separately
            ensemble = self._create_simple_ensemble(method)
        
        ensemble = ensemble.to(self.device)
        ensemble.eval()
        
        logger.info(f"✅ Created {method} ensemble with {len(self.models)} models")
        return ensemble
    
    def _create_simple_ensemble(self, method):
        """Create simple ensemble (voting/averaging)"""
        class SimpleEnsemble(nn.Module):
            def __init__(self, models, method='voting'):
                super().__init__()
                self.models = nn.ModuleList([model_info['model'] for model_info in models.values()])
                self.method = method
                self.model_names = list(models.keys())
                
                # Get AUC scores for weighting
                self.weights = torch.tensor([models[name]['auc'] for name in self.model_names])
                self.weights = self.weights / self.weights.sum()  # Normalize
                
            def forward(self, images, text_input_ids=None, text_attention_mask=None):
                all_logits = []
                
                for i, model in enumerate(self.models):
                    model_name = self.model_names[i]
                    
                    if 'multimodal' in model_name:
                        if text_input_ids is None or text_attention_mask is None:
                            raise ValueError(f"Multimodal model {model_name} requires text inputs")
                        logits = model(images, text_input_ids, text_attention_mask)
                    else:
                        logits = model(images)
                    
                    all_logits.append(logits)
                
                if self.method == 'voting':
                    # Hard voting
                    all_predictions = torch.stack([torch.argmax(logits, dim=1) for logits in all_logits])
                    final_pred = torch.mode(all_predictions, dim=0)[0]
                    return final_pred.unsqueeze(1).float()
                
                elif self.method == 'averaging':
                    # Simple averaging
                    return torch.mean(torch.stack(all_logits), dim=0)
                
                elif self.method == 'weighted':
                    # Weighted averaging
                    weighted_logits = torch.stack([logits * self.weights[i] for i, logits in enumerate(all_logits)])
                    return torch.sum(weighted_logits, dim=0)
                
                else:
                    raise ValueError(f"Unknown ensemble method: {self.method}")
        
        return SimpleEnsemble(self.models, method)
    
    def evaluate_ensemble(self, ensemble, test_loader, method='weighted'):
        """Evaluate ensemble model"""
        logger.info(f"📊 Evaluating {method} ensemble...")
        
        ensemble.eval()
        all_predictions = []
        all_labels = []
        all_probabilities = []
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Evaluating ensemble"):
                images = batch['images'].to(self.device)
                text_input_ids = batch['text_input_ids'].to(self.device)
                text_attention_mask = batch['text_attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                if method == 'weighted' and hasattr(ensemble, 'forward'):
                    # Use MedicalEnsemble
                    outputs, probs, weights = ensemble(images, text_input_ids, text_attention_mask)
                    probabilities = torch.softmax(outputs, dim=1)
                else:
                    # Use simple ensemble
                    outputs = ensemble(images, text_input_ids, text_attention_mask)
                    
                    if method == 'voting':
                        probabilities = outputs
                        predictions = (probabilities > 0.5).long()
                    else:
                        probabilities = torch.softmax(outputs, dim=1)
                        predictions = torch.argmax(outputs, dim=1)
                
                if method != 'voting':
                    predictions = torch.argmax(outputs, dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                if method != 'voting':
                    all_probabilities.extend(probabilities[:, 1].cpu().numpy())
                else:
                    all_probabilities.extend(probabilities.cpu().numpy())
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_predictions)
        
        try:
            auc = roc_auc_score(all_labels, all_probabilities)
        except:
            auc = 0.5
        
        # Classification report
        report = classification_report(all_labels, all_predictions, output_dict=True)
        
        # Confusion matrix
        cm = confusion_matrix(all_labels, all_predictions)
        
        results = {
            'method': method,
            'accuracy': accuracy,
            'auc': auc,
            'classification_report': report,
            'confusion_matrix': cm.tolist(),
            'predictions': all_predictions,
            'labels': all_labels,
            'probabilities': all_probabilities
        }
        
        logger.info(f"✅ Evaluation completed:")
        logger.info(f"   Accuracy: {accuracy:.4f}")
        logger.info(f"   AUC: {auc:.4f}")
        
        return results
    
    def create_visualizations(self, results, method='weighted'):
        """Create evaluation visualizations"""
        logger.info(f"📊 Creating visualizations for {method} ensemble...")
        
        # Confusion Matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(results['confusion_matrix'], annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Negative', 'Positive'], 
                    yticklabels=['Negative', 'Positive'])
        plt.title(f'Confusion Matrix - {method.title()} Ensemble')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(os.path.join(self.ensemble_dir, f'confusion_matrix_{method}.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # ROC Curve
        from sklearn.metrics import roc_curve
        fpr, tpr, _ = roc_curve(results['labels'], results['probabilities'])
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {results["auc"]:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {method.title()} Ensemble')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.ensemble_dir, f'roc_curve_{method}.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"✅ Visualizations saved to {self.ensemble_dir}")
    
    def save_results(self, results, method='weighted'):
        """Save evaluation results"""
        results_path = os.path.join(self.ensemble_dir, f'ensemble_results_{method}.json')
        
        # Convert numpy arrays to lists for JSON serialization
        results_json = results.copy()
        results_json['predictions'] = [int(x) for x in results['predictions']]
        results_json['labels'] = [int(x) for x in results['labels']]
        results_json['probabilities'] = [float(x) for x in results['probabilities']]
        
        with open(results_path, 'w') as f:
            json.dump(results_json, f, indent=2)
        
        logger.info(f"💾 Results saved to {results_path}")
    
    def compare_models(self, test_loader):
        """Compare individual models and ensemble"""
        logger.info("📊 Comparing all models...")
        
        comparison_results = {}
        
        # Evaluate individual models
        for model_name, model_info in self.models.items():
            logger.info(f"🔍 Evaluating {model_name}...")
            
            model = model_info['model']
            model.eval()
            
            all_predictions = []
            all_labels = []
            all_probabilities = []
            
            with torch.no_grad():
                for batch in tqdm(test_loader, desc=f"Evaluating {model_name}"):
                    images = batch['images'].to(self.device)
                    text_input_ids = batch['text_input_ids'].to(self.device)
                    text_attention_mask = batch['text_attention_mask'].to(self.device)
                    labels = batch['label'].to(self.device)
                    
                    if 'multimodal' in model_name:
                        outputs = model(images, text_input_ids, text_attention_mask)
                    else:
                        outputs = model(images)
                    
                    probabilities = torch.softmax(outputs, dim=1)
                    predictions = torch.argmax(outputs, dim=1)
                    
                    all_predictions.extend(predictions.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
                    all_probabilities.extend(probabilities[:, 1].cpu().numpy())
            
            accuracy = accuracy_score(all_labels, all_predictions)
            auc = roc_auc_score(all_labels, all_probabilities)
            
            comparison_results[model_name] = {
                'accuracy': accuracy,
                'auc': auc,
                'type': 'multimodal' if 'multimodal' in model_name else 'image_only'
            }
        
        return comparison_results

def create_and_evaluate_ensemble():
    """Main function to create and evaluate ensemble"""
    print("🏥 Medical Image Classification - Step 4: Create and Test Ensemble")
    print("=" * 70)
    
    # Setup device
    device = torch.device(GPU_CONFIG['device'] if torch.cuda.is_available() else 'cpu')
    logger.info(f"🖥️  Using device: {device}")
    
    # Create ensemble creator
    creator = EnsembleCreator(device)
    
    # Find and load trained models
    model_paths = creator.find_trained_models()
    models = creator.load_individual_models()
    
    if len(models) == 0:
        raise RuntimeError("No models loaded. Please train models first.")
    
    # Create test data loader
    logger.info("📊 Creating test data loader...")
    train_loader, val_loader, test_loader = create_preprocessed_data_loaders()
    
    logger.info(f"📈 Test samples: {len(test_loader.dataset)}")
    
    # Compare individual models
    comparison_results = creator.compare_models(test_loader)
    
    # Test different ensemble methods
    ensemble_results = {}
    
    for method in ENSEMBLE_CONFIG['methods']:
        print(f"\n🚀 Testing {method} ensemble...")
        
        try:
            # Create ensemble
            ensemble = creator.create_ensemble(method)
            
            # Evaluate ensemble
            results = creator.evaluate_ensemble(ensemble, test_loader, method)
            
            # Create visualizations
            creator.create_visualizations(results, method)
            
            # Save results
            creator.save_results(results, method)
            
            ensemble_results[method] = results
            
            print(f"✅ {method} ensemble completed!")
            print(f"   Accuracy: {results['accuracy']:.4f}")
            print(f"   AUC: {results['auc']:.4f}")
            
        except Exception as e:
            logger.error(f"❌ Failed to create {method} ensemble: {e}")
            continue
    
    # Summary
    print(f"\n📊 Model Comparison Summary:")
    print("=" * 50)
    
    # Individual models
    for model_name, results in comparison_results.items():
        print(f"   {model_name}: Accuracy = {results['accuracy']:.4f}, AUC = {results['auc']:.4f}")
    
    print("\n🎯 Ensemble Results:")
    for method, results in ensemble_results.items():
        print(f"   {method}: Accuracy = {results['accuracy']:.4f}, AUC = {results['auc']:.4f}")
    
    # Find best model
    all_results = {**comparison_results, **{f"ensemble_{method}": results for method, results in ensemble_results.items()}}
    best_model = max(all_results.items(), key=lambda x: x[1]['auc'])
    
    print(f"\n🏆 Best Model: {best_model[0]} (AUC: {best_model[1]['auc']:.4f})")
    
    return ensemble_results, comparison_results

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Create and evaluate ensemble models')
    parser.add_argument('--method', type=str, choices=['voting', 'averaging', 'weighted', 'stacking', 'all'], 
                       default='all', help='Ensemble method to test')
    parser.add_argument('--compare-only', action='store_true', help='Only compare individual models')
    
    args = parser.parse_args()
    
    if args.compare_only:
        # Only compare individual models
        device = torch.device(GPU_CONFIG['device'] if torch.cuda.is_available() else 'cpu')
        creator = EnsembleCreator(device)
        
        model_paths = creator.find_trained_models()
        models = creator.load_individual_models()
        
        train_loader, val_loader, test_loader = create_preprocessed_data_loaders()
        comparison_results = creator.compare_models(test_loader)
        
        print("\n📊 Individual Model Comparison:")
        for model_name, results in comparison_results.items():
            print(f"   {model_name}: Accuracy = {results['accuracy']:.4f}, AUC = {results['auc']:.4f}")
    else:
        create_and_evaluate_ensemble()

if __name__ == "__main__":
    main()
