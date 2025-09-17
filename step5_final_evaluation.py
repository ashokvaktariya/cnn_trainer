#!/usr/bin/env python3
"""
Step 5: Final Evaluation and Results Summary
Comprehensive evaluation of all models and generates final report
H200 GPU Server Optimized
"""

import torch
import numpy as np
import pandas as pd
import os
import json
import logging
import time
from pathlib import Path
from tqdm import tqdm
import argparse
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_curve, precision_recall_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

from config import (
    CHECKPOINTS_DIR, RESULTS_DIR, LOGS_DIR, GPU_CONFIG, TRAINING_CONFIG,
    MODEL_CONFIGS, EVALUATION_CONFIG, LOGGING_CONFIG
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
        logging.FileHandler(os.path.join(LOGS_DIR, "final_evaluation.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class FinalEvaluator:
    """Comprehensive final evaluation of all models"""
    
    def __init__(self, device=None):
        self.device = device or torch.device(GPU_CONFIG['device'])
        self.results = {}
        
        # Setup paths
        self.final_results_dir = os.path.join(RESULTS_DIR, "final_evaluation")
        os.makedirs(self.final_results_dir, exist_ok=True)
        
        logger.info(f"🔧 Initialized FinalEvaluator")
        logger.info(f"🖥️  Device: {self.device}")
    
    def load_all_models(self):
        """Load all trained models"""
        logger.info("📂 Loading all trained models...")
        
        models = {}
        model_names = ['image_densenet', 'image_efficientnet', 'multimodal_densenet', 'multimodal_efficientnet']
        
        for model_name in model_names:
            model_path = os.path.join(CHECKPOINTS_DIR, model_name, 'best.pth')
            if os.path.exists(model_path):
                try:
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
                    
                    model.load_state_dict(checkpoint['model_state_dict'])
                    model = model.to(self.device)
                    model.eval()
                    
                    models[model_name] = {
                        'model': model,
                        'auc': checkpoint.get('auc', 0.0),
                        'config': checkpoint.get('config', {}),
                        'type': 'multimodal' if 'multimodal' in model_name else 'image_only'
                    }
                    
                    logger.info(f"✅ Loaded {model_name} (AUC: {checkpoint.get('auc', 0.0):.4f})")
                    
                except Exception as e:
                    logger.error(f"❌ Failed to load {model_name}: {e}")
                    continue
            else:
                logger.warning(f"⚠️  Model not found: {model_name}")
        
        logger.info(f"📊 Loaded {len(models)} models")
        return models
    
    def evaluate_model(self, model, model_name, test_loader):
        """Evaluate a single model comprehensively"""
        logger.info(f"📊 Evaluating {model_name}...")
        
        model.eval()
        all_predictions = []
        all_labels = []
        all_probabilities = []
        all_logits = []
        
        inference_times = []
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc=f"Evaluating {model_name}"):
                start_time = time.time()
                
                images = batch['images'].to(self.device)
                text_input_ids = batch['text_input_ids'].to(self.device)
                text_attention_mask = batch['text_attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                if 'multimodal' in model_name:
                    outputs = model(images, text_input_ids, text_attention_mask)
                else:
                    outputs = model(images)
                
                inference_time = time.time() - start_time
                inference_times.append(inference_time)
                
                probabilities = torch.softmax(outputs, dim=1)
                predictions = torch.argmax(outputs, dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(probabilities[:, 1].cpu().numpy())
                all_logits.extend(outputs.cpu().numpy())
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_predictions)
        auc = roc_auc_score(all_labels, all_probabilities)
        precision = precision_score(all_labels, all_predictions)
        recall = recall_score(all_labels, all_predictions)
        f1 = f1_score(all_labels, all_predictions)
        
        # Classification report
        report = classification_report(all_labels, all_predictions, output_dict=True)
        
        # Confusion matrix
        cm = confusion_matrix(all_labels, all_predictions)
        
        # Inference time statistics
        avg_inference_time = np.mean(inference_times)
        std_inference_time = np.std(inference_times)
        
        results = {
            'model_name': model_name,
            'model_type': 'multimodal' if 'multimodal' in model_name else 'image_only',
            'accuracy': accuracy,
            'auc': auc,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'classification_report': report,
            'confusion_matrix': cm.tolist(),
            'predictions': all_predictions,
            'labels': all_labels,
            'probabilities': all_probabilities,
            'logits': all_logits,
            'inference_stats': {
                'avg_time': avg_inference_time,
                'std_time': std_inference_time,
                'total_samples': len(all_predictions)
            }
        }
        
        logger.info(f"✅ {model_name} evaluation completed:")
        logger.info(f"   Accuracy: {accuracy:.4f}")
        logger.info(f"   AUC: {auc:.4f}")
        logger.info(f"   F1: {f1:.4f}")
        logger.info(f"   Avg inference time: {avg_inference_time:.4f}s")
        
        return results
    
    def evaluate_ensemble_models(self, test_loader):
        """Evaluate ensemble models"""
        logger.info("📊 Evaluating ensemble models...")
        
        ensemble_results = {}
        
        # Check for ensemble results from step 4
        ensemble_dir = os.path.join(RESULTS_DIR, "ensemble")
        
        if os.path.exists(ensemble_dir):
            for result_file in os.listdir(ensemble_dir):
                if result_file.startswith('ensemble_results_') and result_file.endswith('.json'):
                    method = result_file.replace('ensemble_results_', '').replace('.json', '')
                    
                    try:
                        with open(os.path.join(ensemble_dir, result_file), 'r') as f:
                            results = json.load(f)
                        
                        ensemble_results[f"ensemble_{method}"] = {
                            'model_name': f"ensemble_{method}",
                            'model_type': 'ensemble',
                            'accuracy': results['accuracy'],
                            'auc': results['auc'],
                            'predictions': results['predictions'],
                            'labels': results['labels'],
                            'probabilities': results['probabilities']
                        }
                        
                        logger.info(f"✅ Loaded ensemble results for {method}")
                        
                    except Exception as e:
                        logger.error(f"❌ Failed to load ensemble results for {method}: {e}")
        
        return ensemble_results
    
    def create_comprehensive_report(self, individual_results, ensemble_results):
        """Create comprehensive evaluation report"""
        logger.info("📊 Creating comprehensive report...")
        
        # Combine all results
        all_results = {**individual_results, **ensemble_results}
        
        # Create summary dataframe
        summary_data = []
        for model_name, results in all_results.items():
            summary_data.append({
                'Model': model_name,
                'Type': results['model_type'],
                'Accuracy': results['accuracy'],
                'AUC': results['auc'],
                'Precision': results.get('precision', 0.0),
                'Recall': results.get('recall', 0.0),
                'F1-Score': results.get('f1_score', 0.0)
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df.sort_values('AUC', ascending=False)
        
        # Save summary
        summary_path = os.path.join(self.final_results_dir, 'model_comparison_summary.csv')
        summary_df.to_csv(summary_path, index=False)
        
        # Create detailed report
        report = {
            'evaluation_timestamp': datetime.now().isoformat(),
            'total_models_evaluated': len(all_results),
            'dataset_info': {
                'test_samples': len(individual_results[list(individual_results.keys())[0]]['predictions']),
                'config': TRAINING_CONFIG
            },
            'model_summary': summary_df.to_dict('records'),
            'best_models': {
                'overall_best': summary_df.iloc[0].to_dict(),
                'best_image_only': summary_df[summary_df['Type'] == 'image_only'].iloc[0].to_dict() if len(summary_df[summary_df['Type'] == 'image_only']) > 0 else None,
                'best_multimodal': summary_df[summary_df['Type'] == 'multimodal'].iloc[0].to_dict() if len(summary_df[summary_df['Type'] == 'multimodal']) > 0 else None,
                'best_ensemble': summary_df[summary_df['Type'] == 'ensemble'].iloc[0].to_dict() if len(summary_df[summary_df['Type'] == 'ensemble']) > 0 else None
            },
            'detailed_results': all_results
        }
        
        # Save detailed report
        report_path = os.path.join(self.final_results_dir, 'comprehensive_evaluation_report.json')
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"💾 Report saved to {report_path}")
        
        return summary_df, report
    
    def create_visualizations(self, summary_df, individual_results):
        """Create comprehensive visualizations"""
        logger.info("📊 Creating visualizations...")
        
        # Set style
        plt.style.use('seaborn-v0_8')
        
        # 1. Model Performance Comparison
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Accuracy comparison
        axes[0, 0].bar(range(len(summary_df)), summary_df['Accuracy'])
        axes[0, 0].set_title('Model Accuracy Comparison')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].set_xticks(range(len(summary_df)))
        axes[0, 0].set_xticklabels(summary_df['Model'], rotation=45, ha='right')
        
        # AUC comparison
        axes[0, 1].bar(range(len(summary_df)), summary_df['AUC'])
        axes[0, 1].set_title('Model AUC Comparison')
        axes[0, 1].set_ylabel('AUC')
        axes[0, 1].set_xticks(range(len(summary_df)))
        axes[0, 1].set_xticklabels(summary_df['Model'], rotation=45, ha='right')
        
        # F1-Score comparison
        axes[1, 0].bar(range(len(summary_df)), summary_df['F1-Score'])
        axes[1, 0].set_title('Model F1-Score Comparison')
        axes[1, 0].set_ylabel('F1-Score')
        axes[1, 0].set_xticks(range(len(summary_df)))
        axes[1, 0].set_xticklabels(summary_df['Model'], rotation=45, ha='right')
        
        # Model type distribution
        type_counts = summary_df['Type'].value_counts()
        axes[1, 1].pie(type_counts.values, labels=type_counts.index, autopct='%1.1f%%')
        axes[1, 1].set_title('Model Type Distribution')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.final_results_dir, 'model_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. ROC Curves for all models
        plt.figure(figsize=(12, 8))
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(individual_results)))
        
        for i, (model_name, results) in enumerate(individual_results.items()):
            fpr, tpr, _ = roc_curve(results['labels'], results['probabilities'])
            auc_score = results['auc']
            
            plt.plot(fpr, tpr, color=colors[i], lw=2, 
                    label=f'{model_name} (AUC = {auc_score:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves Comparison')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.final_results_dir, 'roc_curves_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Confusion Matrices
        n_models = len(individual_results)
        cols = min(3, n_models)
        rows = (n_models + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
        if rows == 1:
            axes = axes.reshape(1, -1)
        
        for i, (model_name, results) in enumerate(individual_results.items()):
            row = i // cols
            col = i % cols
            
            cm = np.array(results['confusion_matrix'])
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[row, col],
                       xticklabels=['Negative', 'Positive'], 
                       yticklabels=['Negative', 'Positive'])
            axes[row, col].set_title(f'{model_name}\n(Acc: {results["accuracy"]:.3f})')
            axes[row, col].set_ylabel('True Label')
            axes[row, col].set_xlabel('Predicted Label')
        
        # Hide empty subplots
        for i in range(n_models, rows * cols):
            row = i // cols
            col = i % cols
            axes[row, col].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.final_results_dir, 'confusion_matrices.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"✅ Visualizations saved to {self.final_results_dir}")
    
    def print_final_summary(self, summary_df, report):
        """Print final summary to console"""
        print("\n" + "="*80)
        print("🏥 MEDICAL IMAGE CLASSIFICATION - FINAL EVALUATION RESULTS")
        print("="*80)
        
        print(f"\n📊 Evaluation Summary:")
        print(f"   Total Models Evaluated: {report['total_models_evaluated']}")
        print(f"   Test Samples: {report['dataset_info']['test_samples']}")
        print(f"   Evaluation Time: {report['evaluation_timestamp']}")
        
        print(f"\n🏆 BEST MODELS:")
        print(f"   Overall Best: {report['best_models']['overall_best']['Model']} (AUC: {report['best_models']['overall_best']['AUC']:.4f})")
        
        if report['best_models']['best_image_only']:
            best_img = report['best_models']['best_image_only']
            print(f"   Best Image-Only: {best_img['Model']} (AUC: {best_img['AUC']:.4f})")
        
        if report['best_models']['best_multimodal']:
            best_mm = report['best_models']['best_multimodal']
            print(f"   Best Multimodal: {best_mm['Model']} (AUC: {best_mm['AUC']:.4f})")
        
        if report['best_models']['best_ensemble']:
            best_ens = report['best_models']['best_ensemble']
            print(f"   Best Ensemble: {best_ens['Model']} (AUC: {best_ens['AUC']:.4f})")
        
        print(f"\n📈 DETAILED RESULTS:")
        print(summary_df.to_string(index=False, float_format='%.4f'))
        
        print(f"\n📁 Results saved to: {self.final_results_dir}")
        print("="*80)

def run_final_evaluation():
    """Main function for final evaluation"""
    print("🏥 Medical Image Classification - Step 5: Final Evaluation")
    print("=" * 70)
    
    # Setup device
    device = torch.device(GPU_CONFIG['device'] if torch.cuda.is_available() else 'cpu')
    logger.info(f"🖥️  Using device: {device}")
    
    # Initialize evaluator
    evaluator = FinalEvaluator(device)
    
    # Load all models
    models = evaluator.load_all_models()
    
    if len(models) == 0:
        raise RuntimeError("No models found. Please train models first.")
    
    # Create test data loader
    logger.info("📊 Creating test data loader...")
    train_loader, val_loader, test_loader = create_preprocessed_data_loaders()
    
    logger.info(f"📈 Test samples: {len(test_loader.dataset)}")
    
    # Evaluate individual models
    individual_results = {}
    for model_name, model_info in models.items():
        results = evaluator.evaluate_model(model_info['model'], model_name, test_loader)
        individual_results[model_name] = results
    
    # Evaluate ensemble models
    ensemble_results = evaluator.evaluate_ensemble_models(test_loader)
    
    # Create comprehensive report
    summary_df, report = evaluator.create_comprehensive_report(individual_results, ensemble_results)
    
    # Create visualizations
    evaluator.create_visualizations(summary_df, individual_results)
    
    # Print final summary
    evaluator.print_final_summary(summary_df, report)
    
    return summary_df, report

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Final evaluation of all models')
    parser.add_argument('--skip-visualizations', action='store_true', help='Skip creating visualizations')
    parser.add_argument('--models', nargs='+', help='Specific models to evaluate')
    
    args = parser.parse_args()
    
    run_final_evaluation()

if __name__ == "__main__":
    main()
