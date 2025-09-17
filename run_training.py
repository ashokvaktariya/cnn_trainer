#!/usr/bin/env python3
"""
Medical Image Classification Training Script
Trains 5 models: 2 image-only, 2 multimodal, 1 ensemble
"""

import torch
import argparse
import os
from train_models import train_all_models, evaluate_models

def main():
    parser = argparse.ArgumentParser(description='Train Medical Image Classification Models')
    parser.add_argument('--csv_file', type=str, default='dicom_image_url_file.csv',
                       help='Path to CSV file with medical data')
    parser.add_argument('--save_dir', type=str, default='./checkpoints',
                       help='Directory to save model checkpoints')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (cuda/cpu/auto)')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--evaluate_only', action='store_true',
                       help='Only evaluate trained models')
    
    args = parser.parse_args()
    
    # Setup device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print("🏥 Medical Image Classification Training Pipeline")
    print("=" * 60)
    print(f"📁 CSV File: {args.csv_file}")
    print(f"💾 Save Directory: {args.save_dir}")
    print(f"🔧 Device: {device}")
    print(f"📦 Batch Size: {args.batch_size}")
    print(f"🔄 Epochs: {args.num_epochs}")
    print("=" * 60)
    
    # Check if CSV file exists
    if not os.path.exists(args.csv_file):
        print(f"❌ Error: CSV file '{args.csv_file}' not found!")
        return
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    if args.evaluate_only:
        # Only evaluate existing models
        print("📊 Evaluating existing models...")
        
        model_paths = {
            'image_densenet': f"{args.save_dir}/image_densenet_best.pth",
            'image_efficientnet': f"{args.save_dir}/image_efficientnet_best.pth",
            'multimodal_densenet': f"{args.save_dir}/multimodal_densenet_best.pth",
            'multimodal_efficientnet': f"{args.save_dir}/multimodal_efficientnet_best.pth"
        }
        
        # Check if all models exist
        missing_models = [name for name, path in model_paths.items() if not os.path.exists(path)]
        if missing_models:
            print(f"❌ Error: Missing model files: {missing_models}")
            return
        
        results = evaluate_models(args.csv_file, model_paths, device)
        print("✅ Evaluation completed!")
        
    else:
        # Train all models
        print("🚀 Starting training pipeline...")
        
        try:
            trained_models, ensemble_trainer = train_all_models(
                csv_file=args.csv_file,
                device=device,
                save_dir=args.save_dir
            )
            
            print("\n🎉 Training completed successfully!")
            print(f"📁 Models saved in: {args.save_dir}")
            
            # Evaluate ensemble
            print("\n📊 Evaluating ensemble model...")
            model_paths = {
                'image_densenet': f"{args.save_dir}/image_densenet_best.pth",
                'image_efficientnet': f"{args.save_dir}/image_efficientnet_best.pth",
                'multimodal_densenet': f"{args.save_dir}/multimodal_densenet_best.pth",
                'multimodal_efficientnet': f"{args.save_dir}/multimodal_efficientnet_best.pth"
            }
            
            results = evaluate_models(args.csv_file, model_paths, device)
            
            print("\n📈 Final Results Summary:")
            print(f"Accuracy: {results['accuracy']:.4f}")
            print(f"AUC: {results['auc']:.4f}")
            
        except Exception as e:
            print(f"❌ Training failed: {e}")
            return

if __name__ == "__main__":
    main()
