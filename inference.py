#!/usr/bin/env python3
"""
Model Inference Script for Binary Medical Image Classification

This script:
1. Loads trained binary classifier
2. Performs inference on new images
3. Provides confidence scores and predictions
4. Supports batch processing and single image inference
5. Exports model to ONNX for deployment
"""

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import os
import json
import argparse
import logging
from pathlib import Path
import time
from tqdm import tqdm

from models import create_model
from config import (
    TRAINING_CONFIG, GPU_CONFIG, CHECKPOINTS_DIR, RESULTS_DIR,
    MODEL_CONFIGS, BINARY_CONFIG, PREPROCESSING_CONFIG
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BinaryMedicalInference:
    """Binary Medical Image Classification Inference Engine"""
    
    def __init__(self, checkpoint_path, device=None, model_name=None):
        self.device = device or torch.device(GPU_CONFIG['device'])
        self.model_name = model_name or "binary_classifier"
        
        # Load model
        self.model = self._load_model(checkpoint_path)
        
        # Setup preprocessing
        self._setup_preprocessing()
        
        logger.info(f"🎯 Binary Medical Inference initialized")
        logger.info(f"📱 Device: {self.device}")
        logger.info(f"🏷️ Model: {self.model_name}")
    
    def _load_model(self, checkpoint_path):
        """Load model from checkpoint"""
        # Create model architecture
        model_config = MODEL_CONFIGS['binary_classifier']
        model = create_model(
            model_type=model_config['model_type'],
            num_classes=model_config['num_classes'],
            dropout_rate=model_config['dropout_rate'],
            pretrained=False  # Don't use pretrained for inference
        )
        
        # Load checkpoint
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"📂 Model loaded from: {checkpoint_path}")
            
            if 'best_accuracy' in checkpoint:
                logger.info(f"🏆 Best training accuracy: {checkpoint['best_accuracy']:.2f}%")
        else:
            logger.error(f"❌ Checkpoint not found: {checkpoint_path}")
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        model.to(self.device)
        model.eval()
        
        return model
    
    def _setup_preprocessing(self):
        """Setup image preprocessing"""
        self.image_size = PREPROCESSING_CONFIG['image_size']
        self.normalize_mean = PREPROCESSING_CONFIG['normalize_mean']
        self.normalize_std = PREPROCESSING_CONFIG['normalize_std']
        
        logger.info(f"🖼️ Image preprocessing: {self.image_size}")
    
    def _preprocess_image(self, image_path):
        """Preprocess single image for inference"""
        try:
            # Load image
            with Image.open(image_path) as img:
                # Convert to RGB if needed
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Resize image
                img = img.resize(self.image_size, Image.LANCZOS)
                
                # Convert to numpy array
                img_array = np.array(img, dtype=np.float32) / 255.0
                
                # Normalize
                img_array = (img_array - np.array(self.normalize_mean)) / np.array(self.normalize_std)
                
                # Convert to tensor and add batch dimension
                img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)
                
                return img_tensor
                
        except Exception as e:
            logger.error(f"❌ Error preprocessing image {image_path}: {e}")
            return None
    
    def _is_valid_image(self, image_path):
        """Check if image is valid (not blank)"""
        try:
            with Image.open(image_path) as img:
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                img_array = np.array(img)
                
                # Check if all pixels are the same color (blank)
                if len(img_array.shape) == 3:
                    if np.all(img_array == img_array[0, 0]):
                        return False
                    
                    # Check variance - very low variance indicates blank image
                    variance = np.var(img_array)
                    if variance < 10:
                        return False
                else:
                    if np.all(img_array == img_array[0, 0]):
                        return False
                    variance = np.var(img_array)
                    if variance < 10:
                        return False
                
                return True
        except Exception as e:
            logger.warning(f"⚠️ Error checking image {image_path}: {e}")
            return False
    
    def predict_single(self, image_path, return_confidence=True):
        """Predict single image"""
        # Check if image is valid
        if not self._is_valid_image(image_path):
            return {
                'image_path': image_path,
                'prediction': 'INVALID_IMAGE',
                'confidence': 0.0,
                'probability_negative': 0.0,
                'probability_positive': 0.0,
                'error': 'Image appears to be blank or corrupted'
            }
        
        # Preprocess image
        img_tensor = self._preprocess_image(image_path)
        if img_tensor is None:
            return {
                'image_path': image_path,
                'prediction': 'PREPROCESSING_ERROR',
                'confidence': 0.0,
                'probability_negative': 0.0,
                'probability_positive': 0.0,
                'error': 'Failed to preprocess image'
            }
        
        # Move to device
        img_tensor = img_tensor.to(self.device)
        
        # Inference
        with torch.no_grad():
            start_time = time.time()
            outputs = self.model(img_tensor)
            inference_time = time.time() - start_time
            
            # Get probabilities
            probabilities = torch.softmax(outputs, dim=1)
            prob_negative = probabilities[0, 0].item()
            prob_positive = probabilities[0, 1].item()
            
            # Get prediction
            predicted_class = torch.argmax(outputs, dim=1).item()
            prediction = 'POSITIVE' if predicted_class == 1 else 'NEGATIVE'
            
            # Calculate confidence
            confidence = max(prob_negative, prob_positive)
        
        result = {
            'image_path': image_path,
            'prediction': prediction,
            'confidence': confidence,
            'probability_negative': prob_negative,
            'probability_positive': prob_positive,
            'inference_time_ms': inference_time * 1000,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        if return_confidence:
            return result
        else:
            return prediction, confidence
    
    def predict_batch(self, image_paths, batch_size=32):
        """Predict batch of images"""
        logger.info(f"🔍 Predicting {len(image_paths)} images in batches of {batch_size}")
        
        results = []
        
        # Process in batches
        for i in tqdm(range(0, len(image_paths), batch_size), desc="Processing batches"):
            batch_paths = image_paths[i:i + batch_size]
            batch_images = []
            valid_indices = []
            
            # Preprocess batch
            for j, path in enumerate(batch_paths):
                if self._is_valid_image(path):
                    img_tensor = self._preprocess_image(path)
                    if img_tensor is not None:
                        batch_images.append(img_tensor)
                        valid_indices.append(j)
            
            if not batch_images:
                # All images in batch are invalid
                for path in batch_paths:
                    results.append({
                        'image_path': path,
                        'prediction': 'INVALID_IMAGE',
                        'confidence': 0.0,
                        'probability_negative': 0.0,
                        'probability_positive': 0.0,
                        'error': 'Image appears to be blank or corrupted'
                    })
                continue
            
            # Stack batch
            batch_tensor = torch.cat(batch_images, dim=0).to(self.device)
            
            # Inference
            with torch.no_grad():
                start_time = time.time()
                outputs = self.model(batch_tensor)
                inference_time = time.time() - start_time
                
                # Get probabilities
                probabilities = torch.softmax(outputs, dim=1)
                
                # Process results
                for k, (output, prob) in enumerate(zip(outputs, probabilities)):
                    prob_negative = prob[0].item()
                    prob_positive = prob[1].item()
                    
                    predicted_class = torch.argmax(output).item()
                    prediction = 'POSITIVE' if predicted_class == 1 else 'NEGATIVE'
                    confidence = max(prob_negative, prob_positive)
                    
                    results.append({
                        'image_path': batch_paths[valid_indices[k]],
                        'prediction': prediction,
                        'confidence': confidence,
                        'probability_negative': prob_negative,
                        'probability_positive': prob_positive,
                        'inference_time_ms': inference_time * 1000 / len(batch_images),
                        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
                    })
            
            # Add invalid images
            for j, path in enumerate(batch_paths):
                if j not in valid_indices:
                    results.append({
                        'image_path': path,
                        'prediction': 'INVALID_IMAGE',
                        'confidence': 0.0,
                        'probability_negative': 0.0,
                        'probability_positive': 0.0,
                        'error': 'Image appears to be blank or corrupted'
                    })
        
        return results
    
    def predict_directory(self, directory_path, batch_size=32, file_extensions=None):
        """Predict all images in a directory"""
        if file_extensions is None:
            file_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        
        # Find all image files
        image_paths = []
        for ext in file_extensions:
            image_paths.extend(Path(directory_path).glob(f"*{ext}"))
            image_paths.extend(Path(directory_path).glob(f"*{ext.upper()}"))
        
        image_paths = [str(p) for p in image_paths]
        
        logger.info(f"📁 Found {len(image_paths)} images in {directory_path}")
        
        if len(image_paths) == 0:
            logger.warning("⚠️ No images found in directory")
            return []
        
        return self.predict_batch(image_paths, batch_size)
    
    def save_results(self, results, output_path):
        """Save prediction results to file"""
        # Create output directory
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save as JSON
        if output_path.endswith('.json'):
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
        else:
            # Save as CSV
            import pandas as pd
            df = pd.DataFrame(results)
            df.to_csv(output_path, index=False)
        
        logger.info(f"💾 Results saved to: {output_path}")
    
    def export_to_onnx(self, output_path, input_size=(1, 3, 224, 224)):
        """Export model to ONNX format for deployment"""
        logger.info("📦 Exporting model to ONNX format...")
        
        # Create dummy input
        dummy_input = torch.randn(input_size).to(self.device)
        
        # Export to ONNX
        torch.onnx.export(
            self.model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        
        logger.info(f"📦 ONNX model saved to: {output_path}")
        
        # Test ONNX model
        try:
            import onnx
            onnx_model = onnx.load(output_path)
            onnx.checker.check_model(onnx_model)
            logger.info("✅ ONNX model validation passed")
        except ImportError:
            logger.warning("⚠️ ONNX not installed, skipping validation")
        except Exception as e:
            logger.error(f"❌ ONNX model validation failed: {e}")
    
    def get_model_info(self):
        """Get model information"""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        info = {
            'model_name': self.model_name,
            'model_type': MODEL_CONFIGS['binary_classifier']['model_type'],
            'num_classes': MODEL_CONFIGS['binary_classifier']['num_classes'],
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': total_params * 4 / 1024 / 1024,
            'device': str(self.device),
            'image_size': self.image_size
        }
        
        return info

def main():
    """Main inference function"""
    parser = argparse.ArgumentParser(description='Binary medical image classification inference')
    parser.add_argument('--checkpoint', type=str, 
                       default=os.path.join(CHECKPOINTS_DIR, 'binary_classifier_best.pth'),
                       help='Path to model checkpoint')
    parser.add_argument('--input', type=str, required=True,
                       help='Input image path or directory')
    parser.add_argument('--output', type=str, 
                       help='Output file path for results')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for processing')
    parser.add_argument('--export-onnx', type=str,
                       help='Export model to ONNX format')
    
    args = parser.parse_args()
    
    # Initialize inference engine
    inference_engine = BinaryMedicalInference(args.checkpoint)
    
    # Print model info
    model_info = inference_engine.get_model_info()
    logger.info(f"📊 Model Information:")
    logger.info(f"   Model: {model_info['model_type']}")
    logger.info(f"   Parameters: {model_info['total_parameters']:,}")
    logger.info(f"   Size: {model_info['model_size_mb']:.2f} MB")
    
    # Export to ONNX if requested
    if args.export_onnx:
        inference_engine.export_to_onnx(args.export_onnx)
    
    # Run inference
    if os.path.isfile(args.input):
        # Single image
        logger.info(f"🖼️ Processing single image: {args.input}")
        result = inference_engine.predict_single(args.input)
        results = [result]
    elif os.path.isdir(args.input):
        # Directory
        logger.info(f"📁 Processing directory: {args.input}")
        results = inference_engine.predict_directory(args.input, args.batch_size)
    else:
        logger.error(f"❌ Input path not found: {args.input}")
        return
    
    # Print results summary
    valid_results = [r for r in results if r['prediction'] not in ['INVALID_IMAGE', 'PREPROCESSING_ERROR']]
    if valid_results:
        positive_count = sum(1 for r in valid_results if r['prediction'] == 'POSITIVE')
        negative_count = sum(1 for r in valid_results if r['prediction'] == 'NEGATIVE')
        avg_confidence = np.mean([r['confidence'] for r in valid_results])
        
        logger.info(f"📊 Inference Results Summary:")
        logger.info(f"   Total images processed: {len(results)}")
        logger.info(f"   Valid images: {len(valid_results)}")
        logger.info(f"   POSITIVE predictions: {positive_count}")
        logger.info(f"   NEGATIVE predictions: {negative_count}")
        logger.info(f"   Average confidence: {avg_confidence:.4f}")
    
    # Save results if output path provided
    if args.output:
        inference_engine.save_results(results, args.output)
    
    return results

if __name__ == "__main__":
    results = main()
