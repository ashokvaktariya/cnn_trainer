#!/usr/bin/env python3
"""
Simple X-ray Image Classifier for Sampledata
Tests on local sampledata folder first
"""

import os
import json
import shutil
from pathlib import Path
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModel
import logging
from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SimpleXrayClassifier:
    def __init__(self, sampledata_dir="sampledata", output_dir="test_results"):
        """
        Simple X-ray classifier for sampledata
        
        Args:
            sampledata_dir: Directory containing sampledata
            output_dir: Output directory for results
        """
        self.sampledata_dir = Path(sampledata_dir)
        self.output_dir = Path(output_dir)
        
        # Create output directories
        self.xray_dir = self.output_dir / "xray_images"
        self.non_xray_dir = self.output_dir / "non_xray_images"
        
        self.xray_dir.mkdir(parents=True, exist_ok=True)
        self.non_xray_dir.mkdir(parents=True, exist_ok=True)
        
        # Load model
        logger.info("Loading Google/medsiglip-448 model...")
        self.processor = AutoProcessor.from_pretrained("google/medsiglip-448")
        self.model = AutoModel.from_pretrained("google/medsiglip-448")
        
        # Move to GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        
        logger.info(f"Model loaded on device: {self.device}")
    
    def get_all_images(self):
        """Get all image files from sampledata"""
        image_files = []
        
        for root, dirs, files in os.walk(self.sampledata_dir):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    image_files.append(os.path.join(root, file))
        
        logger.info(f"Found {len(image_files)} images in sampledata")
        return image_files
    
    def classify_image(self, image_path):
        """
        Simple classification: Is this an X-ray image?
        
        Returns:
            dict: {'is_xray': bool, 'confidence': float, 'predictions': list}
        """
        try:
            # Load image
            image = Image.open(image_path).convert('RGB')
            
            # Simple text prompts
            text_prompts = [
                "X-ray image of bone",
                "Medical X-ray radiograph", 
                "Bone X-ray scan",
                "Radiological image",
                "Summary report",
                "Text document",
                "Non-medical image"
            ]
            
            # Process
            inputs = self.processor(
                text=text_prompts,
                images=image,
                return_tensors="pt",
                padding=True
            ).to(self.device)
            
            # Get predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits_per_image = outputs.logits_per_image
                probs = torch.softmax(logits_per_image, dim=-1)
            
            # Get top 3 predictions
            top_probs, top_indices = torch.topk(probs, k=3)
            
            results = {
                'is_xray': False,
                'confidence': 0.0,
                'predictions': []
            }
            
            for i in range(3):
                idx = top_indices[0][i].item()
                prob = top_probs[0][i].item()
                text = text_prompts[idx]
                
                results['predictions'].append({
                    'text': text,
                    'confidence': prob
                })
                
                # Check if it's X-ray (first 4 prompts)
                if idx < 4 and prob > 0.25:  # Lower threshold for testing
                    results['is_xray'] = True
                    results['confidence'] = max(results['confidence'], prob)
            
            return results
            
        except Exception as e:
            logger.error(f"Error classifying {image_path}: {e}")
            return {'is_xray': False, 'confidence': 0.0, 'error': str(e)}
    
    def process_images(self):
        """Process all images in sampledata"""
        logger.info("🚀 Starting simple X-ray classification...")
        
        # Get all images
        image_files = self.get_all_images()
        
        if not image_files:
            logger.error("No images found in sampledata!")
            return
        
        # Process each image
        xray_count = 0
        non_xray_count = 0
        results = []
        
        logger.info(f"Processing {len(image_files)} images...")
        
        for image_path in tqdm(image_files, desc="Classifying images"):
            filename = os.path.basename(image_path)
            
            # Classify
            classification = self.classify_image(image_path)
            
            # Copy to appropriate folder
            if classification['is_xray']:
                dest_path = self.xray_dir / filename
                shutil.copy2(image_path, dest_path)
                xray_count += 1
                logger.info(f"✓ X-ray: {filename} (conf: {classification['confidence']:.3f})")
            else:
                dest_path = self.non_xray_dir / filename
                shutil.copy2(image_path, dest_path)
                non_xray_count += 1
                logger.info(f"✗ Non-X-ray: {filename} (conf: {classification['confidence']:.3f})")
            
            # Store results
            results.append({
                'filename': filename,
                'original_path': image_path,
                'is_xray': classification['is_xray'],
                'confidence': classification['confidence'],
                'predictions': classification.get('predictions', [])
            })
        
        # Save results
        results_file = self.output_dir / "classification_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Print summary
        logger.info("=" * 50)
        logger.info("📊 CLASSIFICATION SUMMARY")
        logger.info("=" * 50)
        logger.info(f"Total images processed: {len(image_files)}")
        logger.info(f"X-ray images: {xray_count}")
        logger.info(f"Non-X-ray images: {non_xray_count}")
        logger.info(f"X-ray percentage: {(xray_count/len(image_files)*100):.1f}%")
        logger.info("")
        logger.info(f"📁 X-ray images saved to: {self.xray_dir}")
        logger.info(f"📁 Non-X-ray images saved to: {self.non_xray_dir}")
        logger.info(f"📄 Detailed results: {results_file}")
        
        return results

def main():
    print("🏥 Simple X-ray Classifier for Sampledata")
    print("=" * 50)
    
    # Check if sampledata exists
    if not os.path.exists("sampledata"):
        print("❌ sampledata folder not found!")
        print("Please make sure you're in the right directory")
        return
    
    print("📁 Input: sampledata/")
    print("📁 Output: test_results/")
    print()
    
    # Initialize classifier
    classifier = SimpleXrayClassifier()
    
    # Process images
    results = classifier.process_images()
    
    print("\n✅ Classification complete!")
    print("Check the results in test_results/ folder")

if __name__ == "__main__":
    main()
