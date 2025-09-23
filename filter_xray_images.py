#!/usr/bin/env python3
"""
X-ray Image Filter - Improved Version
Removes document/report images from filtered_images folders, keeping only X-ray images
Uses simple image analysis and optional OCR (more conservative approach)

Required packages:
pip install opencv-python Pillow numpy tqdm

Optional OCR (if available):
pip install pytesseract
For tesseract OCR:
Ubuntu/Debian: sudo apt-get install tesseract-ocr
CentOS/RHEL: sudo yum install tesseract
"""

import os
import shutil
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import logging
from tqdm import tqdm
import argparse
import time

# Try to import pytesseract, but don't fail if not available
try:
    import pytesseract
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False
    print("⚠️ pytesseract not available - using image analysis only")

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class XrayImageFilter:
    """Filter to remove document/report images, keeping only X-ray images"""
    
    def __init__(self, filtered_images_dir="filtered_images"):
        self.filtered_images_dir = Path(filtered_images_dir)
        self.positive_dir = self.filtered_images_dir / "positive_images"
        self.negative_dir = self.filtered_images_dir / "negative_images"
        # Note: We don't filter doubt_images as they might contain mixed content
        
        # Statistics
        self.stats = {
            'total_processed': 0,
            'xray_kept': 0,
            'documents_removed': 0,
            'errors': 0
        }
        
        logger.info(f"🔧 X-ray Image Filter initialized")
        logger.info(f"📁 Filtering directory: {self.filtered_images_dir}")
    
    def is_xray_image(self, image_path):
        """Determine if image is an X-ray (not a document/report) - Conservative approach"""
        try:
            # Check 0: Skip known X-ray UID patterns (already confirmed X-rays)
            filename = image_path.name
            uid = filename.replace('.jpg', '').replace('.jpeg', '').replace('.png', '')
            
            # Known X-ray UID patterns (skip analysis for these)
            known_xray_patterns = [
                '1.2.840.113619',  # Standard X-rays (40.2% of dataset)
                '1.2.392.200036'   # Alternative X-rays (14.6% of dataset)
            ]
            
            # If filename starts with known X-ray pattern, skip analysis
            for pattern in known_xray_patterns:
                if uid.startswith(pattern):
                    logger.debug(f"✅ Known X-ray pattern ({pattern}): {filename}")
                    return True
            
            # Load image
            image = cv2.imread(str(image_path))
            if image is None:
                return False
            
            # Get image dimensions
            height, width = image.shape[:2]
            
            # Check 1: Image size - Very small images are likely not X-rays
            if width < 100 or height < 100:
                logger.debug(f"⚠️ Very small image: {width}x{height} - {image_path.name}")
                return False
            
            # Check 2: Aspect ratio - Extreme ratios suggest documents
            aspect_ratio = width / height
            if aspect_ratio > 5.0 or aspect_ratio < 0.2:  # Very wide or very tall
                logger.debug(f"⚠️ Extreme aspect ratio: {aspect_ratio:.2f} - {image_path.name}")
                return False
            
            # Check 3: Image characteristics analysis
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Calculate basic statistics
            mean_intensity = np.mean(gray)
            std_intensity = np.std(gray)
            
            # Check 4: Very uniform images (like documents) have low std
            if std_intensity < 10:  # Very uniform - likely document
                logger.debug(f"⚠️ Very uniform image (std: {std_intensity:.1f}): {image_path.name}")
                return False
            
            # Check 5: Edge analysis - X-rays have more complex structures
            edges = cv2.Canny(gray, 30, 100)
            edge_density = np.sum(edges > 0) / (width * height)
            
            if edge_density < 0.005:  # Very few edges - likely document
                logger.debug(f"⚠️ Very low edge density: {edge_density:.4f} - {image_path.name}")
                return False
            
            # Check 6: Optional OCR analysis (if available)
            if OCR_AVAILABLE:
                try:
                    # Convert to PIL for OCR
                    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                    
                    # Extract text using OCR
                    text = pytesseract.image_to_string(pil_image, config='--psm 6')
                    text = text.strip().lower()
                    
                    # Only remove if we find clear document patterns
                    document_patterns = [
                        'report', 'summary', 'findings', 'impression', 'conclusion',
                        'patient name', 'date of birth', 'physician', 'radiologist',
                        'study date', 'examination date', 'procedure', 'technique',
                        'clinical history', 'indication', 'complaint', 'symptoms',
                        'diagnosis', 'recommendation', 'follow up', 'next step'
                    ]
                    
                    # Count document patterns
                    doc_patterns_found = sum(1 for pattern in document_patterns if pattern in text)
                    
                    # Only remove if we find multiple document patterns AND lots of text
                    if doc_patterns_found >= 2 and len(text) > 200:
                        logger.debug(f"📄 Document detected ({doc_patterns_found} patterns, {len(text)} chars): {image_path.name}")
                        return False
                    
                    # Check for specific report phrases
                    report_phrases = [
                        'no acute fracture', 'no evidence of', 'unremarkable',
                        'normal study', 'no abnormality', 'within normal limits'
                    ]
                    
                    if any(phrase in text for phrase in report_phrases) and len(text) > 100:
                        logger.debug(f"📄 Report phrase detected: {image_path.name}")
                        return False
                        
                except Exception as e:
                    logger.debug(f"⚠️ OCR error for {image_path.name}: {e}")
                    # If OCR fails, assume it's an X-ray (conservative approach)
                    pass
            
            # Check 7: Color analysis - X-rays are typically grayscale
            # Convert to HSV to check saturation
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            saturation = hsv[:, :, 1]
            mean_saturation = np.mean(saturation)
            
            # Very colorful images might be documents/reports
            if mean_saturation > 50:  # High saturation suggests colored document
                logger.debug(f"⚠️ High color saturation: {mean_saturation:.1f} - {image_path.name}")
                return False
            
            # If all checks pass, it's likely an X-ray
            logger.debug(f"✅ X-ray confirmed: {image_path.name}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error processing {image_path.name}: {e}")
            # Conservative approach: if we can't analyze, keep the image
            return True
    
    def filter_folder(self, folder_path, folder_name):
        """Filter images in a specific folder"""
        if not folder_path.exists():
            logger.warning(f"⚠️ Folder does not exist: {folder_path}")
            return
        
        logger.info(f"🔍 Filtering {folder_name} folder: {folder_path}")
        
        # Get all image files
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            image_files.extend(folder_path.glob(ext))
        
        logger.info(f"📊 Found {len(image_files)} images in {folder_name}")
        
        # Process each image
        for image_path in tqdm(image_files, desc=f"Filtering {folder_name}"):
            self.stats['total_processed'] += 1
            
            try:
                if self.is_xray_image(image_path):
                    self.stats['xray_kept'] += 1
                    logger.debug(f"✅ Kept X-ray: {image_path.name}")
                else:
                    # Remove document/report image
                    image_path.unlink()
                    self.stats['documents_removed'] += 1
                    logger.info(f"🗑️ Removed document: {image_path.name}")
                    
            except Exception as e:
                self.stats['errors'] += 1
                logger.error(f"❌ Error processing {image_path.name}: {e}")
    
    def filter_all_folders(self):
        """Filter all folders"""
        logger.info("🚀 Starting X-ray image filtering")
        
        # Filter positive and negative folders
        self.filter_folder(self.positive_dir, "positive")
        self.filter_folder(self.negative_dir, "negative")
        
        # Print statistics
        logger.info("📊 FILTERING STATISTICS:")
        logger.info(f"   Total images processed: {self.stats['total_processed']}")
        logger.info(f"   X-ray images kept: {self.stats['xray_kept']}")
        logger.info(f"   Document images removed: {self.stats['documents_removed']}")
        logger.info(f"   Processing errors: {self.stats['errors']}")
        
        if self.stats['total_processed'] > 0:
            removal_rate = (self.stats['documents_removed'] / self.stats['total_processed']) * 100
            logger.info(f"   Document removal rate: {removal_rate:.1f}%")
        
        logger.info("🎉 X-ray filtering completed!")
    
    def create_summary_report(self):
        """Create a summary report"""
        report_path = self.filtered_images_dir / "xray_filtering_summary.txt"
        
        with open(report_path, 'w') as f:
            f.write("X-RAY IMAGE FILTERING SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Filtering Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Filtered Directory: {self.filtered_images_dir}\n\n")
            
            f.write("STATISTICS:\n")
            f.write(f"Total images processed: {self.stats['total_processed']}\n")
            f.write(f"X-ray images kept: {self.stats['xray_kept']}\n")
            f.write(f"Document images removed: {self.stats['documents_removed']}\n")
            f.write(f"Processing errors: {self.stats['errors']}\n")
            
            if self.stats['total_processed'] > 0:
                removal_rate = (self.stats['documents_removed'] / self.stats['total_processed']) * 100
                f.write(f"Document removal rate: {removal_rate:.1f}%\n")
            
            f.write("\nFILTERING CRITERIA (Conservative Approach):\n")
            f.write("- Known X-ray patterns: Skip analysis for 1.2.840.113619 and 1.2.392.200036\n")
            f.write("- Image size: Minimum 100x100 pixels\n")
            f.write("- Aspect ratio: Between 0.2 and 5.0\n")
            f.write("- Image uniformity: Standard deviation >10\n")
            f.write("- Edge density: >0.005\n")
            f.write("- Color saturation: <50 (grayscale preference)\n")
            f.write("- OCR analysis: Only removes clear documents with multiple patterns\n")
            f.write("- Conservative approach: Keeps images when in doubt\n")
        
        logger.info(f"📊 Summary report saved: {report_path}")

def main():
    """Main function"""
    import time
    
    parser = argparse.ArgumentParser(description='Filter X-ray images, remove documents/reports')
    parser.add_argument('--input-dir', default='filtered_images', 
                       help='Directory containing filtered images')
    parser.add_argument('--tesseract-path', default=None,
                       help='Path to tesseract executable (if not in PATH)')
    
    args = parser.parse_args()
    
    # Set tesseract path if provided
    if args.tesseract_path:
        pytesseract.pytesseract.tesseract_cmd = args.tesseract_path
    
    # Create filter instance
    xray_filter = XrayImageFilter(filtered_images_dir=args.input_dir)
    
    # Filter images
    xray_filter.filter_all_folders()
    
    # Create summary report
    xray_filter.create_summary_report()
    
    logger.info("✅ X-ray filtering completed successfully!")

if __name__ == "__main__":
    main()
