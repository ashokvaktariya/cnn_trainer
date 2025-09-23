#!/usr/bin/env python3
"""
X-ray Image Filter
Removes document/report images from filtered_images folders, keeping only X-ray images
Uses OCR to detect text and filter out non-X-ray images

Required packages:
pip install opencv-python pytesseract Pillow numpy tqdm

For tesseract OCR:
Ubuntu/Debian: sudo apt-get install tesseract-ocr
CentOS/RHEL: sudo yum install tesseract
Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki
macOS: brew install tesseract
"""

import os
import shutil
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import pytesseract
import logging
from tqdm import tqdm
import argparse
import time

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
        """Determine if image is an X-ray (not a document/report)"""
        try:
            # Load image
            image = cv2.imread(str(image_path))
            if image is None:
                return False
            
            # Convert to PIL for OCR
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            
            # Get image dimensions
            height, width = image.shape[:2]
            
            # Check 1: Image size - X-rays are usually larger than documents
            if width < 200 or height < 200:
                logger.debug(f"⚠️ Small image size: {width}x{height} - {image_path.name}")
                return False
            
            # Check 2: Aspect ratio - X-rays have different aspect ratios than documents
            aspect_ratio = width / height
            if aspect_ratio > 3.0 or aspect_ratio < 0.3:  # Very wide or very tall
                logger.debug(f"⚠️ Unusual aspect ratio: {aspect_ratio:.2f} - {image_path.name}")
                return False
            
            # Check 3: OCR text detection
            try:
                # Extract text using OCR
                text = pytesseract.image_to_string(pil_image, config='--psm 6')
                text = text.strip().lower()
                
                # Check for document indicators
                document_indicators = [
                    'report', 'summary', 'findings', 'impression', 'conclusion',
                    'patient', 'date', 'time', 'physician', 'doctor', 'radiologist',
                    'study', 'examination', 'procedure', 'technique', 'history',
                    'clinical', 'indication', 'reason', 'complaint', 'symptoms',
                    'diagnosis', 'recommendation', 'follow', 'up', 'next',
                    'no acute', 'no evidence', 'unremarkable', 'normal',
                    'fracture', 'dislocation', 'abnormality', 'lesion',
                    'ap', 'pa', 'lateral', 'oblique', 'view', 'views'
                ]
                
                # Count document indicators
                doc_count = sum(1 for indicator in document_indicators if indicator in text)
                
                # If too many document indicators, likely a report
                if doc_count >= 3:
                    logger.debug(f"📄 Document detected ({doc_count} indicators): {image_path.name}")
                    return False
                
                # Check 4: Text density - documents have more text
                if len(text) > 100:  # Lots of text suggests document
                    logger.debug(f"📄 High text density ({len(text)} chars): {image_path.name}")
                    return False
                
                # Check 5: Look for specific patterns that indicate documents
                if any(pattern in text for pattern in ['no acute', 'no evidence', 'unremarkable', 'normal']):
                    if len(text) > 50:  # If it's a full sentence, likely a report
                        logger.debug(f"📄 Report pattern detected: {image_path.name}")
                        return False
                
            except Exception as e:
                logger.warning(f"⚠️ OCR error for {image_path.name}: {e}")
                # If OCR fails, assume it's an X-ray (better to keep than remove)
                pass
            
            # Check 6: Image characteristics - X-rays have specific properties
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Check for X-ray characteristics
            # X-rays typically have:
            # - High contrast between bone and soft tissue
            # - Specific intensity distribution
            # - Less uniform backgrounds than documents
            
            # Calculate image statistics
            mean_intensity = np.mean(gray)
            std_intensity = np.std(gray)
            
            # X-rays typically have moderate mean intensity and good contrast
            if mean_intensity < 30 or mean_intensity > 220:  # Too dark or too bright
                logger.debug(f"⚠️ Unusual intensity: {mean_intensity:.1f} - {image_path.name}")
                return False
            
            if std_intensity < 20:  # Too uniform (like documents)
                logger.debug(f"⚠️ Low contrast: {std_intensity:.1f} - {image_path.name}")
                return False
            
            # Check 7: Edge detection - X-rays have more complex edges
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / (width * height)
            
            if edge_density < 0.01:  # Very few edges (like documents)
                logger.debug(f"⚠️ Low edge density: {edge_density:.4f} - {image_path.name}")
                return False
            
            # If all checks pass, it's likely an X-ray
            logger.debug(f"✅ X-ray confirmed: {image_path.name}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error processing {image_path.name}: {e}")
            return False
    
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
            
            f.write("\nFILTERING CRITERIA:\n")
            f.write("- Image size: Minimum 200x200 pixels\n")
            f.write("- Aspect ratio: Between 0.3 and 3.0\n")
            f.write("- OCR text analysis: <3 document indicators\n")
            f.write("- Text density: <100 characters\n")
            f.write("- Image intensity: Between 30-220\n")
            f.write("- Contrast: Standard deviation >20\n")
            f.write("- Edge density: >0.01\n")
        
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
