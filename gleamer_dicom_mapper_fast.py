#!/usr/bin/env python3
"""
FAST GLEAMER-DICOM Mapper

Optimized version with performance improvements:
- Skip OCR analysis (major bottleneck)
- Skip image content analysis
- Focus on UID-based filtering and metadata labeling
- Process in batches
"""

import os
import json
import logging
import random
import argparse
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import pandas as pd

# Try to import pydicom, but don't fail if not available
try:
    import pydicom
    DICOM_AVAILABLE = True
except ImportError:
    DICOM_AVAILABLE = False
    print("⚠️ pydicom not available - DICOM features disabled")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('gleamer_dicom_mapper_fast.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class FastGleamerDicomMapper:
    """Fast GLEAMER-DICOM mapper with performance optimizations"""
    
    def __init__(self, gleamer_path, dicom_path, output_dir):
        self.gleamer_path = Path(gleamer_path)
        self.dicom_path = Path(dicom_path)
        self.output_dir = Path(output_dir)
        
        # Create output directory
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize data structures
        self.gleamer_images = []
        self.dicom_images = []
        self.mapped_pairs = []
        
        # Statistics
        self.stats = {
            'gleamer_images_found': 0,
            'dicom_images_found': 0,
            'successful_mappings': 0,
            'report_tags_found': 0,
            'positive_with_fracture': 0,
            'positive_without_fracture': 0,
            'negative_cases': 0,
            'summary_reports_filtered': 0,
            'errors': 0
        }
        
        # Known summary report patterns (UID-based filtering)
        self.summary_patterns = [
            '1.2.250.1',  # GLEAMER summary reports
            '1.2.840.10008.5.1.4.1.1.7'  # Secondary Capture SOP Class
        ]
        
        logger.info("🚀 Fast GLEAMER-DICOM Mapper initialized")
    
    def find_gleamer_images(self):
        """Find all GLEAMER images"""
        logger.info("🔍 Finding GLEAMER images...")
        
        if not self.gleamer_path.exists():
            logger.error(f"GLEAMER path does not exist: {self.gleamer_path}")
            return
        
        # Find all image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
        for ext in image_extensions:
            self.gleamer_images.extend(self.gleamer_path.glob(f'*{ext}'))
            self.gleamer_images.extend(self.gleamer_path.glob(f'*{ext.upper()}'))
        
        self.stats['gleamer_images_found'] = len(self.gleamer_images)
        logger.info(f"📊 Found {len(self.gleamer_images)} GLEAMER images")
    
    def find_dicom_images(self):
        """Find all DICOM images"""
        logger.info("🔍 Finding DICOM images...")
        
        if not self.dicom_path.exists():
            logger.error(f"DICOM path does not exist: {self.dicom_path}")
            return
        
        # Find all DICOM files
        dicom_extensions = ['.dcm', '.DCM']
        for ext in dicom_extensions:
            self.dicom_images.extend(self.dicom_path.glob(f'*{ext}'))
        
        self.stats['dicom_images_found'] = len(self.dicom_images)
        logger.info(f"📊 Found {len(self.dicom_images)} DICOM images")
    
    def extract_uid_from_filename(self, filename):
        """Extract UID from filename"""
        # Remove file extension
        name_without_ext = filename.rsplit('.', 1)[0]
        
        # Look for UID pattern (starts with 1.2.)
        parts = name_without_ext.split('_')
        for part in parts:
            if part.startswith('1.2.'):
                return part
        
        return None
    
    def is_summary_report_pattern(self, uid):
        """Check if UID matches known summary report patterns"""
        if not uid:
            return False
        for pattern in self.summary_patterns:
            if uid.startswith(pattern):
                return True
        return False
    
    def extract_dicom_metadata_fast(self, dicom_path):
        """Fast DICOM metadata extraction - only essential fields"""
        try:
            if not DICOM_AVAILABLE:
                return {'read_success': False, 'error': 'pydicom not available'}
            
            ds = pydicom.dcmread(str(dicom_path), stop_before_pixels=True)
            
            # Only extract essential fields for performance
            essential_tags = [
                'SOPInstanceUID', 'StudyDescription', 'SeriesDescription',
                'Findings', 'ClinicalIndication', 'Modality', 'BodyPartExamined'
            ]
            
            metadata = {'read_success': True}
            for tag in essential_tags:
                if hasattr(ds, tag):
                    try:
                        value = getattr(ds, tag)
                        if hasattr(value, '__iter__') and not isinstance(value, str):
                            metadata[tag] = [str(v) for v in value]
                        else:
                            metadata[tag] = str(value)
                    except Exception:
                        metadata[tag] = ''
                else:
                    metadata[tag] = ''
            
            return metadata
            
        except Exception as e:
            return {'read_success': False, 'error': str(e)}
    
    def determine_label_from_report(self, report_info):
        """Determine label from report information"""
        # Check for fracture indicators
        fracture_keywords = ['fracture', 'broken', 'crack', 'break', 'fissure', 'dislocation', 'avulsion']
        
        # Check StudyDescription
        study_desc = report_info['study_description'].lower()
        if any(keyword in study_desc for keyword in fracture_keywords):
            return 'POSITIVE'
        
        # Check Findings - HIGHEST PRIORITY (most reliable)
        findings = report_info['findings'].lower()
        
        # First check for explicit negative findings - THESE OVERRIDE EVERYTHING
        negative_indicators = ['no acute', 'no fracture', 'normal', 'unremarkable', 'negative for fracture', 'no evidence of fracture', 'no radiographic evidence', 'there is no evidence of']
        if any(indicator in findings for indicator in negative_indicators):
            return 'NEGATIVE'
        
        # Then check for positive findings
        if any(keyword in findings for keyword in fracture_keywords):
            return 'POSITIVE'
        
        # Check SeriesDescription - HIGH PRIORITY
        series_desc = report_info['series_description'].lower()
        
        # First check for explicit negative series - THESE OVERRIDE CLINICAL INDICATION
        if 'negative' in series_desc and 'fracture' in series_desc:
            return 'NEGATIVE'
        if 'normal' in series_desc:
            return 'NEGATIVE'
        
        # Then check for positive series
        if any(keyword in series_desc for keyword in fracture_keywords):
            return 'POSITIVE'
        
        # Check ClinicalIndication - ONLY if findings/series don't contradict
        clinical = report_info['clinical_indication'].lower()
        trauma_keywords = ['trauma', 'injury', 'injured', 'fall', 'accident', 'pain', 'hurt', 'damage']
        
        # Check Modality for X-ray specific terms
        modality = report_info['modality'].lower()
        if modality in ['cr', 'dx', 'xr']:  # Computed Radiography, Digital X-ray, X-ray
            # For X-rays, if we have trauma indicators, label as POSITIVE
            if any(keyword in clinical for keyword in trauma_keywords):
                return 'POSITIVE'
        
        # Check BodyPart for trauma-prone areas
        body_part = report_info['body_part'].lower()
        trauma_body_parts = ['wrist', 'ankle', 'hand', 'foot', 'elbow', 'knee', 'shoulder', 'hip', 'spine', 'rib']
        if any(part in body_part for part in trauma_body_parts):
            # If it's a trauma-prone body part and we have clinical indication, label as POSITIVE
            if any(keyword in clinical for keyword in trauma_keywords):
                return 'POSITIVE'
        
        # More lenient approach - if we have any trauma indication, label as POSITIVE
        if any(keyword in clinical for keyword in trauma_keywords):
            return 'POSITIVE'
        
        # Default to NEGATIVE if no clear indicators
        return 'NEGATIVE'
    
    def map_gleamer_to_dicom_fast(self):
        """Fast mapping without image analysis"""
        logger.info("🔗 Fast mapping GLEAMER images to DICOM images...")
        
        # Create UID to DICOM mapping
        dicom_uid_map = {}
        for dicom_path in tqdm(self.dicom_images, desc="Indexing DICOM files"):
            try:
                metadata = self.extract_dicom_metadata_fast(dicom_path)
                if metadata['read_success'] and 'SOPInstanceUID' in metadata:
                    uid = metadata['SOPInstanceUID']
                    dicom_uid_map[uid] = {
                        'path': dicom_path,
                        'metadata': metadata
                    }
            except Exception as e:
                logger.debug(f"Error indexing {dicom_path.name}: {e}")
        
        logger.info(f"📊 Indexed {len(dicom_uid_map)} DICOM files with UIDs")
        
        # Map GLEAMER images to DICOM (FAST - no image analysis)
        for gleamer_path in tqdm(self.gleamer_images, desc="Mapping GLEAMER images"):
            try:
                # Extract UID from GLEAMER filename
                gleamer_uid = self.extract_uid_from_filename(gleamer_path.name)
                
                # Skip summary reports based on UID pattern
                if self.is_summary_report_pattern(gleamer_uid):
                    self.stats['summary_reports_filtered'] += 1
                    continue
                
                if gleamer_uid and gleamer_uid in dicom_uid_map:
                    # Found matching DICOM
                    dicom_info = dicom_uid_map[gleamer_uid]
                    
                    mapping = {
                        'gleamer_path': str(gleamer_path),
                        'gleamer_filename': gleamer_path.name,
                        'gleamer_uid': gleamer_uid,
                        'dicom_path': str(dicom_info['path']),
                        'dicom_filename': dicom_info['path'].name,
                        'dicom_metadata': dicom_info['metadata'],
                        'mapping_success': True
                    }
                    
                    self.mapped_pairs.append(mapping)
                    self.stats['successful_mappings'] += 1
                    
                else:
                    # No matching DICOM found
                    mapping = {
                        'gleamer_path': str(gleamer_path),
                        'gleamer_filename': gleamer_path.name,
                        'gleamer_uid': gleamer_uid,
                        'dicom_path': None,
                        'dicom_filename': None,
                        'dicom_metadata': None,
                        'mapping_success': False
                    }
                    
                    self.mapped_pairs.append(mapping)
                    
            except Exception as e:
                logger.error(f"Error mapping {gleamer_path.name}: {e}")
                self.stats['errors'] += 1
        
        logger.info(f"✅ Successfully mapped {self.stats['successful_mappings']} pairs")
        logger.info(f"📊 Summary reports filtered: {self.stats['summary_reports_filtered']}")
    
    def find_report_tags(self):
        """Find report tags for each image"""
        logger.info("📋 Finding report tags...")
        
        for mapping in tqdm(self.mapped_pairs, desc="Finding report tags"):
            if mapping['mapping_success'] and mapping['dicom_metadata']:
                metadata = mapping['dicom_metadata']
                
                # Extract report information
                report_info = {
                    'study_description': metadata.get('StudyDescription', ''),
                    'series_description': metadata.get('SeriesDescription', ''),
                    'findings': metadata.get('Findings', ''),
                    'clinical_indication': metadata.get('ClinicalIndication', ''),
                    'modality': metadata.get('Modality', ''),
                    'body_part': metadata.get('BodyPartExamined', '')
                }
                
                # Determine label based on report content
                label = self.determine_label_from_report(report_info)
                
                # Debug: Log first few examples
                if self.stats['report_tags_found'] < 5:
                    logger.info(f"🔍 Sample {self.stats['report_tags_found'] + 1}:")
                    logger.info(f"   StudyDescription: '{report_info['study_description']}'")
                    logger.info(f"   Findings: '{report_info['findings']}'")
                    logger.info(f"   ClinicalIndication: '{report_info['clinical_indication']}'")
                    logger.info(f"   SeriesDescription: '{report_info['series_description']}'")
                    logger.info(f"   Modality: '{report_info['modality']}'")
                    logger.info(f"   BodyPart: '{report_info['body_part']}'")
                    logger.info(f"   → Label: {label}")
                
                mapping['report_info'] = report_info
                mapping['label'] = label
                
                self.stats['report_tags_found'] += 1
                
                if label == 'POSITIVE':
                    self.stats['positive_with_fracture'] += 1
                elif label == 'NEGATIVE':
                    self.stats['negative_cases'] += 1
                else:
                    self.stats['positive_without_fracture'] += 1
    
    def handle_positive_without_fracture(self):
        """Handle positive records with no fracture images"""
        logger.info("🔍 Handling positive records with no fracture images...")
        
        # This is simplified - just log the issue
        positive_without_fracture = 0
        for mapping in self.mapped_pairs:
            if mapping.get('label') == 'POSITIVE':
                # Check if this might be a report
                report_info = mapping.get('report_info', {})
                findings = report_info.get('findings', '').lower()
                
                # If findings suggest no fracture, re-label
                if 'no acute' in findings or 'no fracture' in findings or 'normal' in findings:
                    mapping['label'] = 'NEGATIVE'
                    mapping['relabeled'] = True
                    mapping['relabel_reason'] = 'Positive label but findings indicate no fracture'
                    positive_without_fracture += 1
        
        logger.info(f"⚠️ Found {positive_without_fracture} positive cases that might be reports")
        logger.info(f"✅ Re-labeled {positive_without_fracture} cases as NEGATIVE")
    
    def create_dataset_splits(self, train_ratio=0.9, val_ratio=0.1):
        """Create 90/10 train/validation splits"""
        logger.info("📊 Creating 90/10 train/validation splits...")
        
        # Separate by label
        positive_files = [m for m in self.mapped_pairs if m.get('label') == 'POSITIVE']
        negative_files = [m for m in self.mapped_pairs if m.get('label') == 'NEGATIVE']
        
        logger.info(f"Positive files: {len(positive_files)}")
        logger.info(f"Negative files: {len(negative_files)}")
        
        # Shuffle files
        random.shuffle(positive_files)
        random.shuffle(negative_files)
        
        # Calculate split sizes
        pos_train_size = int(len(positive_files) * train_ratio)
        neg_train_size = int(len(negative_files) * train_ratio)
        
        # Split files
        pos_train = positive_files[:pos_train_size]
        pos_val = positive_files[pos_train_size:]
        
        neg_train = negative_files[:neg_train_size]
        neg_val = negative_files[neg_train_size:]
        
        # Create splits
        splits = {
            'train': {'positive': pos_train, 'negative': neg_train},
            'val': {'positive': pos_val, 'negative': neg_val}
        }
        
        # Save split information
        split_info = {
            'total_files': len(self.mapped_pairs),
            'positive_files': len(positive_files),
            'negative_files': len(negative_files),
            'train_positive': len(pos_train),
            'train_negative': len(neg_train),
            'val_positive': len(pos_val),
            'val_negative': len(neg_val),
            'train_ratio': train_ratio,
            'val_ratio': val_ratio
        }
        
        # Save splits to files
        with open(self.output_dir / 'train_positive.json', 'w') as f:
            json.dump(pos_train, f, indent=2)
        
        with open(self.output_dir / 'train_negative.json', 'w') as f:
            json.dump(neg_train, f, indent=2)
        
        with open(self.output_dir / 'val_positive.json', 'w') as f:
            json.dump(pos_val, f, indent=2)
        
        with open(self.output_dir / 'val_negative.json', 'w') as f:
            json.dump(neg_val, f, indent=2)
        
        with open(self.output_dir / 'split_info.json', 'w') as f:
            json.dump(split_info, f, indent=2)
        
        logger.info("📊 Dataset splits created:")
        logger.info(f"  Train: {len(pos_train) + len(neg_train)} files")
        logger.info(f"  Val: {len(pos_val) + len(neg_val)} files")
    
    def save_results(self):
        """Save mapping results"""
        logger.info("💾 Saving results...")
        
        # Save all mappings
        with open(self.output_dir / 'mapped_pairs.json', 'w') as f:
            json.dump(self.mapped_pairs, f, indent=2)
        
        # Save statistics
        with open(self.output_dir / 'mapping_stats.json', 'w') as f:
            json.dump(self.stats, f, indent=2)
        
        logger.info(f"💾 Results saved to: {self.output_dir}")
    
    def print_summary(self):
        """Print mapping summary"""
        logger.info("📊 MAPPING SUMMARY")
        logger.info("=" * 50)
        logger.info(f"GLEAMER images found: {self.stats['gleamer_images_found']}")
        logger.info(f"DICOM images found: {self.stats['dicom_images_found']}")
        logger.info(f"Successful mappings: {self.stats['successful_mappings']}")
        logger.info(f"Report tags found: {self.stats['report_tags_found']}")
        logger.info(f"Positive with fracture: {self.stats['positive_with_fracture']}")
        logger.info(f"Positive without fracture: {self.stats['positive_without_fracture']}")
        logger.info(f"Negative cases: {self.stats['negative_cases']}")
        logger.info(f"Summary reports filtered: {self.stats['summary_reports_filtered']}")
        logger.info(f"Errors: {self.stats['errors']}")
        
        if self.stats['successful_mappings'] > 0:
            mapping_rate = (self.stats['successful_mappings'] / self.stats['gleamer_images_found']) * 100
            logger.info(f"Mapping success rate: {mapping_rate:.1f}%")
        
        # Labeling summary
        logger.info("")
        logger.info("🏷️ LABELING SUMMARY:")
        logger.info(f"   Positive cases: {self.stats['positive_with_fracture']}")
        logger.info(f"   Negative cases: {self.stats['negative_cases']}")
        logger.info(f"   Relabeled cases: {self.stats['positive_without_fracture']}")
        
        if self.stats['positive_with_fracture'] == 0:
            logger.warning("⚠️ No positive cases found! Check labeling criteria.")
            logger.info("💡 Consider:")
            logger.info("   - Check if DICOM metadata contains fracture keywords")
            logger.info("   - Verify ClinicalIndication field has trauma indicators")
            logger.info("   - Review StudyDescription and Findings fields")
        
        logger.info("🎉 Fast mapping and dataset preparation completed!")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Fast GLEAMER-DICOM mapper')
    parser.add_argument('--gleamer-path', default='/mount/civiescaks01storage01/aksfileshare01/CNN/gleamer-images/',
                       help='Path to GLEAMER images directory')
    parser.add_argument('--dicom-path', default='/mount/civiescaks01storage01/aksfileshare01/CNN/gleamer-dicom/',
                       help='Path to DICOM images directory')
    parser.add_argument('--output-dir', default='mapped_dataset_fast',
                       help='Output directory for mapped dataset')
    parser.add_argument('--train-ratio', type=float, default=0.9,
                       help='Training set ratio (default: 0.9 for 90/10 split)')
    parser.add_argument('--val-ratio', type=float, default=0.1,
                       help='Validation set ratio (default: 0.1 for 90/10 split)')
    
    args = parser.parse_args()
    
    # Create mapper instance
    mapper = FastGleamerDicomMapper(
        gleamer_path=args.gleamer_path,
        dicom_path=args.dicom_path,
        output_dir=args.output_dir
    )
    
    # Find images
    mapper.find_gleamer_images()
    mapper.find_dicom_images()
    
    # Map images (FAST)
    mapper.map_gleamer_to_dicom_fast()
    
    # Find report tags
    mapper.find_report_tags()
    
    # Handle positive without fracture issue
    mapper.handle_positive_without_fracture()
    
    # Create dataset splits
    mapper.create_dataset_splits(train_ratio=args.train_ratio, val_ratio=args.val_ratio)
    
    # Save results
    mapper.save_results()
    
    # Print summary
    mapper.print_summary()

if __name__ == "__main__":
    main()
