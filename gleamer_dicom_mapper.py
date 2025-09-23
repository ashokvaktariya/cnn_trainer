#!/usr/bin/env python3
"""
GLEAMER-DICOM Mapper and Dataset Preparer
Maps GLEAMER images to corresponding raw DICOM images
Finds report tags and creates balanced training dataset
"""

import os
import json
import pandas as pd
import shutil
from pathlib import Path
import pydicom
import logging
from datetime import datetime
import argparse
from tqdm import tqdm
import random
from collections import Counter
import re

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GleamerDicomMapper:
    """Maps GLEAMER images to DICOM images and prepares training dataset"""
    
    def __init__(self, gleamer_path="/mount/civiescaks01storage01/aksfileshare01/CNN/gleamer-images/",
                 dicom_path="/mount/civiescaks01storage01/aksfileshare01/CNN/gleamer-images/",
                 output_dir="mapped_dataset"):
        self.gleamer_path = Path(gleamer_path)
        self.dicom_path = Path(dicom_path)
        self.output_dir = Path(output_dir)
        
        self.gleamer_images = []
        self.dicom_images = []
        self.mapped_pairs = []
        self.report_tags = {}
        
        self.stats = {
            'gleamer_images_found': 0,
            'dicom_images_found': 0,
            'successful_mappings': 0,
            'report_tags_found': 0,
            'positive_with_fracture': 0,
            'positive_without_fracture': 0,
            'negative_cases': 0,
            'errors': 0
        }
        
        # Create output directories
        self.train_dir = self.output_dir / "train"
        self.val_dir = self.output_dir / "val"
        
        for split_dir in [self.train_dir, self.val_dir]:
            (split_dir / "positive").mkdir(parents=True, exist_ok=True)
            (split_dir / "negative").mkdir(parents=True, exist_ok=True)
        
        logger.info(f"🔧 GLEAMER-DICOM Mapper initialized")
        logger.info(f"📁 GLEAMER path: {self.gleamer_path}")
        logger.info(f"📁 DICOM path: {self.dicom_path}")
        logger.info(f"📁 Output directory: {self.output_dir}")
    
    def find_gleamer_images(self):
        """Find GLEAMER images (JPG files with specific patterns)"""
        logger.info("🔍 Scanning for GLEAMER images...")
        
        # Look for JPG files
        extensions = ['*.jpg', '*.jpeg', '*.JPG', '*.JPEG']
        
        for ext in extensions:
            files = list(self.gleamer_path.rglob(ext))
            self.gleamer_images.extend(files)
            logger.info(f"Found {len(files)} files with extension {ext}")
        
        # Remove duplicates
        self.gleamer_images = list(set(self.gleamer_images))
        self.stats['gleamer_images_found'] = len(self.gleamer_images)
        
        logger.info(f"📊 Total GLEAMER images found: {len(self.gleamer_images)}")
        return self.gleamer_images
    
    def find_dicom_images(self):
        """Find DICOM images"""
        logger.info("🔍 Scanning for DICOM images...")
        
        # Look for DICOM files
        extensions = ['*.dcm', '*.DCM', '*.dicom', '*.DICOM']
        
        for ext in extensions:
            files = list(self.dicom_path.rglob(ext))
            self.dicom_images.extend(files)
            logger.info(f"Found {len(files)} files with extension {ext}")
        
        # Also look for files without extension
        all_files = list(self.dicom_path.rglob('*'))
        potential_dicom = []
        
        for file_path in all_files[:10000]:  # Limit scan for performance
            if file_path.is_file() and not file_path.suffix:
                try:
                    pydicom.dcmread(str(file_path), stop_before_pixels=True)
                    potential_dicom.append(file_path)
                except:
                    pass
        
        self.dicom_images.extend(potential_dicom)
        
        # Remove duplicates
        self.dicom_images = list(set(self.dicom_images))
        self.stats['dicom_images_found'] = len(self.dicom_images)
        
        logger.info(f"📊 Total DICOM images found: {len(self.dicom_images)}")
        return self.dicom_images
    
    def extract_uid_from_filename(self, filename):
        """Extract UID from filename"""
        # Remove file extension
        name = filename.replace('.jpg', '').replace('.jpeg', '').replace('.dcm', '')
        
        # Extract UID pattern (numbers and dots)
        uid_pattern = r'(\d+(?:\.\d+)*)'
        matches = re.findall(uid_pattern, name)
        
        if matches:
            # Return the longest UID found
            return max(matches, key=len)
        
        return None
    
    def extract_dicom_metadata(self, dicom_path):
        """Extract metadata from DICOM file"""
        try:
            ds = pydicom.dcmread(str(dicom_path), stop_before_pixels=True)
            
            metadata = {
                'file_path': str(dicom_path),
                'file_name': dicom_path.name,
                'read_success': True,
                'error': None
            }
            
            # Extract key DICOM tags
            key_tags = [
                'SOPInstanceUID', 'StudyInstanceUID', 'SeriesInstanceUID',
                'PatientID', 'StudyDate', 'StudyTime', 'StudyDescription',
                'SeriesDescription', 'Modality', 'BodyPartExamined',
                'ImageType', 'Manufacturer', 'InstitutionName',
                'StudyID', 'AccessionNumber', 'SeriesNumber', 'InstanceNumber',
                'Rows', 'Columns', 'BitsAllocated', 'PhotometricInterpretation',
                'SOPClassUID', 'Findings', 'ClinicalIndication'
            ]
            
            for tag in key_tags:
                if hasattr(ds, tag):
                    try:
                        value = getattr(ds, tag)
                        if hasattr(value, '__iter__') and not isinstance(value, str):
                            metadata[tag] = [str(v) for v in value]
                        else:
                            metadata[tag] = str(value)
                    except Exception as e:
                        metadata[tag] = f"Error: {str(e)}"
            
            return metadata
            
        except Exception as e:
            return {
                'file_path': str(dicom_path),
                'file_name': dicom_path.name,
                'read_success': False,
                'error': str(e)
            }
    
    def map_gleamer_to_dicom(self):
        """Map GLEAMER images to corresponding DICOM images"""
        logger.info("🔗 Mapping GLEAMER images to DICOM images...")
        
        # Create UID to DICOM mapping
        dicom_uid_map = {}
        for dicom_path in tqdm(self.dicom_images, desc="Indexing DICOM files"):
            try:
                metadata = self.extract_dicom_metadata(dicom_path)
                if metadata['read_success'] and 'SOPInstanceUID' in metadata:
                    uid = metadata['SOPInstanceUID']
                    dicom_uid_map[uid] = {
                        'path': dicom_path,
                        'metadata': metadata
                    }
            except Exception as e:
                logger.debug(f"Error indexing {dicom_path.name}: {e}")
        
        logger.info(f"📊 Indexed {len(dicom_uid_map)} DICOM files with UIDs")
        
        # Map GLEAMER images to DICOM
        for gleamer_path in tqdm(self.gleamer_images, desc="Mapping GLEAMER images"):
            try:
                # Extract UID from GLEAMER filename
                gleamer_uid = self.extract_uid_from_filename(gleamer_path.name)
                
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
                    'body_part': metadata.get('BodyPartExamined', ''),
                    'manufacturer': metadata.get('Manufacturer', ''),
                    'institution': metadata.get('InstitutionName', '')
                }
                
                # Determine label based on report content
                label = self.determine_label_from_report(report_info)
                
                mapping['report_info'] = report_info
                mapping['label'] = label
                
                self.stats['report_tags_found'] += 1
                
                if label == 'POSITIVE':
                    self.stats['positive_with_fracture'] += 1
                elif label == 'NEGATIVE':
                    self.stats['negative_cases'] += 1
                else:
                    self.stats['positive_without_fracture'] += 1
    
    def determine_label_from_report(self, report_info):
        """Determine label from report information"""
        # Check for fracture indicators
        fracture_keywords = ['fracture', 'broken', 'crack', 'break', 'fissure']
        
        # Check StudyDescription
        study_desc = report_info['study_description'].lower()
        if any(keyword in study_desc for keyword in fracture_keywords):
            return 'POSITIVE'
        
        # Check Findings
        findings = report_info['findings'].lower()
        if any(keyword in findings for keyword in fracture_keywords):
            return 'POSITIVE'
        if 'no acute' in findings or 'no fracture' in findings or 'normal' in findings:
            return 'NEGATIVE'
        
        # Check ClinicalIndication
        clinical = report_info['clinical_indication'].lower()
        trauma_keywords = ['trauma', 'injury', 'fall', 'accident', 'pain']
        if any(keyword in clinical for keyword in trauma_keywords):
            return 'POSITIVE'
        
        # Check SeriesDescription
        series_desc = report_info['series_description'].lower()
        if 'fracture' in series_desc:
            return 'POSITIVE'
        if 'negative' in series_desc:
            return 'NEGATIVE'
        
        # Default to NEGATIVE if no clear indicators
        return 'NEGATIVE'
    
    def create_dataset_splits(self, train_ratio=0.8, val_ratio=0.2):
        """Create 80/20 train/validation splits"""
        logger.info("📊 Creating 80/20 train/validation splits...")
        
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
            'splits': {
                'train': {
                    'positive': len(pos_train),
                    'negative': len(neg_train),
                    'total': len(pos_train) + len(neg_train)
                },
                'val': {
                    'positive': len(pos_val),
                    'negative': len(neg_val),
                    'total': len(pos_val) + len(neg_val)
                }
            }
        }
        
        # Save split info to JSON
        with open(self.output_dir / 'dataset_splits.json', 'w') as f:
            json.dump(split_info, f, indent=2)
        
        logger.info("📊 Dataset splits created:")
        logger.info(f"  Train: {split_info['splits']['train']['total']} files")
        logger.info(f"  Val: {split_info['splits']['val']['total']} files")
        
        return splits
    
    def handle_positive_without_fracture(self):
        """Handle positive records with no fracture images"""
        logger.info("🔍 Handling positive records with no fracture images...")
        
        # Find positive cases that might not have actual fracture images
        problematic_cases = []
        
        for mapping in self.mapped_pairs:
            if mapping.get('label') == 'POSITIVE':
                report_info = mapping.get('report_info', {})
                
                # Check if this is a report/image rather than actual X-ray
                if (report_info.get('manufacturer', '').upper() == 'GLEAMER' or
                    'DERIVED' in str(report_info.get('image_type', [])) or
                    'SECONDARY' in str(report_info.get('image_type', []))):
                    
                    problematic_cases.append(mapping)
        
        logger.info(f"⚠️ Found {len(problematic_cases)} positive cases that might be reports")
        
        # Option 1: Re-label as NEGATIVE
        for case in problematic_cases:
            case['label'] = 'NEGATIVE'
            case['relabeled'] = True
            case['relabel_reason'] = 'Positive case was actually a report/image'
        
        # Update statistics
        self.stats['positive_without_fracture'] = len(problematic_cases)
        self.stats['negative_cases'] += len(problematic_cases)
        self.stats['positive_with_fracture'] -= len(problematic_cases)
        
        logger.info(f"✅ Re-labeled {len(problematic_cases)} cases as NEGATIVE")
    
    def save_results(self):
        """Save all results"""
        # Save mapped pairs
        with open(self.output_dir / 'mapped_pairs.json', 'w') as f:
            json.dump(self.mapped_pairs, f, indent=2)
        
        # Save statistics
        stats_info = {
            'timestamp': datetime.now().isoformat(),
            'gleamer_path': str(self.gleamer_path),
            'dicom_path': str(self.dicom_path),
            'output_dir': str(self.output_dir),
            'statistics': self.stats,
            'summary': {
                'gleamer_images_found': self.stats['gleamer_images_found'],
                'dicom_images_found': self.stats['dicom_images_found'],
                'successful_mappings': self.stats['successful_mappings'],
                'report_tags_found': self.stats['report_tags_found'],
                'positive_with_fracture': self.stats['positive_with_fracture'],
                'positive_without_fracture': self.stats['positive_without_fracture'],
                'negative_cases': self.stats['negative_cases'],
                'errors': self.stats['errors']
            }
        }
        
        with open(self.output_dir / 'mapping_stats.json', 'w') as f:
            json.dump(stats_info, f, indent=2)
        
        logger.info(f"💾 Results saved to: {self.output_dir}")
    
    def print_summary(self):
        """Print processing summary"""
        logger.info("📊 MAPPING SUMMARY")
        logger.info("=" * 50)
        logger.info(f"GLEAMER images found: {self.stats['gleamer_images_found']}")
        logger.info(f"DICOM images found: {self.stats['dicom_images_found']}")
        logger.info(f"Successful mappings: {self.stats['successful_mappings']}")
        logger.info(f"Report tags found: {self.stats['report_tags_found']}")
        logger.info(f"Positive with fracture: {self.stats['positive_with_fracture']}")
        logger.info(f"Positive without fracture: {self.stats['positive_without_fracture']}")
        logger.info(f"Negative cases: {self.stats['negative_cases']}")
        logger.info(f"Errors: {self.stats['errors']}")
        
        if self.stats['successful_mappings'] > 0:
            mapping_rate = (self.stats['successful_mappings'] / self.stats['gleamer_images_found']) * 100
            logger.info(f"Mapping success rate: {mapping_rate:.1f}%")
        
        logger.info("🎉 Mapping and dataset preparation completed!")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Map GLEAMER images to DICOM and prepare training dataset')
    parser.add_argument('--gleamer-path', default='/mount/civiescaks01storage01/aksfileshare01/CNN/gleamer-images/',
                       help='Path to GLEAMER images directory')
    parser.add_argument('--dicom-path', default='/mount/civiescaks01storage01/aksfileshare01/CNN/gleamer-images/',
                       help='Path to DICOM images directory')
    parser.add_argument('--output-dir', default='mapped_dataset',
                       help='Output directory for mapped dataset')
    
    args = parser.parse_args()
    
    # Create mapper instance
    mapper = GleamerDicomMapper(
        gleamer_path=args.gleamer_path,
        dicom_path=args.dicom_path,
        output_dir=args.output_dir
    )
    
    # Find images
    mapper.find_gleamer_images()
    mapper.find_dicom_images()
    
    # Map GLEAMER to DICOM
    mapper.map_gleamer_to_dicom()
    
    # Find report tags
    mapper.find_report_tags()
    
    # Handle positive without fracture issue
    mapper.handle_positive_without_fracture()
    
    # Create dataset splits
    mapper.create_dataset_splits()
    
    # Save results
    mapper.save_results()
    
    # Print summary
    mapper.print_summary()

if __name__ == "__main__":
    main()
