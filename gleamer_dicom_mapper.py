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
import cv2
import numpy as np
from PIL import Image

# Try to import pytesseract, but don't fail if not available
try:
    import pytesseract
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False
    print("⚠️ pytesseract not available - OCR features disabled")

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GleamerDicomMapper:
    """Maps GLEAMER images to DICOM images and prepares training dataset"""
    
    def __init__(self, gleamer_path="/mount/civiescaks01storage01/aksfileshare01/CNN/gleamer-images/",
                 dicom_path="/mount/civiescaks01storage01/aksfileshare01/CNN/gleamer-dicom/",
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
            'summary_reports_filtered': 0,
            'low_quality_filtered': 0,
            'high_confidence_accepted': 0,
            'low_confidence_reviewed': 0,
            'ocr_analysis_performed': 0,
            'image_content_analyzed': 0,
            'errors': 0
        }
        
        # Quality control thresholds
        self.quality_thresholds = {
            'min_confidence_score': 70,
            'max_text_density': 100,
            'min_image_size': (512, 512),
            'min_contrast_std': 20,
            'min_edge_density': 0.01,
            'max_saturation': 50
        }
        
        # Known summary report patterns
        self.summary_patterns = [
            '1.2.250.1',  # GLEAMER summary reports
            '1.2.840.10008.5.1.4.1.1.7'  # Secondary Capture SOP Class
        ]
        
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
        """Find DICOM images - focus on .dcm files only"""
        logger.info("🔍 Scanning for DICOM images (.dcm files only)...")
        
        # Look for .dcm files only
        extensions = ['*.dcm', '*.DCM']
        
        for ext in extensions:
            files = list(self.dicom_path.rglob(ext))
            self.dicom_images.extend(files)
            logger.info(f"Found {len(files)} files with extension {ext}")
        
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
    
    def is_summary_report_pattern(self, uid):
        """Check if UID matches known summary report patterns"""
        if not uid:
            return False
        
        for pattern in self.summary_patterns:
            if uid.startswith(pattern):
                return True
        
        return False
    
    def analyze_image_content(self, image_path):
        """Analyze image content for quality and type detection"""
        try:
            # Load image
            image = cv2.imread(str(image_path))
            if image is None:
                return {
                    'is_xray': False,
                    'quality_score': 0,
                    'has_bone_structures': False,
                    'text_density': 0,
                    'contrast_std': 0,
                    'edge_density': 0,
                    'saturation': 0,
                    'analysis_success': False,
                    'error': 'Could not load image'
                }
            
            # Get image dimensions
            height, width = image.shape[:2]
            
            # Check minimum size
            if width < self.quality_thresholds['min_image_size'][0] or height < self.quality_thresholds['min_image_size'][1]:
                return {
                    'is_xray': False,
                    'quality_score': 0,
                    'has_bone_structures': False,
                    'text_density': 0,
                    'contrast_std': 0,
                    'edge_density': 0,
                    'saturation': 0,
                    'analysis_success': False,
                    'error': 'Image too small'
                }
            
            # Convert to grayscale for analysis
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Calculate image statistics
            mean_intensity = np.mean(gray)
            std_intensity = np.std(gray)
            
            # Check for X-ray characteristics
            is_xray = (30 < mean_intensity < 220) and (std_intensity > self.quality_thresholds['min_contrast_std'])
            
            # Edge analysis for bone structures
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / (width * height)
            has_bone_structures = edge_density > self.quality_thresholds['min_edge_density']
            
            # Color analysis
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            saturation = np.mean(hsv[:, :, 1])
            
            # OCR text analysis (if available)
            text_density = 0
            if OCR_AVAILABLE:
                try:
                    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                    text = pytesseract.image_to_string(pil_image, config='--psm 6')
                    text_density = len(text.strip())
                    self.stats['ocr_analysis_performed'] += 1
                except Exception as e:
                    logger.debug(f"OCR analysis failed for {image_path.name}: {e}")
            
            # Calculate quality score
            quality_score = 0
            if is_xray:
                quality_score += 30
            if has_bone_structures:
                quality_score += 25
            if std_intensity > self.quality_thresholds['min_contrast_std']:
                quality_score += 20
            if edge_density > self.quality_thresholds['min_edge_density']:
                quality_score += 15
            if saturation < self.quality_thresholds['max_saturation']:
                quality_score += 10
            
            self.stats['image_content_analyzed'] += 1
            
            return {
                'is_xray': is_xray,
                'quality_score': quality_score,
                'has_bone_structures': has_bone_structures,
                'text_density': text_density,
                'contrast_std': std_intensity,
                'edge_density': edge_density,
                'saturation': saturation,
                'mean_intensity': mean_intensity,
                'image_size': (width, height),
                'analysis_success': True,
                'error': None
            }
            
        except Exception as e:
            return {
                'is_xray': False,
                'quality_score': 0,
                'has_bone_structures': False,
                'text_density': 0,
                'contrast_std': 0,
                'edge_density': 0,
                'saturation': 0,
                'analysis_success': False,
                'error': str(e)
            }
    
    def calculate_confidence_score(self, mapping):
        """Calculate confidence score for mapping"""
        score = 0
        
        # Mapping success
        if mapping.get('mapping_success', False):
            score += 20
        
        # Metadata consistency
        gleamer_finding = mapping.get('gleamer_finding', '')
        dicom_label = mapping.get('label', '')
        if gleamer_finding == dicom_label:
            score += 25
        
        # Image content analysis
        image_analysis = mapping.get('image_analysis', {})
        if image_analysis.get('analysis_success', False):
            score += image_analysis.get('quality_score', 0) * 0.3  # Scale quality score
        
        # Report consistency
        report_info = mapping.get('report_info', {})
        findings = report_info.get('findings', '').lower()
        clinical = report_info.get('clinical_indication', '').lower()
        
        if mapping.get('label') == 'POSITIVE':
            if 'fracture' in findings or 'trauma' in clinical:
                score += 15
        elif mapping.get('label') == 'NEGATIVE':
            if 'no acute' in findings or 'normal' in findings:
                score += 15
        
        # Required metadata fields
        dicom_metadata = mapping.get('dicom_metadata', {})
        required_fields = ['SOPInstanceUID', 'StudyDescription', 'Modality']
        present_fields = sum(1 for field in required_fields if field in dicom_metadata)
        score += (present_fields / len(required_fields)) * 10
        
        return min(score, 100)  # Cap at 100
    
    def quality_assurance_pipeline(self, mapping):
        """Comprehensive quality assurance pipeline"""
        result = {
            'status': 'PENDING',
            'reason': '',
            'confidence_score': 0,
            'filters_passed': [],
            'filters_failed': []
        }
        
        # Step 1: Basic validation
        if not mapping.get('mapping_success', False):
            result['status'] = 'REJECT'
            result['reason'] = 'Mapping failed'
            result['filters_failed'].append('mapping_validation')
            return result
        
        result['filters_passed'].append('mapping_validation')
        
        # Step 2: UID pattern filtering
        gleamer_uid = mapping.get('gleamer_uid', '')
        if self.is_summary_report_pattern(gleamer_uid):
            result['status'] = 'REJECT'
            result['reason'] = 'Summary report UID pattern'
            result['filters_failed'].append('uid_pattern_filtering')
            self.stats['summary_reports_filtered'] += 1
            return result
        
        result['filters_passed'].append('uid_pattern_filtering')
        
        # Step 3: DICOM metadata filtering
        dicom_metadata = mapping.get('dicom_metadata', {})
        if self.is_summary_report_from_metadata(dicom_metadata):
            result['status'] = 'REJECT'
            result['reason'] = 'DICOM metadata indicates report'
            result['filters_failed'].append('dicom_metadata_filtering')
            self.stats['summary_reports_filtered'] += 1
            return result
        
        result['filters_passed'].append('dicom_metadata_filtering')
        
        # Step 4: Image content analysis
        image_analysis = mapping.get('image_analysis', {})
        if not image_analysis.get('analysis_success', False):
            result['status'] = 'REJECT'
            result['reason'] = 'Image content analysis failed'
            result['filters_failed'].append('image_content_analysis')
            return result
        
        if not image_analysis.get('is_xray', False):
            result['status'] = 'REJECT'
            result['reason'] = 'Not an X-ray image'
            result['filters_failed'].append('xray_validation')
            return result
        
        if image_analysis.get('text_density', 0) > self.quality_thresholds['max_text_density']:
            result['status'] = 'REJECT'
            result['reason'] = 'Excessive text content'
            result['filters_failed'].append('text_density_filtering')
            return result
        
        result['filters_passed'].extend(['image_content_analysis', 'xray_validation', 'text_density_filtering'])
        
        # Step 5: Confidence scoring
        confidence_score = self.calculate_confidence_score(mapping)
        result['confidence_score'] = confidence_score
        
        if confidence_score < self.quality_thresholds['min_confidence_score']:
            result['status'] = 'REVIEW'
            result['reason'] = f'Low confidence score: {confidence_score}'
            result['filters_failed'].append('confidence_threshold')
            self.stats['low_confidence_reviewed'] += 1
            return result
        
        result['filters_passed'].append('confidence_threshold')
        
        # Step 6: Positive record verification
        if mapping.get('label') == 'POSITIVE':
            if not self.verify_positive_record(mapping):
                result['status'] = 'RELABEL'
                result['reason'] = 'Positive record lacks fracture evidence'
                result['filters_failed'].append('positive_verification')
                return result
        
        result['filters_passed'].append('positive_verification')
        
        # All checks passed
        result['status'] = 'ACCEPT'
        result['reason'] = 'Passed all quality checks'
        self.stats['high_confidence_accepted'] += 1
        
        return result
    
    def is_summary_report_from_metadata(self, dicom_metadata):
        """Check if DICOM metadata indicates a summary report"""
        # Check SeriesDescription for report keywords
        series_desc = dicom_metadata.get('SeriesDescription', '').upper()
        report_keywords = ['SUMMARY', 'REPORT', 'FINDINGS', 'IMPRESSION', 'NEGATIVE', 'POSITIVE']
        if any(keyword in series_desc for keyword in report_keywords):
            return True
        
        # Check ImageType for derived/secondary images
        image_type = dicom_metadata.get('ImageType', [])
        if isinstance(image_type, list):
            if 'DERIVED' in image_type or 'SECONDARY' in image_type:
                return True
        
        # Check Manufacturer for GLEAMER (AI reports)
        manufacturer = dicom_metadata.get('Manufacturer', '')
        if 'GLEAMER' in manufacturer.upper():
            return True
        
        # Check SOP Class UID for Secondary Capture
        sop_class = dicom_metadata.get('SOPClassUID', '')
        if sop_class == '1.2.840.10008.5.1.4.1.1.7':  # Secondary Capture
            return True
        
        return False
    
    def verify_positive_record(self, mapping):
        """Verify that positive records have supporting evidence"""
        report_info = mapping.get('report_info', {})
        
        # Count positive indicators
        positive_indicators = 0
        
        # Check StudyDescription
        study_desc = report_info.get('study_description', '').lower()
        if 'fracture' in study_desc:
            positive_indicators += 1
        
        # Check Findings
        findings = report_info.get('findings', '').lower()
        if 'fracture' in findings:
            positive_indicators += 1
        
        # Check ClinicalIndication
        clinical = report_info.get('clinical_indication', '').lower()
        trauma_keywords = ['trauma', 'injury', 'fall', 'accident', 'pain']
        if any(keyword in clinical for keyword in trauma_keywords):
            positive_indicators += 1
        
        # Check SeriesDescription
        series_desc = report_info.get('series_description', '').lower()
        if 'fracture' in series_desc:
            positive_indicators += 1
        
        # Require at least 2 positive indicators for validation
        return positive_indicators >= 2
    
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
        """Map GLEAMER images to corresponding DICOM images with quality control"""
        logger.info("🔗 Mapping GLEAMER images to DICOM images with quality control...")
        
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
        
        # Map GLEAMER images to DICOM with quality control
        for gleamer_path in tqdm(self.gleamer_images, desc="Mapping GLEAMER images"):
            try:
                # Extract UID from GLEAMER filename
                gleamer_uid = self.extract_uid_from_filename(gleamer_path.name)
                
                if gleamer_uid and gleamer_uid in dicom_uid_map:
                    # Found matching DICOM
                    dicom_info = dicom_uid_map[gleamer_uid]
                    
                    # Perform image content analysis
                    image_analysis = self.analyze_image_content(gleamer_path)
                    
                    mapping = {
                        'gleamer_path': str(gleamer_path),
                        'gleamer_filename': gleamer_path.name,
                        'gleamer_uid': gleamer_uid,
                        'dicom_path': str(dicom_info['path']),
                        'dicom_filename': dicom_info['path'].name,
                        'dicom_metadata': dicom_info['metadata'],
                        'image_analysis': image_analysis,
                        'mapping_success': True
                    }
                    
                    # Apply quality assurance pipeline
                    qa_result = self.quality_assurance_pipeline(mapping)
                    mapping['qa_result'] = qa_result
                    
                    # Handle quality assurance results
                    if qa_result['status'] == 'ACCEPT':
                        self.mapped_pairs.append(mapping)
                        self.stats['successful_mappings'] += 1
                    elif qa_result['status'] == 'RELABEL':
                        # Re-label positive cases that lack evidence
                        mapping['label'] = 'NEGATIVE'
                        mapping['relabeled'] = True
                        mapping['relabel_reason'] = qa_result['reason']
                        self.mapped_pairs.append(mapping)
                        self.stats['successful_mappings'] += 1
                        self.stats['positive_without_fracture'] += 1
                    else:
                        # Reject low quality or invalid mappings
                        logger.debug(f"Rejected {gleamer_path.name}: {qa_result['reason']}")
                        if qa_result['status'] == 'REJECT':
                            self.stats['low_quality_filtered'] += 1
                    
                else:
                    # No matching DICOM found
                    mapping = {
                        'gleamer_path': str(gleamer_path),
                        'gleamer_filename': gleamer_path.name,
                        'gleamer_uid': gleamer_uid,
                        'dicom_path': None,
                        'dicom_filename': None,
                        'dicom_metadata': None,
                        'image_analysis': None,
                        'mapping_success': False,
                        'qa_result': {'status': 'REJECT', 'reason': 'No matching DICOM found'}
                    }
                    
                    self.mapped_pairs.append(mapping)
                    
            except Exception as e:
                logger.error(f"Error mapping {gleamer_path.name}: {e}")
                self.stats['errors'] += 1
        
        logger.info(f"✅ Successfully mapped {self.stats['successful_mappings']} pairs")
        logger.info(f"📊 Quality control results:")
        logger.info(f"  Summary reports filtered: {self.stats['summary_reports_filtered']}")
        logger.info(f"  Low quality filtered: {self.stats['low_quality_filtered']}")
        logger.info(f"  High confidence accepted: {self.stats['high_confidence_accepted']}")
        logger.info(f"  Low confidence reviewed: {self.stats['low_confidence_reviewed']}")
    
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
        logger.info("📊 COMPREHENSIVE MAPPING SUMMARY")
        logger.info("=" * 60)
        logger.info(f"GLEAMER images found: {self.stats['gleamer_images_found']}")
        logger.info(f"DICOM images found: {self.stats['dicom_images_found']}")
        logger.info(f"Successful mappings: {self.stats['successful_mappings']}")
        logger.info(f"Report tags found: {self.stats['report_tags_found']}")
        logger.info("")
        logger.info("🔍 QUALITY CONTROL RESULTS:")
        logger.info(f"Summary reports filtered: {self.stats['summary_reports_filtered']}")
        logger.info(f"Low quality filtered: {self.stats['low_quality_filtered']}")
        logger.info(f"High confidence accepted: {self.stats['high_confidence_accepted']}")
        logger.info(f"Low confidence reviewed: {self.stats['low_confidence_reviewed']}")
        logger.info("")
        logger.info("📊 ANALYSIS STATISTICS:")
        logger.info(f"OCR analysis performed: {self.stats['ocr_analysis_performed']}")
        logger.info(f"Image content analyzed: {self.stats['image_content_analyzed']}")
        logger.info("")
        logger.info("🏷️ LABELING RESULTS:")
        logger.info(f"Positive with fracture: {self.stats['positive_with_fracture']}")
        logger.info(f"Positive without fracture: {self.stats['positive_without_fracture']}")
        logger.info(f"Negative cases: {self.stats['negative_cases']}")
        logger.info(f"Errors: {self.stats['errors']}")
        
        if self.stats['successful_mappings'] > 0:
            mapping_rate = (self.stats['successful_mappings'] / self.stats['gleamer_images_found']) * 100
            logger.info(f"Mapping success rate: {mapping_rate:.1f}%")
        
        if self.stats['gleamer_images_found'] > 0:
            filter_rate = ((self.stats['summary_reports_filtered'] + self.stats['low_quality_filtered']) / self.stats['gleamer_images_found']) * 100
            logger.info(f"Overall filter rate: {filter_rate:.1f}%")
        
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
        
        logger.info("🎉 Advanced mapping and dataset preparation completed!")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Map GLEAMER images to DICOM and prepare training dataset')
    parser.add_argument('--gleamer-path', default='/mount/civiescaks01storage01/aksfileshare01/CNN/gleamer-images/',
                       help='Path to GLEAMER images directory')
    parser.add_argument('--dicom-path', default='/mount/civiescaks01storage01/aksfileshare01/CNN/gleamer-dicom/',
                       help='Path to DICOM images directory')
    parser.add_argument('--output-dir', default='mapped_dataset',
                       help='Output directory for mapped dataset')
    parser.add_argument('--train-ratio', type=float, default=0.9,
                       help='Training set ratio (default: 0.9 for 90/10 split)')
    parser.add_argument('--val-ratio', type=float, default=0.1,
                       help='Validation set ratio (default: 0.1 for 90/10 split)')
    
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
    mapper.create_dataset_splits(train_ratio=args.train_ratio, val_ratio=args.val_ratio)
    
    # Save results
    mapper.save_results()
    
    # Print summary
    mapper.print_summary()

if __name__ == "__main__":
    main()
