#!/usr/bin/env python3
"""
DICOM to CSV Converter

Converts DICOM files to CSV format for analysis and verification.
Excludes pixel data for better performance and smaller file size.
"""

import os
import json
import logging
import argparse
from pathlib import Path
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
        logging.FileHandler('dicom_to_csv_converter.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DicomToCsvConverter:
    """Convert DICOM files to CSV format"""
    
    def __init__(self, dicom_path, output_csv):
        self.dicom_path = Path(dicom_path)
        self.output_csv = Path(output_csv)
        
        # Statistics
        self.stats = {
            'dicom_files_found': 0,
            'dicom_files_processed': 0,
            'dicom_files_failed': 0,
            'total_records': 0
        }
        
        logger.info("🔧 DICOM to CSV Converter initialized")
    
    def find_dicom_files(self):
        """Find all DICOM files"""
        logger.info("🔍 Finding DICOM files...")
        
        if not self.dicom_path.exists():
            logger.error(f"DICOM path does not exist: {self.dicom_path}")
            return []
        
        # Find all DICOM files
        dicom_files = []
        dicom_extensions = ['.dcm', '.DCM']
        
        for ext in dicom_extensions:
            dicom_files.extend(self.dicom_path.glob(f'*{ext}'))
        
        self.stats['dicom_files_found'] = len(dicom_files)
        logger.info(f"📊 Found {len(dicom_files)} DICOM files")
        
        return dicom_files
    
    def extract_dicom_metadata(self, dicom_path):
        """Extract metadata from DICOM file (no pixel data)"""
        try:
            if not DICOM_AVAILABLE:
                return {'read_success': False, 'error': 'pydicom not available'}
            
            # Read DICOM file without pixel data for performance
            ds = pydicom.dcmread(str(dicom_path), stop_before_pixels=True)
            
            # Define essential DICOM tags to extract
            essential_tags = [
                # Patient Information
                'PatientID', 'PatientName', 'PatientBirthDate', 'PatientSex', 'PatientAge',
                
                # Study Information
                'StudyInstanceUID', 'StudyID', 'StudyDate', 'StudyTime', 'StudyDescription',
                'AccessionNumber', 'ReferringPhysicianName', 'StudyComments',
                
                # Series Information
                'SeriesInstanceUID', 'SeriesNumber', 'SeriesDescription', 'SeriesDate', 'SeriesTime',
                'Modality', 'BodyPartExamined', 'ImageType', 'Manufacturer', 'ManufacturerModelName',
                'InstitutionName', 'InstitutionAddress', 'StationName',
                
                # Instance Information
                'SOPInstanceUID', 'SOPClassUID', 'InstanceNumber', 'ContentDate', 'ContentTime',
                'AcquisitionDate', 'AcquisitionTime', 'AcquisitionNumber',
                
                # Image Information (metadata only, no pixels)
                'Rows', 'Columns', 'BitsAllocated', 'BitsStored', 'HighBit', 'PixelRepresentation',
                'PhotometricInterpretation', 'SamplesPerPixel', 'PlanarConfiguration',
                'PixelSpacing', 'SliceThickness', 'SpacingBetweenSlices', 'ImageOrientationPatient',
                'ImagePositionPatient', 'SliceLocation', 'TableHeight', 'GantryDetectorTilt',
                
                # Clinical Information
                'ClinicalIndication', 'Findings', 'Impression', 'Diagnosis', 'ProcedureDescription',
                'ContrastBolusAgent', 'ContrastBolusRoute', 'ContrastBolusVolume',
                
                # Technical Information
                'KVP', 'ExposureTime', 'XRayTubeCurrent', 'Exposure', 'FilterMaterial',
                'CollimatorShape', 'FocalSpots', 'GeneratorPower', 'DetectorType',
                
                # Additional Tags
                'SoftwareVersions', 'ProtocolName', 'SequenceName', 'ScanOptions',
                'ScanningSequence', 'SequenceVariant', 'ScanningSequence', 'EchoTime',
                'RepetitionTime', 'FlipAngle', 'MagneticFieldStrength'
            ]
            
            metadata = {
                'file_path': str(dicom_path),
                'file_name': dicom_path.name,
                'file_size': dicom_path.stat().st_size,
                'read_success': True,
                'error': None
            }
            
            # Extract metadata for each tag
            for tag in essential_tags:
                if hasattr(ds, tag):
                    try:
                        value = getattr(ds, tag)
                        
                        # Handle different data types
                        if hasattr(value, '__iter__') and not isinstance(value, (str, bytes)):
                            # Convert lists/arrays to JSON string
                            metadata[tag] = json.dumps([str(v) for v in value])
                        elif isinstance(value, bytes):
                            # Skip binary data
                            metadata[tag] = f"<Binary data: {len(value)} bytes>"
                        else:
                            # Convert to string
                            metadata[tag] = str(value)
                    except Exception as e:
                        metadata[tag] = f"Error: {str(e)}"
                else:
                    metadata[tag] = None
            
            return metadata
            
        except Exception as e:
            return {
                'file_path': str(dicom_path),
                'file_name': dicom_path.name,
                'file_size': dicom_path.stat().st_size if dicom_path.exists() else 0,
                'read_success': False,
                'error': str(e)
            }
    
    def convert_to_csv(self, max_files=None):
        """Convert DICOM files to CSV"""
        logger.info("🔄 Converting DICOM files to CSV...")
        
        # Find DICOM files
        dicom_files = self.find_dicom_files()
        
        if not dicom_files:
            logger.error("No DICOM files found!")
            return False
        
        # Limit files if specified
        if max_files:
            dicom_files = dicom_files[:max_files]
            logger.info(f"📊 Processing first {len(dicom_files)} files")
        
        # Process DICOM files
        all_metadata = []
        
        for dicom_path in tqdm(dicom_files, desc="Processing DICOM files"):
            try:
                metadata = self.extract_dicom_metadata(dicom_path)
                all_metadata.append(metadata)
                
                if metadata['read_success']:
                    self.stats['dicom_files_processed'] += 1
                else:
                    self.stats['dicom_files_failed'] += 1
                    logger.debug(f"Failed to process {dicom_path.name}: {metadata.get('error', 'Unknown error')}")
                
            except Exception as e:
                logger.error(f"Error processing {dicom_path.name}: {e}")
                self.stats['dicom_files_failed'] += 1
        
        # Convert to DataFrame
        logger.info("📊 Converting to DataFrame...")
        df = pd.DataFrame(all_metadata)
        
        # Save to CSV
        logger.info(f"💾 Saving to CSV: {self.output_csv}")
        df.to_csv(self.output_csv, index=False)
        
        self.stats['total_records'] = len(df)
        
        # Print summary
        self.print_summary()
        
        return True
    
    def print_summary(self):
        """Print conversion summary"""
        logger.info("📊 CONVERSION SUMMARY")
        logger.info("=" * 50)
        logger.info(f"DICOM files found: {self.stats['dicom_files_found']}")
        logger.info(f"DICOM files processed: {self.stats['dicom_files_processed']}")
        logger.info(f"DICOM files failed: {self.stats['dicom_files_failed']}")
        logger.info(f"Total records: {self.stats['total_records']}")
        
        if self.stats['dicom_files_found'] > 0:
            success_rate = (self.stats['dicom_files_processed'] / self.stats['dicom_files_found']) * 100
            logger.info(f"Success rate: {success_rate:.1f}%")
        
        logger.info(f"📁 Output file: {self.output_csv}")
        logger.info(f"📊 File size: {self.output_csv.stat().st_size / (1024*1024):.1f} MB")
        
        logger.info("🎉 DICOM to CSV conversion completed!")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Convert DICOM files to CSV')
    parser.add_argument('--dicom-path', default='/mount/civiescaks01storage01/aksfileshare01/CNN/gleamer-dicom/',
                       help='Path to DICOM files directory')
    parser.add_argument('--output-csv', default='dicom_metadata.csv',
                       help='Output CSV file path')
    parser.add_argument('--max-files', type=int, default=None,
                       help='Maximum number of files to process (for testing)')
    
    args = parser.parse_args()
    
    # Create converter instance
    converter = DicomToCsvConverter(
        dicom_path=args.dicom_path,
        output_csv=args.output_csv
    )
    
    # Convert DICOM files to CSV
    success = converter.convert_to_csv(max_files=args.max_files)
    
    if success:
        logger.info("✅ Conversion completed successfully!")
    else:
        logger.error("❌ Conversion failed!")

if __name__ == "__main__":
    main()
