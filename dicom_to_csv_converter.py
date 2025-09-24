#!/usr/bin/env python3
"""
DICOM to CSV Converter (Windows Compatible)

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
    print("WARNING: pydicom not available. Install with: pip install pydicom")

# Configure logging (Windows compatible - no emojis)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DicomToCsvConverter:
    """Converts DICOM files to a CSV containing essential metadata, excluding pixel data."""

    def __init__(self, dicom_path, output_csv, max_files=None):
        self.dicom_path = Path(dicom_path)
        self.output_csv = Path(output_csv)
        self.max_files = max_files
        self.metadata_list = []
        self.processed_count = 0
        self.skipped_count = 0
        self.error_count = 0

        # Define key DICOM tags to extract
        self.key_tags = [
            'SOPInstanceUID', 'StudyInstanceUID', 'SeriesInstanceUID',
            'PatientID', 'StudyDate', 'StudyTime', 'StudyDescription',
            'SeriesDescription', 'Modality', 'BodyPartExamined',
            'ImageType', 'Manufacturer', 'InstitutionName',
            'StudyID', 'AccessionNumber', 'SeriesNumber', 'InstanceNumber',
            'Rows', 'Columns', 'BitsAllocated', 'PhotometricInterpretation',
            'SOPClassUID', 'Findings', 'ClinicalIndication', 'PatientSex',
            'PatientAge', 'KVP', 'XRayTubeCurrent', 'Exposure', 'ExposureInmAs',
            'DistanceSourceToDetector', 'DistanceSourceToPatient', 'ImagerPixelSpacing',
            'PixelSpacing', 'WindowCenter', 'WindowWidth', 'RescaleIntercept', 'RescaleSlope'
        ]
        logger.info(f"DICOM to CSV Converter initialized. Output: {self.output_csv}")

    def find_dicom_files(self):
        """Find all DICOM files in the specified directory."""
        logger.info("Finding DICOM files...")
        
        if not self.dicom_path.exists():
            logger.error(f"DICOM path does not exist: {self.dicom_path}")
            return []

        # Look for .dcm and .DCM files
        dicom_files = list(self.dicom_path.rglob('*.dcm')) + list(self.dicom_path.rglob('*.DCM'))
        logger.info(f"Found {len(dicom_files)} DICOM files")
        
        return dicom_files

    def extract_metadata(self, dicom_file_path):
        """Extracts metadata from a single DICOM file without reading pixel data."""
        try:
            # Read DICOM file, stopping before pixel data for performance
            ds = pydicom.dcmread(str(dicom_file_path), stop_before_pixels=True)

            metadata = {
                'file_path': str(dicom_file_path),
                'file_name': dicom_file_path.name,
                'file_size': dicom_file_path.stat().st_size,
                'read_success': True,
                'error': None
            }

            for tag in self.key_tags:
                if hasattr(ds, tag):
                    try:
                        value = getattr(ds, tag)
                        # Handle multi-valued tags (e.g., ImageType)
                        if hasattr(value, '__iter__') and not isinstance(value, str):
                            metadata[tag] = [str(v) for v in value]
                        else:
                            metadata[tag] = str(value)
                    except Exception as e:
                        metadata[tag] = f"Error: {str(e)}"
                else:
                    metadata[tag] = None  # Tag not present

            return metadata

        except Exception as e:
            self.error_count += 1
            logger.debug(f"Error processing {dicom_file_path.name}: {e}")
            return {
                'file_path': str(dicom_file_path),
                'file_name': dicom_file_path.name,
                'file_size': dicom_file_path.stat().st_size if dicom_file_path.exists() else 0,
                'read_success': False,
                'error': str(e)
            }

    def convert_to_csv(self, max_files=None):
        """Scans DICOM directory and converts metadata to a CSV file."""
        if not DICOM_AVAILABLE:
            logger.error("pydicom is not available. Please install it first.")
            return False

        logger.info("Converting DICOM files to CSV...")
        
        dicom_files = self.find_dicom_files()
        if not dicom_files:
            logger.warning("No DICOM files found!")
            return False

        # Limit files if specified
        if max_files:
            dicom_files = dicom_files[:max_files]
            logger.info(f"Processing first {len(dicom_files)} files")

        # Process each DICOM file
        for dicom_file in tqdm(dicom_files, desc="Processing DICOM files"):
            metadata = self.extract_metadata(dicom_file)
            self.metadata_list.append(metadata)
            self.processed_count += 1

        # Convert to DataFrame and save
        logger.info("Converting to DataFrame...")
        df = pd.DataFrame(self.metadata_list)
        
        # Create output directory if it doesn't exist
        self.output_csv.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving to CSV: {self.output_csv}")
        df.to_csv(self.output_csv, index=False)

        # Print summary
        self.print_summary()
        return True

    def print_summary(self):
        """Prints a summary of the conversion process."""
        logger.info("CONVERSION SUMMARY")
        logger.info("=" * 50)
        logger.info(f"DICOM files found: {len(self.metadata_list)}")
        logger.info(f"DICOM files processed: {sum(1 for m in self.metadata_list if m['read_success'])}")
        logger.info(f"DICOM files failed: {sum(1 for m in self.metadata_list if not m['read_success'])}")
        logger.info(f"Total records: {len(self.metadata_list)}")
        
        success_rate = (sum(1 for m in self.metadata_list if m['read_success']) / len(self.metadata_list)) * 100 if self.metadata_list else 0
        logger.info(f"Success rate: {success_rate:.1f}%")
        logger.info(f"Output file: {self.output_csv}")
        
        if self.output_csv.exists():
            file_size_mb = self.output_csv.stat().st_size / (1024*1024)
            logger.info(f"File size: {file_size_mb:.1f} MB")
        
        logger.info("DICOM to CSV conversion completed!")

def main():
    parser = argparse.ArgumentParser(description='Convert DICOM metadata to CSV.')
    parser.add_argument('--dicom-path', default='/mount/civiescaks01storage01/aksfileshare01/CNN/gleamer-dicom/',
                        help='Path to the DICOM images directory.')
    parser.add_argument('--output-csv', default='dicom_metadata.csv',
                        help='Output CSV file path.')
    parser.add_argument('--max-files', type=int, default=None,
                        help='Maximum number of DICOM files to process (for testing).')
    args = parser.parse_args()

    converter = DicomToCsvConverter(args.dicom_path, args.output_csv, args.max_files)
    success = converter.convert_to_csv(max_files=args.max_files)
    
    if success:
        logger.info("Conversion completed successfully!")
    else:
        logger.error("Conversion failed!")

if __name__ == "__main__":
    main()