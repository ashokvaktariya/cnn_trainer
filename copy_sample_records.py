#!/usr/bin/env python3
"""
Copy Sample Records with Associated Images
Creates a new folder with 3 records and their associated images
"""

import os
import json
import shutil
import pandas as pd
import ast
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SampleRecordCopier:
    def __init__(self, csv_path, image_root, output_dir="sample_records"):
        """
        Initialize the sample record copier
        
        Args:
            csv_path: Path to the CSV file
            image_root: Root directory containing gleamer-images
            output_dir: Output directory for sample records
        """
        self.csv_path = csv_path
        self.image_root = Path(image_root)
        self.output_dir = Path(output_dir)
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def find_image_file(self, filename):
        """Find image file in gleamer-images directory"""
        # Search in root and subdirectories
        for root, dirs, files in os.walk(self.image_root):
            if filename in files:
                return os.path.join(root, filename)
        return None
    
    def copy_record(self, row, record_idx):
        """
        Copy a single record with its associated images
        
        Args:
            row: Pandas row with record data
            record_idx: Index of the record
            
        Returns:
            dict: Copy results
        """
        try:
            # Create record folder
            record_folder = self.output_dir / f"record_{record_idx}"
            record_folder.mkdir(exist_ok=True)
            
            # Copy CSV data
            record_data = {
                'record_index': record_idx,
                'gleamer_finding': row.get('GLEAMER_FINDING', 'UNKNOWN'),
                'study_description': row.get('STUDY_DESCRIPTION', ''),
                'body_part_array': row.get('BODY_PART_ARRAY', ''),
                'clinical_indication': row.get('clinical_indication', ''),
                'exam_technique': row.get('exam_technique', ''),
                'findings': row.get('findings', ''),
                'download_urls': row.get('download_urls', ''),
                'images_found': [],
                'images_not_found': []
            }
            
            # Parse download_urls and copy images
            urls = row['download_urls']
            if pd.notna(urls):
                try:
                    url_list = ast.literal_eval(urls)
                    
                    for url_idx, url in enumerate(url_list):
                        filename = url.split('/')[-1]
                        image_path = self.find_image_file(filename)
                        
                        if image_path:
                            # Copy image to record folder with original filename
                            dest_path = record_folder / filename
                            shutil.copy2(image_path, dest_path)
                            
                            record_data['images_found'].append({
                                'url_index': url_idx,
                                'filename': filename,
                                'url': url
                            })
                            
                            logger.info(f"✓ Copied image: {filename}")
                            
                        else:
                            record_data['images_not_found'].append({
                                'url_index': url_idx,
                                'filename': filename,
                                'url': url
                            })
                            
                            logger.warning(f"✗ Image not found: {filename}")
                
                except Exception as e:
                    logger.error(f"Error parsing URLs for record {record_idx}: {e}")
                    record_data['error'] = str(e)
            
            # Save record data
            record_data_file = record_folder / "record_data.json"
            with open(record_data_file, 'w') as f:
                json.dump(record_data, f, indent=2)
            
            # Save record summary
            summary_file = record_folder / "summary.txt"
            with open(summary_file, 'w') as f:
                f.write(f"Record {record_idx} Summary\n")
                f.write("=" * 30 + "\n\n")
                f.write(f"GLEAMER_FINDING: {record_data['gleamer_finding']}\n")
                f.write(f"STUDY_DESCRIPTION: {record_data['study_description']}\n")
                f.write(f"BODY_PART_ARRAY: {record_data['body_part_array']}\n")
                f.write(f"Images found: {len(record_data['images_found'])}\n")
                f.write(f"Images not found: {len(record_data['images_not_found'])}\n")
                f.write(f"Total URLs: {len(url_list) if 'url_list' in locals() else 0}\n")
            
            logger.info(f"✅ Record {record_idx} copied successfully")
            logger.info(f"   Images found: {len(record_data['images_found'])}")
            logger.info(f"   Images not found: {len(record_data['images_not_found'])}")
            
            return record_data
            
        except Exception as e:
            logger.error(f"Error copying record {record_idx}: {e}")
            return {'error': str(e)}
    
    def copy_sample_records(self, num_records=3, start_idx=0, filter_positive_only=True):
        """
        Copy sample records with their associated images
        
        Args:
            num_records: Number of records to copy
            start_idx: Starting index for records
            filter_positive_only: If True, only copy POSITIVE records
        """
        logger.info(f"🚀 Starting to copy {num_records} sample records...")
        
        # Load CSV data
        logger.info(f"Loading CSV data from: {self.csv_path}")
        df = pd.read_csv(self.csv_path)
        logger.info(f"Loaded {len(df)} records")
        
        # Filter for positive records if requested
        if filter_positive_only:
            positive_df = df[df['GLEAMER_FINDING'] == 'POSITIVE']
            logger.info(f"Found {len(positive_df)} POSITIVE records")
            logger.info(f"Filtering for POSITIVE records only")
            df = positive_df
        
        # Copy records
        all_results = []
        total_images_found = 0
        total_images_not_found = 0
        records_processed = 0
        
        for i in range(num_records):
            record_idx = start_idx + i
            
            if record_idx >= len(df):
                logger.warning(f"Record index {record_idx} exceeds available records ({len(df)})")
                break
            
            # Get the actual row from filtered dataframe
            row = df.iloc[record_idx]
            original_idx = df.index[record_idx]  # Original index in full dataset
            
            logger.info(f"Processing record {original_idx} (POSITIVE: {row['GLEAMER_FINDING']})...")
            result = self.copy_record(row, original_idx)
            all_results.append(result)
            records_processed += 1
            
            if 'images_found' in result:
                total_images_found += len(result['images_found'])
            if 'images_not_found' in result:
                total_images_not_found += len(result['images_not_found'])
        
        # Save overall results
        results_file = self.output_dir / "copy_results.json"
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        # Generate summary report
        self.generate_summary_report(all_results, total_images_found, total_images_not_found, records_processed, filter_positive_only)
        
        logger.info("✅ Sample records copying complete!")
        
        return all_results
    
    def generate_summary_report(self, results, total_images_found, total_images_not_found, records_processed, filter_positive_only):
        """Generate a summary report"""
        
        report_file = self.output_dir / "copy_summary.txt"
        
        with open(report_file, 'w') as f:
            f.write("Sample Records Copy Summary\n")
            f.write("=" * 40 + "\n\n")
            
            if filter_positive_only:
                f.write("🔍 FILTERED FOR POSITIVE RECORDS ONLY\n")
                f.write("-" * 40 + "\n")
            
            f.write(f"Records processed: {records_processed}\n")
            f.write(f"Total images found: {total_images_found}\n")
            f.write(f"Total images not found: {total_images_not_found}\n")
            f.write(f"Success rate: {(total_images_found/(total_images_found+total_images_not_found)*100):.1f}%\n\n")
            
            f.write("Record Details:\n")
            f.write("-" * 20 + "\n")
            for i, result in enumerate(results):
                if 'error' not in result:
                    f.write(f"Record {result['record_index']}: {result['gleamer_finding']} - {len(result['images_found'])} images\n")
                else:
                    f.write(f"Record {result.get('record_index', i)}: ERROR - {result['error']}\n")
            
            f.write(f"\nOutput directory: {self.output_dir}\n")
            f.write(f"Each record folder contains:\n")
            f.write(f"  - record_data.json (detailed data)\n")
            f.write(f"  - summary.txt (quick summary)\n")
            f.write(f"  - *.jpg (associated images with original filenames)\n")
        
        logger.info(f"Summary report saved to: {report_file}")

def main():
    print("📁 Sample Records Copier")
    print("=" * 40)
    
    # Default paths (update these for your setup)
    csv_path = "/mount/civiescaks01storage01/aksfileshare01/CNN/gleamer-images/processed_dicom_image_url_file.csv"  # Server CSV
    image_root = "/mount/civiescaks01storage01/aksfileshare01/CNN/gleamer-images"  # Server gleamer-images folder
    output_dir = "sample_records"
    
    # Check if CSV exists
    if not os.path.exists(csv_path):
        print(f"❌ CSV file not found: {csv_path}")
        print("Please update the csv_path in this script")
        return
    
    print(f"📄 CSV file: {csv_path}")
    print(f"📁 Image root: {image_root}")
    print(f"📁 Output directory: {output_dir}")
    print()
    
    # Ask user for options
    try:
        num_records = int(input("Number of records to copy (default 3): ") or "3")
        start_idx = int(input("Starting record index (default 0): ") or "0")
        filter_positive = input("Filter for POSITIVE records only? (y/n, default y): ").lower().strip()
        filter_positive_only = filter_positive in ['y', 'yes', '']  # Default to True
    except ValueError:
        print("❌ Invalid input. Using defaults: 3 records starting from index 0, POSITIVE only")
        num_records = 3
        start_idx = 0
        filter_positive_only = True
    
    # Initialize copier
    copier = SampleRecordCopier(
        csv_path=csv_path,
        image_root=image_root,
        output_dir=output_dir
    )
    
    # Copy records
    results = copier.copy_sample_records(num_records=num_records, start_idx=start_idx, filter_positive_only=filter_positive_only)
    
    print(f"\n✅ Copying complete!")
    print(f"📊 Results saved in: {output_dir}")
    print(f"📁 Check individual record folders for images and data")

if __name__ == "__main__":
    main()
