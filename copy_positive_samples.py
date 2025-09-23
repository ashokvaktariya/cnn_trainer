#!/usr/bin/env python3
"""
Script to copy positive fracture samples with their associated images
Uses the gleamer-images mount folder and CSV data
"""

import pandas as pd
import os
import shutil
import json
import logging
from pathlib import Path
import argparse

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PositiveSampleCopier:
    def __init__(self, csv_path, image_root, output_dir="positive_samples"):
        self.csv_path = csv_path
        self.image_root = image_root
        self.output_dir = output_dir
        self.data = None
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        logger.info(f"🔧 Initialized Positive Sample Copier")
        logger.info(f"📊 CSV: {self.csv_path}")
        logger.info(f"🖼️ Images: {self.image_root}")
        logger.info(f"📁 Output: {self.output_dir}")
    
    def load_data(self):
        """Load CSV data"""
        logger.info("📊 Loading CSV data...")
        self.data = pd.read_csv(self.csv_path)
        logger.info(f"✅ Loaded {len(self.data)} records")
        
        # Filter for positive samples only
        positive_data = self.data[self.data['GLEAMER_FINDING'] == 'POSITIVE'].copy()
        logger.info(f"🎯 Found {len(positive_data)} positive samples")
        
        self.data = positive_data
        return self.data
    
    def _find_image_file_by_filename(self, filename):
        """Find image file by exact filename in the gleamer-images folder"""
        # Check main directory
        image_path = os.path.join(self.image_root, filename)
        if os.path.exists(image_path):
            return image_path
        
        # Check subdirectories recursively
        for root, dirs, files in os.walk(self.image_root):
            if filename in files:
                return os.path.join(root, filename)
        
        return None
    
    def copy_positive_samples(self, num_samples=None, start_idx=0):
        """Copy positive samples with their images"""
        if self.data is None:
            logger.error("❌ No data loaded. Run load_data() first.")
            return
        
        # Limit samples if specified
        if num_samples:
            end_idx = min(start_idx + num_samples, len(self.data))
            sample_data = self.data.iloc[start_idx:end_idx]
        else:
            sample_data = self.data.iloc[start_idx:]
        
        logger.info(f"🚀 Copying {len(sample_data)} positive samples...")
        
        results = []
        total_images_found = 0
        total_images_not_found = 0
        
        for idx, (_, row) in enumerate(sample_data.iterrows()):
            record_idx = start_idx + idx
            logger.info(f"📋 Processing record {record_idx}...")
            
            # Create record folder
            record_folder = Path(self.output_dir) / f"positive_{record_idx}"
            record_folder.mkdir(exist_ok=True)
            
            # Save record data
            record_data = {
                'record_index': int(record_idx),
                'gleamer_finding': str(row.get('GLEAMER_FINDING', 'POSITIVE')),
                'confidence_score': str(row.get('CONFIDENCE_SCORE', 'N/A')),
                'sop_instance_uid': str(row.get('SOP_INSTANCE_UID_ARRAY', 'N/A')),
                'download_urls': str(row.get('download_urls', 'N/A')),
                'total_images': 0,
                'images_found': 0,
                'images_not_found': 0
            }
            
            # Parse download URLs
            download_urls_string = str(row.get('download_urls', ''))
            image_filenames = []
            
            try:
                # Try to parse as JSON array first
                try:
                    download_urls = json.loads(download_urls_string)
                    if not isinstance(download_urls, list):
                        download_urls = [download_urls]
                except json.JSONDecodeError:
                    # Fallback to comma-separated parsing
                    if ',' in download_urls_string:
                        download_urls = [url.strip().strip('"\'[]') for url in download_urls_string.split(',')]
                    else:
                        download_urls = [download_urls_string.strip().strip('"\'[]')]
                
                # Extract filenames from URLs
                for url in download_urls:
                    if url and url != 'nan':
                        filename = os.path.basename(url.strip())
                        if filename:
                            image_filenames.append(filename)
                
            except Exception as e:
                logger.warning(f"⚠️ Error parsing URLs for record {record_idx}: {e}")
                image_filenames = []
            
            # Copy images
            images_found = 0
            images_not_found = 0
            
            for img_idx, filename in enumerate(image_filenames):
                if filename:
                    image_path = self._find_image_file_by_filename(filename)
                    if image_path:
                        # Copy image with original filename
                        dest_path = record_folder / filename
                        try:
                            shutil.copy2(image_path, dest_path)
                            images_found += 1
                            logger.info(f"✅ Copied: {filename}")
                        except Exception as e:
                            logger.warning(f"⚠️ Error copying {filename}: {e}")
                            images_not_found += 1
                    else:
                        logger.warning(f"❌ Image not found: {filename}")
                        images_not_found += 1
            
            # Update record data
            record_data['total_images'] = len(image_filenames)
            record_data['images_found'] = images_found
            record_data['images_not_found'] = images_not_found
            
            # Save record data as JSON
            with open(record_folder / 'record_data.json', 'w') as f:
                json.dump(record_data, f, indent=2)
            
            # Save record data as text for easy reading
            with open(record_folder / 'record_info.txt', 'w') as f:
                f.write(f"POSITIVE FRACTURE SAMPLE - Record {record_idx}\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"GLEAMER Finding: {record_data['gleamer_finding']}\n")
                f.write(f"Confidence Score: {record_data['confidence_score']}\n")
                f.write(f"Total Images: {record_data['total_images']}\n")
                f.write(f"Images Found: {record_data['images_found']}\n")
                f.write(f"Images Not Found: {record_data['images_not_found']}\n")
                f.write(f"Success Rate: {(images_found/max(len(image_filenames), 1)*100):.1f}%\n\n")
                f.write("Download URLs:\n")
                f.write(f"{record_data['download_urls']}\n\n")
                f.write("SOP Instance UID:\n")
                f.write(f"{record_data['sop_instance_uid']}\n")
            
            results.append(record_data)
            total_images_found += images_found
            total_images_not_found += images_not_found
            
            logger.info(f"✅ Record {record_idx} completed: {images_found}/{len(image_filenames)} images")
        
        # Generate summary report
        self.generate_summary_report(results, total_images_found, total_images_not_found, len(sample_data))
        
        return results
    
    def generate_summary_report(self, results, total_images_found, total_images_not_found, records_processed):
        """Generate summary report"""
        summary_file = os.path.join(self.output_dir, "POSITIVE_SAMPLES_SUMMARY.txt")
        
        with open(summary_file, 'w') as f:
            f.write("🎯 POSITIVE FRACTURE SAMPLES - COPY SUMMARY\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"📊 Records Processed: {records_processed}\n")
            f.write(f"🖼️ Total Images Found: {total_images_found}\n")
            f.write(f"❌ Total Images Not Found: {total_images_not_found}\n")
            
            total_images = total_images_found + total_images_not_found
            if total_images > 0:
                success_rate = (total_images_found / total_images) * 100
                f.write(f"✅ Overall Success Rate: {success_rate:.1f}%\n\n")
            else:
                f.write(f"✅ Overall Success Rate: N/A (no images processed)\n\n")
            
            f.write("📋 Individual Record Summary:\n")
            f.write("-" * 40 + "\n")
            for result in results:
                f.write(f"Record {result['record_index']}: ")
                f.write(f"{result['images_found']}/{result['total_images']} images ")
                if result['total_images'] > 0:
                    rate = (result['images_found'] / result['total_images']) * 100
                    f.write(f"({rate:.1f}%)\n")
                else:
                    f.write("(N/A)\n")
            
            f.write(f"\n📁 Output Directory: {self.output_dir}\n")
            f.write(f"📊 CSV Source: {self.csv_path}\n")
            f.write(f"🖼️ Image Source: {self.image_root}\n")
        
        logger.info(f"📊 Summary report saved: {summary_file}")

def main():
    parser = argparse.ArgumentParser(description='Copy positive fracture samples with images')
    parser.add_argument('--csv', default='/mount/civiescaks01storage01/aksfileshare01/CNN/gleamer-images/processed_dicom_image_url_file.csv',
                       help='Path to CSV file')
    parser.add_argument('--images', default='/mount/civiescaks01storage01/aksfileshare01/CNN/gleamer-images/',
                       help='Path to gleamer-images folder')
    parser.add_argument('--output', default='positive_samples',
                       help='Output directory for positive samples')
    parser.add_argument('--num-samples', type=int, default=None,
                       help='Number of samples to copy (default: all)')
    parser.add_argument('--start-idx', type=int, default=0,
                       help='Starting index for samples')
    
    args = parser.parse_args()
    
    logger.info("🚀 Starting Positive Sample Copy Process")
    logger.info("=" * 50)
    
    # Create copier
    copier = PositiveSampleCopier(
        csv_path=args.csv,
        image_root=args.images,
        output_dir=args.output
    )
    
    # Load data
    copier.load_data()
    
    # Copy samples
    results = copier.copy_positive_samples(
        num_samples=args.num_samples,
        start_idx=args.start_idx
    )
    
    logger.info("🎉 Positive sample copying completed!")
    logger.info(f"📁 Check output in: {args.output}")

if __name__ == "__main__":
    main()
