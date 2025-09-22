#!/usr/bin/env python3
"""
Copy Sample Images from Each UID Series
Copies 5 images from each UID pattern to analyze file types
"""

import os
import json
import shutil
import pandas as pd
import ast
from pathlib import Path
import logging
from collections import defaultdict

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class UIDSampleCopier:
    def __init__(self, csv_path, image_root, output_dir="uid_samples"):
        """
        Initialize the UID sample copier
        
        Args:
            csv_path: Path to the CSV file
            image_root: Root directory containing gleamer-images
            output_dir: Output directory for sample images
        """
        self.csv_path = csv_path
        self.image_root = Path(image_root)
        self.output_dir = Path(output_dir)
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Define UID patterns to analyze
        self.uid_patterns = {
            '1.2.250': 'Summary_Reports',
            '1.2.840.113619': 'Xray_Images_Standard',
            '1.2.392.200036': 'Xray_Images_Alternative',
            '1.2.410.200010': 'Other_Images_German',
            '1.2.840': 'Xray_Images_General'
        }
        
        # Create subdirectories for each pattern
        for pattern, dir_name in self.uid_patterns.items():
            pattern_dir = self.output_dir / dir_name
            pattern_dir.mkdir(exist_ok=True)
        
    def find_image_file(self, filename):
        """Find image file in gleamer-images directory"""
        # Search in root and subdirectories
        for root, dirs, files in os.walk(self.image_root):
            if filename in files:
                return os.path.join(root, filename)
        return None
    
    def copy_uid_samples(self, samples_per_pattern=5):
        """
        Copy sample images from each UID pattern
        
        Args:
            samples_per_pattern: Number of samples to copy per UID pattern
        """
        logger.info(f"🚀 Starting to copy {samples_per_pattern} samples per UID pattern...")
        
        # Load CSV data
        logger.info(f"Loading CSV data from: {self.csv_path}")
        df = pd.read_csv(self.csv_path)
        logger.info(f"Loaded {len(df)} records")
        
        # Collect UIDs by pattern
        uid_collections = defaultdict(list)
        
        logger.info("🔍 Collecting UIDs by pattern...")
        
        for idx, sop_array in df['SOP_INSTANCE_UID_ARRAY'].items():
            try:
                uid_list = ast.literal_eval(sop_array)
                
                for uid in uid_list:
                    # Find matching pattern
                    for pattern in self.uid_patterns.keys():
                        if uid.startswith(pattern):
                            uid_collections[pattern].append(uid)
                            break
                            
            except Exception as e:
                logger.error(f"Error parsing row {idx}: {e}")
        
        # Display collection results
        logger.info("📊 UID Collection Results:")
        for pattern, uids in uid_collections.items():
            logger.info(f"  {pattern}: {len(uids)} UIDs collected")
        
        # Copy samples for each pattern
        results = {}
        
        for pattern, uids in uid_collections.items():
            logger.info(f"📁 Processing pattern: {pattern}")
            
            # Get unique UIDs and limit to samples_per_pattern
            unique_uids = list(set(uids))[:samples_per_pattern]
            pattern_dir = self.output_dir / self.uid_patterns[pattern]
            
            pattern_results = {
                'pattern': pattern,
                'total_uids_available': len(set(uids)),
                'samples_copied': 0,
                'samples_not_found': 0,
                'copied_files': [],
                'not_found_files': []
            }
            
            for i, uid in enumerate(unique_uids):
                # Create filename (assuming .jpg extension)
                filename = f"{uid}.jpg"
                image_path = self.find_image_file(filename)
                
                if image_path:
                    # Copy image
                    dest_filename = f"sample_{i+1}_{filename}"
                    dest_path = pattern_dir / dest_filename
                    shutil.copy2(image_path, dest_path)
                    
                    pattern_results['copied_files'].append({
                        'uid': uid,
                        'filename': filename,
                        'dest_filename': dest_filename
                    })
                    pattern_results['samples_copied'] += 1
                    
                    logger.info(f"  ✓ Copied: {filename} -> {dest_filename}")
                    
                else:
                    pattern_results['not_found_files'].append({
                        'uid': uid,
                        'filename': filename
                    })
                    pattern_results['samples_not_found'] += 1
                    
                    logger.warning(f"  ✗ Not found: {filename}")
            
            results[pattern] = pattern_results
            
            logger.info(f"  📊 Pattern {pattern} summary:")
            logger.info(f"    Copied: {pattern_results['samples_copied']}")
            logger.info(f"    Not found: {pattern_results['samples_not_found']}")
            logger.info(f"    Total available: {pattern_results['total_uids_available']}")
        
        # Save results
        results_file = self.output_dir / "copy_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Generate summary report
        self.generate_summary_report(results)
        
        logger.info("✅ UID sample copying complete!")
        
        return results
    
    def generate_summary_report(self, results):
        """Generate a summary report"""
        
        report_file = self.output_dir / "copy_summary.txt"
        
        with open(report_file, 'w') as f:
            f.write("UID Pattern Sample Copy Summary\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("📊 PATTERN ANALYSIS:\n")
            f.write("-" * 30 + "\n")
            
            for pattern, result in results.items():
                f.write(f"\n🔍 Pattern: {pattern}\n")
                f.write(f"  Description: {self.uid_patterns[pattern]}\n")
                f.write(f"  Total UIDs available: {result['total_uids_available']:,}\n")
                f.write(f"  Samples copied: {result['samples_copied']}\n")
                f.write(f"  Samples not found: {result['samples_not_found']}\n")
                f.write(f"  Success rate: {(result['samples_copied']/(result['samples_copied']+result['samples_not_found'])*100):.1f}%\n")
                
                if result['copied_files']:
                    f.write(f"  Copied files:\n")
                    for file_info in result['copied_files']:
                        f.write(f"    - {file_info['dest_filename']}\n")
            
            f.write(f"\n📁 Output Structure:\n")
            f.write(f"  {self.output_dir}/\n")
            for pattern, dir_name in self.uid_patterns.items():
                f.write(f"    ├── {dir_name}/\n")
                f.write(f"    │   └── sample_*.jpg files\n")
            
            f.write(f"\n📄 Files:\n")
            f.write(f"  - copy_results.json (detailed results)\n")
            f.write(f"  - copy_summary.txt (this summary)\n")
        
        logger.info(f"Summary report saved to: {report_file}")

def main():
    print("🏥 UID Pattern Sample Copier")
    print("=" * 50)
    
    # Default paths (update these for your setup)
    csv_path = "/mount/civiescaks01storage01/aksfileshare01/CNN/gleamer-images/processed_dicom_image_url_file.csv"
    image_root = "/mount/civiescaks01storage01/aksfileshare01/CNN/gleamer-images/gleamer-images"
    output_dir = "uid_samples"
    
    print(f"📄 CSV file: {csv_path}")
    print(f"📁 Image root: {image_root}")
    print(f"📁 Output directory: {output_dir}")
    print()
    
    # Ask user for options
    try:
        samples_per_pattern = int(input("Number of samples per pattern (default 5): ") or "5")
    except ValueError:
        print("❌ Invalid input. Using default: 5 samples per pattern")
        samples_per_pattern = 5
    
    # Initialize copier
    copier = UIDSampleCopier(
        csv_path=csv_path,
        image_root=image_root,
        output_dir=output_dir
    )
    
    # Copy samples
    results = copier.copy_uid_samples(samples_per_pattern=samples_per_pattern)
    
    print(f"\n✅ Copying complete!")
    print(f"📊 Results saved in: {output_dir}")
    print(f"📁 Check subdirectories for sample images from each UID pattern")

if __name__ == "__main__":
    main()
