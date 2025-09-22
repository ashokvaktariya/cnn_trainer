#!/usr/bin/env python3
"""
Data Preparation Script for Binary Medical Image Classification

This script:
1. Filters out blank images and DOUBT cases
2. Balances POSITIVE/NEGATIVE classes
3. Creates clean dataset for binary classification
4. Generates dataset statistics and reports
"""

import pandas as pd
import numpy as np
import os
import logging
from PIL import Image
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import yaml

# Load YAML configuration
with open('config_training.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataPreparator:
    """Data preparation and analysis for binary classification"""
    
    def __init__(self, csv_file=None):
        self.csv_file = csv_file or config['data']['csv_path']
        self.image_root = config['data']['image_root']
        self.output_dir = config['data']['output_dir']
        self.data = None
        self.clean_data = None
        self.stats = {}
        
        logger.info(f"🔧 Initializing Data Preparator with {self.csv_file}")
    
    def load_data(self):
        """Load and initial analysis of CSV data"""
        logger.info("📊 Loading CSV data...")
        
        self.data = pd.read_csv(self.csv_file)
        logger.info(f"✅ Loaded {len(self.data)} records from CSV")
        
        # Initial analysis
        self._analyze_initial_data()
        
        return self.data
    
    def _analyze_initial_data(self):
        """Analyze initial data distribution"""
        logger.info("🔍 Analyzing initial data distribution...")
        
        # GLEAMER_FINDING distribution
        finding_counts = self.data['GLEAMER_FINDING'].value_counts()
        logger.info(f"📊 GLEAMER_FINDING distribution:")
        for finding, count in finding_counts.items():
            percentage = (count / len(self.data)) * 100
            logger.info(f"   {finding}: {count:,} ({percentage:.1f}%)")
        
        self.stats['initial'] = {
            'total_records': len(self.data),
            'gleamer_distribution': dict(finding_counts),
            'columns': list(self.data.columns)
        }
    
    def _find_image_file(self, uid):
        """Find image file for given UID"""
        base_dir = self.image_root  # Use config path
        possible_extensions = ['.jpg', '.jpeg', '.png', '.dcm', '.dicom']
        
        for ext in possible_extensions:
            # Check main directory
            image_path = os.path.join(base_dir, f"{uid}{ext}")
            if os.path.exists(image_path):
                return image_path
            
            # Check subdirectories
            for subdir in ['images', 'data', 'positive', 'negative', 'Negetive', 'Negative 2', 'fracture', 'normal']:
                image_path = os.path.join(base_dir, subdir, f"{uid}{ext}")
                if os.path.exists(image_path):
                    return image_path
        
        return None
    
    def _is_valid_image(self, image_path):
        """Check if image is valid (not blank)"""
        try:
            with Image.open(image_path) as img:
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                img_array = np.array(img)
                
                # Check if all pixels are the same color (blank)
                if len(img_array.shape) == 3:
                    if np.all(img_array == img_array[0, 0]):
                        return False
                    
                    # Check variance - very low variance indicates blank image
                    variance = np.var(img_array)
                    if variance < 10:
                        return False
                else:
                    if np.all(img_array == img_array[0, 0]):
                        return False
                    variance = np.var(img_array)
                    if variance < 10:
                        return False
                
                return True
        except Exception as e:
            logger.warning(f"⚠️ Error checking image {image_path}: {e}")
            return False
    
    def filter_valid_images(self, sample_size=None):
        """Filter data to keep only records with valid images"""
        logger.info("🖼️ Filtering valid images...")
        
        # Use sample for testing if specified
        data_to_process = self.data
        if sample_size and sample_size < len(self.data):
            data_to_process = self.data.sample(n=sample_size, random_state=42)
            logger.info(f"📊 Processing sample of {sample_size} records")
        
        valid_records = []
        total_processed = 0
        valid_images_found = 0
        
        for idx, row in data_to_process.iterrows():
            total_processed += 1
            
            if total_processed % 1000 == 0:
                logger.info(f"   Processed {total_processed}/{len(data_to_process)} records...")
            
            # Parse UIDs
            uid_string = str(row['SOP_INSTANCE_UID_ARRAY'])
            try:
                import json
                # Try to parse as JSON array first
                try:
                    uids = json.loads(uid_string)
                    if not isinstance(uids, list):
                        uids = [uids]
                except json.JSONDecodeError:
                    # Fallback to comma-separated parsing
                    if ',' in uid_string:
                        uids = [uid.strip().strip('"\'[]') for uid in uid_string.split(',')]
                    else:
                        uids = [uid_string.strip().strip('"\'[]')]
                
                # Check for valid images
                has_valid_image = False
                for uid in uids:
                    if uid and uid != 'nan':
                        uid = uid.strip()
                        image_path = self._find_image_file(uid)
                        if image_path and self._is_valid_image(image_path):
                            has_valid_image = True
                            valid_images_found += 1
                            break
                
                if has_valid_image:
                    valid_records.append(row)
                    
            except Exception as e:
                logger.warning(f"⚠️ Error processing row {idx}: {e}")
                continue
        
        self.clean_data = pd.DataFrame(valid_records)
        
        logger.info(f"✅ Image filtering complete:")
        logger.info(f"   Processed: {total_processed:,} records")
        logger.info(f"   Valid images found: {valid_images_found:,}")
        logger.info(f"   Valid records: {len(self.clean_data):,}")
        logger.info(f"   Success rate: {len(self.clean_data)/total_processed*100:.1f}%")
        
        self.stats['image_filtering'] = {
            'total_processed': total_processed,
            'valid_images_found': valid_images_found,
            'valid_records': len(self.clean_data),
            'success_rate': len(self.clean_data)/total_processed*100
        }
        
        return self.clean_data
    
    def filter_binary_labels(self):
        """Filter data for binary classification (exclude DOUBT)"""
        logger.info("🏷️ Filtering for binary classification...")
        
        if self.clean_data is None or len(self.clean_data) == 0:
            logger.error("❌ No clean data available. Run filter_valid_images() first.")
            logger.error("❌ This usually means no valid images were found. Check image paths.")
            return None
        
        # Exclude DOUBT cases
        binary_data = self.clean_data[
            self.clean_data['GLEAMER_FINDING'].isin(['POSITIVE', 'NEGATIVE'])
        ].copy()
        
        # Add binary labels
        binary_data['binary_label'] = binary_data['GLEAMER_FINDING'].map({
            'NEGATIVE': 0,
            'POSITIVE': 1
        })
        
        logger.info(f"✅ Binary filtering complete:")
        logger.info(f"   Original records: {len(self.clean_data):,}")
        logger.info(f"   Binary records: {len(binary_data):,}")
        
        # Class distribution
        class_counts = binary_data['binary_label'].value_counts().sort_index()
        for label, count in class_counts.items():
            label_name = 'NEGATIVE' if label == 0 else 'POSITIVE'
            percentage = (count / len(binary_data)) * 100
            logger.info(f"   {label_name}: {count:,} ({percentage:.1f}%)")
        
        self.clean_data = binary_data
        
        self.stats['binary_filtering'] = {
            'original_records': len(self.clean_data),
            'binary_records': len(binary_data),
            'class_distribution': dict(class_counts)
        }
        
        return self.clean_data
    
    def balance_classes(self, target_ratio=0.5):
        """Balance POSITIVE and NEGATIVE classes"""
        logger.info("⚖️ Balancing classes...")
        
        if self.clean_data is None:
            logger.error("❌ No clean data available.")
            return None
        
        pos_data = self.clean_data[self.clean_data['binary_label'] == 1]
        neg_data = self.clean_data[self.clean_data['binary_label'] == 0]
        
        logger.info(f"📊 Current distribution:")
        logger.info(f"   POSITIVE: {len(pos_data):,}")
        logger.info(f"   NEGATIVE: {len(neg_data):,}")
        
        # Determine target size for each class
        min_class_size = min(len(pos_data), len(neg_data))
        target_size = int(min_class_size * (1 + target_ratio))
        
        # Sample equal number from each class
        if len(pos_data) >= target_size:
            pos_sampled = pos_data.sample(n=target_size, random_state=42)
        else:
            # Oversample POSITIVE class
            pos_sampled = pos_data.sample(n=target_size, replace=True, random_state=42)
        
        if len(neg_data) >= target_size:
            neg_sampled = neg_data.sample(n=target_size, random_state=42)
        else:
            neg_sampled = neg_data.sample(n=target_size, replace=True, random_state=42)
        
        # Combine balanced data
        balanced_data = pd.concat([pos_sampled, neg_sampled], ignore_index=True)
        
        # Shuffle the data
        balanced_data = balanced_data.sample(frac=1, random_state=42).reset_index(drop=True)
        
        logger.info(f"✅ Class balancing complete:")
        logger.info(f"   Final dataset size: {len(balanced_data):,}")
        
        # Final distribution
        final_counts = balanced_data['binary_label'].value_counts().sort_index()
        for label, count in final_counts.items():
            label_name = 'NEGATIVE' if label == 0 else 'POSITIVE'
            percentage = (count / len(balanced_data)) * 100
            logger.info(f"   {label_name}: {count:,} ({percentage:.1f}%)")
        
        self.clean_data = balanced_data
        
        self.stats['class_balancing'] = {
            'final_dataset_size': len(balanced_data),
            'final_class_distribution': dict(final_counts),
            'original_positive': len(pos_data),
            'original_negative': len(neg_data)
        }
        
        return self.clean_data
    
    def save_clean_dataset(self, output_file=None):
        """Save cleaned dataset to file"""
        if self.clean_data is None:
            logger.error("❌ No clean data to save.")
            return None
        
        if output_file is None:
            output_file = os.path.join(PREPROCESSED_DIR, "binary_medical_dataset.csv")
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Save dataset
        self.clean_data.to_csv(output_file, index=False)
        logger.info(f"💾 Clean dataset saved to: {output_file}")
        
        # Save statistics
        stats_file = output_file.replace('.csv', '_stats.json')
        import json
        
        # Convert numpy types to native Python types for JSON serialization
        def convert_numpy_types(obj):
            if isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return obj
        
        # Convert stats to JSON-serializable format
        json_stats = convert_numpy_types(self.stats)
        
        with open(stats_file, 'w') as f:
            json.dump(json_stats, f, indent=2)
        logger.info(f"📊 Statistics saved to: {stats_file}")
        
        return output_file
    
    def generate_report(self, output_dir=None):
        """Generate comprehensive data preparation report"""
        if output_dir is None:
            output_dir = PREPROCESSED_DIR
        
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info("📊 Generating data preparation report...")
        
        # Create visualizations
        self._create_distribution_plots(output_dir)
        
        # Create summary report
        self._create_summary_report(output_dir)
        
        logger.info(f"📋 Report generated in: {output_dir}")
    
    def _create_distribution_plots(self, output_dir):
        """Create distribution plots"""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            plt.style.use('default')
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # 1. Initial GLEAMER_FINDING distribution
            if 'initial' in self.stats:
                finding_dist = self.stats['initial']['gleamer_distribution']
                axes[0, 0].pie(finding_dist.values(), labels=finding_dist.keys(), autopct='%1.1f%%')
                axes[0, 0].set_title('Initial GLEAMER_FINDING Distribution')
            
            # 2. Binary class distribution
            if 'binary_filtering' in self.stats:
                binary_dist = self.stats['binary_filtering']['class_distribution']
                labels = ['NEGATIVE', 'POSITIVE']
                values = [binary_dist.get(0, 0), binary_dist.get(1, 0)]
                axes[0, 1].bar(labels, values)
                axes[0, 1].set_title('Binary Class Distribution')
                axes[0, 1].set_ylabel('Count')
            
            # 3. Final balanced distribution
            if 'class_balancing' in self.stats:
                final_dist = self.stats['class_balancing']['final_class_distribution']
                labels = ['NEGATIVE', 'POSITIVE']
                values = [final_dist.get(0, 0), final_dist.get(1, 0)]
                axes[1, 0].bar(labels, values, color=['lightblue', 'lightcoral'])
                axes[1, 0].set_title('Final Balanced Distribution')
                axes[1, 0].set_ylabel('Count')
            
            # 4. Processing pipeline summary
            pipeline_data = []
            if 'initial' in self.stats:
                pipeline_data.append(('Initial', self.stats['initial']['total_records']))
            if 'image_filtering' in self.stats:
                pipeline_data.append(('After Image Filtering', self.stats['image_filtering']['valid_records']))
            if 'binary_filtering' in self.stats:
                pipeline_data.append(('After Binary Filtering', self.stats['binary_filtering']['binary_records']))
            if 'class_balancing' in self.stats:
                pipeline_data.append(('Final Balanced', self.stats['class_balancing']['final_dataset_size']))
            
            if pipeline_data:
                steps, counts = zip(*pipeline_data)
                axes[1, 1].plot(steps, counts, marker='o', linewidth=2, markersize=8)
                axes[1, 1].set_title('Data Processing Pipeline')
                axes[1, 1].set_ylabel('Record Count')
                axes[1, 1].tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'data_preparation_report.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
        except ImportError:
            logger.warning("⚠️ matplotlib/seaborn not available for plotting")
    
    def _create_summary_report(self, output_dir):
        """Create text summary report"""
        report_file = os.path.join(output_dir, 'data_preparation_summary.txt')
        
        with open(report_file, 'w') as f:
            f.write("📊 BINARY MEDICAL IMAGE CLASSIFICATION - DATA PREPARATION REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            # Initial data
            if 'initial' in self.stats:
                f.write("📋 INITIAL DATA:\n")
                f.write(f"   Total Records: {self.stats['initial']['total_records']:,}\n")
                f.write("   GLEAMER_FINDING Distribution:\n")
                for finding, count in self.stats['initial']['gleamer_distribution'].items():
                    percentage = (count / self.stats['initial']['total_records']) * 100
                    f.write(f"     {finding}: {count:,} ({percentage:.1f}%)\n")
                f.write("\n")
            
            # Image filtering
            if 'image_filtering' in self.stats:
                f.write("🖼️ IMAGE FILTERING:\n")
                f.write(f"   Processed Records: {self.stats['image_filtering']['total_processed']:,}\n")
                f.write(f"   Valid Images Found: {self.stats['image_filtering']['valid_images_found']:,}\n")
                f.write(f"   Valid Records: {self.stats['image_filtering']['valid_records']:,}\n")
                f.write(f"   Success Rate: {self.stats['image_filtering']['success_rate']:.1f}%\n\n")
            
            # Binary filtering
            if 'binary_filtering' in self.stats:
                f.write("🏷️ BINARY FILTERING:\n")
                f.write(f"   Binary Records: {self.stats['binary_filtering']['binary_records']:,}\n")
                f.write("   Class Distribution:\n")
                for label, count in self.stats['binary_filtering']['class_distribution'].items():
                    label_name = 'NEGATIVE' if label == 0 else 'POSITIVE'
                    percentage = (count / self.stats['binary_filtering']['binary_records']) * 100
                    f.write(f"     {label_name}: {count:,} ({percentage:.1f}%)\n")
                f.write("\n")
            
            # Final balanced dataset
            if 'class_balancing' in self.stats:
                f.write("⚖️ FINAL BALANCED DATASET:\n")
                f.write(f"   Total Records: {self.stats['class_balancing']['final_dataset_size']:,}\n")
                f.write("   Final Class Distribution:\n")
                for label, count in self.stats['class_balancing']['final_class_distribution'].items():
                    label_name = 'NEGATIVE' if label == 0 else 'POSITIVE'
                    percentage = (count / self.stats['class_balancing']['final_dataset_size']) * 100
                    f.write(f"     {label_name}: {count:,} ({percentage:.1f}%)\n")
                f.write("\n")
            
            f.write("✅ DATA PREPARATION COMPLETE!\n")
            f.write("Ready for binary classification training.\n")

def main():
    """Main data preparation pipeline"""
    logger.info("🚀 Starting Binary Medical Image Classification Data Preparation")
    
    # Initialize preparator
    preparator = DataPreparator()
    
    # Load data
    preparator.load_data()
    
    # Filter valid images (full dataset processing)
    preparator.filter_valid_images()  # Process full dataset
    
    # Filter for binary classification
    preparator.filter_binary_labels()
    
    # Balance classes
    preparator.balance_classes()
    
    # Save clean dataset
    preparator.save_clean_dataset()
    
    # Generate report
    preparator.generate_report()
    
    logger.info("🎉 Data preparation complete!")
    
    return preparator

if __name__ == "__main__":
    preparator = main()
