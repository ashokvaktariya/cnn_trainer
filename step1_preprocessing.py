#!/usr/bin/env python3
"""
Step 1: Dataset Preprocessing
Validates, preprocesses, and caches the medical imaging dataset
H200 GPU Server Optimized
"""

import pandas as pd
import numpy as np
import torch
import requests
from PIL import Image
import io
import os
import json
import logging
from pathlib import Path
from tqdm import tqdm
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

from config import (
    CSV_FILE, DATA_ROOT, PREPROCESSED_DIR, PREPROCESSING_CONFIG,
    TRAINING_CONFIG, LOGGING_CONFIG
)
from image_alignment_system import ImageAlignmentSystem

# Setup logging
logging.basicConfig(
    level=getattr(logging, LOGGING_CONFIG['log_level']),
    format=LOGGING_CONFIG['log_format'],
    handlers=[
        logging.FileHandler(LOGGING_CONFIG['log_file']),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DatasetPreprocessor:
    """Preprocess and validate the medical imaging dataset"""
    
    def __init__(self):
        self.csv_file = CSV_FILE
        self.preprocessed_dir = PREPROCESSED_DIR
        self.images_dir = os.path.join(DATA_ROOT, "images")
        self.cache_file = os.path.join(self.preprocessed_dir, "preprocessed_data.pkl")
        self.stats_file = os.path.join(self.preprocessed_dir, "dataset_stats.json")
        
        # Create directories
        os.makedirs(self.preprocessed_dir, exist_ok=True)
        os.makedirs(self.images_dir, exist_ok=True)
        
        # Initialize image alignment system
        self.alignment_system = ImageAlignmentSystem(DATA_ROOT, self.images_dir)
        
        logger.info(f"🔧 Initialized DatasetPreprocessor")
        logger.info(f"📁 CSV File: {self.csv_file}")
        logger.info(f"💾 Cache Directory: {self.preprocessed_dir}")
        logger.info(f"🖼️ Images Directory: {self.images_dir}")
    
    def load_dataset(self):
        """Load and validate the CSV dataset"""
        logger.info("📊 Loading dataset...")
        
        if not os.path.exists(self.csv_file):
            raise FileNotFoundError(f"CSV file not found: {self.csv_file}")
        
        # Load CSV
        df = pd.read_csv(self.csv_file)
        logger.info(f"✅ Loaded {len(df)} records from CSV")
        
        # Basic validation
        required_columns = [
            'SOP_INSTANCE_UID_ARRAY', 'download_urls', 'clinical_indication',
            'exam_technique', 'findings', 'GLEAMER_FINDING'
        ]
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Check label distribution
        label_counts = df['GLEAMER_FINDING'].value_counts()
        logger.info(f"📈 Label distribution: {dict(label_counts)}")
        
        return df
    
    def validate_image_url(self, url):
        """Validate if image URL is accessible"""
        try:
            response = requests.head(url, timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def download_and_validate_image(self, url, image_id, max_retries=3):
        """Download and validate a single image, save locally with proper alignment"""
        # Use alignment system to get proper file path
        local_path = self.alignment_system.get_image_path(image_id, url)
            
            # Check if image already exists locally
            if os.path.exists(local_path):
                try:
                    image = Image.open(local_path).convert('RGB')
                    return {
                        'image': np.array(image),
                        'size': image.size,
                        'valid': True,
                        'local_path': local_path,
                        'cached': True
                    }
                except Exception as e:
                    logger.warning(f"Error loading cached image {local_path}: {e}")
                    # Remove corrupted cached file
                    os.remove(local_path)
            
            # Download image if not cached
            for attempt in range(max_retries):
                try:
                    response = requests.get(url, timeout=10)
                    if response.status_code != 200:
                        continue
                    
                    # Try to open as image
                    image = Image.open(io.BytesIO(response.content)).convert('RGB')
                    
                    # Validate size
                    width, height = image.size
                    min_size = PREPROCESSING_CONFIG['min_image_size']
                    max_size = PREPROCESSING_CONFIG['max_image_size']
                    
                    if (width < min_size[0] or height < min_size[1] or 
                        width > max_size[0] or height > max_size[1]):
                        logger.warning(f"Image size {width}x{height} outside valid range")
                        if PREPROCESSING_CONFIG['skip_corrupt_images']:
                            return None
                    
                    # Save image locally
                    image.save(local_path, 'JPEG', quality=95)
                    
                    return {
                        'image': np.array(image),
                        'size': (width, height),
                        'valid': True,
                        'local_path': local_path,
                        'cached': False
                    }
                    
                except Exception as e:
                    logger.warning(f"Error downloading image from {url} (attempt {attempt + 1}): {e}")
                    if attempt == max_retries - 1:
                        return None
            
        except Exception as e:
            logger.error(f"Error processing image {image_id}: {e}")
            return None
        
        return None
    
    def preprocess_batch(self, batch_data):
        """Preprocess a batch of records"""
        processed_batch = []
        
        for idx, row in batch_data.iterrows():
            try:
                # Parse image URLs
                image_urls = eval(row['download_urls']) if isinstance(row['download_urls'], str) else row['download_urls']
                
                # Download and validate images
                images = []
                image_paths = []
                valid_image_count = 0
                
                for i, url in enumerate(image_urls[:TRAINING_CONFIG['max_images_per_study']]):
                    # Create proper image ID using alignment system
                    image_id = self.alignment_system.create_image_id(row, i)
                    image_data = self.download_and_validate_image(url, image_id)
                    
                    if image_data and image_data['valid']:
                        images.append(image_data['image'])
                        image_paths.append(image_data['local_path'])
                        valid_image_count += 1
                        
                        # Register image in alignment system
                        self.alignment_system.register_image(
                            csv_row_index=idx,
                            row_data=row,
                            image_index=i,
                            image_url=url,
                            local_path=image_data['local_path']
                        )
                        
                        # Update image info with file size and dimensions
                        file_size = os.path.getsize(image_data['local_path']) if os.path.exists(image_data['local_path']) else None
                        dimensions = image_data['size'] if 'size' in image_data else None
                        self.alignment_system.update_image_info(image_id, file_size, dimensions)
                    else:
                        # Add black image as placeholder
                        black_image = np.zeros(PREPROCESSING_CONFIG['image_size'] + (3,), dtype=np.uint8)
                        images.append(black_image)
                        image_paths.append(None)
                
                # Prepare comprehensive text data using all available columns
                clinical_text = f"{row['clinical_indication']} {row['exam_technique']} {row['findings']}"
                clinical_text = clinical_text.replace('\n', ' ').strip()
                
                # Prepare label
                label = 1 if row['GLEAMER_FINDING'] == 'POSITIVE' else 0
                
                # Parse and clean additional data
                sop_uids = eval(row['SOP_INSTANCE_UID_ARRAY']) if isinstance(row['SOP_INSTANCE_UID_ARRAY'], str) else row['SOP_INSTANCE_UID_ARRAY']
                body_parts = eval(row['BODY_PART_ARRAY']) if isinstance(row['BODY_PART_ARRAY'], str) else row['BODY_PART_ARRAY']
                
                processed_record = {
                    'index': idx,
                    'images': images,
                    'image_paths': image_paths,  # Store local paths
                    'image_count': valid_image_count,
                    'clinical_text': clinical_text,
                    'label': label,
                    
                    # All CSV columns preserved
                    'sop_instance_uid_array': sop_uids,
                    'accession_number': row['ACCESSION_NUMBER'],
                    'study_instance_uid': row['STUDY_INSTANCE_UID'],
                    'study_description': row['STUDY_DESCRIPTION'],
                    'body_part_array': body_parts,
                    'clinical_indication': row['clinical_indication'],
                    'exam_technique': row['exam_technique'],
                    'findings': row['findings'],
                    'gleamer_finding': row['GLEAMER_FINDING'],
                    'original_urls': image_urls[:TRAINING_CONFIG['max_images_per_study']],
                    
                    # Additional metadata
                    'num_images_available': len(image_urls),
                    'num_images_used': min(len(image_urls), TRAINING_CONFIG['max_images_per_study'])
                }
                
                processed_batch.append(processed_record)
                
            except Exception as e:
                logger.error(f"Error processing record {idx}: {e}")
                continue
        
        return processed_batch
    
    def preprocess_dataset(self, sample_size=None, max_workers=8):
        """Preprocess the entire dataset"""
        logger.info("🚀 Starting dataset preprocessing...")
        
        # Load dataset
        df = self.load_dataset()
        
        # Use sample for testing
        if sample_size:
            df = df.sample(n=min(sample_size, len(df)), random_state=42)
            logger.info(f"📝 Using sample of {len(df)} records")
        
        # Split into batches for parallel processing
        batch_size = 100
        batches = [df.iloc[i:i+batch_size] for i in range(0, len(df), batch_size)]
        
        all_processed = []
        
        # Process batches in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_batch = {
                executor.submit(self.preprocess_batch, batch): batch 
                for batch in batches
            }
            
            for future in tqdm(as_completed(future_to_batch), total=len(batches), desc="Processing batches"):
                try:
                    batch_result = future.result()
                    all_processed.extend(batch_result)
                except Exception as e:
                    logger.error(f"Error processing batch: {e}")
        
        logger.info(f"✅ Preprocessed {len(all_processed)} records")
        
        # Calculate statistics
        stats = self.calculate_statistics(all_processed)
        
        # Save preprocessed data
        if PREPROCESSING_CONFIG['cache_preprocessed']:
            self.save_preprocessed_data(all_processed, stats)
        
        # Save alignment data
        self.alignment_system.save_alignment_data()
        
        # Print alignment statistics
        alignment_stats = self.alignment_system.get_alignment_statistics()
        logger.info(f"📊 Alignment Statistics:")
        logger.info(f"   Total Images: {alignment_stats['total_images']}")
        logger.info(f"   Unique CSV Rows: {alignment_stats['unique_csv_rows']}")
        logger.info(f"   Avg Images per Row: {alignment_stats['avg_images_per_row']:.2f}")
        
        return all_processed, stats
    
    def calculate_statistics(self, processed_data):
        """Calculate dataset statistics"""
        logger.info("📊 Calculating dataset statistics...")
        
        total_records = len(processed_data)
        total_images = sum(record['image_count'] for record in processed_data)
        
        label_counts = {}
        for record in processed_data:
            label = record['label']
            label_counts[label] = label_counts.get(label, 0) + 1
        
        # Image size statistics
        image_sizes = []
        for record in processed_data:
            for image in record['images']:
                if image.size > 0:
                    image_sizes.append(image.shape[:2])
        
        # Body part distribution
        body_parts = []
        study_descriptions = []
        exam_techniques = []
        
        for record in processed_data:
            body_parts.extend(record['body_part_array'] if isinstance(record['body_part_array'], list) else [])
            study_descriptions.append(record['study_description'])
            exam_techniques.append(record['exam_technique'])
        
        stats = {
            'total_records': total_records,
            'total_images': total_images,
            'avg_images_per_record': total_images / total_records if total_records > 0 else 0,
            'label_distribution': label_counts,
            'positive_ratio': label_counts.get(1, 0) / total_records if total_records > 0 else 0,
            
            # Image statistics
            'image_size_distribution': {
                'unique_sizes': len(set(image_sizes)),
                'most_common_sizes': pd.Series(image_sizes).value_counts().head(10).to_dict()
            },
            
            # Body part distribution
            'body_part_distribution': pd.Series(body_parts).value_counts().head(20).to_dict(),
            
            # Study description distribution
            'study_description_distribution': pd.Series(study_descriptions).value_counts().head(20).to_dict(),
            
            # Exam technique distribution
            'exam_technique_distribution': pd.Series(exam_techniques).value_counts().head(20).to_dict(),
            
            # Image availability statistics
            'images_per_study_stats': {
                'min_images': min([record['num_images_available'] for record in processed_data]) if processed_data else 0,
                'max_images': max([record['num_images_available'] for record in processed_data]) if processed_data else 0,
                'avg_images_available': np.mean([record['num_images_available'] for record in processed_data]) if processed_data else 0
            },
            
            'preprocessing_config': PREPROCESSING_CONFIG,
            'timestamp': time.time()
        }
        
        logger.info(f"📈 Statistics calculated:")
        logger.info(f"   Total records: {stats['total_records']}")
        logger.info(f"   Total images: {stats['total_images']}")
        logger.info(f"   Avg images per record: {stats['avg_images_per_record']:.2f}")
        logger.info(f"   Label distribution: {stats['label_distribution']}")
        logger.info(f"   Positive ratio: {stats['positive_ratio']:.3f}")
        
        return stats
    
    def save_preprocessed_data(self, processed_data, stats):
        """Save preprocessed data to disk"""
        logger.info("💾 Saving preprocessed data...")
        
        # Save processed data
        with open(self.cache_file, 'wb') as f:
            pickle.dump(processed_data, f)
        
        # Save statistics
        with open(self.stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"✅ Saved preprocessed data to {self.cache_file}")
        logger.info(f"✅ Saved statistics to {self.stats_file}")
    
    def load_preprocessed_data(self):
        """Load preprocessed data from disk"""
        if not os.path.exists(self.cache_file):
            logger.warning("No preprocessed data found. Run preprocessing first.")
            return None, None
        
        logger.info("📂 Loading preprocessed data...")
        
        with open(self.cache_file, 'rb') as f:
            processed_data = pickle.load(f)
        
        with open(self.stats_file, 'r') as f:
            stats = json.load(f)
        
        logger.info(f"✅ Loaded {len(processed_data)} preprocessed records")
        return processed_data, stats
    
    def validate_preprocessing(self, processed_data):
        """Validate preprocessed data"""
        logger.info("🔍 Validating preprocessed data...")
        
        validation_results = {
            'total_records': len(processed_data),
            'valid_images': 0,
            'invalid_images': 0,
            'text_lengths': [],
            'label_distribution': {}
        }
        
        for record in processed_data:
            # Count valid images
            validation_results['valid_images'] += record['image_count']
            validation_results['invalid_images'] += len(record['images']) - record['image_count']
            
            # Text length
            validation_results['text_lengths'].append(len(record['clinical_text']))
            
            # Label distribution
            label = record['label']
            validation_results['label_distribution'][label] = validation_results['label_distribution'].get(label, 0) + 1
        
        logger.info(f"✅ Validation complete:")
        logger.info(f"   Valid images: {validation_results['valid_images']}")
        logger.info(f"   Invalid images: {validation_results['invalid_images']}")
        logger.info(f"   Avg text length: {np.mean(validation_results['text_lengths']):.1f}")
        
        return validation_results

def main():
    """Main preprocessing function"""
    print("🏥 Medical Image Classification - Step 1: Dataset Preprocessing")
    print("=" * 70)
    
    # Initialize preprocessor
    preprocessor = DatasetPreprocessor()
    
    try:
        # Check if preprocessed data already exists
        if os.path.exists(preprocessor.cache_file):
            logger.info("📂 Found existing preprocessed data. Loading...")
            processed_data, stats = preprocessor.load_preprocessed_data()
            
            if processed_data:
                # Validate existing data
                validation_results = preprocessor.validate_preprocessing(processed_data)
                
                print(f"\n✅ Preprocessed data loaded successfully!")
                print(f"📊 Records: {validation_results['total_records']}")
                print(f"🖼️  Valid images: {validation_results['valid_images']}")
                print(f"📝 Avg text length: {np.mean(validation_results['text_lengths']):.1f}")
                
                return processed_data, stats
        
        # Run preprocessing
        print("\n🚀 Starting preprocessing...")
        processed_data, stats = preprocessor.preprocess_dataset(
            sample_size=1000,  # Use sample for testing
            max_workers=8
        )
        
        # Validate results
        validation_results = preprocessor.validate_preprocessing(processed_data)
        
        print(f"\n🎉 Preprocessing completed successfully!")
        print(f"📊 Total records processed: {validation_results['total_records']}")
        print(f"🖼️  Valid images: {validation_results['valid_images']}")
        print(f"📝 Average text length: {np.mean(validation_results['text_lengths']):.1f}")
        print(f"📈 Label distribution: {validation_results['label_distribution']}")
        
        return processed_data, stats
        
    except Exception as e:
        logger.error(f"❌ Preprocessing failed: {e}")
        raise

if __name__ == "__main__":
    processed_data, stats = main()
