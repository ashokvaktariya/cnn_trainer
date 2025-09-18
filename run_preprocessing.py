#!/usr/bin/env python3
"""
Preprocessing Runner Script
Run preprocessing with existing images from gleamer directory
"""

import argparse
import sys
import os
from step1_preprocessing import DatasetPreprocessor

def main():
    """Main preprocessing function"""
    parser = argparse.ArgumentParser(description='Run preprocessing with existing images')
    parser.add_argument('--sample-size', type=int, default=None, 
                       help='Number of records to process (default: all)')
    parser.add_argument('--batch-size', type=int, default=1000,
                       help='Batch size for processing')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory for preprocessed data')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    print("🚀 Starting Preprocessing with Existing Images")
    print("=" * 50)
    
    try:
        # Initialize preprocessor
        print("🔧 Initializing preprocessor...")
        preprocessor = DatasetPreprocessor()
        
        # Load dataset
        print("📊 Loading dataset...")
        df = preprocessor.load_dataset()
        
        if len(df) == 0:
            print("❌ No data loaded. Check CSV file and paths.")
            return 1
        
        print(f"✅ Loaded {len(df)} records")
        
        # Apply sample size if specified
        if args.sample_size and args.sample_size < len(df):
            df = df.head(args.sample_size)
            print(f"📝 Processing sample of {len(df)} records")
        
        # Run preprocessing
        print("🔄 Starting preprocessing...")
        processed_data, stats = preprocessor.preprocess_dataset(
            df, 
            batch_size=args.batch_size
        )
        
        # Print results
        print("\n📈 Preprocessing Results:")
        print("=" * 30)
        print(f"   Total Records: {stats['total_records']}")
        print(f"   Total Images: {stats['total_images']}")
        print(f"   Avg Images per Record: {stats['avg_images_per_record']:.2f}")
        print(f"   Label Distribution: {stats['label_distribution']}")
        print(f"   Positive Ratio: {stats['positive_ratio']:.3f}")
        
        if 'body_part_distribution' in stats:
            print(f"   Top Body Parts: {list(stats['body_part_distribution'].keys())[:5]}")
        
        print(f"\n✅ Preprocessing completed successfully!")
        print(f"💾 Data saved to: {preprocessor.preprocessed_dir}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Preprocessing failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
