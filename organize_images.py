#!/usr/bin/env python3
"""
Medical Image Organizer
Reads CSV, extracts image names from URLs, copies images to organized folders
Excludes summary images (1.2.250.1 pattern)
"""

import pandas as pd
import os
import shutil
import ast
from pathlib import Path
from urllib.parse import urlparse
from tqdm import tqdm

def main():
    # Configuration
    csv_file = "processed_dicom_image_url_file.csv"
    mount_path = "/mount/civiescaks01storage01/aksfileshare01/CNN/gleamer-images/"
    output_dir = "filtered_images"
    
    print("🚀 Medical Image Organizer")
    print("=" * 50)
    
    # Create output folders
    positive_dir = Path(output_dir) / "positive_images"
    negative_dir = Path(output_dir) / "negative_images"
    doubt_dir = Path(output_dir) / "doubt_images"
    
    for folder in [positive_dir, negative_dir, doubt_dir]:
        folder.mkdir(parents=True, exist_ok=True)
        print(f"📁 Created: {folder}")
    
    # Load CSV
    print(f"📊 Loading CSV: {csv_file}")
    df = pd.read_csv(csv_file)
    print(f"✅ Loaded {len(df):,} records")
    
    # Statistics
    stats = {'positive': 0, 'negative': 0, 'doubt': 0, 'excluded': 0, 'not_found': 0}
    
    # Process each case
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
        case_id = f"case_{idx:06d}"
        finding = row['GLEAMER_FINDING']
        
        # Parse URLs
        urls_str = row['download_urls']
        try:
            urls = ast.literal_eval(urls_str)
        except:
            urls = [url.strip().strip("'\"") for url in urls_str.split(',')]
        
        if not isinstance(urls, list):
            urls = [urls]
        
        # Choose target folder
        if finding == 'POSITIVE':
            target_dir = positive_dir
        elif finding == 'NEGATIVE':
            target_dir = negative_dir
        elif finding == 'DOUBT':
            target_dir = doubt_dir
        else:
            continue
        
        # Process each image
        image_index = 0
        for url in urls:
            if not url or url == 'nan':
                continue
            
            # Extract filename from URL
            filename = os.path.basename(urlparse(url).path)
            if not filename:
                continue
            
            # Skip summary images (1.2.250.1 pattern)
            uid = filename.replace('.jpg', '').replace('.jpeg', '').replace('.png', '')
            if uid.startswith('1.2.250.1'):
                stats['excluded'] += 1
                continue
            
            # Find image in mount location
            source_path = Path(mount_path) / filename
            if not source_path.exists():
                stats['not_found'] += 1
                continue
            
            # Keep original filename
            target_filename = filename
            target_path = target_dir / target_filename
            
            # Skip if already exists
            if target_path.exists():
                image_index += 1
                continue
            
            # Copy image
            try:
                shutil.copy2(source_path, target_path)
                stats[finding.lower()] += 1
                image_index += 1
            except Exception as e:
                print(f"❌ Error copying {filename}: {e}")
    
    # Print results
    print("\n📊 RESULTS:")
    print(f"   Positive images: {stats['positive']:,}")
    print(f"   Negative images: {stats['negative']:,}")
    print(f"   Doubt images: {stats['doubt']:,}")
    print(f"   Summary images excluded: {stats['excluded']:,}")
    print(f"   Images not found: {stats['not_found']:,}")
    print(f"   Total copied: {sum(stats.values()) - stats['excluded'] - stats['not_found']:,}")
    
    print(f"\n✅ Images organized in: {output_dir}/")
    print("🎉 Done!")

if __name__ == "__main__":
    main()
