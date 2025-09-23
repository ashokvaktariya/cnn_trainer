#!/usr/bin/env python3
import pandas as pd
import os
import shutil
import json

# Paths
CSV_PATH = "processed_dicom_image_url_file.csv"
IMAGE_ROOT = "/mount/civiescaks01storage01/aksfileshare01/CNN/gleamer-images/"
OUTPUT_DIR = "positive_images"

def find_image(filename):
    for root, dirs, files in os.walk(IMAGE_ROOT):
        if filename in files:
            return os.path.join(root, filename)
    return None

def main():
    print("Copying 5 positive records...")
    
    # Load CSV and get 5 positive records
    df = pd.read_csv(CSV_PATH)
    positive_records = df[df['GLEAMER_FINDING'] == 'POSITIVE'].head(5)
    print(f"Found {len(positive_records)} positive records")
    
    # Create output folder
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    for i, (_, row) in enumerate(positive_records.iterrows()):
        # Create subdirectory for each record
        record_dir = os.path.join(OUTPUT_DIR, f"record_{i+1}")
        os.makedirs(record_dir, exist_ok=True)
        
        # Extract image URLs from download_urls column
        urls_str = str(row['download_urls'])
        print(f"Record {i+1} URLs: {urls_str[:100]}...")
        
        try:
            # Try to parse as Python list literal
            urls = eval(urls_str)
            if not isinstance(urls, list):
                urls = [urls]
        except:
            try:
                # Try JSON parsing
                urls = json.loads(urls_str)
                if not isinstance(urls, list):
                    urls = [urls]
            except:
                urls = [urls_str]
        
        print(f"Record {i+1}: Found {len(urls)} URLs")
        
        # Copy each image
        copied_count = 0
        for url in urls:
            if url and url != 'nan':
                filename = os.path.basename(url.strip())
                print(f"Looking for: {filename}")
                image_path = find_image(filename)
                if image_path:
                    shutil.copy2(image_path, os.path.join(record_dir, filename))
                    print(f"Record {i+1}: Copied {filename}")
                    copied_count += 1
                else:
                    print(f"Record {i+1}: NOT FOUND {filename}")
        
        print(f"Record {i+1}: Copied {copied_count} images")
    
    print(f"Done! Images copied to {OUTPUT_DIR}/")

if __name__ == "__main__":
    main()
