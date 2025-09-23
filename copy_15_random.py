#!/usr/bin/env python3
import pandas as pd
import os
import shutil
import json
import random

# Paths
CSV_PATH = "processed_dicom_image_url_file.csv"
IMAGE_ROOT = "/mount/civiescaks01storage01/aksfileshare01/CNN/gleamer-dicom/"
OUTPUT_DIR = "/"

def find_image(filename):
    for root, dirs, files in os.walk(IMAGE_ROOT):
        if filename in files:
            return os.path.join(root, filename)
    return None

def main():
    print("Copying 15 random samples to root...")
    
    # Load CSV and get 15 random records
    df = pd.read_csv(CSV_PATH)
    random_records = df.sample(n=15, random_state=42)
    print(f"Selected {len(random_records)} random records")
    
    copied_count = 0
    
    for i, (_, row) in enumerate(random_records.iterrows()):
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
        
        # Copy each image to root directory
        for url in urls:
            if url and url != 'nan':
                filename = os.path.basename(url.strip())
                print(f"Looking for: {filename}")
                image_path = find_image(filename)
                if image_path:
                    shutil.copy2(image_path, os.path.join(OUTPUT_DIR, filename))
                    print(f"Record {i+1}: Copied {filename}")
                    copied_count += 1
                else:
                    print(f"Record {i+1}: NOT FOUND {filename}")
    
    print(f"Done! Copied {copied_count} images to root directory")

if __name__ == "__main__":
    main()
