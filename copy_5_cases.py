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
    print("Copying 5 records from each category...")
    
    # Load CSV and get 5 records from each category
    df = pd.read_csv(CSV_PATH)
    positive_records = df[df['GLEAMER_FINDING'] == 'POSITIVE'].head(5)
    negative_records = df[df['GLEAMER_FINDING'] == 'NEGATIVE'].head(5)
    doubt_records = df[df['GLEAMER_FINDING'] == 'DOUBT'].head(5)
    
    print(f"Found {len(positive_records)} positive records")
    print(f"Found {len(negative_records)} negative records")
    print(f"Found {len(doubt_records)} doubt records")
    
    # Create output folder
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Process positive records
    for i, (_, row) in enumerate(positive_records.iterrows()):
        # Create subdirectory for each record
        record_dir = os.path.join(OUTPUT_DIR, f"positive_{i+1}")
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
        
        print(f"Positive {i+1}: Copied {copied_count} images")
    
    # Process negative records
    for i, (_, row) in enumerate(negative_records.iterrows()):
        # Create subdirectory for each record
        record_dir = os.path.join(OUTPUT_DIR, f"negative_{i+1}")
        os.makedirs(record_dir, exist_ok=True)
        
        # Extract image URLs from download_urls column
        urls_str = str(row['download_urls'])
        print(f"Negative {i+1} URLs: {urls_str[:100]}...")
        
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
        
        print(f"Negative {i+1}: Found {len(urls)} URLs")
        
        # Copy each image
        copied_count = 0
        for url in urls:
            if url and url != 'nan':
                filename = os.path.basename(url.strip())
                print(f"Looking for: {filename}")
                image_path = find_image(filename)
                if image_path:
                    shutil.copy2(image_path, os.path.join(record_dir, filename))
                    print(f"Negative {i+1}: Copied {filename}")
                    copied_count += 1
                else:
                    print(f"Negative {i+1}: NOT FOUND {filename}")
        
        print(f"Negative {i+1}: Copied {copied_count} images")
    
    # Process doubt records
    for i, (_, row) in enumerate(doubt_records.iterrows()):
        # Create subdirectory for each record
        record_dir = os.path.join(OUTPUT_DIR, f"doubt_{i+1}")
        os.makedirs(record_dir, exist_ok=True)
        
        # Extract image URLs from download_urls column
        urls_str = str(row['download_urls'])
        print(f"Doubt {i+1} URLs: {urls_str[:100]}...")
        
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
        
        print(f"Doubt {i+1}: Found {len(urls)} URLs")
        
        # Copy each image
        copied_count = 0
        for url in urls:
            if url and url != 'nan':
                filename = os.path.basename(url.strip())
                print(f"Looking for: {filename}")
                image_path = find_image(filename)
                if image_path:
                    shutil.copy2(image_path, os.path.join(record_dir, filename))
                    print(f"Doubt {i+1}: Copied {filename}")
                    copied_count += 1
                else:
                    print(f"Doubt {i+1}: NOT FOUND {filename}")
        
        print(f"Doubt {i+1}: Copied {copied_count} images")
    
    print(f"Done! Images copied to {OUTPUT_DIR}/")

if __name__ == "__main__":
    main()
