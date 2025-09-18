#!/usr/bin/env python3
"""
Test script to verify existing images can be loaded from gleamer directory
"""

import os
import pandas as pd
from config import CSV_FILE, EXISTING_IMAGES_DIR

def test_existing_images():
    """Test loading existing images from gleamer directory"""
    print("🔍 Testing Existing Images from Gleamer Directory")
    print("=" * 60)
    
    # Check if CSV file exists
    if not os.path.exists(CSV_FILE):
        print(f"❌ CSV file not found: {CSV_FILE}")
        return False
    
    print(f"✅ CSV file found: {CSV_FILE}")
    
    # Check if images directory exists
    if not os.path.exists(EXISTING_IMAGES_DIR):
        print(f"❌ Images directory not found: {EXISTING_IMAGES_DIR}")
        return False
    
    print(f"✅ Images directory found: {EXISTING_IMAGES_DIR}")
    
    # List some files in the directory
    try:
        files = os.listdir(EXISTING_IMAGES_DIR)
        image_files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        print(f"📊 Found {len(image_files)} image files in directory")
        
        if len(image_files) > 0:
            print(f"📁 Sample files:")
            for i, file in enumerate(image_files[:5]):
                print(f"   {i+1}. {file}")
            if len(image_files) > 5:
                print(f"   ... and {len(image_files) - 5} more files")
    except Exception as e:
        print(f"❌ Error listing directory: {e}")
        return False
    
    # Test CSV loading
    try:
        df = pd.read_csv(CSV_FILE)
        print(f"✅ CSV loaded successfully: {len(df)} records")
        
        # Test first few rows
        print(f"\n📊 Testing first 3 rows:")
        for i in range(min(3, len(df))):
            row = df.iloc[i]
            print(f"\n   Row {i+1}:")
            print(f"   Accession: {row['ACCESSION_NUMBER']}")
            print(f"   Label: {row['GLEAMER_FINDING']}")
            
            # Parse URLs
            try:
                urls = eval(row['download_urls'])
                print(f"   URLs: {len(urls)} images")
                
                # Test finding first image
                if len(urls) > 0:
                    first_url = urls[0]
                    filename = first_url.split('/')[-1]
                    image_path = os.path.join(EXISTING_IMAGES_DIR, filename)
                    
                    if os.path.exists(image_path):
                        print(f"   ✅ Image found: {filename}")
                    else:
                        print(f"   ❌ Image not found: {filename}")
                        # Try alternative locations
                        alt_paths = [
                            os.path.join(EXISTING_IMAGES_DIR, "images", filename),
                            os.path.join(EXISTING_IMAGES_DIR, "data", filename),
                        ]
                        found = False
                        for alt_path in alt_paths:
                            if os.path.exists(alt_path):
                                print(f"   ✅ Image found at: {alt_path}")
                                found = True
                                break
                        if not found:
                            print(f"   ❌ Image not found in any location")
                            
            except Exception as e:
                print(f"   ❌ Error parsing URLs: {e}")
                
    except Exception as e:
        print(f"❌ Error loading CSV: {e}")
        return False
    
    print(f"\n🎉 Test completed successfully!")
    print(f"✅ Ready to use existing images from: {EXISTING_IMAGES_DIR}")
    return True

if __name__ == "__main__":
    test_existing_images()
