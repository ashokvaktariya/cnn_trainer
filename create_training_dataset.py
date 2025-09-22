#!/usr/bin/env python3
"""
Create Training Dataset
Filter UID and download_urls columns to keep only X-ray images
Keep all other columns including GLEAMER_FINDING
"""

import pandas as pd
import json
import ast

def create_training_dataset(csv_path, output_path):
    """Create training dataset with filtered UIDs but all other data intact"""
    print("🏥 CREATING TRAINING DATASET")
    print("=" * 50)
    
    # Load CSV
    df = pd.read_csv(csv_path)
    print(f"📊 Original records: {len(df):,}")
    
    # Define X-ray UID patterns from README.md
    xray_patterns = [
        '1.2.840.113619',  # Standard X-rays
        '1.2.392.200036',  # Alternative X-rays  
        '1.2.840'          # General X-rays
    ]
    
    print("🎯 X-ray UID patterns to keep:")
    for pattern in xray_patterns:
        print(f"  - {pattern}")
    print()
    
    # Process each record
    filtered_records = []
    total_removed_1_2_250 = 0
    records_with_xrays = 0
    
    for idx, row in df.iterrows():
        try:
            # Parse UIDs and URLs
            sop_uids = ast.literal_eval(row['SOP_INSTANCE_UID_ARRAY'])
            download_urls = ast.literal_eval(row['download_urls'])
            
            # Filter for X-ray images only (exclude 1.2.250 and keep only X-ray patterns)
            xray_uids = []
            xray_urls = []
            
            for uid, url in zip(sop_uids, download_urls):
                # Exclude 1.2.250 UIDs (summary reports)
                if uid.startswith('1.2.250'):
                    total_removed_1_2_250 += 1
                    continue
                
                # Keep only X-ray UID patterns
                is_xray = False
                for pattern in xray_patterns:
                    if uid.startswith(pattern):
                        is_xray = True
                        break
                
                if is_xray:
                    xray_uids.append(uid)
                    xray_urls.append(url)
            
            # Only keep records with X-ray images
            if xray_uids:
                new_row = row.copy()
                new_row['SOP_INSTANCE_UID_ARRAY'] = json.dumps(xray_uids)
                new_row['download_urls'] = json.dumps(xray_urls)
                filtered_records.append(new_row)
                records_with_xrays += 1
                
        except Exception as e:
            print(f"Error processing record {idx}: {e}")
    
    # Create filtered DataFrame
    filtered_df = pd.DataFrame(filtered_records)
    
    print(f"📊 Filtered records: {len(filtered_df):,}")
    print(f"📊 Records removed: {len(df) - len(filtered_df):,}")
    print(f"📊 1.2.250 UIDs removed: {total_removed_1_2_250:,}")
    print(f"📊 Records with X-rays: {records_with_xrays:,}")
    
    # Save filtered CSV
    filtered_df.to_csv(output_path, index=False)
    print(f"💾 Saved to: {output_path}")
    
    # Generate summary
    generate_summary(filtered_df, xray_patterns)
    
    return filtered_df

def generate_summary(df, xray_patterns):
    """Generate filtering summary"""
    print("\n📊 TRAINING DATASET SUMMARY")
    print("=" * 40)
    
    # Check GLEAMER_FINDING distribution
    if 'GLEAMER_FINDING' in df.columns:
        finding_dist = df['GLEAMER_FINDING'].value_counts()
        print("🏥 GLEAMER_FINDING Distribution:")
        for finding, count in finding_dist.items():
            percentage = (count / len(df)) * 100
            print(f"  {finding}: {count:,} ({percentage:.1f}%)")
        print()
    
    # Count X-ray images by pattern
    total_images = 0
    uid_patterns = {}
    
    for idx, row in df.iterrows():
        try:
            sop_uids = ast.literal_eval(row['SOP_INSTANCE_UID_ARRAY'])
            total_images += len(sop_uids)
            
            for uid in sop_uids:
                if uid.startswith('1.2.840.113619'):
                    uid_patterns['1.2.840.113619'] = uid_patterns.get('1.2.840.113619', 0) + 1
                elif uid.startswith('1.2.392.200036'):
                    uid_patterns['1.2.392.200036'] = uid_patterns.get('1.2.392.200036', 0) + 1
                elif uid.startswith('1.2.840'):
                    uid_patterns['1.2.840'] = uid_patterns.get('1.2.840', 0) + 1
                else:
                    uid_patterns['other'] = uid_patterns.get('other', 0) + 1
                    
        except:
            pass
    
    print(f"📊 Total X-ray images: {total_images:,}")
    print(f"📊 Average images per record: {total_images/len(df):.1f}")
    print()
    
    print("🎯 X-ray UID Pattern Distribution:")
    for pattern in xray_patterns:
        count = uid_patterns.get(pattern, 0)
        percentage = (count / total_images) * 100 if total_images > 0 else 0
        print(f"  {pattern}: {count:,} ({percentage:.1f}%)")
    
    other_count = uid_patterns.get('other', 0)
    if other_count > 0:
        other_percentage = (other_count / total_images) * 100 if total_images > 0 else 0
        print(f"  other: {other_count:,} ({other_percentage:.1f}%)")

def main():
    """Main function"""
    print("🏥 CREATE TRAINING DATASET")
    print("=" * 50)
    
    csv_path = "processed_dicom_image_url_file.csv"
    output_path = "training_dataset.csv"
    
    print(f"📄 Input: {csv_path}")
    print(f"📄 Output: {output_path}")
    print()
    
    try:
        filtered_df = create_training_dataset(csv_path, output_path)
        print("\n✅ Training dataset created successfully!")
        print(f"📊 Final dataset: {len(filtered_df):,} records")
        print("✅ All columns preserved including GLEAMER_FINDING")
        print("✅ Only X-ray UIDs kept (1.2.840.113619, 1.2.392.200036, 1.2.840)")
        print("✅ All 1.2.250 UIDs (summary reports) removed")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
