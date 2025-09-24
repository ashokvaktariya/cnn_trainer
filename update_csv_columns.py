import pandas as pd
import os

print("=== UPDATING CSV COLUMN NAMES ===")

# List of CSV files to update
csv_files = [
    "final_dataset_cnn.csv",
    "final_dataset_cnn_train.csv", 
    "final_dataset_cnn_val.csv"
]

for csv_file in csv_files:
    if os.path.exists(csv_file):
        print(f"\n📊 Processing {csv_file}...")
        
        # Read the CSV file
        df = pd.read_csv(csv_file)
        print(f"   Original columns: {list(df.columns)}")
        
        # Check if gleamer_finding column exists
        if 'gleamer_finding' in df.columns:
            # Rename gleamer_finding to label
            df = df.rename(columns={'gleamer_finding': 'label'})
            print(f"   ✅ Renamed 'gleamer_finding' to 'label'")
        else:
            print(f"   ⚠️ 'gleamer_finding' column not found")
        
        # Also rename jpg_filename to image_path for consistency
        if 'jpg_filename' in df.columns:
            df = df.rename(columns={'jpg_filename': 'image_path'})
            print(f"   ✅ Renamed 'jpg_filename' to 'image_path'")
        
        # Update file_path column to match image_path if it exists
        if 'file_path' in df.columns and 'image_path' in df.columns:
            df['file_path'] = df['image_path']
            print(f"   ✅ Updated 'file_path' to match 'image_path'")
        
        print(f"   Updated columns: {list(df.columns)}")
        
        # Save the updated CSV
        df.to_csv(csv_file, index=False)
        print(f"   💾 Saved updated {csv_file}")
        
        # Show label distribution
        if 'label' in df.columns:
            label_counts = df['label'].value_counts()
            print(f"   📊 Label distribution:")
            for label, count in label_counts.items():
                label_name = 'NEGATIVE' if label == 'NEGATIVE' else 'POSITIVE'
                percentage = (count / len(df)) * 100
                print(f"      {label_name}: {count:,} ({percentage:.1f}%)")
    else:
        print(f"   ❌ File {csv_file} not found")

print(f"\n=== COLUMN UPDATE COMPLETE ===")
