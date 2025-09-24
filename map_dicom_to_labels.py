#!/usr/bin/env python3
import pandas as pd

def create_training_csv():
    """Create training CSV with filename and labels from DICOM metadata"""
    
    # File paths
    dicom_metadata_file = "test/dicom_metadata.csv"
    output_file = "training_data_with_labels.csv"
    simplified_output = "filename_labels.csv"
    
    print("📖 Reading DICOM metadata...")
    dicom_df = pd.read_csv(dicom_metadata_file, low_memory=False)
    print(f"DICOM records: {len(dicom_df)}")
    
    # Extract labels from SeriesDescription
    print("🎯 Extracting labels from SeriesDescription...")
    mapped_data = []
    
    for idx, row in dicom_df.iterrows():
        # Get label from SeriesDescription
        series_desc = str(row.get('SeriesDescription', ''))
        
        if 'POSITIVE' in series_desc.upper():
            label = 'POSITIVE'
        elif 'NEGATIVE' in series_desc.upper():
            label = 'NEGATIVE'
        else:
            label = 'UNKNOWN'
        
        # Create record
        record = {
            'file_path': row['file_path'],
            'file_name': row['file_name'],
            'label': label,
            'study_description': row.get('StudyDescription', ''),
            'body_part': row.get('BodyPartExamined', ''),
            'modality': row.get('Modality', ''),
            'manufacturer': row.get('Manufacturer', ''),
            'sop_instance_uid': row.get('SOPInstanceUID', ''),
            'study_instance_uid': row.get('StudyInstanceUID', ''),
            'series_description': series_desc,
            'patient_id': row.get('PatientID', ''),
            'study_date': row.get('StudyDate', ''),
            'accession_number': row.get('AccessionNumber', '')
        }
        
        mapped_data.append(record)
    
    # Create DataFrame
    mapped_df = pd.DataFrame(mapped_data)
    
    print(f"✅ Processed {len(mapped_df)} records")
    
    # Show label distribution
    label_counts = mapped_df['label'].value_counts()
    print("\n📊 Label Distribution:")
    for label, count in label_counts.items():
        print(f"  {label}: {count}")
    
    # Save full dataset
    mapped_df.to_csv(output_file, index=False)
    print(f"💾 Saved full dataset to: {output_file}")
    
    # Create simplified version with just filename and label
    simplified_df = mapped_df[['file_name', 'label']].copy()
    simplified_df.to_csv(simplified_output, index=False)
    print(f"💾 Saved simplified dataset to: {simplified_output}")
    
    # Show sample of simplified data
    print("\n📋 Sample of simplified data:")
    print(simplified_df.head(10))
    
    return mapped_df, simplified_df

if __name__ == "__main__":
    mapped_df, simplified_df = create_training_csv()
