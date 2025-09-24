#!/usr/bin/env python3
import os
import csv
import pandas as pd
from pathlib import Path
import pydicom

def extract_dicom_to_csv():
    """Extract DICOM metadata from .dcm files and save to CSV"""
    
    # Paths
    dicom_path = "/mount/civiescaks01storage01/aksfileshare01/CNN/gleamer-dicom/"
    output_csv = "dicom_metadata.csv"
    
    print(f"Scanning DICOM files in: {dicom_path}")
    
    # Find all .dcm files
    dicom_files = []
    for root, dirs, files in os.walk(dicom_path):
        for file in files:
            if file.lower().endswith('.dcm'):
                dicom_files.append(os.path.join(root, file))
    
    print(f"Found {len(dicom_files)} .dcm files")
    
    # Extract metadata from each file
    all_data = []
    
    for i, file_path in enumerate(dicom_files):
        print(f"Processing {i+1}/{len(dicom_files)}: {os.path.basename(file_path)}")
        
        try:
            # Read DICOM file without pixel data for speed, force=True to handle missing headers
            ds = pydicom.dcmread(file_path, stop_before_pixels=True, force=True)
            
            # Convert metadata to dictionary using the method you mentioned
            metadata_dict = {elem.keyword: elem.value for elem in ds if elem.keyword}
            
            # Extract common metadata with fallback to empty string
            metadata = {
                'file_path': file_path,
                'file_name': os.path.basename(file_path),
                'file_size': os.path.getsize(file_path),
                'PatientID': str(metadata_dict.get('PatientID', '')),
                'PatientName': str(metadata_dict.get('PatientName', '')),
                'StudyInstanceUID': str(metadata_dict.get('StudyInstanceUID', '')),
                'SeriesInstanceUID': str(metadata_dict.get('SeriesInstanceUID', '')),
                'SOPInstanceUID': str(metadata_dict.get('SOPInstanceUID', '')),
                'StudyDate': str(metadata_dict.get('StudyDate', '')),
                'StudyTime': str(metadata_dict.get('StudyTime', '')),
                'Modality': str(metadata_dict.get('Modality', '')),
                'BodyPartExamined': str(metadata_dict.get('BodyPartExamined', '')),
                'Manufacturer': str(metadata_dict.get('Manufacturer', '')),
                'Rows': str(metadata_dict.get('Rows', '')),
                'Columns': str(metadata_dict.get('Columns', '')),
                'PixelSpacing': str(metadata_dict.get('PixelSpacing', '')),
                'SliceThickness': str(metadata_dict.get('SliceThickness', '')),
                'WindowCenter': str(metadata_dict.get('WindowCenter', '')),
                'WindowWidth': str(metadata_dict.get('WindowWidth', '')),
                'InstitutionName': str(metadata_dict.get('InstitutionName', '')),
                'SeriesDescription': str(metadata_dict.get('SeriesDescription', '')),
                'StudyDescription': str(metadata_dict.get('StudyDescription', ''))
            }
            
            all_data.append(metadata)
            
        except Exception as e:
            print(f"Error reading {file_path}: {str(e)}")
            # Add error record
            all_data.append({
                'file_path': file_path,
                'file_name': os.path.basename(file_path),
                'file_size': os.path.getsize(file_path) if os.path.exists(file_path) else 0,
                'error': str(e)
            })
    
    # Save to CSV
    if all_data:
        df = pd.DataFrame(all_data)
        df.to_csv(output_csv, index=False)
        print(f"✅ Data saved to {output_csv}")
        print(f"Total records: {len(df)}")
    else:
        print("❌ No data to save")

if __name__ == "__main__":
    extract_dicom_to_csv()
